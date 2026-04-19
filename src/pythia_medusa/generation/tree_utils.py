from __future__ import annotations

from dataclasses import dataclass

import torch

TREE_MASK_DOC = """\
Simplified Medusa tree semantics used in this repository.

- Node 0 is the root and represents the currently verified sequence state.
- Every candidate node attends to itself.
- Every candidate node attends to the root.
- Every candidate node attends to its ancestors.
- Sibling or unrelated branches are masked from each other.

The resulting attention mask has shape [num_nodes, num_nodes], where True means
"this query node can attend to that key node".

Phase 3 uses a correctness-first linear tree path, but the helpers here keep the
ancestor / sibling semantics explicit so a fuller GPTNeoX tree implementation
can be swapped in later.
"""


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    future_offset: int


def build_linear_medusa_tree(num_future_steps: int) -> list[TreeNode]:
    nodes = [TreeNode(node_id=0, parent_id=None, depth=0, future_offset=0)]
    parent_id = 0
    for index in range(1, num_future_steps + 1):
        nodes.append(
            TreeNode(
                node_id=index,
                parent_id=parent_id,
                depth=index,
                future_offset=index,
            )
        )
        parent_id = index
    return nodes


def build_tree_attention_mask(nodes: list[TreeNode]) -> torch.Tensor:
    num_nodes = len(nodes)
    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    node_map = {node.node_id: node for node in nodes}

    for query_node in nodes:
        visible = {query_node.node_id, 0}
        current_parent = query_node.parent_id
        while current_parent is not None:
            visible.add(current_parent)
            current_parent = node_map[current_parent].parent_id
        for key_node_id in visible:
            mask[query_node.node_id, key_node_id] = True
    return mask


def build_linear_tree_attention_mask(num_future_steps: int) -> torch.Tensor:
    return build_tree_attention_mask(build_linear_medusa_tree(num_future_steps))


def build_tree_verification_visibility(
    *,
    prefix_attention_mask: torch.Tensor,
    nodes: list[TreeNode],
) -> torch.Tensor:
    if prefix_attention_mask.dim() != 2 or prefix_attention_mask.size(0) != 1:
        raise ValueError(
            "Tree verification currently expects prefix_attention_mask with shape [1, prefix_len]."
        )
    if not nodes or nodes[0].node_id != 0:
        raise ValueError("Tree verification requires nodes to start with the root node 0.")

    prefix_length = int(prefix_attention_mask.shape[-1])
    candidate_nodes = nodes[1:]
    total_length = prefix_length + len(candidate_nodes)
    visibility = torch.zeros(total_length, total_length, dtype=torch.bool, device=prefix_attention_mask.device)

    # Prefix tokens retain ordinary causal visibility and never attend to candidate nodes.
    prefix_visible = torch.tril(
        torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=prefix_attention_mask.device)
    )
    prefix_valid = prefix_attention_mask[0].to(dtype=torch.bool)
    prefix_visible = prefix_visible & prefix_valid.unsqueeze(0) & prefix_valid.unsqueeze(1)
    visibility[:prefix_length, :prefix_length] = prefix_visible

    node_map = {node.node_id: node for node in nodes}
    node_to_seq_index = {
        node.node_id: prefix_length + index for index, node in enumerate(candidate_nodes)
    }

    for node in candidate_nodes:
        seq_index = node_to_seq_index[node.node_id]
        visibility[seq_index, :prefix_length] = prefix_valid

        current_id: int | None = node.node_id
        while current_id is not None and current_id != 0:
            ancestor_index = node_to_seq_index[current_id]
            visibility[seq_index, ancestor_index] = True
            current_id = node_map[current_id].parent_id

    return visibility


def build_tree_verification_attention_mask(
    *,
    prefix_attention_mask: torch.Tensor,
    nodes: list[TreeNode],
    dtype: torch.dtype,
) -> torch.Tensor:
    visibility = build_tree_verification_visibility(
        prefix_attention_mask=prefix_attention_mask,
        nodes=nodes,
    )
    additive_mask = torch.full(
        (1, 1, visibility.size(0), visibility.size(1)),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=visibility.device,
    )
    additive_mask.masked_fill_(visibility.unsqueeze(0).unsqueeze(0), 0.0)
    return additive_mask
