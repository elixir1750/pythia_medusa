from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MedusaLossOutput:
    loss: torch.Tensor
    per_head_losses: list[torch.Tensor]
    per_head_accuracies: list[float]
    per_head_valid_tokens: list[int]


def compute_medusa_loss(
    medusa_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    head_weights: list[float] | None = None,
) -> MedusaLossOutput:
    num_heads, batch_size, sequence_length, vocab_size = medusa_logits.shape
    if labels.shape != (batch_size, sequence_length):
        raise ValueError(
            "Labels must align with medusa logits sequence shape: "
            f"expected {(batch_size, sequence_length)}, got {tuple(labels.shape)}"
        )

    weights = head_weights or [1.0] * num_heads
    if len(weights) != num_heads:
        raise ValueError("head_weights length must match medusa logits head dimension.")

    per_head_losses: list[torch.Tensor] = []
    per_head_accuracies: list[float] = []
    per_head_valid_tokens: list[int] = []
    weighted_loss_sum = medusa_logits.new_zeros((), dtype=torch.float32)
    total_weight = 0.0

    for head_index in range(num_heads):
        offset = head_index + 1
        if sequence_length <= offset:
            head_loss = medusa_logits.new_zeros((), dtype=torch.float32)
            per_head_losses.append(head_loss)
            per_head_accuracies.append(0.0)
            per_head_valid_tokens.append(0)
            continue

        head_logits = medusa_logits[head_index, :, : sequence_length - offset, :]
        head_targets = labels[:, offset:]
        valid_mask = head_targets != ignore_index
        valid_tokens = int(valid_mask.sum().item())

        if valid_tokens == 0:
            head_loss = medusa_logits.new_zeros((), dtype=torch.float32)
            head_accuracy = 0.0
        else:
            flat_logits = head_logits.reshape(-1, vocab_size).float()
            flat_targets = head_targets.reshape(-1)
            head_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=ignore_index,
                reduction="sum",
            ) / valid_tokens
            predictions = head_logits.argmax(dim=-1)
            correct = ((predictions == head_targets) & valid_mask).sum().item()
            head_accuracy = float(correct) / float(valid_tokens)
            weighted_loss_sum = weighted_loss_sum + head_loss * weights[head_index]
            total_weight += float(weights[head_index])

        per_head_losses.append(head_loss)
        per_head_accuracies.append(head_accuracy)
        per_head_valid_tokens.append(valid_tokens)

    total_loss = weighted_loss_sum / total_weight if total_weight > 0 else weighted_loss_sum
    return MedusaLossOutput(
        loss=total_loss,
        per_head_losses=per_head_losses,
        per_head_accuracies=per_head_accuracies,
        per_head_valid_tokens=per_head_valid_tokens,
    )
