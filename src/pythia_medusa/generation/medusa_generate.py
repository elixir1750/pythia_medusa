from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from pythia_medusa.config import DEFAULT_MODEL_NAME
from pythia_medusa.data.prompt_sets import PromptExample, resolve_prompts
from pythia_medusa.generation.candidate_utils import CandidateBundle, build_candidate_bundle
from pythia_medusa.generation.posterior_utils import AcceptanceResult, compute_acceptance
from pythia_medusa.generation.tree_utils import (
    TreeNode,
    build_linear_medusa_tree,
    build_tree_verification_attention_mask,
)
from pythia_medusa.models import MedusaModel
from pythia_medusa.utils.io import write_structured_output


@dataclass(frozen=True)
class MedusaGenerationConfig:
    model_name: str = DEFAULT_MODEL_NAME
    checkpoint_path: str | None = None
    max_new_tokens: int = 64
    greedy: bool = True
    temperature: float = 0.0
    device: str | None = None
    medusa_num_heads: int = 3
    medusa_num_layers: int = 1
    verify_mode: str = "tree"


def infer_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def load_medusa_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_path: str | None = None,
    medusa_num_heads: int = 3,
    medusa_num_layers: int = 1,
    device: str | None = None,
) -> tuple[MedusaModel, Any, str]:
    runtime_device = infer_device(device)
    if checkpoint_path:
        model = MedusaModel.from_medusa_checkpoint(checkpoint_path)
    else:
        model = MedusaModel.from_pretrained(
            model_name,
            medusa_num_heads=medusa_num_heads,
            medusa_num_layers=medusa_num_layers,
        )
    model = model.to(runtime_device)
    model.eval()
    tokenizer = model.get_tokenizer()
    if tokenizer is None:
        raise ValueError("MedusaModel must expose a tokenizer for generation.")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, runtime_device


def _ensure_2d_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids with shape [batch, seq], got {tuple(input_ids.shape)}")
    if input_ids.size(0) != 1:
        raise ValueError("This correctness-first Medusa generator currently supports batch size 1.")
    return input_ids


def _append_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    append_tensor = torch.tensor([token_ids], dtype=input_ids.dtype, device=input_ids.device)
    append_mask = torch.ones_like(append_tensor, dtype=attention_mask.dtype, device=attention_mask.device)
    return (
        torch.cat([input_ids, append_tensor], dim=-1),
        torch.cat([attention_mask, append_mask], dim=-1),
    )


def _build_tree_verify_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_token_ids: list[int],
    *,
    nodes: list[TreeNode],
    model: MedusaModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded_input_ids, expanded_attention_mask = _append_token_ids(
        input_ids,
        attention_mask,
        candidate_token_ids,
    )
    reference = next(model.base_model.parameters(), None)
    mask_dtype = reference.dtype if reference is not None else torch.float32
    tree_attention_mask = build_tree_verification_attention_mask(
        prefix_attention_mask=attention_mask,
        nodes=nodes,
        dtype=mask_dtype,
    )
    position_ids = torch.arange(
        expanded_input_ids.shape[-1],
        device=expanded_input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    return expanded_input_ids, tree_attention_mask, position_ids


def _build_verify_trace(
    candidate_token_ids: list[int],
    verified_token_ids: list[int],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for step, candidate_token_id in enumerate(candidate_token_ids, start=1):
        verified_token_id = verified_token_ids[step - 1] if step - 1 < len(verified_token_ids) else None
        trace.append(
            {
                "step": step,
                "candidate_token_id": candidate_token_id,
                "verified_token_id": verified_token_id,
                "matched": candidate_token_id == verified_token_id,
            }
        )
        if candidate_token_id != verified_token_id:
            break
    return trace


def _verify_candidate_prefix(
    model: MedusaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_bundle: CandidateBundle,
) -> tuple[list[int], list[dict[str, Any]]]:
    working_input_ids = input_ids
    working_attention_mask = attention_mask
    verified_token_ids: list[int] = []
    trace: list[dict[str, Any]] = []

    for step, candidate_token_id in enumerate(candidate_bundle.candidate_token_ids, start=1):
        base_outputs = model(
            input_ids=working_input_ids,
            attention_mask=working_attention_mask,
        )
        next_logits = base_outputs.logits[:, -1, :][0]
        verified_token_id = int(torch.argmax(next_logits).item())
        verified_token_ids.append(verified_token_id)
        trace.append(
            {
                "step": step,
                "candidate_token_id": candidate_token_id,
                "verified_token_id": verified_token_id,
                "matched": candidate_token_id == verified_token_id,
            }
        )
        if candidate_token_id != verified_token_id:
            break
        working_input_ids, working_attention_mask = _append_token_ids(
            working_input_ids,
            working_attention_mask,
            [candidate_token_id],
        )
    return verified_token_ids, trace


def _verify_candidate_prefix_tree(
    model: MedusaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_bundle: CandidateBundle,
) -> tuple[list[int], list[dict[str, Any]]]:
    candidate_token_ids = candidate_bundle.candidate_token_ids
    if not candidate_token_ids:
        return [], []

    tree_nodes = build_linear_medusa_tree(len(candidate_token_ids))
    verify_input_ids, tree_attention_mask, position_ids = _build_tree_verify_inputs(
        input_ids,
        attention_mask,
        candidate_token_ids,
        nodes=tree_nodes,
        model=model,
    )
    base_outputs = model(
        input_ids=verify_input_ids,
        attention_mask=tree_attention_mask,
        position_ids=position_ids,
    )
    logits = base_outputs.logits[0]
    prefix_length = int(input_ids.shape[-1])
    verified_token_ids = [
        int(torch.argmax(logits[prefix_length - 1 + step]).item())
        for step in range(len(candidate_token_ids))
    ]
    return verified_token_ids, _build_verify_trace(candidate_token_ids, verified_token_ids)


def _verify_candidate_bundle(
    model: MedusaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_bundle: CandidateBundle,
    *,
    verify_mode: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    if verify_mode == "serial":
        return _verify_candidate_prefix(model, input_ids, attention_mask, candidate_bundle)
    if verify_mode == "tree":
        return _verify_candidate_prefix_tree(model, input_ids, attention_mask, candidate_bundle)
    raise ValueError(f"Unsupported verify_mode: {verify_mode}")


def medusa_generate(
    model: MedusaModel,
    tokenizer: Any,
    prompt: str,
    *,
    config: MedusaGenerationConfig,
    device: str | None = None,
) -> dict[str, Any]:
    runtime_device = infer_device(device or config.device)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = _ensure_2d_input_ids(encoded["input_ids"].to(runtime_device))
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(runtime_device)

    prompt_token_count = int(input_ids.shape[-1])
    generated_token_count = 0
    acceptance_lengths: list[int] = []
    rounds: list[dict[str, Any]] = []
    tree_nodes = build_linear_medusa_tree(config.medusa_num_heads)

    start_time = time.perf_counter()
    while generated_token_count < config.max_new_tokens:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            medusa_forward=True,
            output_orig=True,
        )
        baseline_logits = outputs.logits[:, -1, :][0]
        medusa_logits = outputs.medusa_logits[:, 0, -1, :]
        candidate_bundle = build_candidate_bundle(
            baseline_logits,
            medusa_logits,
            greedy=config.greedy,
            temperature=config.temperature if not config.greedy else 0.0,
        )
        verified_token_ids, verify_trace = _verify_candidate_bundle(
            model,
            input_ids,
            attention_mask,
            candidate_bundle,
            verify_mode=config.verify_mode,
        )
        acceptance = compute_acceptance(
            candidate_bundle.candidate_token_ids,
            verified_token_ids,
            candidate_bundle.baseline_token_id,
        )
        remaining_budget = config.max_new_tokens - generated_token_count
        tokens_to_append = acceptance.tokens_to_append[:remaining_budget]
        input_ids, attention_mask = _append_token_ids(input_ids, attention_mask, tokens_to_append)

        generated_token_count += len(tokens_to_append)
        acceptance_lengths.append(acceptance.accepted_length)
        rounds.append(
            {
                "round_index": len(rounds) + 1,
                "baseline_token_id": candidate_bundle.baseline_token_id,
                "candidate_token_ids": candidate_bundle.candidate_token_ids,
                "verified_token_ids": acceptance.verified_token_ids,
                "accepted_length": acceptance.accepted_length,
                "tokens_appended": tokens_to_append,
                "tree_depth": len(tree_nodes) - 1,
                "verify_mode": config.verify_mode,
                "verify_trace": verify_trace,
            }
        )

        if not tokens_to_append:
            break
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and tokens_to_append[-1] == eos_token_id:
            break

    latency_sec = time.perf_counter() - start_time
    generated_ids = input_ids[:, prompt_token_count:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    histogram = Counter(acceptance_lengths)

    return {
        "model": config.model_name,
        "checkpoint_path": config.checkpoint_path,
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": int(generated_ids.shape[-1]),
        "total_tokens": int(input_ids.shape[-1]),
        "latency_sec": latency_sec,
        "tokens_per_sec": float(generated_ids.shape[-1]) / latency_sec if latency_sec > 0 else 0.0,
        "rounds": len(rounds),
        "accept_lengths": acceptance_lengths,
        "average_accept_length": sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0,
        "accept_length_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "tree_nodes": [node.__dict__ for node in tree_nodes],
        "round_traces": rounds,
        "greedy": config.greedy,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "verify_mode": config.verify_mode,
    }


def medusa_generate_from_prompts(
    model: MedusaModel,
    tokenizer: Any,
    prompts: list[PromptExample],
    *,
    config: MedusaGenerationConfig,
    device: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        result = medusa_generate(model, tokenizer, prompt.text, config=config, device=device)
        result.update(
            {
                "bucket": prompt.bucket,
                "name": prompt.name,
                "source": prompt.source,
            }
        )
        rows.append(result)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run correctness-first Medusa generation.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-set")
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--medusa-num-heads", type=int, default=3)
    parser.add_argument("--medusa-num-layers", type=int, default=1)
    parser.add_argument("--verify-mode", choices=("tree", "serial"), default="tree")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = resolve_prompts(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        prompt_set=args.prompt_set,
        limit=args.limit,
    )
    config = MedusaGenerationConfig(
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy or args.temperature <= 0.0,
        temperature=args.temperature,
        device=args.device,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        verify_mode=args.verify_mode,
    )
    model, tokenizer, runtime_device = load_medusa_model(
        model_name=config.model_name,
        checkpoint_path=config.checkpoint_path,
        medusa_num_heads=config.medusa_num_heads,
        medusa_num_layers=config.medusa_num_layers,
        device=config.device,
    )
    results = medusa_generate_from_prompts(
        model,
        tokenizer,
        prompts,
        config=config,
        device=runtime_device,
    )
    payload = results if len(results) > 1 else results[0]
    if args.output:
        write_structured_output(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
