from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from pythia_medusa.config import DEFAULT_MODEL_NAME
from pythia_medusa.data.prompt_sets import load_text_examples
from pythia_medusa.generation.medusa_generate import infer_device
from pythia_medusa.models import MedusaModel
from pythia_medusa.training.losses import compute_medusa_loss
from pythia_medusa.utils.io import append_csv_row, write_json


def evaluate_medusa_heads(
    model: MedusaModel,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    runtime_device = infer_device(device)
    totals: dict[str, float] = {}
    steps = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(runtime_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(runtime_device)

        labels = input_ids.clone()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            medusa_forward=True,
        )
        loss_output = compute_medusa_loss(outputs.medusa_logits, labels)
        metrics = {"loss": float(loss_output.loss.item())}
        for index, (head_loss, head_accuracy, valid_tokens) in enumerate(
            zip(
                loss_output.per_head_losses,
                loss_output.per_head_accuracies,
                loss_output.per_head_valid_tokens,
            ),
            start=1,
        ):
            metrics[f"head{index}_loss"] = float(head_loss.item())
            metrics[f"head{index}_acc"] = head_accuracy
            metrics[f"head{index}_valid_tokens"] = float(valid_tokens)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1

    if steps == 0:
        raise ValueError("No examples were available for Medusa head evaluation.")
    return {
        "num_examples": steps,
        **{key: value / steps for key, value in totals.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Medusa head metrics on a text dataset.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MedusaModel.from_medusa_checkpoint(args.checkpoint_path)
    tokenizer = model.get_tokenizer()
    if tokenizer is None:
        raise ValueError("Checkpoint load did not provide a tokenizer.")
    model.to(infer_device(args.device))
    model.eval()

    texts = load_text_examples(
        args.dataset,
        text_field=args.text_field,
        max_examples=args.max_examples,
    )
    summary = evaluate_medusa_heads(
        model,
        tokenizer,
        texts,
        device=args.device,
        max_length=args.max_length,
    )
    summary.update(
        {
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "model": args.model,
        }
    )
    write_json(summary, args.output)
    if args.csv_output:
        append_csv_row(summary, args.csv_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
