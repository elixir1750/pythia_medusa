import argparse
import json
import math
from typing import Any

from pythia_medusa.config import DEFAULT_MODEL_NAME, EvaluationConfig
from pythia_medusa.data.prompt_sets import load_text_examples
from pythia_medusa.generation.base_generate import (
    infer_device,
    load_model_and_tokenizer,
    move_batch_to_device,
    require_runtime_dependencies,
)
from pythia_medusa.utils.io import append_csv_row, write_json

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    torch = None
    F = None


def evaluate_language_modeling(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    if torch is None or F is None:
        raise ImportError(
            "Language-model evaluation requires torch. Install the project requirements first."
        )

    runtime_device = infer_device(device)
    total_loss = 0.0
    total_tokens = 0
    evaluated_examples = 0
    skipped_examples = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        encoded = move_batch_to_device(encoded, runtime_device)
        if int(encoded["input_ids"].shape[-1]) < 2:
            skipped_examples += 1
            continue

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = encoded["input_ids"][:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

        token_count = int(labels.numel())
        total_loss += float(loss.item())
        total_tokens += token_count
        evaluated_examples += 1

    if total_tokens == 0:
        raise ValueError("No evaluable tokens found in dataset.")

    average_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(average_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
        "num_examples": evaluated_examples,
        "skipped_examples": skipped_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher-forcing language-model evaluation with Pythia."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="Path to output json summary.")
    parser.add_argument("--csv-output", help="Optional CSV file to append a summary row to.")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--device")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-length", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    config = EvaluationConfig(
        model_name=args.model,
        device=args.device,
        max_examples=args.max_examples,
        max_length=args.max_length,
        text_field=args.text_field,
    )
    texts = load_text_examples(
        args.dataset,
        text_field=config.text_field,
        max_examples=config.max_examples,
    )
    model, tokenizer, runtime_device = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
    )
    summary = evaluate_language_modeling(
        model,
        tokenizer,
        texts,
        device=runtime_device,
        max_length=config.max_length,
    )
    summary.update(
        {
            "model": config.model_name,
            "dataset": args.dataset,
            "text_field": config.text_field,
        }
    )

    write_json(summary, args.output)
    if args.csv_output:
        append_csv_row(summary, args.csv_output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
