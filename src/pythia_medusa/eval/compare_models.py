from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pythia_medusa.config import DEFAULT_MODEL_NAME, GenerationConfig
from pythia_medusa.data.prompt_sets import load_text_examples, resolve_prompts
from pythia_medusa.eval.eval_heads import evaluate_medusa_heads
from pythia_medusa.eval.eval_language_modeling import evaluate_language_modeling
from pythia_medusa.generation.base_generate import generate_from_prompts, load_model_and_tokenizer
from pythia_medusa.generation.medusa_generate import (
    MedusaGenerationConfig,
    load_medusa_model,
    medusa_generate_from_prompts,
)
from pythia_medusa.models import MedusaModel
from pythia_medusa.utils.io import append_csv_row, write_json, write_jsonl

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    yaml = None


def compare_generation_outputs(
    baseline_model: Any,
    baseline_tokenizer: Any,
    medusa_model: MedusaModel,
    medusa_tokenizer: Any,
    prompts: list[Any],
    *,
    baseline_config: GenerationConfig,
    medusa_config: MedusaGenerationConfig,
    device: str | None = None,
) -> list[dict[str, Any]]:
    baseline_rows = generate_from_prompts(
        baseline_model,
        baseline_tokenizer,
        prompts,
        config=baseline_config,
        device=device,
    )
    medusa_rows = medusa_generate_from_prompts(
        medusa_model,
        medusa_tokenizer,
        prompts,
        config=medusa_config,
        device=device,
    )

    comparison_rows = []
    for baseline_row, medusa_row in zip(baseline_rows, medusa_rows):
        comparison_rows.append(
            {
                "prompt": baseline_row["prompt"],
                "bucket": baseline_row.get("bucket"),
                "name": baseline_row.get("name"),
                "baseline_text": baseline_row["generated_text"],
                "medusa_text": medusa_row["generated_text"],
                "baseline_generated_tokens": baseline_row["generated_tokens"],
                "medusa_generated_tokens": medusa_row["generated_tokens"],
                "baseline_latency_sec": baseline_row["latency_sec"],
                "medusa_latency_sec": medusa_row["latency_sec"],
                "medusa_average_accept_length": medusa_row["average_accept_length"],
                "verify_mode": medusa_row["verify_mode"],
            }
        )
    return comparison_rows


def compare_models(
    baseline_model: Any,
    baseline_tokenizer: Any,
    medusa_model: MedusaModel,
    medusa_tokenizer: Any,
    texts: list[str],
    prompts: list[Any],
    *,
    baseline_config: GenerationConfig,
    medusa_config: MedusaGenerationConfig,
    device: str | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    baseline_lm = evaluate_language_modeling(
        baseline_model,
        baseline_tokenizer,
        texts,
        device=device,
        max_length=max_length,
    )
    medusa_lm = evaluate_language_modeling(
        medusa_model,
        medusa_tokenizer,
        texts,
        device=device,
        max_length=max_length,
    )
    head_metrics = evaluate_medusa_heads(
        medusa_model,
        medusa_tokenizer,
        texts,
        device=device,
        max_length=max_length,
    )
    generation_rows = compare_generation_outputs(
        baseline_model,
        baseline_tokenizer,
        medusa_model,
        medusa_tokenizer,
        prompts,
        baseline_config=baseline_config,
        medusa_config=medusa_config,
        device=device,
    )
    return {
        "baseline_lm": baseline_lm,
        "medusa_lm": medusa_lm,
        "head_metrics": head_metrics,
        "generation_compare": generation_rows,
        "verify_mode": medusa_config.verify_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and Medusa models on the same data.")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-set")
    parser.add_argument("--text-field")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device")
    parser.add_argument("--verify-mode", choices=("tree", "serial"))
    return parser.parse_args()


def load_compare_config(config_path: str | None, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config_path:
        if yaml is None:
            raise ImportError("PyYAML is required to read comparison config files.")
        with Path(config_path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping in config file {config_path}, got {type(loaded)!r}")
        payload.update(loaded)
    for key, value in cli_overrides.items():
        if value is not None:
            payload[key] = value
    return payload


def main() -> None:
    args = parse_args()
    config = load_compare_config(
        args.config,
        {
            "model": args.model,
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "output_dir": args.output_dir,
            "prompt": args.prompt,
            "prompt_file": args.prompt_file,
            "prompt_set": args.prompt_set,
            "text_field": args.text_field,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "verify_mode": args.verify_mode,
        },
    )
    config.setdefault("model", DEFAULT_MODEL_NAME)
    config.setdefault("text_field", "text")
    config.setdefault("max_new_tokens", 32)
    required_keys = ("model", "checkpoint_path", "dataset", "output_dir")
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required comparison config values: {', '.join(missing)}")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, baseline_tokenizer, runtime_device = load_model_and_tokenizer(
        config["model"],
        device=config.get("device"),
    )
    medusa_model, medusa_tokenizer, _ = load_medusa_model(
        model_name=config["model"],
        checkpoint_path=config["checkpoint_path"],
        device=runtime_device,
    )

    texts = load_text_examples(
        config["dataset"],
        text_field=config.get("text_field", "text"),
        max_examples=config.get("max_examples"),
    )
    prompts = resolve_prompts(
        prompt=config.get("prompt"),
        prompt_file=config.get("prompt_file"),
        prompt_set=config.get("prompt_set") or "all",
        limit=config.get("max_examples"),
    )
    baseline_config = GenerationConfig(
        model_name=config["model"],
        max_new_tokens=config.get("max_new_tokens", 32),
        greedy=True,
        temperature=0.0,
        device=runtime_device,
    )
    medusa_config = MedusaGenerationConfig(
        model_name=config["model"],
        checkpoint_path=config["checkpoint_path"],
        max_new_tokens=config.get("max_new_tokens", 32),
        greedy=True,
        temperature=0.0,
        device=runtime_device,
        medusa_num_heads=medusa_model.medusa_config.medusa_num_heads,
        medusa_num_layers=medusa_model.medusa_config.medusa_num_layers,
        verify_mode=config.get("verify_mode", "tree"),
    )
    summary = compare_models(
        baseline_model,
        baseline_tokenizer,
        medusa_model,
        medusa_tokenizer,
        texts,
        prompts,
        baseline_config=baseline_config,
        medusa_config=medusa_config,
        device=runtime_device,
        max_length=config.get("max_length"),
    )
    summary.update(
        {
            "model": config["model"],
            "checkpoint_path": config["checkpoint_path"],
            "dataset": config["dataset"],
            "verify_mode": medusa_config.verify_mode,
        }
    )
    write_json(summary, output_dir / "comparison_summary.json")
    write_jsonl(summary["generation_compare"], output_dir / "generation_compare.jsonl")
    append_csv_row(
        {
            "model": config["model"],
            "checkpoint_path": config["checkpoint_path"],
            "dataset": config["dataset"],
            "verify_mode": medusa_config.verify_mode,
            **summary["baseline_lm"],
            **{f"medusa_{key}": value for key, value in summary["medusa_lm"].items()},
            **{f"head_eval_{key}": value for key, value in summary["head_metrics"].items()},
        },
        output_dir / "comparison_summary.csv",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
