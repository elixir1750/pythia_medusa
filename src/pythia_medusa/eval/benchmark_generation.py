from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pythia_medusa.config import DEFAULT_MODEL_NAME, GenerationConfig
from pythia_medusa.data.prompt_sets import resolve_prompts
from pythia_medusa.generation.base_generate import generate_from_prompts, load_model_and_tokenizer
from pythia_medusa.generation.medusa_generate import (
    MedusaGenerationConfig,
    load_medusa_model,
    medusa_generate_from_prompts,
)
from pythia_medusa.utils.io import append_csv_row, write_json, write_jsonl
from pythia_medusa.utils.profiling import measure_repeated, summarize_benchmark_runs

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    yaml = None


def _run_baseline_once(
    model: Any,
    tokenizer: Any,
    prompts: list[Any],
    *,
    config: GenerationConfig,
    device: str | None = None,
) -> dict[str, Any]:
    rows = generate_from_prompts(model, tokenizer, prompts, config=config, device=device)
    total_latency = sum(row["latency_sec"] for row in rows)
    total_tokens = sum(row["generated_tokens"] for row in rows)
    return {
        "rows": rows,
        "latency_sec": total_latency,
        "output_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
    }


def _run_medusa_once(
    model: Any,
    tokenizer: Any,
    prompts: list[Any],
    *,
    config: MedusaGenerationConfig,
    device: str | None = None,
) -> dict[str, Any]:
    rows = medusa_generate_from_prompts(model, tokenizer, prompts, config=config, device=device)
    total_latency = sum(row["latency_sec"] for row in rows)
    total_tokens = sum(row["generated_tokens"] for row in rows)
    accept_lengths = [row["average_accept_length"] for row in rows]
    return {
        "rows": rows,
        "latency_sec": total_latency,
        "output_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "avg_accept_length": sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0,
    }


def benchmark_generation(
    baseline_model: Any,
    baseline_tokenizer: Any,
    medusa_model: Any,
    medusa_tokenizer: Any,
    prompts: list[Any],
    *,
    baseline_config: GenerationConfig,
    medusa_config: MedusaGenerationConfig,
    repeat: int = 5,
    warmup: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    baseline_runs = measure_repeated(
        lambda: _run_baseline_once(
            baseline_model,
            baseline_tokenizer,
            prompts,
            config=baseline_config,
            device=device,
        ),
        repeat=repeat,
        warmup=warmup,
    )
    medusa_runs = measure_repeated(
        lambda: _run_medusa_once(
            medusa_model,
            medusa_tokenizer,
            prompts,
            config=medusa_config,
            device=device,
        ),
        repeat=repeat,
        warmup=warmup,
    )

    summary = {
        "baseline": summarize_benchmark_runs(
            baseline_runs,
            latency_key="latency_sec",
            throughput_key="tokens_per_sec",
        ),
        "medusa": summarize_benchmark_runs(
            medusa_runs,
            latency_key="latency_sec",
            throughput_key="tokens_per_sec",
        ),
        "benchmark_metadata": {
            "num_prompts": len(prompts),
            "repeat": repeat,
            "warmup": warmup,
            "verify_mode": medusa_config.verify_mode,
        },
        "baseline_runs": baseline_runs,
        "medusa_runs": medusa_runs,
    }
    medusa_accept_lengths = [run.get("avg_accept_length", 0.0) for run in medusa_runs]
    summary["medusa"]["avg_accept_length"] = (
        sum(medusa_accept_lengths) / len(medusa_accept_lengths) if medusa_accept_lengths else 0.0
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs Medusa generation timing.")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-set")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--device")
    parser.add_argument("--verify-mode", choices=("tree", "serial"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_benchmark_config(config_path: str | None, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config_path:
        if yaml is None:
            raise ImportError("PyYAML is required to read benchmark config files.")
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
    config = load_benchmark_config(
        args.config,
        {
            "model": args.model,
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir,
            "prompt": args.prompt,
            "prompt_file": args.prompt_file,
            "prompt_set": args.prompt_set,
            "max_new_tokens": args.max_new_tokens,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "device": args.device,
            "verify_mode": args.verify_mode,
            "limit": args.limit,
        },
    )
    config.setdefault("model", DEFAULT_MODEL_NAME)
    config.setdefault("prompt_set", "all")
    config.setdefault("max_new_tokens", 32)
    config.setdefault("repeat", 5)
    config.setdefault("warmup", 1)
    required_keys = ("model", "checkpoint_path", "output_dir")
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required benchmark config values: {', '.join(missing)}")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = resolve_prompts(
        prompt=config.get("prompt"),
        prompt_file=config.get("prompt_file"),
        prompt_set=config.get("prompt_set", "all"),
        limit=config.get("limit"),
    )
    baseline_model, baseline_tokenizer, runtime_device = load_model_and_tokenizer(
        config["model"],
        device=config.get("device"),
    )
    medusa_model, medusa_tokenizer, _ = load_medusa_model(
        model_name=config["model"],
        checkpoint_path=config["checkpoint_path"],
        device=runtime_device,
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
    summary = benchmark_generation(
        baseline_model,
        baseline_tokenizer,
        medusa_model,
        medusa_tokenizer,
        prompts,
        baseline_config=baseline_config,
        medusa_config=medusa_config,
        repeat=config.get("repeat", 5),
        warmup=config.get("warmup", 1),
        device=runtime_device,
    )
    write_json(summary, output_dir / "benchmark_summary.json")
    write_jsonl(summary["baseline_runs"], output_dir / "baseline_runs.jsonl")
    write_jsonl(summary["medusa_runs"], output_dir / "medusa_runs.jsonl")
    append_csv_row(
        {
            "model": config["model"],
            "checkpoint_path": config["checkpoint_path"],
            "avg_latency_sec_baseline": summary["baseline"]["avg_latency_sec"],
            "avg_tokens_per_sec_baseline": summary["baseline"]["avg_tokens_per_sec"],
            "avg_latency_sec_medusa": summary["medusa"]["avg_latency_sec"],
            "avg_tokens_per_sec_medusa": summary["medusa"]["avg_tokens_per_sec"],
            "avg_accept_length_medusa": summary["medusa"]["avg_accept_length"],
            "verify_mode": summary["benchmark_metadata"]["verify_mode"],
        },
        output_dir / "benchmark_summary.csv",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
