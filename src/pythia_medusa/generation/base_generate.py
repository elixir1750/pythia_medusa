import argparse
import json
import time
from collections.abc import Mapping
from typing import Any

from pythia_medusa.config import DEFAULT_MODEL_NAME, GenerationConfig
from pythia_medusa.data.prompt_sets import PromptExample, resolve_prompts
from pythia_medusa.utils.io import write_structured_output

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    AutoModelForCausalLM = None
    AutoTokenizer = None


def require_runtime_dependencies() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "Baseline generation requires torch and transformers. Install the project requirements first."
        )


def infer_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str | None = None,
):
    require_runtime_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    runtime_device = infer_device(device)
    model = model.to(runtime_device)
    model.eval()
    return model, tokenizer, runtime_device


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    greedy: bool,
    tokenizer: Any,
) -> dict[str, Any]:
    do_sample = not greedy and temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 1.0,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    return generation_kwargs


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_example: PromptExample,
    *,
    config: GenerationConfig,
    device: str | None = None,
) -> dict[str, Any]:
    runtime_device = infer_device(device or config.device)
    encoded = tokenizer(prompt_example.text, return_tensors="pt")
    encoded = move_batch_to_device(encoded, runtime_device)

    prompt_token_count = int(encoded["input_ids"].shape[-1])
    generation_kwargs = build_generation_kwargs(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        greedy=config.greedy,
        tokenizer=tokenizer,
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)
    latency_sec = time.perf_counter() - start_time

    sequence = output_ids[0]
    generated_ids = sequence[prompt_token_count:]
    full_text = tokenizer.decode(sequence, skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_token_count = int(generated_ids.shape[-1])
    total_token_count = int(sequence.shape[-1])

    return {
        "model": config.model_name,
        "prompt": prompt_example.text,
        "generated_text": generated_text,
        "full_text": full_text,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": generated_token_count,
        "total_tokens": total_token_count,
        "latency_sec": latency_sec,
        "tokens_per_sec": generated_token_count / latency_sec if latency_sec > 0 else 0.0,
        "temperature": config.temperature,
        "greedy": config.greedy,
        "max_new_tokens": config.max_new_tokens,
        "bucket": prompt_example.bucket,
        "name": prompt_example.name,
        "source": prompt_example.source,
    }


def generate_from_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[PromptExample],
    *,
    config: GenerationConfig,
    device: str | None = None,
) -> list[dict[str, Any]]:
    return [
        generate_one(model, tokenizer, prompt_example, config=config, device=device)
        for prompt_example in prompts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline generation with Pythia.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-set", help="Built-in prompt set name or 'all'.")
    parser.add_argument("--output", help="Optional .json or .jsonl output path.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device")
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
    config = GenerationConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        greedy=args.greedy,
        device=args.device,
    )
    model, tokenizer, runtime_device = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
    )
    results = generate_from_prompts(
        model,
        tokenizer,
        prompts,
        config=config,
        device=runtime_device,
    )

    if args.output:
        write_structured_output(results if len(results) > 1 else results[0], args.output)

    print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
