from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pythia_medusa.config import DEFAULT_MODEL_NAME
from pythia_medusa.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class DatasetPreset:
    hf_path: str
    hf_name: str | None
    train_split: str
    valid_split: str
    text_field: str = "text"
    description: str = ""


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "wikitext-2": DatasetPreset(
        hf_path="wikitext",
        hf_name="wikitext-2-raw-v1",
        train_split="train",
        valid_split="validation",
        description="Small natural-text baseline close to language modeling homework scale.",
    ),
    "wikitext-103": DatasetPreset(
        hf_path="wikitext",
        hf_name="wikitext-103-raw-v1",
        train_split="train",
        valid_split="validation",
        description="Larger WikiText variant for a stronger natural-text stage1 run.",
    ),
    "pg19": DatasetPreset(
        hf_path="pg19",
        hf_name=None,
        train_split="train",
        valid_split="validation",
        description="Long-form book corpus; useful when you want richer continuation-style text.",
    ),
}


def list_dataset_presets() -> list[str]:
    return sorted(DATASET_PRESETS)


def get_dataset_preset(name: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(list_dataset_presets())
        raise ValueError(f"Unknown dataset preset '{name}'. Available: {available}") from exc


def normalize_text(text: str) -> str:
    normalized_lines: list[str] = []
    blank_pending = False

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line:
            if blank_pending and normalized_lines:
                normalized_lines.append("")
            normalized_lines.append(line)
            blank_pending = False
        else:
            blank_pending = True

    return "\n".join(normalized_lines).strip()


def merge_text_fragments(
    texts: Iterable[str],
    *,
    min_chars: int = 800,
    separator: str = "\n\n",
) -> list[str]:
    if min_chars <= 0:
        raise ValueError("min_chars must be positive.")

    merged: list[str] = []
    pending: list[str] = []
    pending_chars = 0

    def flush_pending() -> None:
        nonlocal pending_chars
        if not pending:
            return
        merged.append(separator.join(pending).strip())
        pending.clear()
        pending_chars = 0

    for text in texts:
        cleaned = normalize_text(text)
        if not cleaned:
            continue

        cleaned_length = len(cleaned)
        if cleaned_length >= min_chars:
            flush_pending()
            merged.append(cleaned)
            continue

        pending.append(cleaned)
        pending_chars += cleaned_length
        if pending_chars >= min_chars:
            flush_pending()

    flush_pending()
    return merged


def chunk_text_by_char_length(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int = 0,
    min_chunk_chars: int = 80,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    length = len(cleaned)

    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end
        if hard_end < length:
            best_break = cleaned.rfind(" ", start + max(1, chunk_size // 2), hard_end)
            if best_break > start:
                end = best_break

        chunk = cleaned[start:end].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

        if hard_end >= length:
            break

        start = max(end - chunk_overlap, start + 1)
        while start < length and cleaned[start].isspace():
            start += 1

    return chunks


def chunk_texts_with_tokenizer(
    texts: Iterable[str],
    *,
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    min_chunk_tokens: int = 32,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    if min_chunk_tokens <= 0:
        raise ValueError("min_chunk_tokens must be positive.")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise ImportError(
            "transformers is required for tokenizer-based chunking. "
            "Install requirements or use --chunker chars."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    for text in texts:
        cleaned = normalize_text(text)
        if not cleaned:
            continue

        token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if len(token_ids) < min_chunk_tokens:
            continue

        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start : start + chunk_size]
            if len(chunk_ids) < min_chunk_tokens:
                continue
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if chunk_text:
                chunks.append(chunk_text)

    return chunks


def examples_to_jsonl_records(texts: Iterable[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for text in texts:
        cleaned = normalize_text(text)
        if cleaned:
            records.append({"text": cleaned})
    return records


def _limit_items(items: list[str], max_items: int | None) -> list[str]:
    if max_items is None:
        return items
    if max_items < 0:
        raise ValueError("max_items must be non-negative when provided.")
    return items[:max_items]


def _load_hf_texts(
    preset: DatasetPreset,
    *,
    split: str,
    cache_dir: str | None = None,
    max_records: int | None = None,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise ImportError(
            "datasets is required to download corpora. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(
        preset.hf_path,
        preset.hf_name,
        split=split,
        cache_dir=cache_dir,
    )

    texts: list[str] = []
    for index, record in enumerate(dataset):
        value = record.get(preset.text_field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if max_records is not None and index + 1 >= max_records:
            break
    return texts


def prepare_split_records(
    raw_texts: Iterable[str],
    *,
    chunker: str,
    merge_min_chars: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_tokens: int,
    min_chunk_chars: int,
    tokenizer_name: str,
    max_examples: int | None = None,
) -> list[dict[str, str]]:
    merged_documents = merge_text_fragments(raw_texts, min_chars=merge_min_chars)

    if chunker == "tokenizer":
        chunked = chunk_texts_with_tokenizer(
            merged_documents,
            tokenizer_name=tokenizer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_tokens=min_chunk_tokens,
        )
    elif chunker == "chars":
        chunked = []
        for document in merged_documents:
            chunked.extend(
                chunk_text_by_char_length(
                    document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_chars=min_chunk_chars,
                )
            )
    else:
        raise ValueError(f"Unsupported chunker '{chunker}'.")

    return _limit_items(examples_to_jsonl_records(chunked), max_examples)


def prepare_hf_dataset(
    *,
    preset_name: str,
    train_output: str | Path,
    valid_output: str | Path,
    manifest_output: str | Path | None = None,
    cache_dir: str | None = None,
    chunker: str = "tokenizer",
    tokenizer_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    merge_min_chars: int = 800,
    min_chunk_tokens: int = 32,
    min_chunk_chars: int = 80,
    max_train_records: int | None = 2000,
    max_valid_records: int | None = 400,
    max_train_examples: int | None = None,
    max_valid_examples: int | None = None,
) -> dict[str, Any]:
    preset = get_dataset_preset(preset_name)

    train_texts = _load_hf_texts(
        preset,
        split=preset.train_split,
        cache_dir=cache_dir,
        max_records=max_train_records,
    )
    valid_texts = _load_hf_texts(
        preset,
        split=preset.valid_split,
        cache_dir=cache_dir,
        max_records=max_valid_records,
    )

    train_records = prepare_split_records(
        train_texts,
        chunker=chunker,
        merge_min_chars=merge_min_chars,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_tokens=min_chunk_tokens,
        min_chunk_chars=min_chunk_chars,
        tokenizer_name=tokenizer_name,
        max_examples=max_train_examples,
    )
    valid_records = prepare_split_records(
        valid_texts,
        chunker=chunker,
        merge_min_chars=merge_min_chars,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_tokens=min_chunk_tokens,
        min_chunk_chars=min_chunk_chars,
        tokenizer_name=tokenizer_name,
        max_examples=max_valid_examples,
    )

    if not train_records:
        raise ValueError("Prepared training split is empty. Relax filters or raise record limits.")
    if not valid_records:
        raise ValueError("Prepared validation split is empty. Relax filters or raise record limits.")

    train_output_path = write_jsonl(train_records, train_output)
    valid_output_path = write_jsonl(valid_records, valid_output)

    summary = {
        "preset": preset_name,
        "preset_config": asdict(preset),
        "chunker": chunker,
        "tokenizer_name": tokenizer_name if chunker == "tokenizer" else None,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "merge_min_chars": merge_min_chars,
        "min_chunk_tokens": min_chunk_tokens if chunker == "tokenizer" else None,
        "min_chunk_chars": min_chunk_chars if chunker == "chars" else None,
        "train_output": str(train_output_path),
        "valid_output": str(valid_output_path),
        "num_train_records": len(train_records),
        "num_valid_records": len(valid_records),
        "max_train_records": max_train_records,
        "max_valid_records": max_valid_records,
    }

    if manifest_output is not None:
        manifest_path = write_json(summary, manifest_output)
        summary["manifest_output"] = str(manifest_path)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a natural-text corpus and export Medusa training data as JSONL."
    )
    parser.add_argument("--preset", default="wikitext-2", choices=list_dataset_presets())
    parser.add_argument("--train-output", default="data/train.jsonl")
    parser.add_argument("--valid-output", default="data/valid.jsonl")
    parser.add_argument("--manifest-output", default="data/dataset_manifest.json")
    parser.add_argument("--cache-dir")
    parser.add_argument("--chunker", choices=("tokenizer", "chars"), default="tokenizer")
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--merge-min-chars", type=int, default=800)
    parser.add_argument("--min-chunk-tokens", type=int, default=32)
    parser.add_argument("--min-chunk-chars", type=int, default=80)
    parser.add_argument("--max-train-records", type=int, default=2000)
    parser.add_argument("--max-valid-records", type=int, default=400)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-valid-examples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_hf_dataset(
        preset_name=args.preset,
        train_output=args.train_output,
        valid_output=args.valid_output,
        manifest_output=args.manifest_output,
        cache_dir=args.cache_dir,
        chunker=args.chunker,
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        merge_min_chars=args.merge_min_chars,
        min_chunk_tokens=args.min_chunk_tokens,
        min_chunk_chars=args.min_chunk_chars,
        max_train_records=args.max_train_records,
        max_valid_records=args.max_valid_records,
        max_train_examples=args.max_train_examples,
        max_valid_examples=args.max_valid_examples,
    )

    print(f"Prepared {summary['preset']} dataset.")
    print(f"Train records: {summary['num_train_records']} -> {summary['train_output']}")
    print(f"Valid records: {summary['num_valid_records']} -> {summary['valid_output']}")
    if summary.get("manifest_output"):
        print(f"Manifest: {summary['manifest_output']}")


if __name__ == "__main__":
    main()
