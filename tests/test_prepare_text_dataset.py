import json
import sys
import types

from pythia_medusa.data.prepare_text_dataset import (
    merge_text_fragments,
    normalize_text,
    prepare_hf_dataset,
    prepare_split_records,
)
from pythia_medusa.utils.io import read_jsonl


def test_normalize_and_merge_text_fragments_preserve_paragraph_structure():
    raw_texts = [
        "  First line. \n\n Second line.  ",
        "\nThird line.",
        "Standalone paragraph with enough content.",
    ]

    normalized = normalize_text(raw_texts[0])
    merged = merge_text_fragments(raw_texts, min_chars=30)

    assert normalized == "First line.\n\nSecond line."
    assert merged[0] == "First line.\n\nSecond line.\n\nThird line."
    assert merged[1] == "Standalone paragraph with enough content."


def test_prepare_split_records_char_chunker_returns_jsonl_ready_examples():
    records = prepare_split_records(
        [
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
            "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega.",
        ],
        chunker="chars",
        merge_min_chars=20,
        chunk_size=40,
        chunk_overlap=0,
        min_chunk_tokens=4,
        min_chunk_chars=10,
        tokenizer_name="unused",
        max_examples=None,
    )

    assert records
    assert all("text" in record for record in records)
    assert all(len(record["text"]) >= 10 for record in records)


def test_prepare_hf_dataset_writes_train_valid_and_manifest(tmp_path, monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [index + 1 for index, _ in enumerate(text.split())]

        def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return " ".join(f"tok{token_id}" for token_id in token_ids)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: FakeTokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def fake_load_hf_texts(preset, *, split, cache_dir=None, max_records=None):
        if split == preset.train_split:
            return [
                "one two three four five six seven eight",
                "nine ten eleven twelve thirteen fourteen",
            ]
        return ["alpha beta gamma delta epsilon zeta eta theta"]

    monkeypatch.setattr(
        "pythia_medusa.data.prepare_text_dataset._load_hf_texts",
        fake_load_hf_texts,
    )

    summary = prepare_hf_dataset(
        preset_name="wikitext-2",
        train_output=tmp_path / "train.jsonl",
        valid_output=tmp_path / "valid.jsonl",
        manifest_output=tmp_path / "manifest.json",
        chunker="tokenizer",
        tokenizer_name="fake-tokenizer",
        chunk_size=4,
        chunk_overlap=0,
        merge_min_chars=10,
        min_chunk_tokens=2,
        min_chunk_chars=10,
        max_train_records=10,
        max_valid_records=10,
    )

    train_records = read_jsonl(tmp_path / "train.jsonl")
    valid_records = read_jsonl(tmp_path / "valid.jsonl")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert summary["num_train_records"] == len(train_records)
    assert summary["num_valid_records"] == len(valid_records)
    assert train_records[0]["text"] == "tok1 tok2 tok3 tok4"
    assert valid_records[0]["text"] == "tok1 tok2 tok3 tok4"
    assert manifest["preset"] == "wikitext-2"
    assert manifest["chunker"] == "tokenizer"
