from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from pythia_medusa.data.prompt_sets import load_text_examples


@dataclass(frozen=True)
class TextExample:
    text: str


@dataclass(frozen=True)
class TokenizedTextExample:
    input_ids: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.examples = [TextExample(text=text) for text in texts]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.examples[index].text}


class TokenizedTextDataset(Dataset):
    def __init__(self, token_ids: list[torch.Tensor]) -> None:
        self.examples = [TokenizedTextExample(input_ids=ids) for ids in token_ids]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"input_ids": self.examples[index].input_ids}


def _tokenize_one(
    text: str,
    *,
    tokenizer: Any,
    seq_len: int,
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encoded["input_ids"]
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(
            f"Tokenizer must return input_ids with shape [1, seq], got {tuple(input_ids.shape)}"
        )
    return input_ids[0].to(dtype=torch.long, device="cpu")


def load_text_dataset(
    path: str,
    *,
    text_field: str = "text",
    max_examples: int | None = None,
) -> TextDataset | TokenizedTextDataset:
    texts = load_text_examples(
        path,
        text_field=text_field,
        max_examples=max_examples,
    )
    return TextDataset(texts)


def load_training_dataset(
    path: str,
    *,
    tokenizer: Any,
    seq_len: int,
    text_field: str = "text",
    max_examples: int | None = None,
    pretokenize: bool = True,
) -> TextDataset | TokenizedTextDataset:
    texts = load_text_examples(
        path,
        text_field=text_field,
        max_examples=max_examples,
    )
    if not pretokenize:
        return TextDataset(texts)
    return TokenizedTextDataset(
        [
            _tokenize_one(
                text,
                tokenizer=tokenizer,
                seq_len=seq_len,
            )
            for text in texts
        ]
    )
