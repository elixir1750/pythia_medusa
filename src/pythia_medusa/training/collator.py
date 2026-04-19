from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class MedusaTrainingCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        label_pad_token_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.label_pad_token_id = label_pad_token_id

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            if hasattr(tokenizer, "eos_token") and eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                pad_token_id = tokenizer.pad_token_id
            else:
                pad_token_id = 0
        self.pad_token_id = int(pad_token_id)

    def _tokenize_one(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.seq_len,
        )
        input_ids = encoded["input_ids"]
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                f"Tokenizer must return input_ids with shape [1, seq], got {tuple(input_ids.shape)}"
            )
        return input_ids[0]

    @staticmethod
    def _coerce_input_ids(input_ids: Any) -> torch.Tensor:
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() != 1:
                raise ValueError(
                    f"Expected pretokenized input_ids with shape [seq], got {tuple(input_ids.shape)}"
                )
            return input_ids.to(dtype=torch.long, device="cpu")
        if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
            tensor = torch.tensor(list(input_ids), dtype=torch.long)
            if tensor.dim() != 1:
                raise ValueError(
                    f"Expected pretokenized input_ids with shape [seq], got {tuple(tensor.shape)}"
                )
            return tensor
        raise TypeError(f"Unsupported pretokenized input_ids type: {type(input_ids)!r}")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = []
        for example in examples:
            if "input_ids" in example:
                sequences.append(self._coerce_input_ids(example["input_ids"]))
            else:
                sequences.append(self._tokenize_one(example["text"]))
        max_length = max(sequence.size(0) for sequence in sequences)

        input_ids = torch.full(
            (len(sequences), max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)

        for row, sequence in enumerate(sequences):
            length = int(sequence.size(0))
            input_ids[row, :length] = sequence
            attention_mask[row, :length] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
