from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pythia_medusa.models.medusa_config import (
    MEDUSA_HEADS_NAME,
    MEDUSA_METADATA_NAME,
    MedusaConfig,
)
from pythia_medusa.models.medusa_heads import MedusaHeadStack

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class MedusaForwardOutput:
    medusa_logits: torch.Tensor | None
    logits: torch.Tensor | None
    hidden_states: torch.Tensor
    base_outputs: Any


class MedusaModel(nn.Module):
    """Wrap a causal LM and expose baseline and Medusa forward paths."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        medusa_config: MedusaConfig,
        tokenizer: Any | None = None,
        medusa_heads: MedusaHeadStack | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.medusa_config = medusa_config
        self.tokenizer = tokenizer
        self.medusa_heads = medusa_heads or MedusaHeadStack(medusa_config)
        self._align_medusa_heads_to_base_model()

    def _reference_parameter(self) -> torch.nn.Parameter | torch.Tensor | None:
        try:
            return next(self.base_model.parameters())
        except StopIteration:
            pass

        lm_head = self.lm_head
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
        return None

    def _align_medusa_heads_to_base_model(self) -> None:
        reference = self._reference_parameter()
        if reference is None:
            return
        self.medusa_heads.to(device=reference.device, dtype=torch.float32)

    def _project_with_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm_head = self.lm_head
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor):
            bias = getattr(lm_head, "bias", None)
            projected_bias = None
            if isinstance(bias, torch.Tensor):
                projected_bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
            return F.linear(
                hidden_states,
                weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
                projected_bias,
            )
        return lm_head(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path: str,
        *,
        medusa_num_heads: int = 3,
        medusa_num_layers: int = 1,
        medusa_hidden_size: int | None = None,
        tokenizer: Any | None = None,
        base_model: nn.Module | None = None,
        **hf_kwargs: Any,
    ) -> "MedusaModel":
        if base_model is None:
            if AutoModelForCausalLM is None or AutoTokenizer is None:
                raise ImportError(
                    "Loading a pretrained MedusaModel requires transformers to be installed."
                )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                **hf_kwargs,
            )
        if tokenizer is None and AutoTokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        medusa_config = MedusaConfig.from_base_model_config(
            base_model_name_or_path,
            getattr(base_model, "config", None),
            medusa_num_heads=medusa_num_heads,
            medusa_num_layers=medusa_num_layers,
            medusa_hidden_size=medusa_hidden_size,
        )
        return cls(
            base_model=base_model,
            medusa_config=medusa_config,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_medusa_checkpoint(
        cls,
        load_directory: str | Path,
        *,
        tokenizer: Any | None = None,
        base_model: nn.Module | None = None,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
        **hf_kwargs: Any,
    ) -> "MedusaModel":
        medusa_config = MedusaConfig.from_pretrained(load_directory)
        model = cls.from_pretrained(
            medusa_config.base_model_name_or_path,
            medusa_num_heads=medusa_config.medusa_num_heads,
            medusa_num_layers=medusa_config.medusa_num_layers,
            medusa_hidden_size=medusa_config.medusa_hidden_size,
            tokenizer=tokenizer,
            base_model=base_model,
            **hf_kwargs,
        )
        model.medusa_config = medusa_config
        model.load_medusa_heads(load_directory, map_location=map_location, strict=strict)
        return model

    @property
    def lm_head(self) -> nn.Module:
        if hasattr(self.base_model, "get_output_embeddings"):
            output_embeddings = self.base_model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            raise AttributeError("Base model does not expose lm_head or get_output_embeddings().")
        return lm_head

    def get_tokenizer(self) -> Any | None:
        return self.tokenizer

    def _run_base_model(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        return self.base_model(*args, **kwargs)

    @staticmethod
    def _extract_last_hidden_state(base_outputs: Any) -> torch.Tensor:
        hidden_states = getattr(base_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Base model outputs did not include hidden_states.")
        if isinstance(hidden_states, tuple):
            if not hidden_states:
                raise ValueError("Base model returned an empty hidden_states tuple.")
            return hidden_states[-1]
        return hidden_states

    def forward(
        self,
        *args: Any,
        medusa_forward: bool = False,
        output_orig: bool = False,
        **kwargs: Any,
    ) -> Any:
        base_outputs = self._run_base_model(*args, **kwargs)
        if not medusa_forward and not output_orig:
            return base_outputs

        last_hidden_state = self._extract_last_hidden_state(base_outputs)
        medusa_logits = None
        if medusa_forward:
            medusa_reference = next(self.medusa_heads.parameters(), None)
            if medusa_reference is not None and (
                last_hidden_state.dtype != medusa_reference.dtype
                or last_hidden_state.device != medusa_reference.device
            ):
                last_hidden_state = last_hidden_state.to(
                    device=medusa_reference.device,
                    dtype=medusa_reference.dtype,
                )
            medusa_hidden_states = self.medusa_heads(last_hidden_state)
            medusa_logits = self._project_with_lm_head(medusa_hidden_states)
        original_logits = None
        if output_orig:
            original_logits = getattr(base_outputs, "logits", None)
            if original_logits is None:
                original_logits = self._project_with_lm_head(last_hidden_state)
        return MedusaForwardOutput(
            medusa_logits=medusa_logits,
            logits=original_logits,
            hidden_states=last_hidden_state,
            base_outputs=base_outputs,
        )

    def save_medusa_checkpoint(
        self,
        save_directory: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        save_tokenizer: bool = False,
    ) -> dict[str, Path]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = self.medusa_config.save_pretrained(save_path)
        heads_path = save_path / MEDUSA_HEADS_NAME
        torch.save(
            {
                "state_dict": self.medusa_heads.state_dict(),
                "metadata": metadata or {},
                "medusa_config": self.medusa_config.to_dict(),
            },
            heads_path,
        )

        metadata_path = save_path / MEDUSA_METADATA_NAME
        metadata_payload = {
            "base_model_name_or_path": self.medusa_config.base_model_name_or_path,
            "medusa_num_heads": self.medusa_config.medusa_num_heads,
            "medusa_num_layers": self.medusa_config.medusa_num_layers,
            "hidden_size": self.medusa_config.hidden_size,
            "vocab_size": self.medusa_config.vocab_size,
            "medusa_hidden_size": self.medusa_config.medusa_hidden_size,
            "version": self.medusa_config.version,
            "tree_name": self.medusa_config.tree_name,
            "metadata": metadata or {},
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        if save_tokenizer and self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(save_path))

        return {
            "config": config_path,
            "heads": heads_path,
            "metadata": metadata_path,
        }

    def load_medusa_heads(
        self,
        load_directory: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
    ) -> Any:
        checkpoint_path = Path(load_directory) / MEDUSA_HEADS_NAME
        payload = torch.load(checkpoint_path, map_location=map_location)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        return self.medusa_heads.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def load_checkpoint_metadata(load_directory: str | Path) -> dict[str, Any]:
        metadata_path = Path(load_directory) / MEDUSA_METADATA_NAME
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected checkpoint metadata object in {metadata_path}, got {type(payload)!r}"
            )
        return payload
