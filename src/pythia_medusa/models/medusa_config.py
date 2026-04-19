from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MEDUSA_CONFIG_NAME = "medusa_config.json"
MEDUSA_HEADS_NAME = "medusa_heads.pt"
MEDUSA_METADATA_NAME = "medusa_metadata.json"


@dataclass(frozen=True)
class MedusaConfig:
    base_model_name_or_path: str
    medusa_num_heads: int
    medusa_num_layers: int
    hidden_size: int
    vocab_size: int
    medusa_hidden_size: int | None = None
    version: str = "medusa-v1"
    tree_name: str | None = None

    def __post_init__(self) -> None:
        if self.medusa_num_heads <= 0:
            raise ValueError("medusa_num_heads must be positive.")
        if self.medusa_num_layers < 0:
            raise ValueError("medusa_num_layers must be non-negative.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.medusa_hidden_size is not None and self.medusa_hidden_size <= 0:
            raise ValueError("medusa_hidden_size must be positive when provided.")

    @property
    def resolved_medusa_hidden_size(self) -> int:
        return self.medusa_hidden_size or self.hidden_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedusaConfig":
        return cls(**data)

    @classmethod
    def from_base_model_config(
        cls,
        base_model_name_or_path: str,
        base_config: Any,
        *,
        medusa_num_heads: int = 3,
        medusa_num_layers: int = 1,
        medusa_hidden_size: int | None = None,
        version: str = "medusa-v1",
        tree_name: str | None = None,
    ) -> "MedusaConfig":
        hidden_size = getattr(base_config, "hidden_size", None)
        vocab_size = getattr(base_config, "vocab_size", None)
        if hidden_size is None or vocab_size is None:
            raise ValueError(
                "Base model config must expose hidden_size and vocab_size for MedusaConfig."
            )
        return cls(
            base_model_name_or_path=base_model_name_or_path,
            medusa_num_heads=medusa_num_heads,
            medusa_num_layers=medusa_num_layers,
            hidden_size=int(hidden_size),
            vocab_size=int(vocab_size),
            medusa_hidden_size=medusa_hidden_size,
            version=version,
            tree_name=tree_name,
        )

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config_path = save_path / MEDUSA_CONFIG_NAME
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return config_path

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "MedusaConfig":
        config_path = Path(load_directory) / MEDUSA_CONFIG_NAME
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected config object in {config_path}, got {type(payload)!r}")
        return cls.from_dict(payload)
