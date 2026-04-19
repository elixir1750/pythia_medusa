"""Training utilities for Medusa heads."""

from .collator import MedusaTrainingCollator
from .dataset import TokenizedTextDataset, TextDataset, load_text_dataset, load_training_dataset
from .losses import MedusaLossOutput, compute_medusa_loss

__all__ = [
    "TextDataset",
    "TokenizedTextDataset",
    "load_text_dataset",
    "load_training_dataset",
    "MedusaTrainingCollator",
    "MedusaLossOutput",
    "compute_medusa_loss",
    "TrainingConfig",
    "MedusaTrainer",
    "freeze_base_model_parameters",
    "count_parameters",
]


def __getattr__(name: str):
    if name in {"TrainingConfig", "MedusaTrainer", "freeze_base_model_parameters", "count_parameters"}:
        from .trainer import (
            MedusaTrainer,
            TrainingConfig,
            count_parameters,
            freeze_base_model_parameters,
        )

        mapping = {
            "TrainingConfig": TrainingConfig,
            "MedusaTrainer": MedusaTrainer,
            "freeze_base_model_parameters": freeze_base_model_parameters,
            "count_parameters": count_parameters,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
