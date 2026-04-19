"""Baseline utilities for the staged Pythia Medusa implementation."""

from .config import DEFAULT_MODEL_NAME, EvaluationConfig, GenerationConfig

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EvaluationConfig",
    "GenerationConfig",
]

try:
    from .models import MedusaConfig, MedusaForwardOutput, MedusaHeadStack, MedusaModel
except ImportError:  # pragma: no cover - optional while torch/transformers are absent
    pass
else:
    __all__.extend(
        [
            "MedusaConfig",
            "MedusaHeadStack",
            "MedusaForwardOutput",
            "MedusaModel",
        ]
    )
