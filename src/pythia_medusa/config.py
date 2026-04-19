from dataclasses import dataclass

DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = 64
    temperature: float = 0.8
    greedy: bool = False
    device: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str | None = None
    max_examples: int | None = None
    max_length: int | None = None
    text_field: str = "text"
