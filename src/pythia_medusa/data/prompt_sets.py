from dataclasses import asdict, dataclass
from pathlib import Path

from pythia_medusa.utils.io import read_jsonl

PROMPT_FIELDS = ("prompt", "text", "input")


@dataclass(frozen=True)
class PromptExample:
    text: str
    bucket: str
    name: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PROMPT_SETS: dict[str, list[PromptExample]] = {
    "short_factual": [
        PromptExample(
            text="The capital of France is",
            bucket="short_factual",
            name="capital_france",
            source="builtin",
        ),
        PromptExample(
            text="Water freezes at",
            bucket="short_factual",
            name="water_freezes",
            source="builtin",
        ),
        PromptExample(
            text="The largest planet in our solar system is",
            bucket="short_factual",
            name="largest_planet",
            source="builtin",
        ),
    ],
    "short_continuation": [
        PromptExample(
            text="Once upon a time, in a city powered by wind,",
            bucket="short_continuation",
            name="wind_city",
            source="builtin",
        ),
        PromptExample(
            text="The email began with an apology, but by the second paragraph",
            bucket="short_continuation",
            name="email_apology",
            source="builtin",
        ),
        PromptExample(
            text="At the edge of the forest stood a cabin that",
            bucket="short_continuation",
            name="forest_cabin",
            source="builtin",
        ),
    ],
    "medium_reasoning": [
        PromptExample(
            text="If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?",
            bucket="medium_reasoning",
            name="train_speed",
            source="builtin",
        ),
        PromptExample(
            text="A recipe needs 3 eggs for 12 cookies. How many eggs are needed for 20 cookies?",
            bucket="medium_reasoning",
            name="cookie_ratio",
            source="builtin",
        ),
        PromptExample(
            text="Explain why the sum of two odd numbers is always even.",
            bucket="medium_reasoning",
            name="odd_plus_odd",
            source="builtin",
        ),
    ],
    "medium_instruction": [
        PromptExample(
            text="Write three bullet points that explain why recycling matters.",
            bucket="medium_instruction",
            name="recycling_bullets",
            source="builtin",
        ),
        PromptExample(
            text="Summarize the benefits of regular exercise in two sentences.",
            bucket="medium_instruction",
            name="exercise_summary",
            source="builtin",
        ),
        PromptExample(
            text="Draft a polite reminder email asking for project feedback.",
            bucket="medium_instruction",
            name="feedback_email",
            source="builtin",
        ),
    ],
}


def list_prompt_sets() -> list[str]:
    return sorted(PROMPT_SETS)


def get_prompt_set(name: str) -> list[PromptExample]:
    try:
        return PROMPT_SETS[name]
    except KeyError as exc:
        available = ", ".join(list_prompt_sets())
        raise ValueError(f"Unknown prompt set '{name}'. Available: {available}") from exc


def flatten_prompt_sets() -> list[PromptExample]:
    prompts: list[PromptExample] = []
    for set_name in list_prompt_sets():
        prompts.extend(get_prompt_set(set_name))
    return prompts


def load_prompts_from_jsonl(path: str | Path) -> list[PromptExample]:
    prompts: list[PromptExample] = []
    for index, record in enumerate(read_jsonl(path)):
        prompt_text = None
        for field in PROMPT_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                prompt_text = value.strip()
                break
        if prompt_text is None:
            raise ValueError(
                f"Could not find any of {PROMPT_FIELDS} in prompt file {path} record {index}"
            )
        bucket = str(record.get("bucket", "file"))
        name = str(record.get("name", f"prompt_{index}"))
        prompts.append(
            PromptExample(
                text=prompt_text,
                bucket=bucket,
                name=name,
                source=str(path),
            )
        )
    return prompts


def resolve_prompts(
    *,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    prompt_set: str | None = None,
    limit: int | None = None,
) -> list[PromptExample]:
    sources_selected = sum(
        value is not None and value != "" for value in (prompt, prompt_file, prompt_set)
    )
    if sources_selected == 0:
        raise ValueError("Provide one of --prompt, --prompt-file, or --prompt-set.")
    if sources_selected > 1:
        raise ValueError("Use only one prompt source at a time.")

    if prompt is not None:
        prompts = [
            PromptExample(
                text=prompt,
                bucket="cli",
                name="cli_prompt",
                source="cli",
            )
        ]
    elif prompt_file is not None:
        prompts = load_prompts_from_jsonl(prompt_file)
    elif prompt_set == "all":
        prompts = flatten_prompt_sets()
    else:
        prompts = get_prompt_set(prompt_set or "")

    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def load_text_examples(
    path: str | Path,
    *,
    text_field: str = "text",
    max_examples: int | None = None,
) -> list[str]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        texts = []
        for index, record in enumerate(read_jsonl(dataset_path)):
            value = record.get(text_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Expected non-empty '{text_field}' in {path} record {index}"
                )
            texts.append(value)
    else:
        with dataset_path.open("r", encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]

    if max_examples is not None:
        texts = texts[:max_examples]
    return texts
