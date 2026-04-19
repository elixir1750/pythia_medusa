from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pythia_medusa.config import GenerationConfig
from pythia_medusa.data.prompt_sets import PromptExample
from pythia_medusa.eval.benchmark_generation import benchmark_generation
from pythia_medusa.eval.compare_models import compare_models
from pythia_medusa.models.medusa_config import MedusaConfig
from pythia_medusa.models.medusa_model import MedusaModel
from pythia_medusa.generation.medusa_generate import MedusaGenerationConfig
from pythia_medusa.utils.io import append_csv_row, write_json


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        token_ids = [((ord(char) - 31) % 20) + 2 for char in text] or [self.eos_token_id]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(chr(((token_id - 2) % 26) + 97))
        return "".join(chars) or "x"


class DummyBaseCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=8, vocab_size=21):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        **kwargs,
    ):
        embedded = self.embedding(input_ids)
        hidden = self.proj(embedded)
        logits = self.lm_head(hidden)
        hidden_states = (embedded, hidden) if output_hidden_states else None
        if return_dict:
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)
        return logits, hidden_states

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        extra = torch.full(
            (input_ids.size(0), max_new_tokens),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, extra], dim=1)


def build_models():
    tokenizer = DummyTokenizer()
    baseline = DummyBaseCausalLM()
    medusa_base = DummyBaseCausalLM()
    medusa_config = MedusaConfig.from_base_model_config(
        "dummy-base",
        medusa_base.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    medusa = MedusaModel(base_model=medusa_base, medusa_config=medusa_config, tokenizer=tokenizer)
    return baseline, tokenizer, medusa, tokenizer


def test_compare_models_and_export(tmp_path: Path):
    baseline_model, baseline_tokenizer, medusa_model, medusa_tokenizer = build_models()
    prompts = [
        PromptExample(text="Alpha", bucket="test", name="p1", source="unit"),
        PromptExample(text="Beta", bucket="test", name="p2", source="unit"),
    ]
    texts = ["alpha beta", "gamma delta"]

    summary = compare_models(
        baseline_model,
        baseline_tokenizer,
        medusa_model,
        medusa_tokenizer,
        texts,
        prompts,
        baseline_config=GenerationConfig(
            model_name="dummy-base",
            max_new_tokens=4,
            greedy=True,
            temperature=0.0,
        ),
        medusa_config=MedusaGenerationConfig(
            model_name="dummy-base",
            max_new_tokens=4,
            greedy=True,
            temperature=0.0,
            verify_mode="tree",
        ),
        device="cpu",
    )

    assert "baseline_lm" in summary
    assert "medusa_lm" in summary
    assert "head_metrics" in summary
    assert len(summary["generation_compare"]) == 2
    assert summary["verify_mode"] == "tree"

    json_path = write_json(summary, tmp_path / "summary.json")
    csv_path = append_csv_row({"loss": summary["baseline_lm"]["loss"]}, tmp_path / "summary.csv")

    assert json_path.exists()
    assert csv_path.exists()


def test_benchmark_generation_summary():
    baseline_model, baseline_tokenizer, medusa_model, medusa_tokenizer = build_models()
    prompts = [PromptExample(text="Alpha", bucket="test", name="p1", source="unit")]

    summary = benchmark_generation(
        baseline_model,
        baseline_tokenizer,
        medusa_model,
        medusa_tokenizer,
        prompts,
        baseline_config=GenerationConfig(
            model_name="dummy-base",
            max_new_tokens=4,
            greedy=True,
            temperature=0.0,
        ),
        medusa_config=MedusaGenerationConfig(
            model_name="dummy-base",
            max_new_tokens=4,
            greedy=True,
            temperature=0.0,
            verify_mode="tree",
        ),
        repeat=2,
        warmup=0,
        device="cpu",
    )

    assert summary["baseline"]["num_runs"] == 2
    assert summary["medusa"]["num_runs"] == 2
    assert summary["medusa"]["avg_accept_length"] >= 0.0
    assert summary["benchmark_metadata"]["verify_mode"] == "tree"
