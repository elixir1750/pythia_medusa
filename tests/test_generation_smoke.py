from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pythia_medusa.config import GenerationConfig
from pythia_medusa.data.prompt_sets import PromptExample
from pythia_medusa.eval.eval_language_modeling import evaluate_language_modeling
from pythia_medusa.generation import base_generate
from pythia_medusa.generation.base_generate import generate_one


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

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


class DummyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )
        next_tokens = torch.roll(input_ids, shifts=-1, dims=1)
        logits.scatter_(2, next_tokens.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)

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


def test_load_model_and_tokenizer_smoke(monkeypatch):
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 1
        eos_token = "<eos>"
        pad_token = None

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "dummy-model"
            return FakeTokenizer()

    class FakeModel:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "dummy-model"
            return FakeModel()

    monkeypatch.setattr(base_generate, "AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(base_generate, "AutoModelForCausalLM", FakeModelLoader)

    model, tokenizer, device = base_generate.load_model_and_tokenizer(
        "dummy-model",
        device="cpu",
    )

    assert device == "cpu"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert model.device == "cpu"
    assert model.eval_called is True


def test_generate_one_returns_non_empty_text():
    model = DummyCausalLM()
    tokenizer = DummyTokenizer()
    prompt = PromptExample(
        text="The capital of France is",
        bucket="short_factual",
        name="capital_france",
        source="test",
    )
    config = GenerationConfig(max_new_tokens=6, temperature=0.8, greedy=False)

    result = generate_one(model, tokenizer, prompt, config=config, device="cpu")

    assert result["prompt"] == prompt.text
    assert result["generated_tokens"] == 6
    assert result["generated_text"]
    assert result["latency_sec"] >= 0.0


def test_evaluate_language_modeling_returns_finite_metrics():
    model = DummyCausalLM()
    tokenizer = DummyTokenizer()

    summary = evaluate_language_modeling(
        model,
        tokenizer,
        ["hello world", "tiny dataset"],
        device="cpu",
    )

    assert summary["num_examples"] == 2
    assert summary["num_tokens"] > 0
    assert summary["loss"] >= 0.0
    assert summary["perplexity"] > 0.0
