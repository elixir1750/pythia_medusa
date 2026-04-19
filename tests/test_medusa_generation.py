from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pythia_medusa.generation.medusa_generate import MedusaGenerationConfig, medusa_generate
from pythia_medusa.generation.tree_utils import (
    build_linear_tree_attention_mask,
    build_tree_verification_visibility,
)
from pythia_medusa.models.medusa_config import MedusaConfig
from pythia_medusa.models.medusa_model import MedusaModel


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
    def __init__(self, hidden_size=8, vocab_size=32):
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
        next_tokens = torch.full(
            (input_ids.size(0), max_new_tokens),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, next_tokens], dim=1)


def build_medusa_model() -> MedusaModel:
    tokenizer = DummyTokenizer()
    base_model = DummyBaseCausalLM()
    config = MedusaConfig.from_base_model_config(
        "dummy-base",
        base_model.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    return MedusaModel(base_model=base_model, medusa_config=config, tokenizer=tokenizer)


def test_tree_mask_has_root_and_ancestor_visibility():
    mask = build_linear_tree_attention_mask(3)

    assert mask.shape == (4, 4)
    assert bool(mask[3, 0]) is True
    assert bool(mask[3, 2]) is True
    assert bool(mask[1, 3]) is False


def test_tree_verification_visibility_extends_prefix_causally():
    prefix_attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    visibility = build_tree_verification_visibility(
        prefix_attention_mask=prefix_attention_mask,
        nodes=[
            SimpleNamespace(node_id=0, parent_id=None, depth=0, future_offset=0),
            SimpleNamespace(node_id=1, parent_id=0, depth=1, future_offset=1),
            SimpleNamespace(node_id=2, parent_id=1, depth=2, future_offset=2),
        ],
    )

    assert visibility.shape == (5, 5)
    assert bool(visibility[0, 0]) is True
    assert bool(visibility[0, 3]) is False
    assert bool(visibility[3, 0]) is True
    assert bool(visibility[3, 3]) is True
    assert bool(visibility[3, 4]) is False
    assert bool(visibility[4, 3]) is True


def test_medusa_generation_returns_text_and_valid_accept_lengths():
    model = build_medusa_model()
    tokenizer = model.get_tokenizer()
    config = MedusaGenerationConfig(
        model_name="dummy-base",
        max_new_tokens=4,
        greedy=True,
        temperature=0.0,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )

    result = medusa_generate(model, tokenizer, "tiny prompt", config=config, device="cpu")

    assert result["generated_text"]
    assert result["generated_tokens"] > 0
    assert result["rounds"] >= 1
    assert len(result["accept_lengths"]) == result["rounds"]
    assert all(0 <= value <= 3 for value in result["accept_lengths"])


def test_tree_verify_matches_serial_verify_on_linear_candidates():
    model = build_medusa_model()
    tokenizer = model.get_tokenizer()
    serial_config = MedusaGenerationConfig(
        model_name="dummy-base",
        max_new_tokens=4,
        greedy=True,
        temperature=0.0,
        medusa_num_heads=3,
        medusa_num_layers=1,
        verify_mode="serial",
    )
    tree_config = MedusaGenerationConfig(
        model_name="dummy-base",
        max_new_tokens=4,
        greedy=True,
        temperature=0.0,
        medusa_num_heads=3,
        medusa_num_layers=1,
        verify_mode="tree",
    )

    serial_result = medusa_generate(model, tokenizer, "tiny prompt", config=serial_config, device="cpu")
    tree_result = medusa_generate(model, tokenizer, "tiny prompt", config=tree_config, device="cpu")

    assert serial_result["generated_text"] == tree_result["generated_text"]
    assert serial_result["accept_lengths"] == tree_result["accept_lengths"]
    assert tree_result["verify_mode"] == "tree"
