from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pythia_medusa.models.medusa_config import MEDUSA_HEADS_NAME, MedusaConfig
from pythia_medusa.models.medusa_model import MedusaModel


class DummyBaseCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=8, vocab_size=17):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True):
        embedded = self.embedding(input_ids)
        hidden = self.proj(embedded)
        logits = self.lm_head(hidden)
        hidden_states = (embedded, hidden) if output_hidden_states else None
        if return_dict:
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)
        return logits, hidden_states


def build_model() -> MedusaModel:
    base_model = DummyBaseCausalLM()
    config = MedusaConfig.from_base_model_config(
        "dummy-base",
        base_model.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    return MedusaModel(base_model=base_model, medusa_config=config, tokenizer={"kind": "dummy"})


def test_medusa_wrapper_output_shapes():
    model = build_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    outputs = model(input_ids=input_ids, medusa_forward=True, output_orig=True)

    assert outputs.medusa_logits is not None
    assert outputs.logits is not None
    assert outputs.hidden_states.shape == (2, 4, 8)
    assert outputs.logits.shape == (2, 4, 17)
    assert outputs.medusa_logits.shape == (3, 2, 4, 17)


def test_output_orig_only_path_returns_original_logits():
    model = build_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    outputs = model(input_ids=input_ids, output_orig=True)

    assert outputs.medusa_logits is None
    assert outputs.logits is not None
    assert outputs.logits.shape == (1, 3, 17)


def test_medusa_checkpoint_roundtrip(tmp_path: Path):
    model = build_model()

    with torch.no_grad():
        first_param = next(model.medusa_heads.parameters())
        first_param.fill_(0.25)

    saved_paths = model.save_medusa_checkpoint(
        tmp_path,
        metadata={"step": 7, "note": "phase2"},
    )
    assert saved_paths["heads"].name == MEDUSA_HEADS_NAME

    restored = build_model()
    restored.load_medusa_heads(tmp_path)
    restored_metadata = restored.load_checkpoint_metadata(tmp_path)

    original_state = model.medusa_heads.state_dict()
    restored_state = restored.medusa_heads.state_dict()
    for name, tensor in original_state.items():
        assert torch.equal(tensor, restored_state[name])

    assert restored_metadata["metadata"]["step"] == 7
    assert restored_metadata["base_model_name_or_path"] == "dummy-base"


def test_medusa_heads_stay_float32_even_if_base_model_is_float16():
    base_model = DummyBaseCausalLM().to(dtype=torch.float16)
    config = MedusaConfig.from_base_model_config(
        "dummy-base",
        base_model.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    model = MedusaModel(base_model=base_model, medusa_config=config, tokenizer={"kind": "dummy"})

    head_param = next(model.medusa_heads.parameters())

    assert head_param.dtype == torch.float32
