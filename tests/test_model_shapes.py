from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pythia_medusa.models.medusa_config import MedusaConfig
from pythia_medusa.models.medusa_heads import MedusaHeadStack


def test_medusa_head_stack_output_shape():
    config = MedusaConfig(
        base_model_name_or_path="dummy-base",
        medusa_num_heads=3,
        medusa_num_layers=1,
        hidden_size=16,
        vocab_size=50,
    )
    heads = MedusaHeadStack(config)
    hidden_states = torch.randn(2, 5, 16)

    medusa_hidden_states = heads(hidden_states)

    assert medusa_hidden_states.shape == (3, 2, 5, 16)


def test_medusa_head_stack_projects_back_to_base_hidden_size():
    config = MedusaConfig(
        base_model_name_or_path="dummy-base",
        medusa_num_heads=3,
        medusa_num_layers=1,
        hidden_size=16,
        vocab_size=50,
        medusa_hidden_size=8,
    )
    heads = MedusaHeadStack(config)
    hidden_states = torch.randn(2, 5, 16)

    medusa_hidden_states = heads(hidden_states)

    assert medusa_hidden_states.shape == (3, 2, 5, 16)


def test_medusa_config_save_load_roundtrip(tmp_path: Path):
    config = MedusaConfig(
        base_model_name_or_path="EleutherAI/pythia-70m-deduped",
        medusa_num_heads=3,
        medusa_num_layers=1,
        hidden_size=64,
        vocab_size=128,
        medusa_hidden_size=32,
        version="phase2-test",
        tree_name="debug-tree",
    )

    config.save_pretrained(tmp_path)
    restored = MedusaConfig.from_pretrained(tmp_path)

    assert restored == config
