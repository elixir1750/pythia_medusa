from types import SimpleNamespace

import pytest
from torch.optim import AdamW
from torch.utils.data import DataLoader

torch = pytest.importorskip("torch")

from pythia_medusa.models.medusa_config import MedusaConfig
from pythia_medusa.models.medusa_model import MedusaModel
from pythia_medusa.training.collator import MedusaTrainingCollator
from pythia_medusa.training.dataset import TokenizedTextDataset, TextDataset
from pythia_medusa.training.losses import compute_medusa_loss
from pythia_medusa.training.trainer import (
    MedusaTrainer,
    count_parameters,
    freeze_base_model_parameters,
)


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


class DummyBaseCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=8, vocab_size=24):
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


def build_model_and_batch(*, pretokenized: bool = False):
    tokenizer = DummyTokenizer()
    base_model = DummyBaseCausalLM()
    config = MedusaConfig.from_base_model_config(
        "dummy-base",
        base_model.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    model = MedusaModel(base_model=base_model, medusa_config=config, tokenizer=tokenizer)
    dataset = (
        TokenizedTextDataset(
            [
                torch.tensor([2, 3, 4, 5], dtype=torch.long),
                torch.tensor([6, 7, 8], dtype=torch.long),
                torch.tensor([9, 10], dtype=torch.long),
            ]
        )
        if pretokenized
        else TextDataset(["alpha", "beta", "gamma"])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=MedusaTrainingCollator(tokenizer, seq_len=16),
    )
    batch = next(iter(dataloader))
    return model, dataloader, batch


def test_only_medusa_params_require_grad():
    model, _, _ = build_model_and_batch()
    freeze_base_model_parameters(model)
    stats = count_parameters(model)

    assert stats["trainable_params"] > 0
    assert stats["trainable_params"] < stats["total_params"]
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    assert all(parameter.requires_grad for parameter in model.medusa_heads.parameters())


def test_one_batch_forward_backward_returns_finite_loss(tmp_path):
    model, dataloader, _ = build_model_and_batch()
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        valid_dataloader=None,
        device="cpu",
        output_dir=str(tmp_path),
    )

    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)

    assert metrics["loss"] >= 0.0
    assert all(model_param.grad is None for model_param in model.base_model.parameters())
    assert any(model_param.grad is not None for model_param in model.medusa_heads.parameters())


def test_one_batch_forward_backward_with_pretokenized_dataset(tmp_path):
    model, dataloader, _ = build_model_and_batch(pretokenized=True)
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        valid_dataloader=None,
        device="cpu",
        output_dir=str(tmp_path),
    )

    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)

    assert metrics["loss"] >= 0.0
    assert batch["input_ids"].dtype == torch.long


def test_trainer_runs_with_progress_disabled(tmp_path):
    model, dataloader, _ = build_model_and_batch()
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        device="cpu",
        output_dir=str(tmp_path),
        show_progress=False,
    )

    summary = trainer.train(num_epochs=1, log_every=1)

    assert "final_train_metrics" in summary
    assert "final_valid_metrics" in summary
    assert "eval_history" in summary
    assert summary["final_train_metrics"]["loss"] >= 0.0


def test_trainer_stops_at_max_steps(tmp_path):
    model, dataloader, _ = build_model_and_batch(pretokenized=True)
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        device="cpu",
        output_dir=str(tmp_path),
        show_progress=False,
    )

    summary = trainer.train(num_epochs=3, log_every=1, max_steps=2)

    assert len(summary["history"]) == 2
    assert summary["max_steps_reached"] is True


def test_visual_dashboard_is_written_from_run_summary(tmp_path):
    model, dataloader, _ = build_model_and_batch()
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        device="cpu",
        output_dir=str(tmp_path),
        show_progress=False,
    )

    summary = trainer.train(num_epochs=1, log_every=1)
    payload = {
        "config": {"model": "dummy-base", "write_dashboard": True},
        "parameter_stats": count_parameters(model),
        "summary": summary,
    }

    from pythia_medusa.training.visualize_training import write_training_dashboard

    dashboard_path = write_training_dashboard(
        payload,
        output_path=tmp_path / "training_dashboard.html",
    )

    assert dashboard_path.exists()
    content = dashboard_path.read_text(encoding="utf-8")
    assert "Pythia Medusa Training Dashboard" in content
    assert "Head Accuracy" in content


def test_compute_medusa_loss_avoids_nan_from_overflowing_logits_sum():
    medusa_logits = torch.full((3, 1, 4, 8), 10000.0, dtype=torch.float16)
    labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    loss_output = compute_medusa_loss(medusa_logits, labels)

    assert torch.isfinite(loss_output.loss)
    assert all(torch.isfinite(loss) for loss in loss_output.per_head_losses)


def test_float16_base_model_keeps_medusa_params_finite_after_optimizer_step():
    tokenizer = DummyTokenizer()
    base_model = DummyBaseCausalLM().to(dtype=torch.float16)
    config = MedusaConfig.from_base_model_config(
        "dummy-base",
        base_model.config,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    model = MedusaModel(base_model=base_model, medusa_config=config, tokenizer=tokenizer)
    freeze_base_model_parameters(model)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5)
    dataset = TokenizedTextDataset(
        [
            torch.tensor([2, 3, 4, 5], dtype=torch.long),
            torch.tensor([6, 7, 8, 9], dtype=torch.long),
        ]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=MedusaTrainingCollator(tokenizer, seq_len=16),
    )

    batch = next(iter(dataloader))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        medusa_forward=True,
    )
    loss_output = compute_medusa_loss(outputs.medusa_logits, batch["labels"])
    optimizer.zero_grad()
    loss_output.loss.backward()
    optimizer.step()

    assert all(torch.isfinite(parameter).all() for parameter in model.medusa_heads.parameters())
