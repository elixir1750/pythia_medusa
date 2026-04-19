from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pythia_medusa.config import DEFAULT_MODEL_NAME
from pythia_medusa.models import MedusaModel
from pythia_medusa.training.collator import MedusaTrainingCollator
from pythia_medusa.training.dataset import TextDataset, TokenizedTextDataset, load_training_dataset
from pythia_medusa.training.losses import MedusaLossOutput, compute_medusa_loss
from pythia_medusa.utils.io import append_csv_row, write_json

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing-deps environments
    yaml = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency for friendlier CLI progress
    tqdm = None


@dataclass(frozen=True)
class TrainingConfig:
    model: str = DEFAULT_MODEL_NAME
    dataset: str = ""
    valid_dataset: str | None = None
    output_dir: str = "outputs/medusa_train"
    text_field: str = "text"
    seq_len: int = 128
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-3
    num_epochs: int = 1
    max_steps: int | None = None
    max_examples: int | None = None
    medusa_num_heads: int = 3
    medusa_num_layers: int = 1
    medusa_hidden_size: int | None = None
    device: str | None = None
    log_every: int = 10
    save_tokenizer: bool = False
    show_progress: bool = True
    pretokenize_dataset: bool = True


def freeze_base_model_parameters(model: MedusaModel) -> None:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.medusa_heads.parameters():
        parameter.requires_grad = True


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
        "trainable_pct": float(trainable_params) / float(total_params) if total_params else 0.0,
    }


def _format_dataset_mode(pretokenize_dataset: bool) -> str:
    return "pretokenized" if pretokenize_dataset else "on-the-fly tokenization"


def _infer_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _loss_output_to_metrics(loss_output: MedusaLossOutput) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "loss": float(loss_output.loss.item()),
    }
    for index, (head_loss, head_accuracy, valid_tokens) in enumerate(
        zip(
            loss_output.per_head_losses,
            loss_output.per_head_accuracies,
            loss_output.per_head_valid_tokens,
        ),
        start=1,
    ):
        metrics[f"head{index}_loss"] = float(head_loss.item())
        metrics[f"head{index}_acc"] = head_accuracy
        metrics[f"head{index}_valid_tokens"] = valid_tokens
    return metrics


class MedusaTrainer:
    def __init__(
        self,
        model: MedusaModel,
        *,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        device: str = "cpu",
        grad_accum: int = 1,
        output_dir: str = "outputs/medusa_train",
        save_tokenizer: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.base_device = device
        self.head_device = "cpu" if device == "mps" else device
        self.grad_accum = grad_accum
        self.output_dir = Path(output_dir)
        self.save_tokenizer = save_tokenizer
        self.show_progress = show_progress
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.base_model.to(self.base_device)
        self.model.medusa_heads.to(self.head_device, dtype=torch.float32)
        if self.base_device != self.head_device and self.show_progress:
            print(
                f"Training split devices: base_model={self.base_device}, medusa_heads={self.head_device}",
                flush=True,
            )
        self.model.base_model.eval()
        self.model.medusa_heads.train()

    def _build_progress(
        self,
        iterable: Any,
        *,
        desc: str,
        total: int | None = None,
        leave: bool = False,
    ) -> Any:
        if self.show_progress and tqdm is not None:
            return tqdm(iterable, desc=desc, total=total, leave=leave)
        if self.show_progress:
            print(desc, flush=True)
        return iterable

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        batch_on_base = _move_batch_to_device(batch, self.base_device)
        labels = batch["labels"].to(self.head_device)
        with torch.no_grad():
            base_outputs = self.model.base_model(
                input_ids=batch_on_base["input_ids"],
                attention_mask=batch_on_base["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden_state = self.model._extract_last_hidden_state(base_outputs).detach()
        medusa_hidden_input = last_hidden_state.to(self.head_device, dtype=torch.float32)
        medusa_hidden_states = self.model.medusa_heads(medusa_hidden_input)
        medusa_logits = self.model._project_with_lm_head(medusa_hidden_states)
        loss_output = compute_medusa_loss(medusa_logits, labels)
        if not torch.isfinite(loss_output.loss):
            raise FloatingPointError(
                "Encountered non-finite Medusa loss before backward(). "
                "Try a smaller learning rate or inspect per-head losses."
            )
        loss_output.loss.backward()
        return _loss_output_to_metrics(loss_output)

    def _eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        batch_on_base = _move_batch_to_device(batch, self.base_device)
        labels = batch["labels"].to(self.head_device)
        with torch.no_grad():
            base_outputs = self.model.base_model(
                input_ids=batch_on_base["input_ids"],
                attention_mask=batch_on_base["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_state = self.model._extract_last_hidden_state(base_outputs).detach()
            medusa_hidden_input = last_hidden_state.to(self.head_device, dtype=torch.float32)
            medusa_hidden_states = self.model.medusa_heads(medusa_hidden_input)
            medusa_logits = self.model._project_with_lm_head(medusa_hidden_states)
            loss_output = compute_medusa_loss(medusa_logits, labels)
        if not torch.isfinite(loss_output.loss):
            raise FloatingPointError(
                "Encountered non-finite Medusa loss during evaluation."
            )
        return _loss_output_to_metrics(loss_output)

    def _run_eval(self, dataloader: DataLoader, *, desc: str = "Validation") -> dict[str, Any]:
        self.model.base_model.eval()
        self.model.medusa_heads.eval()
        totals: dict[str, float] = {}
        steps = 0
        progress = self._build_progress(
            dataloader,
            desc=desc,
            total=len(dataloader),
            leave=False,
        )
        for batch in progress:
            metrics = self._eval_step(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
            if tqdm is not None and hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{metrics['loss']:.4f}")
        self.model.base_model.eval()
        self.model.medusa_heads.train()
        if steps == 0:
            return {}
        return {key: value / steps for key, value in totals.items()}

    def save_checkpoint(
        self,
        *,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
    ) -> dict[str, Path]:
        checkpoint_dir = self.output_dir / f"checkpoint-epoch{epoch:02d}-step{step:05d}"
        paths = self.model.save_medusa_checkpoint(
            checkpoint_dir,
            metadata={
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
            },
            save_tokenizer=self.save_tokenizer,
        )
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        paths["optimizer"] = optimizer_path
        return paths

    def train(
        self,
        *,
        num_epochs: int = 1,
        log_every: int = 10,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        global_step = 0
        stop_training = False
        self.optimizer.zero_grad()

        for epoch in range(1, num_epochs + 1):
            progress = self._build_progress(
                self.train_dataloader,
                desc=f"Epoch {epoch}/{num_epochs}",
                total=len(self.train_dataloader),
                leave=True,
            )
            for batch in progress:
                global_step += 1
                train_metrics = self.train_step(batch)

                if global_step % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                row = {
                    "epoch": epoch,
                    "step": global_step,
                    **train_metrics,
                }
                history.append(row)
                if tqdm is not None and hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        step=global_step,
                        loss=f"{train_metrics['loss']:.4f}",
                        head1_acc=f"{train_metrics['head1_acc']:.3f}",
                    )
                if log_every > 0 and global_step % log_every == 0:
                    append_csv_row(row, self.output_dir / "train_log.csv")
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break

            if global_step % self.grad_accum != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            eval_metrics = (
                self._run_eval(self.valid_dataloader, desc=f"Eval {epoch}/{num_epochs}")
                if self.valid_dataloader
                else {}
            )
            checkpoint_metrics = {
                "train": history[-1] if history else {},
                "valid": eval_metrics,
            }
            self.save_checkpoint(epoch=epoch, step=global_step, metrics=checkpoint_metrics)
            if stop_training:
                break

        summary = {
            "history": history,
            "final_train_metrics": history[-1] if history else {},
            "final_valid_metrics": (
                self._run_eval(self.valid_dataloader, desc="Final validation")
                if self.valid_dataloader
                else {}
            ),
            "max_steps_reached": stop_training,
        }
        write_json(summary, self.output_dir / "training_summary.json")
        return summary


def load_training_config(config_path: str | None, cli_overrides: dict[str, Any]) -> TrainingConfig:
    payload: dict[str, Any] = {}
    if config_path:
        if yaml is None:
            raise ImportError("PyYAML is required to read trainer config files.")
        with Path(config_path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping in config file {config_path}, got {type(loaded)!r}")
        payload.update(loaded)

    for key, value in cli_overrides.items():
        if value is not None:
            payload[key] = value
    return TrainingConfig(**payload)


def build_dataloader(
    dataset: TextDataset | TokenizedTextDataset,
    *,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MedusaTrainingCollator(tokenizer, seq_len=seq_len),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Medusa heads on top of frozen Pythia.")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--valid-dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--text-field")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--medusa-num-heads", type=int)
    parser.add_argument("--medusa-num-layers", type=int)
    parser.add_argument("--medusa-hidden-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--save-tokenizer", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-pretokenize-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(
        args.config,
        {
            "model": args.model,
            "dataset": args.dataset,
            "valid_dataset": args.valid_dataset,
            "output_dir": args.output_dir,
            "text_field": args.text_field,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "max_examples": args.max_examples,
            "medusa_num_heads": args.medusa_num_heads,
            "medusa_num_layers": args.medusa_num_layers,
            "medusa_hidden_size": args.medusa_hidden_size,
            "device": args.device,
            "log_every": args.log_every,
            "save_tokenizer": args.save_tokenizer or None,
            "show_progress": False if args.no_progress else None,
            "pretokenize_dataset": False if args.no_pretokenize_dataset else None,
        },
    )
    if not config.dataset:
        raise ValueError("Training requires a dataset path via --dataset or config.")

    model = MedusaModel.from_pretrained(
        config.model,
        medusa_num_heads=config.medusa_num_heads,
        medusa_num_layers=config.medusa_num_layers,
        medusa_hidden_size=config.medusa_hidden_size,
    )
    freeze_base_model_parameters(model)
    parameter_stats = count_parameters(model)

    tokenizer = model.get_tokenizer()
    if tokenizer is None:
        raise ValueError("Training requires a tokenizer from MedusaModel.")

    dataset_mode = _format_dataset_mode(config.pretokenize_dataset)
    print(f"Loading train dataset ({dataset_mode})...", flush=True)
    train_dataset = load_training_dataset(
        config.dataset,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        text_field=config.text_field,
        max_examples=config.max_examples,
        pretokenize=config.pretokenize_dataset,
    )
    if config.valid_dataset:
        print(f"Loading valid dataset ({dataset_mode})...", flush=True)
    valid_dataset = (
        load_training_dataset(
            config.valid_dataset,
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            text_field=config.text_field,
            max_examples=config.max_examples,
            pretokenize=config.pretokenize_dataset,
        )
        if config.valid_dataset
        else None
    )
    train_dataloader = build_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        shuffle=True,
    )
    valid_dataloader = (
        build_dataloader(
            valid_dataset,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            shuffle=False,
        )
        if valid_dataset is not None
        else None
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
    )
    trainer = MedusaTrainer(
        model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        device=_infer_device(config.device),
        grad_accum=config.grad_accum,
        output_dir=config.output_dir,
        save_tokenizer=config.save_tokenizer,
        show_progress=config.show_progress,
    )
    summary = trainer.train(
        num_epochs=config.num_epochs,
        log_every=config.log_every,
        max_steps=config.max_steps,
    )
    payload = {
        "config": asdict(config),
        "parameter_stats": parameter_stats,
        "summary": summary,
    }
    write_json(payload, Path(config.output_dir) / "run_summary.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
