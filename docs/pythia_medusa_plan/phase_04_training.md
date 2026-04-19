# Phase 4 - Training

## Purpose

Train Medusa-1 heads on top of frozen Pythia-70M.

This phase should make the model learn future-token prediction without updating the base model weights.

## Scope

Implement:
- dataset pipeline
- collator
- Medusa training loss
- trainer script
- checkpoint save/load

The training objective should match Medusa-1 style training:
- freeze the base model
- train only the additional heads

## Required files

Recommended files:
- `src/pythia_medusa/training/dataset.py`
- `src/pythia_medusa/training/collator.py`
- `src/pythia_medusa/training/losses.py`
- `src/pythia_medusa/training/trainer.py`
- `configs/train/medusa_train_small.yaml`
- `scripts/train_medusa1.sh`

## Data requirements

Minimal accepted formats:
- `.txt`
- `.jsonl`

Each sample should be convertible into:
- `input_ids`
- `attention_mask`
- `labels`

Start with a small corpus for debugging.

## Training objective

Recommended minimal loss:
- head 0 predicts token at offset `+1`
- head 1 predicts token at offset `+2`
- head 2 predicts token at offset `+3`

Loss design:
- compute cross-entropy loss for each head
- average or weighted-average the losses

Optional:
- ignore positions that do not have enough future tokens
- log per-head loss separately

## Freezing policy

Required:
- freeze all base model parameters
- train only Medusa-head parameters

Log at startup:
- total parameter count
- trainable parameter count
- trainable parameter percentage

## Minimal config suggestion

- model: `EleutherAI/pythia-70m-deduped`
- seq_len: `128`
- batch_size: `8`
- grad_accum: `1` or `2`
- lr: `1e-3`
- medusa_num_heads: `3`
- medusa_num_layers: `1`

## Validation metrics

During training, record:
- overall Medusa loss
- per-head loss
- per-head accuracy

## Checkpoint contents

Checkpoint folder should contain:
- Medusa config
- Medusa head weights
- tokenizer files if needed
- training metadata

Optional:
- optimizer state
- scheduler state

## Tests

Add smoke tests for:
- one batch forward + backward
- only Medusa params require grad
- loss is finite

## Deliverables

Phase 4 is complete when:
- training runs for at least one short epoch
- checkpoints are saved
- a trained checkpoint can be reloaded
- per-head validation metrics are available

## Notes for Codex

- Start with correctness, not distributed complexity.
- Use small debug runs first.
- Do not optimize data loading or mixed precision too early.
