#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/medusa_train_small.yaml}"

python -m pythia_medusa.training.trainer --config "${CONFIG_PATH}"
