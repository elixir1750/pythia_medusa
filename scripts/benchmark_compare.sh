#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-70m-deduped}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/medusa_train_small/checkpoint-epoch01-step00001}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/benchmark_compare}"
PROMPT_SET="${PROMPT_SET:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
REPEAT_COUNT="${REPEAT_COUNT:-5}"
WARMUP_COUNT="${WARMUP_COUNT:-1}"

python -m pythia_medusa.eval.benchmark_generation \
  --model "${MODEL_NAME}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --prompt-set "${PROMPT_SET}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --repeat "${REPEAT_COUNT}" \
  --warmup "${WARMUP_COUNT}"
