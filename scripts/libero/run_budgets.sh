#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after downloading data and generating the full cache.
# Default to the mask policy. Set POLICIES="rgb boxes masks" to compare all variants.
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICIES="${POLICIES:-masks}"
EPOCHS="${EPOCHS:-5}"
export HF_HOME="${HF_HOME:-$PWD/checkpoints/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/checkpoints/torch}"
for fraction in 0.10 0.25 0.50 1.00; do
  for policy in $POLICIES; do
    run_name="libero_${policy}_${fraction}_seed42"
    "$PYTHON_BIN" scripts/libero/run_grounded.py --policy "$policy" \
      --train-fraction "$fraction" --epochs "$EPOCHS" --device cuda --run-name "$run_name"
    "$PYTHON_BIN" scripts/libero/run_grounded.py --mode evaluate \
      --checkpoint "checkpoints/$run_name/best.pt" --device cuda --run-name "$run_name"
  done
done
