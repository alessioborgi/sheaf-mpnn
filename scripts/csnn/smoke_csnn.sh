#!/bin/bash
# CSNN wiring smoke test: every reproduction preset for 1 fold x 5 epochs.
# Catches registry/preset/shape regressions in minutes; the resulting numbers
# mean nothing. For the real 10-fold reproduction, use run_csnn_repro.sh.
#
# Usage (from anywhere):
#   bash scripts/csnn/smoke_csnn.sh
cd "$(dirname "$0")/../.." || exit 1

PRESETS="roman_empire_csnn amazon_ratings_csnn minesweeper_csnn questions_csnn squirrel_csnn squirrel_filtered_csnn tolokers_csnn"

for p in $PRESETS; do
  echo "===== $(date -Is)  smoke --preset $p ====="
  sheaf run --preset "$p" --cv.folds 1 --optim.epochs 5 \
    || { echo "!!! FAILED: $p"; exit 1; }
done
echo "===== CSNN SMOKE OK $(date -Is) ====="
