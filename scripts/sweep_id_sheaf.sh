#!/bin/bash
# Run every Id-NSD (ISN) sweep sequentially: the seven Pei et al. datasets kept
# for the paper grid, then the seven Platonov et al. ones. Id-NSD pins all
# restriction maps to the identity, so each dataset has exactly one config.
#
# Usage (inside tmux, from the repo root):
#   bash scripts/sweep_id_sheaf.sh                    # all 14 datasets
#   bash scripts/sweep_id_sheaf.sh pei                # Pei et al. only
#   bash scripts/sweep_id_sheaf.sh platonov           # Platonov et al. only
#   bash scripts/sweep_id_sheaf.sh cora texas         # an explicit subset
cd "$(dirname "$0")/.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefer the project venv so the script works without a pre-activated shell.
SHEAF="sheaf"
command -v "$SHEAF" >/dev/null 2>&1 || SHEAF="./.venv/bin/sheaf"

# squirrel and chameleon are excluded: only their filtered Platonov versions run.
PEI="cora citeseer pubmed film cornell texas wisconsin"
PLATONOV="roman_empire amazon_ratings minesweeper tolokers questions squirrel_filtered chameleon_filtered"

case "${1:-all}" in
  all)      DATASETS="$PEI $PLATONOV" ;;
  pei)      DATASETS="$PEI" ;;
  platonov) DATASETS="$PLATONOV" ;;
  *)        DATASETS="$*" ;;
esac

for ds in $DATASETS; do
  f="configs/$ds/id_sheaf/id_sheaf.yaml"
  if [ ! -e "$f" ]; then
    echo "!!! no Id-NSD config for $ds, skipping"
    continue
  fi
  echo "===== $(date -Is)  $f  --preset ${ds}_id_sheaf ====="
  "$SHEAF" sweep --yaml-path "$f" --preset "${ds}_id_sheaf" \
    || echo "!!! FAILED: $f (continuing with the next sweep)"
done
echo "===== ALL ID-NSD SWEEPS DONE $(date -Is) ====="
