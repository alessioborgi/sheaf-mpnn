#!/bin/bash
# Run every ConnNSD sweep sequentially: the nine Pei et al. datasets followed
# by the seven Platonov et al. ones. ConnNSD derives its transports from the
# raw features, so each dataset has exactly one config (no variant grid).
#
# Usage (inside tmux, from the repo root):
#   bash scripts/sweep_conn_nsd.sh                    # all 16 datasets
#   bash scripts/sweep_conn_nsd.sh pei                # Pei et al. only
#   bash scripts/sweep_conn_nsd.sh platonov           # Platonov et al. only
#   bash scripts/sweep_conn_nsd.sh cora texas         # an explicit subset
cd "$(dirname "$0")/.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PEI="cora citeseer pubmed chameleon squirrel film cornell texas wisconsin"
PLATONOV="roman_empire amazon_ratings minesweeper tolokers questions squirrel_filtered chameleon_filtered"

case "${1:-all}" in
  all)      DATASETS="$PEI $PLATONOV" ;;
  pei)      DATASETS="$PEI" ;;
  platonov) DATASETS="$PLATONOV" ;;
  *)        DATASETS="$*" ;;
esac

for ds in $DATASETS; do
  f="configs/$ds/conn_nsd/conn.yaml"
  if [ ! -e "$f" ]; then
    echo "!!! no ConnNSD config for $ds, skipping"
    continue
  fi
  echo "===== $(date -Is)  $f  --preset ${ds}_conn_nsd ====="
  sheaf sweep --yaml-path "$f" --preset "${ds}_conn_nsd" \
    || echo "!!! FAILED: $f (continuing with the next sweep)"
done
echo "===== ALL CONN-NSD SWEEPS DONE $(date -Is) ====="
