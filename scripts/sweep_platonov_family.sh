#!/bin/bash
# Run the full Platonov-et-al. sweep grid sequentially: every dataset x every
# restriction-map config of the given model families (default: poly_nsd).
# The --preset is read from each config's own documented header, which is the
# only source of the dataset+variant identity (see README: always pass it).
#
# Usage (inside tmux, from the repo root):
#   bash scripts/sweep_platonov_family.sh              # poly_nsd
#   bash scripts/sweep_platonov_family.sh nsd          # nsd instead
#   bash scripts/sweep_platonov_family.sh nsd poly_nsd # both, in sequence
#   bash scripts/sweep_platonov_family.sh csnn         # cooperative sheaf NNs
cd "$(dirname "$0")/.." || exit 1

DATASETS="roman_empire amazon_ratings minesweeper tolokers questions squirrel_filtered chameleon_filtered"
FAMILIES="${*:-poly_nsd}"

for fam in $FAMILIES; do
  for ds in $DATASETS; do
    for f in configs/"$ds"/"$fam"/*.yaml; do
      [ -e "$f" ] || { echo "!!! no configs for $ds/$fam, skipping"; continue; }
      p=$(grep -oE '\-\-preset [a-z_]+' "$f" | head -1 | awk '{print $2}')
      if [ -z "$p" ]; then
        echo "!!! $f documents no preset, skipping (identity would be wrong)"
        continue
      fi
      echo "===== $(date -Is)  $f  --preset $p ====="
      sheaf sweep --yaml-path "$f" --preset "$p" \
        || echo "!!! FAILED: $f (continuing with the next sweep)"
    done
  done
done
echo "===== ALL SWEEPS DONE $(date -Is) ====="
