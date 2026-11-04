#!/bin/bash
# Full 10-fold CSNN reproduction on the reference datasets.
#
# Reference numbers (arXiv:2507.00647, mean +/- std over 10 folds):
#   roman_empire    92.63 +/- 0.50  accuracy
#   amazon_ratings  52.07 +/- 1.00  accuracy
#   minesweeper     99.07 +/- 0.25  ROC AUC
#   tolokers        85.45 +/- 0.53  ROC AUC
#   questions       79.31 +/- 1.22  ROC AUC
#   squirrel        41.18 +/- 2.23  accuracy (the paper's directed-Wikipedia
#                                   table, not the Platonov table)
#
# Known data-pipeline deviations from the reference; expect drift, especially
# where directionality matters:
#   * Graph directedness: the reference trains on the directed graphs; this
#     pipeline symmetrises edges (to_undirected in exp/data.py), which blunts
#     CSNN's in/out-Laplacian asymmetry.
#   * squirrel identity: the reference "squirrel" is the directed filtered
#     Wikipedia graph -- closer to squirrel_filtered here than to the Geom-GCN
#     squirrel. Both presets run below; compare both against the paper.
#   * Binary head: the reference uses a single logit + BCE on
#     minesweeper/questions/tolokers; this pipeline uses a 2-logit softmax.
#     The ROC AUC metric itself is computed the same way on both sides.
#
# Usage (inside tmux, from the repo root):
#   bash scripts/csnn/run_csnn_repro.sh
cd "$(dirname "$0")/../.." || exit 1

PRESETS="roman_empire_csnn amazon_ratings_csnn minesweeper_csnn questions_csnn squirrel_csnn squirrel_filtered_csnn tolokers_csnn"

for p in $PRESETS; do
  echo "===== $(date -Is)  repro --preset $p ====="
  sheaf run --preset "$p" \
    || echo "!!! FAILED: $p (continuing with the next dataset)"
done
echo "===== CSNN REPRO DONE $(date -Is) ====="
