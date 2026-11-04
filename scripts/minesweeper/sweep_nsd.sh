#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/minesweeper/nsd/diagonal.yaml --preset minesweeper_nsd_diagonal
sheaf sweep --yaml-path configs/minesweeper/nsd/general.yaml --preset minesweeper_nsd_general
sheaf sweep --yaml-path configs/minesweeper/nsd/low_rank.yaml --preset minesweeper_nsd_low_rank
sheaf sweep --yaml-path configs/minesweeper/nsd/orthogonal.yaml --preset minesweeper_nsd_orthogonal

sudo shutdown -t 120
