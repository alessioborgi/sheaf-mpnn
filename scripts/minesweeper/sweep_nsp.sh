#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/minesweeper/nsp/diagonal.yaml --preset minesweeper_nsp_diagonal
sheaf sweep --yaml-path configs/minesweeper/nsp/general.yaml --preset minesweeper_nsp_general
sheaf sweep --yaml-path configs/minesweeper/nsp/low_rank.yaml --preset minesweeper_nsp_low_rank
sheaf sweep --yaml-path configs/minesweeper/nsp/orthogonal.yaml --preset minesweeper_nsp_orthogonal

sudo shutdown -t 120
