#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/minesweeper/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/minesweeper/nsd/general.yaml
sheaf sweep --yaml-path configs/minesweeper/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/minesweeper/nsd/orthogonal.yaml
