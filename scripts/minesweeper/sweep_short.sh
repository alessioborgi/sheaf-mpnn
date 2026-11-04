#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/minesweeper/nsd/orthogonal.yaml --preset minesweeper_nsd_orthogonal
sheaf sweep --yaml-path configs/minesweeper/nsp/orthogonal.yaml --preset minesweeper_nsp_orthogonal
sheaf sweep --yaml-path configs/minesweeper/sheaf_adj/orthogonal.yaml --preset minesweeper_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/minesweeper/sheaf_attn/res_sheafan.yaml --preset minesweeper_res_sheafan

sudo shutdown -t 120
