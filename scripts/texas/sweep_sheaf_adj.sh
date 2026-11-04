#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/texas/sheaf_attn/res_sheafan.yaml --preset texas_res_sheafan
sheaf sweep --yaml-path configs/texas/sheaf_adj/diagonal.yaml --preset texas_sheaf_adj_diagonal
sheaf sweep --yaml-path configs/texas/sheaf_adj/general.yaml --preset texas_sheaf_adj_general
sheaf sweep --yaml-path configs/texas/sheaf_adj/low_rank.yaml --preset texas_sheaf_adj_low_rank
sheaf sweep --yaml-path configs/texas/sheaf_adj/orthogonal.yaml --preset texas_sheaf_adj_orthogonal

sudo shutdown -t 120
