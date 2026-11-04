#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cora/sheaf_attn/res_sheafan.yaml --preset cora_res_sheafan
sheaf sweep --yaml-path configs/cora/sheaf_adj/diagonal.yaml --preset cora_sheaf_adj_diagonal
sheaf sweep --yaml-path configs/cora/sheaf_adj/general.yaml --preset cora_sheaf_adj_general
sheaf sweep --yaml-path configs/cora/sheaf_adj/low_rank.yaml --preset cora_sheaf_adj_low_rank
sheaf sweep --yaml-path configs/cora/sheaf_adj/orthogonal.yaml --preset cora_sheaf_adj_orthogonal

sudo shutdown -t 120
