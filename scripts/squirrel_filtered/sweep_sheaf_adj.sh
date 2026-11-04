#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_attn/res_sheafan.yaml --preset squirrel_filtered_res_sheafan
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_adj/diagonal.yaml --preset squirrel_filtered_sheaf_adj_diagonal
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_adj/general.yaml --preset squirrel_filtered_sheaf_adj_general
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_adj/low_rank.yaml --preset squirrel_filtered_sheaf_adj_low_rank
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_adj/orthogonal.yaml --preset squirrel_filtered_sheaf_adj_orthogonal

sudo shutdown -t 120
