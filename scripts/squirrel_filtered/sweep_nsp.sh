#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/squirrel_filtered/nsp/diagonal.yaml --preset squirrel_filtered_nsp_diagonal
sheaf sweep --yaml-path configs/squirrel_filtered/nsp/general.yaml --preset squirrel_filtered_nsp_general
sheaf sweep --yaml-path configs/squirrel_filtered/nsp/low_rank.yaml --preset squirrel_filtered_nsp_low_rank
sheaf sweep --yaml-path configs/squirrel_filtered/nsp/orthogonal.yaml --preset squirrel_filtered_nsp_orthogonal

sudo shutdown -t 120
