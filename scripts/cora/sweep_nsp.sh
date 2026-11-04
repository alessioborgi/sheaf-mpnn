#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cora/nsp/diagonal.yaml --preset cora_nsp_diagonal
sheaf sweep --yaml-path configs/cora/nsp/general.yaml --preset cora_nsp_general
sheaf sweep --yaml-path configs/cora/nsp/low_rank.yaml --preset cora_nsp_low_rank
sheaf sweep --yaml-path configs/cora/nsp/orthogonal.yaml --preset cora_nsp_orthogonal

sudo shutdown -t 120
