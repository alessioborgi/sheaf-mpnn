#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cora/nsd/diagonal.yaml --preset cora_nsd_diagonal
sheaf sweep --yaml-path configs/cora/nsd/general.yaml --preset cora_nsd_general
sheaf sweep --yaml-path configs/cora/nsd/low_rank.yaml --preset cora_nsd_low_rank
sheaf sweep --yaml-path configs/cora/nsd/orthogonal.yaml --preset cora_nsd_orthogonal

sudo shutdown -t 120
