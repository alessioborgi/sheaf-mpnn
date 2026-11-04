#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/wisconsin/nsp/diagonal.yaml --preset wisconsin_nsp_diagonal
sheaf sweep --yaml-path configs/wisconsin/nsp/general.yaml --preset wisconsin_nsp_general
sheaf sweep --yaml-path configs/wisconsin/nsp/low_rank.yaml --preset wisconsin_nsp_low_rank
sheaf sweep --yaml-path configs/wisconsin/nsp/orthogonal.yaml --preset wisconsin_nsp_orthogonal

sudo shutdown -t 120
