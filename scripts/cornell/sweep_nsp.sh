#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cornell/nsp/diagonal.yaml --preset cornell_nsp_diagonal
sheaf sweep --yaml-path configs/cornell/nsp/general.yaml --preset cornell_nsp_general
sheaf sweep --yaml-path configs/cornell/nsp/low_rank.yaml --preset cornell_nsp_low_rank
sheaf sweep --yaml-path configs/cornell/nsp/orthogonal.yaml --preset cornell_nsp_orthogonal

sudo shutdown -t 120
