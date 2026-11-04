#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/citeseer/nsp/diagonal.yaml --preset citeseer_nsp_diagonal
sheaf sweep --yaml-path configs/citeseer/nsp/general.yaml --preset citeseer_nsp_general
sheaf sweep --yaml-path configs/citeseer/nsp/low_rank.yaml --preset citeseer_nsp_low_rank
sheaf sweep --yaml-path configs/citeseer/nsp/orthogonal.yaml --preset citeseer_nsp_orthogonal

sudo shutdown -t 120
