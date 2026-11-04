#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/chameleon_filtered/nsp/diagonal.yaml --preset chameleon_filtered_nsp_diagonal
sheaf sweep --yaml-path configs/chameleon_filtered/nsp/general.yaml --preset chameleon_filtered_nsp_general
sheaf sweep --yaml-path configs/chameleon_filtered/nsp/low_rank.yaml --preset chameleon_filtered_nsp_low_rank
sheaf sweep --yaml-path configs/chameleon_filtered/nsp/orthogonal.yaml --preset chameleon_filtered_nsp_orthogonal

sudo shutdown -t 120
