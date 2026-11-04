#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/chameleon_filtered/nsd/diagonal.yaml --preset chameleon_filtered_nsd_diagonal
sheaf sweep --yaml-path configs/chameleon_filtered/nsd/general.yaml --preset chameleon_filtered_nsd_general
sheaf sweep --yaml-path configs/chameleon_filtered/nsd/low_rank.yaml --preset chameleon_filtered_nsd_low_rank
sheaf sweep --yaml-path configs/chameleon_filtered/nsd/orthogonal.yaml --preset chameleon_filtered_nsd_orthogonal

sudo shutdown -t 120
