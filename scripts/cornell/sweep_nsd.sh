#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cornell/nsd/diagonal.yaml --preset cornell_nsd_diagonal
sheaf sweep --yaml-path configs/cornell/nsd/general.yaml --preset cornell_nsd_general
sheaf sweep --yaml-path configs/cornell/nsd/low_rank.yaml --preset cornell_nsd_low_rank
sheaf sweep --yaml-path configs/cornell/nsd/orthogonal.yaml --preset cornell_nsd_orthogonal

sudo shutdown -t 120
