#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/wisconsin/nsd/diagonal.yaml --preset wisconsin_nsd_diagonal
sheaf sweep --yaml-path configs/wisconsin/nsd/general.yaml --preset wisconsin_nsd_general
sheaf sweep --yaml-path configs/wisconsin/nsd/low_rank.yaml --preset wisconsin_nsd_low_rank
sheaf sweep --yaml-path configs/wisconsin/nsd/orthogonal.yaml --preset wisconsin_nsd_orthogonal

sudo shutdown -t 120
