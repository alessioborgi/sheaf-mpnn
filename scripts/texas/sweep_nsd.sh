#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/texas/nsd/diagonal.yaml --preset texas_nsd_diagonal
sheaf sweep --yaml-path configs/texas/nsd/general.yaml --preset texas_nsd_general
sheaf sweep --yaml-path configs/texas/nsd/low_rank.yaml --preset texas_nsd_low_rank
sheaf sweep --yaml-path configs/texas/nsd/orthogonal.yaml --preset texas_nsd_orthogonal

sudo shutdown -t 120
