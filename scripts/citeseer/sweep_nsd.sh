#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/citeseer/nsd/diagonal.yaml --preset citeseer_nsd_diagonal
sheaf sweep --yaml-path configs/citeseer/nsd/general.yaml --preset citeseer_nsd_general
sheaf sweep --yaml-path configs/citeseer/nsd/low_rank.yaml --preset citeseer_nsd_low_rank
sheaf sweep --yaml-path configs/citeseer/nsd/orthogonal.yaml --preset citeseer_nsd_orthogonal

sudo shutdown -t 120
