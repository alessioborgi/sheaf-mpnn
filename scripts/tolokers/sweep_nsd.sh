#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/tolokers/nsd/diagonal.yaml --preset tolokers_nsd_diagonal
sheaf sweep --yaml-path configs/tolokers/nsd/general.yaml --preset tolokers_nsd_general
sheaf sweep --yaml-path configs/tolokers/nsd/low_rank.yaml --preset tolokers_nsd_low_rank
sheaf sweep --yaml-path configs/tolokers/nsd/orthogonal.yaml --preset tolokers_nsd_orthogonal

sudo shutdown -t 120
