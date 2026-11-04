#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/tolokers/nsp/diagonal.yaml --preset tolokers_nsp_diagonal
sheaf sweep --yaml-path configs/tolokers/nsp/general.yaml --preset tolokers_nsp_general
sheaf sweep --yaml-path configs/tolokers/nsp/low_rank.yaml --preset tolokers_nsp_low_rank
sheaf sweep --yaml-path configs/tolokers/nsp/orthogonal.yaml --preset tolokers_nsp_orthogonal

sudo shutdown -t 120
