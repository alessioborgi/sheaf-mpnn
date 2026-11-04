#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/texas/nsp/diagonal.yaml --preset texas_nsp_diagonal
sheaf sweep --yaml-path configs/texas/nsp/general.yaml --preset texas_nsp_general
sheaf sweep --yaml-path configs/texas/nsp/low_rank.yaml --preset texas_nsp_low_rank
sheaf sweep --yaml-path configs/texas/nsp/orthogonal.yaml --preset texas_nsp_orthogonal

sudo shutdown -t 120
