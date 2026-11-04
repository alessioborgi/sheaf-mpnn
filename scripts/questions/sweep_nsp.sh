#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/questions/nsp/diagonal.yaml --preset questions_nsp_diagonal
sheaf sweep --yaml-path configs/questions/nsp/general.yaml --preset questions_nsp_general
sheaf sweep --yaml-path configs/questions/nsp/low_rank.yaml --preset questions_nsp_low_rank
sheaf sweep --yaml-path configs/questions/nsp/orthogonal.yaml --preset questions_nsp_orthogonal

sudo shutdown -t 120
