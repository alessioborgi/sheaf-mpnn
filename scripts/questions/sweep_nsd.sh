#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/questions/nsd/diagonal.yaml --preset questions_nsd_diagonal
sheaf sweep --yaml-path configs/questions/nsd/general.yaml --preset questions_nsd_general
sheaf sweep --yaml-path configs/questions/nsd/low_rank.yaml --preset questions_nsd_low_rank
sheaf sweep --yaml-path configs/questions/nsd/orthogonal.yaml --preset questions_nsd_orthogonal

sudo shutdown -t 120
