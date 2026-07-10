#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/tolokers/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/tolokers/nsd/general.yaml
sheaf sweep --yaml-path configs/tolokers/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/tolokers/nsd/orthogonal.yaml
