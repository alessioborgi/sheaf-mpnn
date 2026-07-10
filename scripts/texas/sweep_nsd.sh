#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/texas/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/texas/nsd/general.yaml
sheaf sweep --yaml-path configs/texas/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/texas/nsd/orthogonal.yaml
