#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cora/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/cora/nsd/general.yaml
sheaf sweep --yaml-path configs/cora/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/cora/nsd/orthogonal.yaml
