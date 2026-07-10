#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/wisconsin/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/wisconsin/nsd/general.yaml
sheaf sweep --yaml-path configs/wisconsin/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/wisconsin/nsd/orthogonal.yaml
