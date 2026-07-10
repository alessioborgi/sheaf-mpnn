#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cornell/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/cornell/nsd/general.yaml
sheaf sweep --yaml-path configs/cornell/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/cornell/nsd/orthogonal.yaml
