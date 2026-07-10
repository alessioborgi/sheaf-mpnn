#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/squirrel_filtered/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/general.yaml
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/orthogonal.yaml
