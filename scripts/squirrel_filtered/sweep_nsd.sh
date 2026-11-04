#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/squirrel_filtered/nsd/diagonal.yaml --preset squirrel_filtered_nsd_diagonal
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/general.yaml --preset squirrel_filtered_nsd_general
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/low_rank.yaml --preset squirrel_filtered_nsd_low_rank
sheaf sweep --yaml-path configs/squirrel_filtered/nsd/orthogonal.yaml --preset squirrel_filtered_nsd_orthogonal

sudo shutdown -t 120
