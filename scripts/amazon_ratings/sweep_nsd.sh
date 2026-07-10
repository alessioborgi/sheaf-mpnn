#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/amazon_ratings/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/amazon_ratings/nsd/general.yaml
sheaf sweep --yaml-path configs/amazon_ratings/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/amazon_ratings/nsd/orthogonal.yaml
