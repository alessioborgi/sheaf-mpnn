#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/amazon_ratings/nsd/diagonal.yaml --preset amazon_ratings_nsd_diagonal
sheaf sweep --yaml-path configs/amazon_ratings/nsd/general.yaml --preset amazon_ratings_nsd_general
sheaf sweep --yaml-path configs/amazon_ratings/nsd/low_rank.yaml --preset amazon_ratings_nsd_low_rank
sheaf sweep --yaml-path configs/amazon_ratings/nsd/orthogonal.yaml --preset amazon_ratings_nsd_orthogonal

sudo shutdown -t 120
