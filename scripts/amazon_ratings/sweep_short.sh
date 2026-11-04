#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/amazon_ratings/nsd/orthogonal.yaml --preset amazon_ratings_nsd_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/nsp/orthogonal.yaml --preset amazon_ratings_nsp_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/sheaf_adj/orthogonal.yaml --preset amazon_ratings_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/sheaf_attn/res_sheafan.yaml --preset amazon_ratings_res_sheafan

sudo shutdown -t 120
