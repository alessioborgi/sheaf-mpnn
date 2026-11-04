#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cornell/nsd/orthogonal.yaml --preset cornell_nsd_orthogonal
sheaf sweep --yaml-path configs/cornell/nsp/orthogonal.yaml --preset cornell_nsp_orthogonal
sheaf sweep --yaml-path configs/cornell/sheaf_adj/orthogonal.yaml --preset cornell_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/cornell/sheaf_attn/res_sheafan.yaml --preset cornell_res_sheafan

sudo shutdown -t 120
