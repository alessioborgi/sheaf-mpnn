#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/wisconsin/nsd/orthogonal.yaml --preset wisconsin_nsd_orthogonal
sheaf sweep --yaml-path configs/wisconsin/nsp/orthogonal.yaml --preset wisconsin_nsp_orthogonal
sheaf sweep --yaml-path configs/wisconsin/sheaf_adj/orthogonal.yaml --preset wisconsin_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/wisconsin/sheaf_attn/res_sheafan.yaml --preset wisconsin_res_sheafan

sudo shutdown -t 120
