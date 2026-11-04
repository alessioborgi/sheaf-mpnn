#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/texas/nsd/orthogonal.yaml --preset texas_nsd_orthogonal
sheaf sweep --yaml-path configs/texas/nsp/orthogonal.yaml --preset texas_nsp_orthogonal
sheaf sweep --yaml-path configs/texas/sheaf_adj/orthogonal.yaml --preset texas_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/texas/sheaf_attn/res_sheafan.yaml --preset texas_res_sheafan

sudo shutdown -t 120
