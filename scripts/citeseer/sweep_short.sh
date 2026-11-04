#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/citeseer/nsd/orthogonal.yaml --preset citeseer_nsd_orthogonal
sheaf sweep --yaml-path configs/citeseer/nsp/orthogonal.yaml --preset citeseer_nsp_orthogonal
sheaf sweep --yaml-path configs/citeseer/sheaf_adj/orthogonal.yaml --preset citeseer_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/citeseer/sheaf_attn/res_sheafan.yaml --preset citeseer_res_sheafan

sudo shutdown -t 120
