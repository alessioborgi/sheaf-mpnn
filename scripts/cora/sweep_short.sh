#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/cora/nsd/orthogonal.yaml --preset cora_nsd_orthogonal
sheaf sweep --yaml-path configs/cora/nsp/orthogonal.yaml --preset cora_nsp_orthogonal
sheaf sweep --yaml-path configs/cora/sheaf_adj/orthogonal.yaml --preset cora_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/cora/sheaf_attn/res_sheafan.yaml --preset cora_res_sheafan

sudo shutdown -t 120
