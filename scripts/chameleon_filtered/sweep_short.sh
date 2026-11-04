#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/chameleon_filtered/nsd/orthogonal.yaml --preset chameleon_filtered_nsd_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/nsp/orthogonal.yaml --preset chameleon_filtered_nsp_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/sheaf_adj/orthogonal.yaml --preset chameleon_filtered_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/sheaf_attn/res_sheafan.yaml --preset chameleon_filtered_res_sheafan

sudo shutdown -t 120
