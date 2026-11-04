#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/tolokers/nsd/orthogonal.yaml --preset tolokers_nsd_orthogonal
sheaf sweep --yaml-path configs/tolokers/nsp/orthogonal.yaml --preset tolokers_nsp_orthogonal
sheaf sweep --yaml-path configs/tolokers/sheaf_adj/orthogonal.yaml --preset tolokers_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/tolokers/sheaf_attn/res_sheafan.yaml --preset tolokers_res_sheafan

sudo shutdown -t 120
