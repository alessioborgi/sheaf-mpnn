#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/roman_empire/nsd/orthogonal.yaml --preset roman_empire_nsd_orthogonal
sheaf sweep --yaml-path configs/roman_empire/nsp/orthogonal.yaml --preset roman_empire_nsp_orthogonal
sheaf sweep --yaml-path configs/roman_empire/sheaf_adj/orthogonal.yaml --preset roman_empire_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/roman_empire/sheaf_attn/res_sheafan.yaml --preset roman_empire_res_sheafan

sudo shutdown -t 120
