#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/film/nsd/orthogonal.yaml --preset film_nsd_orthogonal
sheaf sweep --yaml-path configs/film/nsp/orthogonal.yaml --preset film_nsp_orthogonal
sheaf sweep --yaml-path configs/film/sheaf_adj/orthogonal.yaml --preset film_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/film/sheaf_attn/res_sheafan.yaml --preset film_res_sheafan

sudo shutdown -t 120
