#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/pubmed/nsd/orthogonal.yaml --preset pubmed_nsd_orthogonal
sheaf sweep --yaml-path configs/pubmed/nsp/orthogonal.yaml --preset pubmed_nsp_orthogonal
sheaf sweep --yaml-path configs/pubmed/sheaf_adj/orthogonal.yaml --preset pubmed_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/pubmed/sheaf_attn/res_sheafan.yaml --preset pubmed_res_sheafan

sudo shutdown -t 120
