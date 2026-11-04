#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/pubmed/nsp/diagonal.yaml --preset pubmed_nsp_diagonal
sheaf sweep --yaml-path configs/pubmed/nsp/general.yaml --preset pubmed_nsp_general
sheaf sweep --yaml-path configs/pubmed/nsp/low_rank.yaml --preset pubmed_nsp_low_rank
sheaf sweep --yaml-path configs/pubmed/nsp/orthogonal.yaml --preset pubmed_nsp_orthogonal

sudo shutdown -t 120
