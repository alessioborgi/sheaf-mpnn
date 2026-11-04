#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/pubmed/nsd/diagonal.yaml --preset pubmed_nsd_diagonal
sheaf sweep --yaml-path configs/pubmed/nsd/general.yaml --preset pubmed_nsd_general
sheaf sweep --yaml-path configs/pubmed/nsd/low_rank.yaml --preset pubmed_nsd_low_rank
sheaf sweep --yaml-path configs/pubmed/nsd/orthogonal.yaml --preset pubmed_nsd_orthogonal

sudo shutdown -t 120
