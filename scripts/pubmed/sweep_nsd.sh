#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/pubmed/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/pubmed/nsd/general.yaml
sheaf sweep --yaml-path configs/pubmed/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/pubmed/nsd/orthogonal.yaml
