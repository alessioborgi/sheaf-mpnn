#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/roman_empire/nsd/diagonal.yaml
sheaf sweep --yaml-path configs/roman_empire/nsd/general.yaml
sheaf sweep --yaml-path configs/roman_empire/nsd/low_rank.yaml
sheaf sweep --yaml-path configs/roman_empire/nsd/orthogonal.yaml
