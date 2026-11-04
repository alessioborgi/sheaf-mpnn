#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/roman_empire/nsd/diagonal.yaml --preset roman_empire_nsd_diagonal
sheaf sweep --yaml-path configs/roman_empire/nsd/general.yaml --preset roman_empire_nsd_general
sheaf sweep --yaml-path configs/roman_empire/nsd/low_rank.yaml --preset roman_empire_nsd_low_rank
sheaf sweep --yaml-path configs/roman_empire/nsd/orthogonal.yaml --preset roman_empire_nsd_orthogonal

sudo shutdown -t 120
