#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/roman_empire/nsp/diagonal.yaml --preset roman_empire_nsp_diagonal
sheaf sweep --yaml-path configs/roman_empire/nsp/general.yaml --preset roman_empire_nsp_general
sheaf sweep --yaml-path configs/roman_empire/nsp/low_rank.yaml --preset roman_empire_nsp_low_rank
sheaf sweep --yaml-path configs/roman_empire/nsp/orthogonal.yaml --preset roman_empire_nsp_orthogonal

sudo shutdown -t 120
