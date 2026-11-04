#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/film/nsp/diagonal.yaml --preset film_nsp_diagonal
sheaf sweep --yaml-path configs/film/nsp/general.yaml --preset film_nsp_general
sheaf sweep --yaml-path configs/film/nsp/low_rank.yaml --preset film_nsp_low_rank
sheaf sweep --yaml-path configs/film/nsp/orthogonal.yaml --preset film_nsp_orthogonal

sudo shutdown -t 120
