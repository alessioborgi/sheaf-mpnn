#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/film/nsd/diagonal.yaml --preset film_nsd_diagonal
sheaf sweep --yaml-path configs/film/nsd/general.yaml --preset film_nsd_general
sheaf sweep --yaml-path configs/film/nsd/low_rank.yaml --preset film_nsd_low_rank
sheaf sweep --yaml-path configs/film/nsd/orthogonal.yaml --preset film_nsd_orthogonal

sudo shutdown -t 120
