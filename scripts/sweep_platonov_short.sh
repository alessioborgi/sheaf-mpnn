#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sheaf sweep --yaml-path configs/roman_empire/nsd/orthogonal.yaml --preset roman_empire_nsd_orthogonal
sheaf sweep --yaml-path configs/roman_empire/nsp/orthogonal.yaml --preset roman_empire_nsp_orthogonal
sheaf sweep --yaml-path configs/roman_empire/sheaf_adj/orthogonal.yaml --preset roman_empire_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/roman_empire/sheaf_attn/res_sheafan.yaml --preset roman_empire_res_sheafan

sheaf sweep --yaml-path configs/amazon_ratings/nsd/orthogonal.yaml --preset amazon_ratings_nsd_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/nsp/orthogonal.yaml --preset amazon_ratings_nsp_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/sheaf_adj/orthogonal.yaml --preset amazon_ratings_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/amazon_ratings/sheaf_attn/res_sheafan.yaml --preset amazon_ratings_res_sheafan

sheaf sweep --yaml-path configs/minesweeper/nsd/orthogonal.yaml --preset minesweeper_nsd_orthogonal
sheaf sweep --yaml-path configs/minesweeper/nsp/orthogonal.yaml --preset minesweeper_nsp_orthogonal
sheaf sweep --yaml-path configs/minesweeper/sheaf_adj/orthogonal.yaml --preset minesweeper_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/minesweeper/sheaf_attn/res_sheafan.yaml --preset minesweeper_res_sheafan

sheaf sweep --yaml-path configs/tolokers/nsd/orthogonal.yaml --preset tolokers_nsd_orthogonal
sheaf sweep --yaml-path configs/tolokers/nsp/orthogonal.yaml --preset tolokers_nsp_orthogonal
sheaf sweep --yaml-path configs/tolokers/sheaf_adj/orthogonal.yaml --preset tolokers_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/tolokers/sheaf_attn/res_sheafan.yaml --preset tolokers_res_sheafan

sheaf sweep --yaml-path configs/questions/nsd/orthogonal.yaml --preset questions_nsd_orthogonal
sheaf sweep --yaml-path configs/questions/nsp/orthogonal.yaml --preset questions_nsp_orthogonal
sheaf sweep --yaml-path configs/questions/sheaf_adj/orthogonal.yaml --preset questions_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/questions/sheaf_attn/res_sheafan.yaml --preset questions_res_sheafan

sheaf sweep --yaml-path configs/squirrel_filtered/nsd/orthogonal.yaml --preset squirrel_filtered_nsd_orthogonal
sheaf sweep --yaml-path configs/squirrel_filtered/nsp/orthogonal.yaml --preset squirrel_filtered_nsp_orthogonal
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_adj/orthogonal.yaml --preset squirrel_filtered_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/squirrel_filtered/sheaf_attn/res_sheafan.yaml --preset squirrel_filtered_res_sheafan

sheaf sweep --yaml-path configs/chameleon_filtered/nsd/orthogonal.yaml --preset chameleon_filtered_nsd_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/nsp/orthogonal.yaml --preset chameleon_filtered_nsp_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/sheaf_adj/orthogonal.yaml --preset chameleon_filtered_sheaf_adj_orthogonal
sheaf sweep --yaml-path configs/chameleon_filtered/sheaf_attn/res_sheafan.yaml --preset chameleon_filtered_res_sheafan

sudo shutdown -t 120
