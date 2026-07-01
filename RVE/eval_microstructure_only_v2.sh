#!/usr/bin/env bash
set -euo pipefail
python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --checkpoint runs/aeno_rve32_microstructure_only_v2_nodal_residual/best.pt \
  --out_dir runs/aeno_rve32_microstructure_only_v2_nodal_residual/fem_compare_all_test \
  --num_samples 41 \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 1.0 \
  --nu_values 0.33 \
  --fem_downsample 1 \
  --save_npz
