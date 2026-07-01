#!/usr/bin/env bash
set -euo pipefail
python train.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --out_dir runs/aeno_rve32_microstructure_only_v2_nodal_residual \
  --crop_size 32 \
  --epochs 300 \
  --batch_size 4 \
  --base_channels 20 \
  --lr 2e-4 \
  --eps0 0.01 \
  --Es_min 1.0 --Es_max 1.0 \
  --nu_min 0.33 --nu_max 0.33 \
  --porosity_balance_strength 0.3 \
  --high_nu_prob 0.0 \
  --residual_weight 1e-3 \
  --residual_ramp_epochs 25 \
  --interface_residual_alpha 2.0
