#!/bin/bash
#SBATCH --job-name=aeno-rve-sens
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rve-sens-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT/RVE"
mkdir -p ../supplementary_runs/rve_sensitivity logs

NAMES=(alpha1e5 alpha1e3 bubble070 bubble100 threshold015 threshold025)
ALPHAS=(1e-5 1e-3 1e-4 1e-4 1e-4 1e-4)
BUBBLES=(0.85 0.85 0.70 1.00 0.85 0.85)
DATASETS=(
  data/bristol_25um_rve32_spatial_gray.npz
  data/bristol_25um_rve32_spatial_gray.npz
  data/bristol_25um_rve32_spatial_gray.npz
  data/bristol_25um_rve32_spatial_gray.npz
  data/bristol_25um_rve32_threshold015.npz
  data/bristol_25um_rve32_threshold025.npz
)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
BUBBLE=${BUBBLES[$SLURM_ARRAY_TASK_ID]}
DATA=${DATASETS[$SLURM_ARRAY_TASK_ID]}
RUN="../supplementary_runs/rve_sensitivity/${NAME}_seed0"

python train.py \
  --data "$DATA" --out_dir "$RUN" \
  --crop_size 32 --epochs 300 --batch_size 4 --base_channels 20 \
  --lr 2e-4 --eps0 0.01 --alpha_void "$ALPHA" \
  --Es_min 1.0 --Es_max 1.0 --nu_min 0.33 --nu_max 0.33 \
  --porosity_balance_strength 0.3 --high_nu_prob 0.0 \
  --residual_weight 5e-4 --residual_ramp_epochs 25 \
  --interface_residual_alpha 2.0 --bubble_power "$BUBBLE" --seed 0

python eval_fem_compare.py \
  --data "$DATA" --checkpoint "$RUN/best.pt" --out_dir "$RUN/eval_test" \
  --num_samples 41 --crop_size 32 --eps0 0.01 --alpha_void "$ALPHA" \
  --Es_values 1.0 --nu_values 0.33 --fem_downsample 1 \
  --timing_warmup 10 --timing_repeats 30 --no_plots
