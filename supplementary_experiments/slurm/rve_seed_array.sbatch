#!/bin/bash
#SBATCH --job-name=aeno-rve-seeds
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rve-seeds-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT/RVE"
mkdir -p ../supplementary_runs/rve logs

SEEDS=(0 1 2 0 1 2)
OBJECTIVES=(full full full energy_only energy_only energy_only)
WEIGHTS=(5e-4 5e-4 5e-4 0 0 0)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OBJECTIVE=${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}
WEIGHT=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
RUN="../supplementary_runs/rve/${OBJECTIVE}_seed${SEED}"

python train.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --out_dir "$RUN" \
  --crop_size 32 --epochs 300 --batch_size 4 --base_channels 20 \
  --lr 2e-4 --eps0 0.01 \
  --Es_min 1.0 --Es_max 1.0 --nu_min 0.33 --nu_max 0.33 \
  --porosity_balance_strength 0.3 --high_nu_prob 0.0 \
  --residual_weight "$WEIGHT" --residual_ramp_epochs 25 \
  --interface_residual_alpha 2.0 --bubble_power 0.85 \
  --seed "$SEED"

python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --checkpoint "$RUN/best.pt" \
  --out_dir "$RUN/eval_test" \
  --num_samples 41 --crop_size 32 --eps0 0.01 \
  --Es_values 1.0 --nu_values 0.33 --fem_downsample 1 \
  --timing_warmup 10 --timing_repeats 30 --no_plots
