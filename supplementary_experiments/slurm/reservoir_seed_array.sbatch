#!/bin/bash
#SBATCH --job-name=aeno-res-seeds
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/res-seeds-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT"
mkdir -p supplementary_runs/reservoir_seeds logs
SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python re/run_reservoir_experiments.py \
  --domain reservoir --mode train_eval --variants full \
  --output_dir supplementary_runs/reservoir_seeds \
  --benchmark_cases re/reservoir_benchmark_cases.csv \
  --seed "$SEED" --timing_warmup 10 --timing_repeats 30
