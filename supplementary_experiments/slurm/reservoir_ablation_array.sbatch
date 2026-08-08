#!/bin/bash
#SBATCH --job-name=aeno-res-ablation
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/res-ablation-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT"
mkdir -p supplementary_runs/reservoir_ablation logs
VARIANTS=(full soft_bc concat_cond highfreq_pe random_sampling)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

python re/run_reservoir_experiments.py \
  --domain reservoir --mode train_eval --variants "$VARIANT" \
  --output_dir supplementary_runs/reservoir_ablation \
  --benchmark_cases re/reservoir_benchmark_cases.csv \
  --seed 42 --timing_warmup 10 --timing_repeats 30
