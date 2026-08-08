#!/bin/bash
#SBATCH --job-name=aeno-res-fd
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/res-fd-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT"
mkdir -p supplementary_runs/reservoir_fd logs
NAMES=(h2p5e3 h1e2)
VALUES=(0.0025 0.01)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
VALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}

python re/run_reservoir_experiments.py \
  --domain reservoir --mode train_eval --variants full \
  --output_dir "supplementary_runs/reservoir_fd/$NAME" \
  --benchmark_cases re/reservoir_benchmark_cases.csv \
  --seed 42 --fd_h "$VALUE" --timing_warmup 10 --timing_repeats 30
