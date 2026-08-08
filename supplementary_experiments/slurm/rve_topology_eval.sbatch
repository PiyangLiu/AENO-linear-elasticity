#!/bin/bash
#SBATCH --job-name=aeno-rve-topology
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rve-topology-%j.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT/RVE"
mkdir -p ../supplementary_runs/rve_topology

CHECKPOINT=${CHECKPOINT:-../supplementary_runs/rve/full_seed0/best.pt}
OUT=../supplementary_runs/rve_topology/eval_synthetic

python eval_fem_compare.py \
  --data data/synthetic_void_rve32_test.npz \
  --checkpoint "$CHECKPOINT" --out_dir "$OUT" \
  --num_samples 91 --split test --crop_size 32 --eps0 0.01 \
  --Es_values 1.0 --nu_values 0.33 --fem_downsample 1 \
  --timing_warmup 10 --timing_repeats 30 --no_plots

cd "$ROOT"
python supplementary_experiments/analyze_topology_descriptors.py \
  --data RVE/data/synthetic_void_rve32_test.npz \
  --metrics supplementary_runs/rve_topology/eval_synthetic/fem_comparison_metrics.csv \
  --out_dir supplementary_runs/rve_topology/descriptor_analysis
