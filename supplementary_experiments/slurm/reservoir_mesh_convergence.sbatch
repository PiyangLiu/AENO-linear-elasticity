#!/bin/bash
#SBATCH --job-name=aeno-res-mesh
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/res-mesh-%j.out

set -euo pipefail

ROOT=${ROOT:-$PWD}
cd "$ROOT"
mkdir -p supplementary_runs/reservoir_mesh

python re/reservoir_mesh_convergence.py \
  --cases re/reservoir_benchmark_cases.csv \
  --case_ids 1 13 25 39 50 \
  --grids 20x20x5 40x40x10 80x80x20 \
  --out_dir supplementary_runs/reservoir_mesh
