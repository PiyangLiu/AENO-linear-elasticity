# AENO supplementary experiments

This directory contains reproducible drivers and aggregation tools for the
additional validation experiments requested during manuscript revision. All
new outputs are written below `supplementary_runs/`; the reported legacy
checkpoints and logs are not overwritten.

## Statistical protocol

- The complete reservoir and CT-RVE models use three independent seeds.
- The CT-RVE energy-only comparison changes only `residual_weight` from
  `5e-4` to `0`; all other training resources remain fixed.
- Sensitivity studies use one predeclared seed and change one numerical
  parameter at a time.
- CT-RVE checkpoints are selected by the minimum label-free validation
  objective. Reservoir runs use the final checkpoint after a fixed 400-epoch
  budget. Test errors are never used for selection.
- Latency measurements use single-sample inference after warm-up, explicit
  CUDA synchronization and 30 repeated measurements.

The revision reservoir benchmark uses the explicit 50-case table in
`re/reservoir_benchmark_cases.csv`. These values are the fixed, two-decimal
case definitions recorded in the original batch-evaluation log. Treat the
newly generated metrics as the reproducible revision results; do not force the
new summaries to reproduce the legacy rounded metrics exactly.

## Cluster preparation

From the repository root, activate the cluster environment and create the
directories required before Slurm opens its output files:

```bash
export ROOT=$PWD
mkdir -p logs supplementary_runs
pip install -r RVE/requirements.txt
```

Edit the `#SBATCH` account/partition lines locally if required by the cluster.

For threshold sensitivity, retain the original crop origins and split labels
while changing only the global segmentation quantile:

```bash
cd "$ROOT/RVE"
python make_threshold_sensitivity_dataset.py \
  --tif_dir 25micron_60min \
  --base_data data/bristol_25um_rve32_spatial_gray.npz \
  --out data/bristol_25um_rve32_threshold015.npz \
  --threshold_quantile 0.015 --store_gray

python make_threshold_sensitivity_dataset.py \
  --tif_dir 25micron_60min \
  --base_data data/bristol_25um_rve32_spatial_gray.npz \
  --out data/bristol_25um_rve32_threshold025.npz \
  --threshold_quantile 0.025 --store_gray
cd "$ROOT"
```

The raw Bristol TIFF slices are needed only for these two dataset-generation
commands and must not be redistributed.

## Submit the experiment matrix

### Linux server without a scheduler

On a single-GPU Linux server, run the complete matrix sequentially with:

```bash
nohup python supplementary_experiments/run_all_experiments.py \
  --root "$ROOT" --gpu 0 \
  > logs/all_experiments.out 2>&1 &
echo $! > logs/all_experiments.pid
```

The runner skips tasks whose completion markers already exist, so the same
command can resume an interrupted suite. Add `--force` only when every result
must be regenerated.

### Slurm

```bash
sbatch supplementary_experiments/slurm/rve_seed_array.sbatch
sbatch supplementary_experiments/slurm/rve_sensitivity_array.sbatch
sbatch supplementary_experiments/slurm/reservoir_seed_array.sbatch
sbatch supplementary_experiments/slurm/reservoir_ablation_array.sbatch
sbatch supplementary_experiments/slurm/reservoir_fd_array.sbatch
sbatch supplementary_experiments/slurm/reservoir_mesh_convergence.sbatch
```

After `full_seed0` exists, submit the synthetic topology analysis:

```bash
sbatch supplementary_experiments/slurm/rve_topology_eval.sbatch
```

## Aggregate manuscript tables

After all jobs finish:

```bash
python supplementary_experiments/summarize_runs.py \
  --rve_root supplementary_runs/rve \
  --rve_sensitivity_root supplementary_runs/rve_sensitivity \
  --reservoir_root supplementary_runs/reservoir_seeds \
  --reservoir_ablation_root supplementary_runs/reservoir_ablation \
  --reservoir_fd_root supplementary_runs/reservoir_fd \
  --out_dir supplementary_runs/tables
```

The generated files map to the manuscript as follows:

- `S6_morphology_error_associations.csv`: morphology associations.
- `S7_objective_comparison.csv`: energy-only versus energy plus equilibrium.
- `S8_reservoir_ablation.csv`: reservoir ablations.
- `S9_rve_sensitivity_runs.csv` and `S9_reservoir_fd_sensitivity.csv`:
  numerical sensitivity results.
- `S10_seed_variability.csv`: independent-seed variability.
- `S11_cost_break_even.csv`: training cost, latency and break-even calls.
- `reservoir_mesh/S13_mesh_convergence.csv`: reservoir FEM convergence.

## Resource estimate

The primary matrix contains 12 CT-RVE training jobs and 10 reservoir training
jobs. The mesh-convergence job is CPU-only and has the largest memory request.
If the cluster has fewer GPUs, reduce Slurm array concurrency, for example
`--array=0-5%2`, without changing the statistical protocol.
