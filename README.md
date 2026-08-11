# Numerics-informed neural operators for linear elasticity

This repository contains the code, processed benchmark data, trained-model
outputs and reproducibility workflows for the manuscript *Numerics-informed
neural operators for efficient linear elasticity simulations*.

The admissible expected-energy neural operator (AENO) constructs learned
linear-elastic solvers around the numerical formulation of each problem
family. Dirichlet conditions define the admissible displacement space,
mechanics objectives provide training signals without paired finite-element
field labels, and problem-matched numerical operators recover strain, stress
and effective quantities. FEM solutions are used as evaluation references,
not as paired training labels.

## Validation scope

The repository contains three complementary validation regimes:

- `MMS`: manufactured-solution verification against analytical displacement,
  strain, stress and energy fields;
- `re`: interpolation over held-out constitutive calibration and boundary
  controls for a fixed reservoir prior;
- `RVE`: spatially held-out `32^3` CT-derived RVEs from one Bristol XCT volume,
  together with zero-shot synthetic topology-shift tests.

These regimes have distinct claim boundaries. The CT-RVE test split evaluates
same-volume spatial holdout and is not evidence of independent-specimen
transfer. The synthetic geometries probe the limits of topology extrapolation
and are reported separately from the CT-RVE holdout accuracy.

## Repository structure

```text
MMS/
  mms.py                         Complete manufactured-solution workflow.

re/
  run_reservoir_experiments.py  Final reservoir training/evaluation entry point.
  reservoir_benchmark_cases.csv Fixed 50-case evaluation set.
  reservoir_mesh_convergence.py FEM reference-grid convergence study.
  ab/123.py                     Reservoir model, variants and evaluation code.
  ab/001.py                     Ablation-result plotting and table export.

RVE/
  aeno_rve/                     Model, features, elasticity and Hex8 FEM modules.
  data/                          Processed CT-RVE and synthetic benchmark data.
  25micron_60min/README.md       Third-party-data provenance and reconstruction note.
  make_dataset.py                Bristol CT-RVE dataset construction.
  make_threshold_sensitivity_dataset.py
                                  Segmentation-threshold sensitivity datasets.
  make_dataset_synthetic_void_rve32.py
                                  Synthetic topology-shift data generation.
  train.py                       Final label-free CT-RVE training entry point.
  eval_fem_compare.py            FEM comparison, timing and metric export.
  requirements.txt               Python dependencies.

supplementary_experiments/
  run_all_experiments.py         Resumable single-GPU experiment runner.
  summarize_runs.py              Supplementary-table aggregation.
  analyze_topology_descriptors.py
                                  Morphology/error association analysis.
  slurm/                         Slurm scripts for the revision experiment matrix.

supplementary_runs/
  tables/                        Source CSV files for Supplementary Tables S6--S11.
```

## Environment

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r RVE/requirements.txt
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The reported runs used Python 3.10.18, PyTorch 2.6.0 with CUDA 12.4,
SciPy 1.15.3 and NumPy 1.26.4. Neural-network runs used one NVIDIA L20 GPU.
Exact bitwise agreement can depend on the PyTorch, CUDA and cuDNN versions and
the selected deterministic kernels.

For the commands below, define the repository root on Linux:

```bash
export ROOT=$PWD
mkdir -p supplementary_runs logs
```

## Data

The CT-RVE benchmark is derived from the public University of Bristol dataset:

> *X-Ray Computed Tomography Scans of a Composite Laminate with Voids*  
> https://doi.org/10.5523/bris.1ywqtm9gy6fgl2lwz17ukn2obr

The raw Bristol TIF slices are third-party data and are not redistributed by
this repository. `RVE/25micron_60min/` must contain only its provenance README
in the public release. Users who reconstruct the processed benchmark must
obtain the raw volume from the original Bristol record and comply with its
reuse terms.

The processed files required by the reported workflows are under `RVE/data/`:

- `bristol_25um_rve32_spatial_gray.npz`, its JSON metadata and index CSV;
- `bristol_25um_rve32_threshold015.npz` and matching metadata/index files;
- `bristol_25um_rve32_threshold025.npz` and matching metadata/index files;
- `synthetic_void_rve32_test.npz`, its index CSV and overview image.

The principal CT-RVE dataset contains 237 training, 47 validation and 41 test
samples separated by non-overlapping crop-start layers along the XCT volume's
`z` direction. See `RVE/25micron_60min/README.md` for provenance, segmentation
and split details.

## Manufactured-solution benchmark

The MMS program performs the FEM calculation, AENO training, analytical-field
evaluation, metric reporting and figure generation in one run:

```bash
cd "$ROOT/MMS"
python mms.py \
  --timing-file ../supplementary_runs/mms_training_timing.json
```

The optimization uses 1,200 Adam epochs followed by at most 200 L-BFGS
iterations. `--training-only` records training timing without running the full
verification and is not the complete manuscript-reproduction command.

## Reservoir geostress benchmark

The final reservoir results use three independent seeds and the fixed 50-case
table in `re/reservoir_benchmark_cases.csv`. From the repository root, run:

```bash
python re/run_reservoir_experiments.py \
  --domain reservoir \
  --mode train_eval \
  --variants full \
  --output_dir supplementary_runs/reservoir_seeds \
  --benchmark_cases re/reservoir_benchmark_cases.csv \
  --seeds 42 43 44 \
  --device cuda \
  --epochs 400 \
  --batch_size 16 \
  --n_pts 8192 \
  --fd_h 0.005 \
  --timing_warmup 10 \
  --timing_repeats 30
```

Use `--mode train` and `--mode eval` with the same remaining arguments to run
training and evaluation separately. Reservoir checkpoints are taken after the
fixed 400-epoch budget; the 50 test cases are not used for model selection.

## CT-RVE benchmark

Run the final three-seed workflow from `RVE/`:

```bash
cd "$ROOT/RVE"

for SEED in 0 1 2; do
  RUN="../supplementary_runs/rve/full_seed${SEED}"

  python train.py \
    --data data/bristol_25um_rve32_spatial_gray.npz \
    --out_dir "$RUN" \
    --crop_size 32 \
    --epochs 300 \
    --batch_size 4 \
    --base_channels 20 \
    --lr 2e-4 \
    --weight_decay 1e-6 \
    --eps0 0.01 \
    --Es_min 1.0 --Es_max 1.0 \
    --nu_min 0.33 --nu_max 0.33 \
    --alpha_void 1e-4 \
    --porosity_balance_strength 0.3 \
    --high_nu_prob 0.0 \
    --residual_weight 5e-4 \
    --residual_ramp_epochs 25 \
    --interface_residual_alpha 2.0 \
    --residual_scale force \
    --bubble_power 0.85 \
    --seed "$SEED"

  python eval_fem_compare.py \
    --data data/bristol_25um_rve32_spatial_gray.npz \
    --checkpoint "$RUN/best.pt" \
    --out_dir "$RUN/eval_test" \
    --num_samples 41 \
    --split test \
    --crop_size 32 \
    --eps0 0.01 \
    --Es_values 1.0 \
    --nu_values 0.33 \
    --fem_downsample 1 \
    --timing_warmup 10 \
    --timing_repeats 30 \
    --no_plots
done
```

The CT-RVE training objective contains expected elastic energy and a normalized
free-degree-of-freedom equilibrium term with weight `5e-4`. No paired FEM
displacement, strain or stress fields are used during training. For each seed,
`best.pt` is selected by the minimum label-free validation objective; test
errors are not used for checkpoint or hyperparameter selection.

To export all seed-0 fields and per-sample plots, repeat the evaluation without
`--no_plots` and add `--save_npz`.

## Synthetic topology-shift tests

The synthetic geometries are evaluated zero-shot using the seed-0 CT-RVE
checkpoint:

```bash
cd "$ROOT/RVE"

python eval_fem_compare.py \
  --data data/synthetic_void_rve32_test.npz \
  --checkpoint ../supplementary_runs/rve/full_seed0/best.pt \
  --out_dir ../supplementary_runs/rve_topology/eval_synthetic \
  --num_samples 91 \
  --split test \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 1.0 \
  --nu_values 0.33 \
  --fem_downsample 1 \
  --timing_warmup 10 \
  --timing_repeats 30 \
  --no_plots

cd "$ROOT"
python supplementary_experiments/analyze_topology_descriptors.py \
  --data RVE/data/synthetic_void_rve32_test.npz \
  --metrics supplementary_runs/rve_topology/eval_synthetic/fem_comparison_metrics.csv \
  --out_dir supplementary_runs/rve_topology/descriptor_analysis
```

## Complete supplementary experiment matrix

The revision matrix includes independent seeds, the energy-only objective
comparison, reservoir ablations, numerical sensitivities, topology/error
associations, timing and reservoir mesh convergence. On a single-GPU Linux
server, run:

```bash
cd "$ROOT"
python supplementary_experiments/run_all_experiments.py \
  --root "$ROOT" \
  --gpu 0
```

The runner is resumable and skips completed tasks unless `--force` is supplied.
Slurm commands and resource estimates are documented in
`supplementary_experiments/README.md`.

After all experiments complete, regenerate the supplementary source tables:

```bash
python supplementary_experiments/summarize_runs.py \
  --rve_root supplementary_runs/rve \
  --rve_sensitivity_root supplementary_runs/rve_sensitivity \
  --reservoir_root supplementary_runs/reservoir_seeds \
  --reservoir_ablation_root supplementary_runs/reservoir_ablation \
  --reservoir_fd_root supplementary_runs/reservoir_fd \
  --out_dir supplementary_runs/tables
```

## Trained weights and reported outputs

The public repository retains source code, processed benchmark inputs and
compact source CSV/JSON files. The complete trained checkpoints, training logs,
per-case metrics and generated supplementary outputs are distributed as the
GitHub Release asset `supplementary_runs_v1.0.0.zip`:

https://github.com/PiyangLiu/AENO-linear-elasticity/releases

Intermediate epoch checkpoints are not required. For CT-RVE reproduction,
retain `best.pt`, `config.json`, `training_log.csv`, `run_summary.json` and the
evaluation CSV/JSON files for each reported seed. For the reservoir, retain the
three final checkpoints, training metadata, per-case CSV files and evaluation
summaries.

## Legacy entry points

The following files record earlier development workflows and are not the
authoritative entry points for the final manuscript results:

- `re/train_reservoir_operator.py` and `re/eval_reservoir_operator.py`;
- `RVE/train_microstructure_only_v2.sh` and
  `RVE/eval_microstructure_only_v2.sh`;
- the tuning commands in `RVE/README.md` and `RVE/README_V2.md`.

Use the commands in this root README and in
`supplementary_experiments/README.md` for manuscript reproduction.

## Reproducibility notes

- Manufactured-solution and reservoir neural calculations use float64.
- The CT-RVE network and differentiable stiffness application use float32;
  SciPy FEM references and NumPy metrics use float64.
- Latency measurements use single-sample inference, explicit CUDA
  synchronization, 10 warm-up calls and 30 timed repetitions.
- The CT-RVE FEM reference uses diagonally preconditioned conjugate gradients
  with relative tolerance `1e-8` and at most 2,000 iterations.
- The reservoir FEM reference uses the same solver class with relative
  tolerance `1e-6`.
- Timing excludes file input/output, rendering and host-transfer overhead as
  specified in the manuscript and Supplementary Information.

## Citation

Please cite the manuscript and the archived software release:

```text
Liu, P., Wang, J., Sun, S., Zhang, Z., Zhang, J., Zhang, K. & Zhang, L.
Numerics-informed neural operators for efficient linear elasticity simulations.
Manuscript prepared for Nature Computational Science (2026). DOI: to be assigned.
```

When using the CT-RVE data, also cite the original Bristol XCT dataset:

```text
X-Ray Computed Tomography Scans of a Composite Laminate with Voids.
University of Bristol. https://doi.org/10.5523/bris.1ywqtm9gy6fgl2lwz17ukn2obr
```

## Licence

See `licence.md`. The repository licence does not override the attribution or
reuse terms of third-party datasets, including the Bristol XCT source data.
