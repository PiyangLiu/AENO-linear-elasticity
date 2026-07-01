# AENO benchmark code for linear-elastic neural operators

This repository contains the code and benchmark assets used for the manuscript
on the admissible expected-energy neural operator (AENO), a physics-structured
neural operator for repeated solution of fixed-domain linear-elastic
boundary-value problem families.

AENO is designed to predict displacement fields while preserving mechanical
admissibility. The reported experiments use label-free mechanics objectives
rather than paired finite-element field labels for training. Finite-element
solutions are used as references for evaluation and benchmarking.

## Repository structure

```text
MMS/
  mms.py                         Manufactured-solution verification.

re/
  train_reservoir_operator.py    Reservoir geostress operator training.
  eval_reservoir_operator.py     Reservoir FEM comparison/evaluation.
  reservoir_operator_smooth.pth  Trained reservoir checkpoint.
  Batch_Evaluation_Log.txt       Batch evaluation log.
  Reservoir_Inversion_Case_*.png Example evaluation figures.

RVE/
  aeno_rve/                      CT-RVE AENO model, elasticity and FEM modules.
  data/                          Processed Bristol CT-RVE and synthetic tests.
  runs/                          Reported checkpoints and evaluation outputs.
  make_dataset.py                Bristol CT-RVE dataset construction.
  make_dataset_synthetic_void_rve32.py
                                  Synthetic topology-shift benchmark generation.
  train.py                       Label-free CT-RVE AENO training.
  eval_fem_compare.py            FEM comparison and field/error export.
  requirements.txt               Python dependencies for the RVE workflow.
```

The three benchmark directories correspond to the three validation regimes in
the manuscript:

- `MMS`: manufactured-solution verification of displacement, strain, stress and
  energy recovery.
- `re`: held-out parameter-space prediction for the reservoir geostress
  benchmark.
- `RVE`: same-volume spatial holdout on Bristol CT-derived RVEs, plus synthetic
  topology-shift stress tests.

## Scientific scope

The code is intended to reproduce the computational evidence for AENO within
the regimes tested in the manuscript. In particular:

- the reservoir benchmark evaluates interpolation over held-out physical
  controls;
- the CT-RVE benchmark evaluates spatially held-out crops from the same
  tomographic volume;
- the synthetic RVE cases are topology-shift stress tests and should be treated
  as probes of extrapolation limits rather than evidence of broad
  microstructure generalization.

## Environment

The RVE workflow was run with Python and PyTorch. From the repository root:

```bash
cd RVE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The reservoir and manufactured-solution scripts use the same core scientific
Python stack: NumPy, SciPy, PyTorch and Matplotlib.

## Data

The CT-RVE benchmark is derived from the public University of Bristol X-ray
computed tomography dataset:

> X-Ray Computed Tomography Scans of a Composite Laminate with Voids,
> https://doi.org/10.5523/bris.1ywqtm9gy6fgl2lwz17ukn2obr

The processed RVE files used by the reported workflow are in `RVE/data/`:

- `bristol_25um_rve32_spatial_gray.npz`
- `bristol_25um_rve32_spatial_gray.json`
- `bristol_25um_rve32_spatial_gray.index.csv`
- `synthetic_void_rve32_test.npz`
- `synthetic_void_rve32_test.index.csv`

The raw Bristol XCT data are third-party data and are not relicensed by this
repository. If this repository is archived publicly, cite the original Bristol
dataset DOI and retain any required attribution and reuse terms from the
source dataset.

## Manufactured-solution benchmark

Run from the repository root:

```bash
cd MMS
python mms.py
```

This script verifies the AENO formulation on manufactured linear-elastic
solutions where analytical fields are available.

## Reservoir geostress benchmark

Run from the repository root:

```bash
cd re
python train_reservoir_operator.py
python eval_reservoir_operator.py
```

`train_reservoir_operator.py` trains the reservoir operator using an
energy-based mechanics objective over randomly sampled material and boundary
controls. `eval_reservoir_operator.py` loads `reservoir_operator_smooth.pth`,
runs the FEM comparison and writes the reservoir case figures and batch log.

## CT-RVE benchmark

Run all commands from `RVE/`.

### Dataset construction

If reconstructing the processed Bristol RVE dataset from local TIFF slices:

```bash
python make_dataset.py \
  --tif_dir 25micron_60min \
  --out data/bristol_25um_rve32_spatial_gray.npz \
  --crop_size 32 \
  --stride 32 \
  --target_porosity 0.02 \
  --min_porosity 0.001 \
  --max_rves 0 \
  --split_mode spatial_z_blocks
```

### Reported CT-RVE training run

The reported CT-RVE model was trained with the following command:

```bash
python train.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --out_dir runs/aeno_rve32_v2_res5e4_bubble085 \
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
  --residual_weight 5e-4 \
  --residual_ramp_epochs 25 \
  --interface_residual_alpha 2.0 \
  --bubble_power 0.85
```

The CT-RVE training objective combines expected elastic energy with a small
equilibrium residual penalty. No paired FEM displacement, strain or stress
labels are used in training.

### Main CT-RVE FEM comparison

The main held-out CT-RVE test comparison was run with:

```bash
python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --checkpoint runs/aeno_rve32_v2_res5e4_bubble085/best.pt \
  --out_dir runs/aeno_rve32_v2_res5e4_bubble085/fem_compare_all_test \
  --num_samples 41 \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 1.0 \
  --nu_values 0.33 \
  --fem_downsample 1 \
  --save_npz
```

The output directory contains per-sample FEM comparisons, exported fields and
summary metrics used for the CT-RVE results in the manuscript.

### Synthetic topology-shift tests

Synthetic RVE data are generated by:

```bash
python make_dataset_synthetic_void_rve32.py
```

They are evaluated using the same FEM-comparison machinery as the Bristol
spatial holdout. These cases are intended to reveal robustness and failure
modes under geometry shift, especially for derivative-sensitive strain and
stress-concentration quantities.

## Reported outputs

The reported CT-RVE run directory is:

```text
RVE/runs/aeno_rve32_v2_res5e4_bubble085/
```

Important files include:

- `best.pt`: checkpoint used for reported evaluation;
- `config.json`: training configuration;
- `training_log.csv`: training trace;
- `fem_compare_all_test/`: held-out Bristol CT-RVE FEM comparison outputs;
- `eval_synthetic_ood/`: synthetic topology-shift evaluation outputs.

For a public release, intermediate epoch checkpoints can be omitted if
`best.pt`, `config.json`, `training_log.csv`, evaluation outputs and source data
are retained.

## Reproducibility notes

- Randomness enters neural-network initialization, data loading and stochastic
  sampling. Exact bitwise reproduction may depend on the GPU, PyTorch version
  and CUDA/cuDNN settings.
- The CT-RVE test split is a same-volume spatial holdout and should not be
  interpreted as independent-specimen transfer.
- FEM references are used only for evaluation in the reported CT-RVE workflow.
  The training objective is label-free with respect to FEM field labels.
- The synthetic topology-shift benchmark should be reported separately from
  same-volume Bristol CT-RVE accuracy.

## Citation

If you use this repository, please cite the manuscript:

```text
[Author list], [Admissible energy training enables label-free neural operators for linear elasticity], [Journal], [2026], [DOI when available].
```

Please also cite the Bristol XCT source dataset when using the CT-RVE data:

```text
X-Ray Computed Tomography Scans of a Composite Laminate with Voids.
University of Bristol. https://doi.org/10.5523/bris.1ywqtm9gy6fgl2lwz17ukn2obr
```

## Licence

See `licence.md`. The repository licence does not override the terms of
third-party datasets, including the Bristol XCT source data.
