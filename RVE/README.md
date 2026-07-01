# AENO-RVE Bristol XCT benchmark — V1 tuning version

This version keeps the AENO framework unchanged:

\[
 u_\theta(x)=u_{affine}(x)+b(x)\hat u_\theta(x),\qquad
 \min_\theta \mathbb{E}_{\chi_v,E_s,\nu_s}[\Pi(u_\theta;\chi_v,E_s,\nu_s)].
\]

It only changes the implementation details that were responsible for the current weak points in the absolute-error figures.

## V1 changes

1. **Nondimensionalized `E_s` treatment**
   - The network no longer receives `E_s` as an input channel.
   - `E_s` still enters the energy, stress, FEM comparison and `E_eff` calculation analytically.
   - This does **not** change the generalization target. The model is still tested over `E_s`; the scaling is now handled physically instead of learned.

2. **Stronger local-prior encoding**
   The input channels are now:

   ```text
   chi_void, local_phi_3, local_phi_7, sdf_void, interface, boundary_distance, x, y, z, nu_norm
   ```

   `sdf_void`, `interface`, and `boundary_distance` are computed from each RVE crop and are returned by the dataset loader.

3. **Mild sampling correction rather than aggressive resampling**
   - Default `--porosity_balance_strength 0.5` mixes the natural RVE distribution with inverse-frequency porosity-bin balancing.
   - Default `--high_nu_prob 0.35` mildly increases exposure to high-`nu_s` samples during training only.
   - Validation and testing still use the natural held-out split.

4. **Replication padding in the 3D CNN**
   This reduces artificial boundary stripes in stress recovery.

## Dataset construction

Use the same spatially held-out split:

```bash
python make_dataset.py \
  --tif_dir /wangjie/bristol/25micron_60min \
  --out data/bristol_25um_rve32_spatial.npz \
  --crop_size 32 \
  --stride 32 \
  --target_porosity 0.02 \
  --min_porosity 0.001 \
  --max_rves 0 \
  --split_mode spatial_z_blocks
```

## Training V1

```bash
python train.py \
  --data data/bristol_25um_rve32_spatial.npz \
  --out_dir runs/aeno_rve32_spatial_v1 \
  --crop_size 32 \
  --epochs 300 \
  --batch_size 4 \
  --base_channels 20 \
  --lr 2e-4 \
  --eps0 0.01 \
  --Es_min 0.5 --Es_max 2.0 \
  --nu_min 0.25 --nu_max 0.40 \
  --porosity_balance_strength 0.5 \
  --high_nu_prob 0.35 \
  --high_nu_min 0.35
```

If GPU memory is limited, use `--base_channels 16` or `--batch_size 2`.

## Quick FEM comparison

```bash
python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial.npz \
  --checkpoint runs/aeno_rve32_spatial_v1/best.pt \
  --out_dir runs/aeno_rve32_spatial_v1/fem_compare_grid \
  --num_samples 6 \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 0.5 1.0 2.0 \
  --nu_values 0.25 0.33 0.40 \
  --fem_downsample 1 \
  --save_npz
```

## Full test set FEM comparison

```bash
python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial.npz \
  --checkpoint runs/aeno_rve32_spatial_v1/best.pt \
  --out_dir runs/aeno_rve32_spatial_v1/fem_compare_all_test \
  --num_samples 41 \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 0.5 1.0 2.0 \
  --nu_values 0.25 0.33 0.40 \
  --fem_downsample 1 \
  --save_npz
```

## What to compare against the previous version

Focus on:

- `rel_l2_u`
- `rel_l2_eps`
- `rel_l2_sig`
- `rel_l2_vm`
- `rel_err_Eeff`
- `rel_err_Ksigma95`
- the parity plots for `E_eff` and `K_sigma95`
- absolute-error panels for RVE 324, 300, 308 and 284

Expected improvements should mainly appear in high-porosity and high-`nu_s` cases, and in reduced boundary/stripe artifacts.
