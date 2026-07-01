# Bristol RVE AENO V2: nodal Hex8 + equilibrium residual

This package keeps the microstructure-only Bristol RVE target fixed:

```text
chi_void(x) -> u(x), epsilon(x), sigma(x), E_eff, K_sigma95
E_s = 1.0, nu_s = 0.33, eps_xx_bar = 0.01
```

Main changes relative to V1b:

1. `AdmissibleAENO` now returns nodal displacement `u_nodal` with shape `[B,3,N+1,N+1,N+1]` for `N^3` voxel/Hex8 elements.
2. AENO strain/stress recovery is element-centered and uses the same Hex8 center B-matrix and node ordering as `aeno_rve/fem.py`.
3. Training uses a label-free objective

```text
L = mean(Pi(u_theta)/E_s) + lambda_r mean(||r_free(u_theta)||^2 / scale)
r = K(chi_void,E,nu) u_theta
```

where `K` is assembled implicitly by Hex8 element internal forces. No FEM displacement/stress labels are used in training.
4. The residual loss supports interface weighting through `w_e = 1 + alpha I_interface,e` with default `alpha=2`.

Recommended first training run:

```bash
python train.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --out_dir runs/aeno_rve32_microstructure_only_v2_nodal_residual \
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
  --residual_weight 1e-3 \
  --residual_ramp_epochs 25 \
  --interface_residual_alpha 2.0
```

Recommended evaluation:

```bash
python eval_fem_compare.py \
  --data data/bristol_25um_rve32_spatial_gray.npz \
  --checkpoint runs/aeno_rve32_microstructure_only_v2_nodal_residual/best.pt \
  --out_dir runs/aeno_rve32_microstructure_only_v2_nodal_residual/fem_compare_all_test \
  --num_samples 41 \
  --crop_size 32 \
  --eps0 0.01 \
  --Es_values 1.0 \
  --nu_values 0.33 \
  --fem_downsample 1 \
  --save_npz
```

If the residual term dominates the energy term, retry `--residual_weight 5e-4`. If the stress and K_sigma95 remain too smooth after the first run, retry `--residual_weight 2e-3` while keeping `--interface_residual_alpha 2.0`.
