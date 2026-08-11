#!/usr/bin/env python3
"""Reservoir Hex8 mesh-convergence study for Supplementary Table S13."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load_suite():
    path = Path(__file__).resolve().parent / "ab" / "123.py"
    spec = importlib.util.spec_from_file_location("reservoir_ablation_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_grid(text):
    parts = [int(v) for v in text.lower().split("x")]
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("Grid must have the form Nx x Ny x Nz, for example 40x40x10")
    return tuple(parts)


def relative_l2(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.linalg.norm((prediction - reference).ravel()) / (np.linalg.norm(reference.ravel()) + 1e-30))


def interpolate_nodal(U, cfg, query_nodes):
    axes = (
        np.linspace(0.0, cfg.Lx, cfg.Nx + 1),
        np.linspace(0.0, cfg.Ly, cfg.Ny + 1),
        np.linspace(0.0, cfg.Lz, cfg.Nz + 1),
    )
    values = U.reshape(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz + 1, 3)
    out = np.empty((len(query_nodes), 3), dtype=np.float64)
    for component in range(3):
        interp = RegularGridInterpolator(axes, values[..., component], bounds_error=True)
        out[:, component] = interp(query_nodes)
    return out.reshape(-1)


def interpolate_cell_field(field, cfg, query_centers):
    axes = (
        np.linspace(cfg.Lx / (2 * cfg.Nx), cfg.Lx - cfg.Lx / (2 * cfg.Nx), cfg.Nx),
        np.linspace(cfg.Ly / (2 * cfg.Ny), cfg.Ly - cfg.Ly / (2 * cfg.Ny), cfg.Ny),
        np.linspace(cfg.Lz / (2 * cfg.Nz), cfg.Lz - cfg.Lz / (2 * cfg.Nz), cfg.Nz),
    )
    return RegularGridInterpolator(axes, field, bounds_error=False, fill_value=None)(query_centers)


def solve_case(suite, V_np, case, grid):
    cfg = suite.ReservoirDomainConfig(Nx=grid[0], Ny=grid[1], Nz=grid[2])
    nodes, U, elapsed = suite.run_fem_reservoir(
        V_np, case["c_scale"], case["c_shift"], case["du_x"], case["dv_y"], cfg
    )
    exx, eyy, ezz = suite.recover_center_normal_strains(U, cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz)
    centers = suite.reservoir_center_points(cfg)
    lin = np.linspace(0.0, 1.0, V_np.shape[-1])
    prior_interp = RegularGridInterpolator((lin, lin, lin), V_np, bounds_error=False, fill_value=None)
    normalized = centers / np.array([cfg.Lx, cfg.Ly, cfg.Lz])
    E = (case["c_scale"] * prior_interp(normalized[:, [2, 1, 0]]) + case["c_shift"]).reshape(grid)
    lam = E * cfg.nu_const / ((1.0 + cfg.nu_const) * (1.0 - 2.0 * cfg.nu_const))
    mu = E / (2.0 * (1.0 + cfg.nu_const))
    sxx = lam * (exx + eyy + ezz) + 2.0 * mu * exx
    return {"cfg": cfg, "nodes": nodes, "centers": centers, "U": U, "exx": exx, "sxx": sxx, "time_s": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).resolve().parent / "reservoir_benchmark_cases.csv"))
    parser.add_argument("--case_ids", type=int, nargs="+", default=[1, 13, 25, 39, 50])
    parser.add_argument("--grids", type=parse_grid, nargs="+", default=[(20, 20, 5), (40, 40, 10), (80, 80, 20)])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    suite = load_suite()
    all_cases = {row["case_id"]: row for row in suite.load_reservoir_cases(args.cases)}
    cases = [all_cases[i] for i in args.case_ids]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V_np = suite.generate_seismic_data(32)[0, 0].cpu().numpy()

    rows = []
    for case in cases:
        solutions = {grid: solve_case(suite, V_np, case, grid) for grid in args.grids}
        finest_grid = args.grids[-1]
        fine = solutions[finest_grid]
        for grid in args.grids:
            current = solutions[grid]
            if grid == finest_grid:
                u_diff = exx_diff = sxx_diff = 0.0
            else:
                u_ref = interpolate_nodal(fine["U"], fine["cfg"], current["nodes"])
                exx_ref = interpolate_cell_field(fine["exx"], fine["cfg"], current["centers"])
                sxx_ref = interpolate_cell_field(fine["sxx"], fine["cfg"], current["centers"])
                u_diff = relative_l2(u_ref, current["U"])
                exx_diff = relative_l2(exx_ref, current["exx"].reshape(-1))
                sxx_diff = relative_l2(sxx_ref, current["sxx"].reshape(-1))
            rows.append({
                "case_id": case["case_id"],
                "grid": "x".join(map(str, grid)),
                "degrees_of_freedom": 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1),
                "u_difference_pct": 100.0 * u_diff,
                "exx_difference_pct": 100.0 * exx_diff,
                "sxx_difference_pct": 100.0 * sxx_diff,
                "fem_time_s": current["time_s"],
                "reference_grid": "x".join(map(str, finest_grid)),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "mesh_convergence_cases.csv", index=False)
    summary = frame.groupby("grid", sort=False).agg(
        degrees_of_freedom=("degrees_of_freedom", "first"),
        u_difference_mean_pct=("u_difference_pct", "mean"),
        u_difference_std_pct=("u_difference_pct", "std"),
        exx_difference_mean_pct=("exx_difference_pct", "mean"),
        exx_difference_std_pct=("exx_difference_pct", "std"),
        sxx_difference_mean_pct=("sxx_difference_pct", "mean"),
        sxx_difference_std_pct=("sxx_difference_pct", "std"),
        fem_time_mean_s=("fem_time_s", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "S13_mesh_convergence.csv", index=False)
    with open(out_dir / "mesh_convergence_config.json", "w", encoding="utf-8") as f:
        json.dump({"case_ids": args.case_ids, "grids": args.grids, "reference_grid": args.grids[-1]}, f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
