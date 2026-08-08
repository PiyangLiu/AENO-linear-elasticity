#!/usr/bin/env python3
"""Aggregate AENO revision runs into Supplementary Tables S7--S11."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RVE_METRICS = {
    "displacement_error_pct": "displacement",
    "strain_error_pct": "strain",
    "stress_error_pct": "stress",
    "effective_modulus_error_pct": "effective_modulus",
    "Ksigma95_error_pct": "Ksigma95",
    "equilibrium_metric": "equilibrium_metric",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_up(path, name):
    for parent in [path.parent, *path.parents]:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def collect_rve(root):
    records = []
    if not root:
        return pd.DataFrame()
    for path in Path(root).rglob("evaluation_summary.json"):
        evaluation = read_json(path)
        train_path = find_up(path, "run_summary.json")
        config_path = find_up(path, "config.json")
        training = read_json(train_path) if train_path else {}
        config = read_json(config_path) if config_path else {}
        record = {
            "run_dir": str(train_path.parent if train_path else path.parent),
            "seed": int(evaluation.get("seed", training.get("seed", config.get("seed", -1)))),
            "objective": evaluation.get("objective", training.get("objective", "unknown")),
            "residual_weight": float(evaluation.get("residual_weight", config.get("residual_weight", np.nan))),
            "alpha_void": float(config.get("alpha_void", np.nan)),
            "bubble_power": float(config.get("bubble_power", np.nan)),
            "data": config.get("data", ""),
            "num_parameters": evaluation.get("num_parameters", training.get("num_parameters")),
            "training_time_s": training.get("training_time_s", np.nan),
            "preprocessing_time_s": training.get("dataset_and_feature_setup_time_s", np.nan),
            "aeno_latency_s": evaluation["statistics"]["aeno_latency_s"]["median"],
            "fem_time_s": evaluation["statistics"]["fem_total_time_s"]["median"],
        }
        for source, target in RVE_METRICS.items():
            record[target] = evaluation["statistics"][source]["mean"]
        records.append(record)
    return pd.DataFrame(records)


def collect_reservoir(root):
    records = []
    if not root:
        return pd.DataFrame()
    for path in Path(root).rglob("reservoir_*_eval_summary.json"):
        obj = read_json(path)
        match = re.search(r"seed(\d+)", str(path))
        seed = int(match.group(1)) if match else int(obj.get("seed", -1))
        train_candidates = list(path.parent.glob("reservoir_*_train.json"))
        training = read_json(train_candidates[0]) if train_candidates else {}
        records.append({
            "run_dir": str(path.parent),
            "variant": obj["variant"],
            "seed": seed,
            "num_parameters": obj.get("num_parameters"),
            "training_time_s": training.get("seconds", np.nan),
            "preprocessing_time_s": 0.0,
            "displacement": obj["avg_u_err_pct"],
            "strain": obj["avg_exx_err_pct"],
            "stress": obj["avg_sxx_err_pct"],
            "aeno_latency_s": obj.get("median_op_time_s", obj["avg_op_time_s"]),
            "fem_time_s": obj.get("median_fem_time_s", obj["avg_fem_time_s"]),
            "max_bc_violation": obj["max_bc_violation"],
            "fd_h": training.get("ablation_config", {}).get("fd_h", np.nan),
            "num_cases": obj["num_cases"],
        })
    return pd.DataFrame(records)


def format_mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{np.mean(values):.6g} +/- {np.std(values, ddof=1) if len(values) > 1 else 0.0:.6g}"


def objective_table(frame):
    rows = []
    for objective, group in frame.groupby("objective"):
        row = {"objective": objective, "n_seeds": len(group)}
        for metric in RVE_METRICS.values():
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd_across_seeds"] = group[metric].std(ddof=1)
            row[f"{metric}_formatted"] = format_mean_sd(group[metric])
        rows.append(row)
    result = pd.DataFrame(rows)
    if {"energy_only", "energy_plus_equilibrium"}.issubset(set(result["objective"])):
        energy = result.set_index("objective").loc["energy_only", "stress_mean"]
        full = result.set_index("objective").loc["energy_plus_equilibrium", "stress_mean"]
        result["stress_absolute_reduction_pct_points"] = energy - full
        result["stress_relative_reduction_pct"] = 100.0 * (energy - full) / energy
    return result


def seed_table(rve, reservoir):
    rows = []
    if not reservoir.empty:
        group = reservoir[reservoir["variant"] == "full"]
        row = {"model": "Reservoir AENO", "n_seeds": len(group)}
        for metric in ["displacement", "strain", "stress", "training_time_s"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    if not rve.empty:
        group = rve[rve["objective"] == "energy_plus_equilibrium"]
        row = {"model": "CT-RVE AENO", "n_seeds": len(group)}
        for metric in ["displacement", "strain", "stress", "effective_modulus", "Ksigma95", "training_time_s"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(rve, reservoir):
    rows = []
    groups = []
    if not reservoir.empty:
        groups.append(("Reservoir", reservoir[reservoir["variant"] == "full"]))
    if not rve.empty:
        groups.append(("CT-RVE", rve[rve["objective"] == "energy_plus_equilibrium"]))
    for name, group in groups:
        if group.empty:
            continue
        train = group["training_time_s"].mean()
        preprocess = group["preprocessing_time_s"].mean()
        latency = group["aeno_latency_s"].median()
        fem = group["fem_time_s"].median()
        rows.append({
            "problem": name,
            "n_runs": len(group),
            "training_time_s_mean": train,
            "preprocessing_time_s_mean": preprocess,
            "aeno_latency_s_median": latency,
            "aeno_throughput_samples_per_s": 1.0 / latency,
            "fem_time_s_median": fem,
            "speedup": fem / latency,
            "break_even_calls": (train + preprocess) / (fem - latency),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rve_root", help="Primary CT-RVE full and energy-only seed runs.")
    parser.add_argument("--rve_sensitivity_root")
    parser.add_argument("--reservoir_root", help="Primary reservoir full-model seed runs.")
    parser.add_argument("--reservoir_ablation_root")
    parser.add_argument("--reservoir_fd_root")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rve = collect_rve(args.rve_root)
    rve_sensitivity = collect_rve(args.rve_sensitivity_root)
    reservoir = collect_reservoir(args.reservoir_root)
    reservoir_ablation = collect_reservoir(args.reservoir_ablation_root)
    reservoir_fd = collect_reservoir(args.reservoir_fd_root)
    if not rve.empty:
        rve.to_csv(out / "all_rve_runs.csv", index=False)
        objective_table(rve).to_csv(out / "S7_objective_comparison.csv", index=False)
    if not rve_sensitivity.empty:
        baseline = rve[(rve["objective"] == "energy_plus_equilibrium") & (rve["seed"] == 0)].copy()
        pd.concat([baseline, rve_sensitivity], ignore_index=True).to_csv(out / "S9_rve_sensitivity_runs.csv", index=False)
    if not reservoir_ablation.empty:
        reservoir_ablation.to_csv(out / "S8_reservoir_ablation.csv", index=False)
    if not reservoir_fd.empty:
        baseline = reservoir[(reservoir["variant"] == "full") & (reservoir["seed"] == 42)].copy()
        pd.concat([baseline, reservoir_fd], ignore_index=True).to_csv(out / "S9_reservoir_fd_sensitivity.csv", index=False)
    seed_table(rve, reservoir).to_csv(out / "S10_seed_variability.csv", index=False)
    cost_table(rve, reservoir).to_csv(out / "S11_cost_break_even.csv", index=False)
    print(f"Wrote supplementary summaries to {out}")


if __name__ == "__main__":
    main()
