#!/usr/bin/env python3
"""Run all AENO revision experiments sequentially without a scheduler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


RVE_SEED_RUNS = [
    (0, "rve/full_seed0/eval_test/evaluation_summary.json"),
    (1, "rve/full_seed1/eval_test/evaluation_summary.json"),
    (2, "rve/full_seed2/eval_test/evaluation_summary.json"),
    (3, "rve/energy_only_seed0/eval_test/evaluation_summary.json"),
    (4, "rve/energy_only_seed1/eval_test/evaluation_summary.json"),
    (5, "rve/energy_only_seed2/eval_test/evaluation_summary.json"),
]

RVE_SENSITIVITY_RUNS = [
    (0, "rve_sensitivity/alpha1e5_seed0/eval_test/evaluation_summary.json"),
    (1, "rve_sensitivity/alpha1e3_seed0/eval_test/evaluation_summary.json"),
    (2, "rve_sensitivity/bubble070_seed0/eval_test/evaluation_summary.json"),
    (3, "rve_sensitivity/bubble100_seed0/eval_test/evaluation_summary.json"),
    (4, "rve_sensitivity/threshold015_seed0/eval_test/evaluation_summary.json"),
    (5, "rve_sensitivity/threshold025_seed0/eval_test/evaluation_summary.json"),
]

RESERVOIR_SEED_RUNS = [
    (0, "reservoir_seeds/reservoir_full_seed42/reservoir_full_eval_summary.json"),
    (1, "reservoir_seeds/reservoir_full_seed43/reservoir_full_eval_summary.json"),
    (2, "reservoir_seeds/reservoir_full_seed44/reservoir_full_eval_summary.json"),
]

RESERVOIR_ABLATION_RUNS = [
    (0, "reservoir_ablation/reservoir_full_seed42/reservoir_full_eval_summary.json"),
    (1, "reservoir_ablation/reservoir_soft_bc_seed42/reservoir_soft_bc_eval_summary.json"),
    (2, "reservoir_ablation/reservoir_concat_cond_seed42/reservoir_concat_cond_eval_summary.json"),
    (3, "reservoir_ablation/reservoir_highfreq_pe_seed42/reservoir_highfreq_pe_eval_summary.json"),
    (4, "reservoir_ablation/reservoir_random_sampling_seed42/reservoir_random_sampling_eval_summary.json"),
]

RESERVOIR_FD_RUNS = [
    (0, "reservoir_fd/h2p5e3/reservoir_full_seed42/reservoir_full_eval_summary.json"),
    (1, "reservoir_fd/h1e2/reservoir_full_seed42/reservoir_full_eval_summary.json"),
]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def message(text: str) -> None:
    print(f"[{timestamp()}] {text}", flush=True)


def save_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    message("RUN: " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, env=env, check=True)


def run_array_group(
    root: Path,
    output_root: Path,
    script: Path,
    runs: list[tuple[int, str]],
    env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> None:
    for task_id, marker_relative in runs:
        marker = output_root / marker_relative
        if marker.exists() and not force:
            message(f"SKIP completed task {task_id}: {marker}")
            continue
        task_env = env.copy()
        task_env["SLURM_ARRAY_TASK_ID"] = str(task_id)
        run_command(["bash", str(script)], task_env, dry_run)
        if not dry_run and not marker.exists():
            raise RuntimeError(f"Task {task_id} exited without producing completion marker: {marker}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; default 0.")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even when completion markers exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--skip_threshold_check",
        action="store_true",
        help="Do not require the 1.5% and 2.5% threshold datasets. Tasks 4 and 5 will still fail if submitted without them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = root / "supplementary_runs"
    log_root = root / "logs"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "all_experiments_status.json"

    required = [
        root / "RVE" / "data" / "bristol_25um_rve32_spatial_gray.npz",
        root / "RVE" / "data" / "synthetic_void_rve32_test.npz",
        root / "re" / "reservoir_benchmark_cases.csv",
    ]
    if not args.skip_threshold_check:
        required.extend([
            root / "RVE" / "data" / "bristol_25um_rve32_threshold015.npz",
            root / "RVE" / "data" / "bristol_25um_rve32_threshold025.npz",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required experiment inputs:\n" + "\n".join(missing))

    env = os.environ.copy()
    env["ROOT"] = str(root)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    scripts = root / "supplementary_experiments" / "slurm"

    status = {
        "state": "running",
        "pid": os.getpid(),
        "root": str(root),
        "gpu": str(args.gpu),
        "python": sys.executable,
        "started_at": timestamp(),
        "current_stage": None,
    }

    def record_status() -> None:
        if not args.dry_run:
            save_status(status_path, status)

    record_status()

    stages = [
        ("CT-RVE seeds and objective comparison", scripts / "rve_seed_array.sbatch", RVE_SEED_RUNS),
        ("CT-RVE numerical sensitivity", scripts / "rve_sensitivity_array.sbatch", RVE_SENSITIVITY_RUNS),
        ("Reservoir independent seeds", scripts / "reservoir_seed_array.sbatch", RESERVOIR_SEED_RUNS),
        ("Reservoir ablations", scripts / "reservoir_ablation_array.sbatch", RESERVOIR_ABLATION_RUNS),
        ("Reservoir finite-difference sensitivity", scripts / "reservoir_fd_array.sbatch", RESERVOIR_FD_RUNS),
    ]

    try:
        message(f"AENO experiment suite started with Python {sys.executable}; GPU selector {args.gpu}")
        for stage_name, script, runs in stages:
            status["current_stage"] = stage_name
            record_status()
            message("===== " + stage_name + " =====")
            run_array_group(root, output_root, script, runs, env, args.force, args.dry_run)

        status["current_stage"] = "Reservoir mesh convergence"
        record_status()
        mesh_marker = output_root / "reservoir_mesh" / "S13_mesh_convergence.csv"
        if mesh_marker.exists() and not args.force:
            message(f"SKIP completed mesh convergence: {mesh_marker}")
        else:
            run_command(["bash", str(scripts / "reservoir_mesh_convergence.sbatch")], env, args.dry_run)
            if not args.dry_run and not mesh_marker.exists():
                raise RuntimeError(f"Mesh convergence did not produce {mesh_marker}")

        status["current_stage"] = "Synthetic topology evaluation and associations"
        record_status()
        topology_marker = output_root / "rve_topology" / "descriptor_analysis" / "main_text_descriptor_association.json"
        if topology_marker.exists() and not args.force:
            message(f"SKIP completed topology analysis: {topology_marker}")
        else:
            run_command(["bash", str(scripts / "rve_topology_eval.sbatch")], env, args.dry_run)
            if not args.dry_run and not topology_marker.exists():
                raise RuntimeError(f"Topology analysis did not produce {topology_marker}")

        status["current_stage"] = "Aggregate supplementary tables"
        record_status()
        summary_command = [
            sys.executable,
            str(root / "supplementary_experiments" / "summarize_runs.py"),
            "--rve_root", str(output_root / "rve"),
            "--rve_sensitivity_root", str(output_root / "rve_sensitivity"),
            "--reservoir_root", str(output_root / "reservoir_seeds"),
            "--reservoir_ablation_root", str(output_root / "reservoir_ablation"),
            "--reservoir_fd_root", str(output_root / "reservoir_fd"),
            "--out_dir", str(output_root / "tables"),
        ]
        run_command(summary_command, env, args.dry_run)

        status.update({"state": "completed", "current_stage": None, "completed_at": timestamp()})
        record_status()
        message("ALL AENO EXPERIMENTS COMPLETED")
    except Exception as exc:
        status.update({
            "state": "failed",
            "failed_at": timestamp(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        record_status()
        message(f"FAILED during {status.get('current_stage')}: {exc}")
        raise


if __name__ == "__main__":
    main()
