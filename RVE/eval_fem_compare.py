import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from aeno_rve.data import RVENPZDataset
from aeno_rve.model import AdmissibleAENO
from aeno_rve.elasticity import compute_fields
from aeno_rve.fem import solve_fem_kubc, block_average_3d


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Compare AENO predictions against Hex8 FEM references and plot absolute errors.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--crop_size", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=3, help="Number of held-out RVE samples for FEM comparison")
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--eps0", type=float, default=None)
    p.add_argument("--alpha_void", type=float, default=None)
    p.add_argument("--Es_values", type=float, nargs="+", default=[1.0])
    p.add_argument("--nu_values", type=float, nargs="+", default=[0.33])
    p.add_argument("--fem_downsample", type=int, default=1, help="Block-average factor for FEM reference. Default 1 compares AENO and FEM directly on the same 32^3 grid; use >1 only for larger grids.")
    p.add_argument("--fem_tol", type=float, default=1e-8)
    p.add_argument("--fem_maxiter", type=int, default=2000)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_npz", action="store_true", help="Save fields for each compared case")
    p.add_argument("--verbose_fem", action="store_true")
    return p.parse_args()


def to_np(x):
    return x.detach().cpu().numpy()


def mid_slice(arr):
    return arr[arr.shape[0] // 2]


def rel_l2(pred, ref):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm((pred - ref).ravel()) / (np.linalg.norm(ref.ravel()) + EPS))


def abs_rel_scalar(pred, ref):
    return float(abs(pred - ref)), float(abs(pred - ref) / (abs(ref) + EPS))


def stack_components(dct):
    return np.stack([dct[k] for k in ["xx", "yy", "zz", "xy", "xz", "yz"]], axis=0)


def coarsen_prediction(u_np, eps_np, sig_np, vm_np, factor):
    u_c = block_average_3d(u_np, factor)
    eps_c = {k: block_average_3d(v, factor) for k, v in eps_np.items()}
    sig_c = {k: block_average_3d(v, factor) for k, v in sig_np.items()}
    vm_c = block_average_3d(vm_np, factor)
    return u_c, eps_c, sig_c, vm_c


def common_vlim(a, b):
    lo = min(float(np.nanmin(a)), float(np.nanmin(b)))
    hi = max(float(np.nanmax(a)), float(np.nanmax(b)))
    if abs(hi - lo) < 1e-14:
        hi = lo + 1e-14
    return lo, hi


def plot_compare_panel(chi, fem, pred, out_path, title, gray=None):
    rows = [
        (r"$u_x$", fem["ux"], pred["ux"], "jet"),
        (r"$\epsilon_{xx}$", fem["exx"], pred["exx"], "viridis"),
        (r"$\sigma_{xx}$", fem["sxx"], pred["sxx"], "coolwarm"),
        (r"$\sigma_{vM}$", fem["vm"], pred["vm"], "coolwarm"),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 12.0), constrained_layout=True)

    for r, (name, f_ref, f_pred, cmap) in enumerate(rows):
        ref_sl = mid_slice(f_ref)
        pred_sl = mid_slice(f_pred)
        err_sl = np.abs(pred_sl - ref_sl)

        vmin, vmax = common_vlim(ref_sl, pred_sl)

        im0 = axes[r, 0].imshow(
            ref_sl,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        im1 = axes[r, 1].imshow(
            pred_sl,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        im2 = axes[r, 2].imshow(
            err_sl,
            origin="lower",
            cmap="Reds",
        )

        axes[r, 0].set_ylabel(name)

        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

        if r == 0:
            axes[r, 0].set_title("FEM")
            axes[r, 1].set_title("AENO")
            axes[r, 2].set_title("absolute error")

        fig.colorbar(im0, ax=axes[r, 0], shrink=0.72)
        fig.colorbar(im1, ax=axes[r, 1], shrink=0.72)
        fig.colorbar(im2, ax=axes[r, 2], shrink=0.72)

    fig.suptitle(title + "\nmid-z slices; FEM uses the same KUBC as AENO", fontsize=11)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    if gray is not None:
        raw = mid_slice(gray)
        void = mid_slice(chi)

        fig2, axes2 = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)

        im0 = axes2[0].imshow(raw, origin="lower", cmap="gray")
        axes2[0].set_title("raw XCT crop slice")

        im1 = axes2[1].imshow(void, origin="lower", vmin=0, vmax=1, cmap="gray_r")
        axes2[1].set_title("void indicator used by FEM")

        axes2[2].imshow(raw, origin="lower", cmap="gray")
        axes2[2].imshow(
            np.ma.masked_where(void <= 0.5, void),
            origin="lower",
            alpha=0.65,
            vmin=0,
            vmax=1,
            cmap="Reds",
        )
        axes2[2].set_title("threshold overlay")

        for ax in axes2:
            ax.set_xticks([])
            ax.set_yticks([])

        fig2.colorbar(im0, ax=axes2[0], shrink=0.75)
        fig2.colorbar(im1, ax=axes2[1], shrink=0.75)
    else:
        fig2, ax = plt.subplots(figsize=(4.0, 3.8), constrained_layout=True)
        im = ax.imshow(mid_slice(chi), origin="lower", vmin=0, vmax=1, cmap="gray_r")
        ax.set_title("void fraction / indicator used by FEM")
        ax.set_xticks([])
        ax.set_yticks([])
        fig2.colorbar(im, ax=ax, shrink=0.8)

    fig2.savefig(out_path.with_name(out_path.stem + "_microstructure.png"), dpi=220)
    plt.close(fig2)


def parity_plot(df, ref_col, pred_col, out_path, label):
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    ax.scatter(df[ref_col], df[pred_col], s=32)
    lo = min(df[ref_col].min(), df[pred_col].min())
    hi = max(df[ref_col].max(), df[pred_col].max())
    pad = 0.05 * (hi - lo + EPS)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    ax.set_xlabel(f"FEM {label}")
    ax.set_ylabel(f"AENO {label}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def bar_error_plot(df, out_path):
    metrics = ["rel_l2_u", "rel_l2_eps", "rel_l2_sig", "rel_err_Eeff", "rel_err_Ksigma95"]
    means = [df[m].mean() for m in metrics]
    stds = [df[m].std(ddof=0) for m in metrics]
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("relative error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)



def error_by_porosity_bin_plot(df, out_path):
    if "porosity_bin" not in df.columns or df["porosity_bin"].nunique() == 0:
        return
    metrics = ["rel_l2_u", "rel_l2_eps", "rel_l2_sig", "rel_err_Eeff", "rel_err_Ksigma95"]
    ordered_bins = ["phi<1%", "1%<=phi<2%", "phi>=2%"]
    bins = [b for b in ordered_bins if (df["porosity_bin"] == b).any()]
    if not bins:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.8), constrained_layout=True)
    for ax, m in zip(axes, metrics):
        data = [df.loc[df["porosity_bin"] == b, m].dropna().values for b in bins]
        ax.boxplot(data, labels=bins, showfliers=False)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("relative error")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    crop_size = int(args.crop_size if args.crop_size is not None else cfg.get("crop_size", 64))
    eps0 = float(args.eps0 if args.eps0 is not None else cfg.get("eps0", 0.01))
    alpha_void = float(args.alpha_void if args.alpha_void is not None else cfg.get("alpha_void", 1e-4))
    base_channels = int(args.base_channels if args.base_channels is not None else cfg.get("base_channels", 12))
    Es_range = (float(cfg.get("Es_min", min(args.Es_values))), float(cfg.get("Es_max", max(args.Es_values))))
    nu_range = (float(cfg.get("nu_min", min(args.nu_values))), float(cfg.get("nu_max", max(args.nu_values))))

    if crop_size % args.fem_downsample != 0:
        raise ValueError("crop_size must be divisible by fem_downsample")
    n_fem = crop_size // args.fem_downsample
    if n_fem > 48:
        print(f"WARNING: FEM grid has {n_fem}^3 elements. This may require substantial memory/time. For grids larger than 32^3, consider --fem_downsample 2 or 4.")

    model = AdmissibleAENO(n=crop_size, base=base_channels, eps0=eps0, Es_range=Es_range, nu_range=nu_range, bubble_power=float(cfg.get("bubble_power", 1.0))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = RVENPZDataset(args.data, split=args.split, Es_range=Es_range, nu_range=nu_range, seed=123)
    n = min(args.num_samples, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int).tolist()
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False)
    spacing = None  # V2 recovers strain/stress from nodal Hex8 B-matrix; spacing is ignored.

    rows = []
    for bidx, batch in enumerate(loader):
        chi_t = batch["chi_void"].to(device)
        chi_np_full = to_np(chi_t[0]).astype(np.float64)
        chi_fem = block_average_3d(chi_np_full, args.fem_downsample)
        gray_np_full = to_np(batch["gray"][0]).astype(np.float64) if "gray" in batch else None
        gray_fem = block_average_3d(gray_np_full, args.fem_downsample) if gray_np_full is not None else None
        porosity = float(batch["porosity"].item())
        porosity_bin_id = int(batch.get("porosity_bin_id", torch.tensor([-1])).item())
        porosity_bin = {0: "phi<1%", 1: "1%<=phi<2%", 2: "phi>=2%"}.get(porosity_bin_id, "unknown")
        rve_id = int(batch["rve_id"].item())

        for Es in args.Es_values:
            for nu in args.nu_values:
                Es_t = torch.tensor([Es], dtype=torch.float32, device=device)
                nu_t = torch.tensor([nu], dtype=torch.float32, device=device)

                t_pred0 = time.perf_counter()
                with torch.no_grad():
                    u_nodal = model(chi_t, Es_t, nu_t, batch.get("sdf_void", None).to(device) if "sdf_void" in batch else None, batch.get("interface", None).to(device) if "interface" in batch else None, batch.get("boundary_distance", None).to(device) if "boundary_distance" in batch else None)
                    fields = compute_fields(u_nodal, chi_t, Es_t, nu_t, eps0=eps0, spacing=spacing, alpha_void=alpha_void)
                pred_time = time.perf_counter() - t_pred0

                u_nodal_pred = to_np(u_nodal[0]).astype(np.float64)
                u_pred = to_np(fields["u_center"][0]).astype(np.float64)
                eps_pred = {k: to_np(v[0]).astype(np.float64) for k, v in fields["eps"].items()}
                sig_pred = {k: to_np(v[0]).astype(np.float64) for k, v in fields["sig"].items()}
                vm_pred = to_np(fields["von_mises"][0]).astype(np.float64)
                u_pred_c, eps_pred_c, sig_pred_c, vm_pred_c = coarsen_prediction(u_pred, eps_pred, sig_pred, vm_pred, args.fem_downsample)

                fem = solve_fem_kubc(
                    chi_fem,
                    Es=Es,
                    nu_s=nu,
                    eps0=eps0,
                    alpha_void=alpha_void,
                    tol=args.fem_tol,
                    maxiter=args.fem_maxiter,
                    verbose=args.verbose_fem,
                )

                pred_Eeff = float(sig_pred_c["xx"].mean() / eps0)
                pred_K = float(np.quantile(vm_pred_c.reshape(-1), 0.95) / (vm_pred_c.mean() + EPS))

                abs_Eeff, rel_Eeff = abs_rel_scalar(pred_Eeff, fem.Eeff)
                abs_K, rel_K = abs_rel_scalar(pred_K, fem.Ksigma95)

                eps_pred_stack = stack_components(eps_pred_c)
                sig_pred_stack = stack_components(sig_pred_c)
                eps_ref_stack = stack_components(fem.eps)
                sig_ref_stack = stack_components(fem.sig)

                row = {
                    "rve_id": rve_id,
                    "sample_order": bidx,
                    "porosity_full": porosity,
                    "porosity_bin": porosity_bin,
                    "porosity_fem": float(chi_fem.mean()),
                    "Es": float(Es),
                    "nu_s": float(nu),
                    "fem_downsample": int(args.fem_downsample),
                    "fem_n_elements_per_axis": int(n_fem),
                    "pred_time_s": float(pred_time),
                    "fem_total_time_s": fem.solve_info["total_time_s"],
                    "fem_assembly_time_s": fem.solve_info["assembly_time_s"],
                    "fem_solve_time_s": fem.solve_info["solve_time_s"],
                    "fem_cg_info": fem.solve_info["cg_info"],
                    "Eeff_fem": float(fem.Eeff),
                    "Eeff_aeno": float(pred_Eeff),
                    "abs_err_Eeff": abs_Eeff,
                    "rel_err_Eeff": rel_Eeff,
                    "Ksigma95_fem": float(fem.Ksigma95),
                    "Ksigma95_aeno": float(pred_K),
                    "abs_err_Ksigma95": abs_K,
                    "rel_err_Ksigma95": rel_K,
                    "rel_l2_ux": rel_l2(u_pred_c[0], fem.u_center[0]),
                    "rel_l2_u": rel_l2(u_pred_c, fem.u_center),
                    "rel_l2_u_nodal": rel_l2(np.moveaxis(u_nodal_pred, 0, -1), fem.u_nodal),
                    "rel_l2_exx": rel_l2(eps_pred_c["xx"], fem.eps["xx"]),
                    "rel_l2_eps": rel_l2(eps_pred_stack, eps_ref_stack),
                    "rel_l2_sxx": rel_l2(sig_pred_c["xx"], fem.sig["xx"]),
                    "rel_l2_sig": rel_l2(sig_pred_stack, sig_ref_stack),
                    "rel_l2_vm": rel_l2(vm_pred_c, fem.von_mises),
                    "mean_abs_err_ux": float(np.mean(np.abs(u_pred_c[0] - fem.u_center[0]))),
                    "mean_abs_err_exx": float(np.mean(np.abs(eps_pred_c["xx"] - fem.eps["xx"]))),
                    "mean_abs_err_sxx": float(np.mean(np.abs(sig_pred_c["xx"] - fem.sig["xx"]))),
                    "mean_abs_err_vm": float(np.mean(np.abs(vm_pred_c - fem.von_mises))),
                }
                rows.append(row)

                tag = f"rve{rve_id}_Es{Es:g}_nu{nu:g}_fem{n_fem}"
                plot_compare_panel(
                    chi_fem,
                    fem={"ux": fem.u_center[0], "exx": fem.eps["xx"], "sxx": fem.sig["xx"], "vm": fem.von_mises},
                    pred={"ux": u_pred_c[0], "exx": eps_pred_c["xx"], "sxx": sig_pred_c["xx"], "vm": vm_pred_c},
                    out_path=out_dir / f"fem_aeno_abs_error_{tag}.png",
                    title=(
                        f"RVE {rve_id}, {porosity_bin}, Es={Es:g}, nu={nu:g}, phi={porosity:.3f}; "
                        f"relL2(u)={row['rel_l2_u']:.2e}, relL2(sig)={row['rel_l2_sig']:.2e}"
                    ),
                    gray=gray_fem,
                )

                if args.save_npz:
                    np.savez_compressed(
                        out_dir / f"fields_{tag}.npz",
                        chi_fem=chi_fem,
                        u_fem=fem.u_center,
                        ux_aeno=u_pred_c[0],
                        u_aeno=u_pred_c,
                        u_nodal_fem=fem.u_nodal,
                        u_nodal_aeno=np.moveaxis(u_nodal_pred, 0, -1),
                        exx_fem=fem.eps["xx"],
                        exx_aeno=eps_pred_c["xx"],
                        sxx_fem=fem.sig["xx"],
                        sxx_aeno=sig_pred_c["xx"],
                        vm_fem=fem.von_mises,
                        vm_aeno=vm_pred_c,
                    )
                print(f"[done] {tag}: relL2_u={row['rel_l2_u']:.3e}, relL2_sig={row['rel_l2_sig']:.3e}, relEeff={row['rel_err_Eeff']:.3e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fem_comparison_metrics.csv", index=False)
    if len(df) > 0:
        parity_plot(df, "Eeff_fem", "Eeff_aeno", out_dir / "parity_Eeff_fem_vs_aeno.png", r"$E_{eff}$")
        parity_plot(df, "Ksigma95_fem", "Ksigma95_aeno", out_dir / "parity_Ksigma95_fem_vs_aeno.png", r"$K_{\sigma,95}$")
        bar_error_plot(df, out_dir / "summary_relative_errors.png")
        error_by_porosity_bin_plot(df, out_dir / "relative_errors_by_porosity_bin.png")
        df.groupby("porosity_bin")[["rel_l2_u", "rel_l2_eps", "rel_l2_sig", "rel_err_Eeff", "rel_err_Ksigma95"]].agg(["mean", "std", "count"]).to_csv(out_dir / "fem_errors_by_porosity_bin.csv")

    with open(out_dir / "fem_eval_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(df.head())
    print(f"Saved FEM comparison metrics and figures to: {out_dir}")


if __name__ == "__main__":
    main()
