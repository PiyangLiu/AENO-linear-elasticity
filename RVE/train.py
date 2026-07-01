import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from aeno_rve.data import RVENPZDataset
from aeno_rve.model import AdmissibleAENO
from aeno_rve.elasticity import fem_energy_residual_objective


def parse_args():
    p = argparse.ArgumentParser(description="Train label-free nodal AENO RVE elasticity model with Hex8 equilibrium residual.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--crop_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--base_channels", type=int, default=20)
    p.add_argument("--eps0", type=float, default=0.01)
    p.add_argument("--Es_min", type=float, default=1.0)
    p.add_argument("--Es_max", type=float, default=1.0)
    p.add_argument("--nu_min", type=float, default=0.33)
    p.add_argument("--nu_max", type=float, default=0.33)
    p.add_argument("--alpha_void", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--balance_porosity_bins", action="store_true", help="Backward-compatible alias for mild porosity-balanced sampling.")
    p.add_argument("--porosity_balance_strength", type=float, default=0.3, help="0=natural sampling, 1=full inverse-frequency bin balancing.")
    p.add_argument("--high_nu_prob", type=float, default=0.0, help="Keep 0 for the current microstructure-only fixed-nu experiment.")
    p.add_argument("--high_nu_min", type=float, default=0.35)
    p.add_argument("--bubble_power", type=float, default=1.0)
    p.add_argument("--residual_weight", type=float, default=1e-3, help="Weight lambda_r for normalized free-DOF residual loss. Set 0 for pure energy training.")
    p.add_argument("--residual_ramp_epochs", type=int, default=25, help="Linearly ramp residual_weight during early training; 0 disables ramp.")
    p.add_argument("--interface_residual_alpha", type=float, default=2.0, help="Element interface weight w_e=1+alpha*I_interface,e used in residual loss.")
    p.add_argument("--residual_scale", type=str, default="force", choices=["force", "energy", "none"])
    return p.parse_args()


def _to_device_optional(batch, key, device):
    return batch[key].to(device, non_blocking=True) if key in batch else None


def loss_terms(model, batch, args, device, residual_multiplier=1.0):
    chi = batch["chi_void"].to(device, non_blocking=True)
    Es = batch["Es"].to(device, non_blocking=True)
    nu = batch["nu_s"].to(device, non_blocking=True)
    sdf = _to_device_optional(batch, "sdf_void", device)
    interface = _to_device_optional(batch, "interface", device)
    bd = _to_device_optional(batch, "boundary_distance", device)

    u = model(chi, Es, nu, sdf_void=sdf, interface=interface, boundary_distance=bd)
    obj = fem_energy_residual_objective(
        u,
        chi,
        Es,
        nu,
        eps0=args.eps0,
        alpha_void=args.alpha_void,
        interface=interface,
        interface_alpha=args.interface_residual_alpha,
        residual_scale=args.residual_scale,
    )
    energy_norm_per = obj["energy_per"] / Es.reshape(-1)
    energy_loss = energy_norm_per.mean()
    residual_loss = obj["residual_per"].mean()
    amp_reg = 1e-8 * (u ** 2).mean()
    total = energy_loss + float(args.residual_weight) * float(residual_multiplier) * residual_loss + amp_reg
    return total, {
        "energy_loss": energy_loss.detach(),
        "residual_loss": residual_loss.detach(),
        "residual_raw": obj["residual_raw"].mean().detach(),
        "amp_reg": amp_reg.detach(),
    }


def evaluate_objective(model, loader, args, device):
    model.eval()
    losses = []
    energies = []
    residuals = []
    residual_raw = []
    with torch.no_grad():
        for batch in loader:
            loss, terms = loss_terms(model, batch, args, device, residual_multiplier=1.0)
            bs = batch["chi_void"].shape[0]
            losses.append(loss.detach().cpu().repeat(bs))
            energies.append(terms["energy_loss"].detach().cpu().repeat(bs))
            residuals.append(terms["residual_loss"].detach().cpu().repeat(bs))
            residual_raw.append(terms["residual_raw"].detach().cpu().repeat(bs))
    return {
        "val_loss": torch.cat(losses).mean().item(),
        "val_energy": torch.cat(energies).mean().item(),
        "val_residual": torch.cat(residuals).mean().item(),
        "val_residual_raw": torch.cat(residual_raw).mean().item(),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    Es_range = (args.Es_min, args.Es_max)
    nu_range = (args.nu_min, args.nu_max)
    train_ds = RVENPZDataset(
        args.data,
        split="train",
        Es_range=Es_range,
        nu_range=nu_range,
        seed=args.seed,
        high_nu_prob=args.high_nu_prob,
        high_nu_min=args.high_nu_min,
    )
    val_ds = RVENPZDataset(args.data, split="val", Es_range=Es_range, nu_range=nu_range, seed=args.seed + 1)

    balance_strength = float(args.porosity_balance_strength if args.balance_porosity_bins else max(0.0, args.porosity_balance_strength))
    balance_strength = min(max(balance_strength, 0.0), 1.0)
    if balance_strength > 0:
        bins = train_ds.porosity_bin_id
        counts = np.bincount(bins, minlength=3).astype(np.float64)
        inv = counts.mean() / np.maximum(counts[bins], 1.0)
        weights = (1.0 - balance_strength) * np.ones_like(inv, dtype=np.float64) + balance_strength * inv
        sampler = WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = AdmissibleAENO(
        n=args.crop_size,
        base=args.base_channels,
        eps0=args.eps0,
        Es_range=Es_range,
        nu_range=nu_range,
        bubble_power=args.bubble_power,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cfg = vars(args).copy()
    cfg.update({
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "aeno_v2": "nodal_u_hex8_center_recovery_energy_plus_free_dof_residual",
        "u_output": "nodal [B,3,N+1,N+1,N+1]",
        "strain_stress_recovery": "Hex8 center B matrix, same node ordering as FEM",
    })
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    log_rows = []
    for ep in range(1, args.epochs + 1):
        model.train()
        ramp = 1.0 if args.residual_ramp_epochs <= 0 else min(1.0, ep / float(args.residual_ramp_epochs))
        running = {"loss": 0.0, "energy": 0.0, "residual": 0.0, "residual_raw": 0.0}
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {ep:04d}/{args.epochs}")
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            loss, terms = loss_terms(model, batch, args, device, residual_multiplier=ramp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = batch["chi_void"].shape[0]
            running["loss"] += float(loss.detach().cpu()) * bs
            running["energy"] += float(terms["energy_loss"].detach().cpu()) * bs
            running["residual"] += float(terms["residual_loss"].detach().cpu()) * bs
            running["residual_raw"] += float(terms["residual_raw"].detach().cpu()) * bs
            n_seen += bs
            pbar.set_postfix(
                loss=running["loss"] / max(1, n_seen),
                E=running["energy"] / max(1, n_seen),
                R=running["residual"] / max(1, n_seen),
                ramp=ramp,
            )
        sched.step()

        train_loss = running["loss"] / max(1, n_seen)
        train_energy = running["energy"] / max(1, n_seen)
        train_residual = running["residual"] / max(1, n_seen)
        train_residual_raw = running["residual_raw"] / max(1, n_seen)
        val = evaluate_objective(model, val_loader, args, device)
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_energy": train_energy,
            "train_residual": train_residual,
            "train_residual_raw": train_residual_raw,
            **val,
            "lr": sched.get_last_lr()[0],
            "residual_ramp": ramp,
            "effective_residual_weight": args.residual_weight * ramp,
        }
        log_rows.append(row)
        print(
            f"epoch={ep} train={train_loss:.6e} E={train_energy:.6e} R={train_residual:.6e} "
            f"val={val['val_loss']:.6e} valE={val['val_energy']:.6e} valR={val['val_residual']:.6e}"
        )

        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            torch.save({"model": model.state_dict(), "args": cfg, "epoch": ep, "val_loss": val["val_loss"]}, out_dir / "best.pt")
        if ep % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": cfg, "epoch": ep, "val_loss": val["val_loss"]}, out_dir / f"epoch_{ep:04d}.pt")

    import pandas as pd
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    print(f"Best validation objective: {best_val:.6e}")
    print(f"Saved best checkpoint to: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
