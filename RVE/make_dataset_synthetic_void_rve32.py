#!/usr/bin/env python3
"""Generate synthetic 3D void-geometry test sets for AENO.

The output NPZ is compatible with the Bristol AENO RVENPZDataset format:
    chi_void: uint8 [N, n, n, n], 1=void/pore, 0=solid
    split:    string [N], usually all "test"
    porosity: float32 [N]
    origins:  int32 [N,3]

Additional metadata fields are included for analysis:
    geometry_type, geometry_family, porosity_bin_id

Two geometry families are generated:
    1) bristol_like: ellipsoids, clusters, elongated voids, boundary-near voids
    2) strong_ood: extruded rings/maze, checkerboard, tree/branches, channels, lattice

This script creates synthetic test data only. The intended use is zero-shot testing:
    train on Bristol 25micron_60min, evaluate on this synthetic NPZ with eval_fem_compare.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None


# -----------------------------
# Basic geometry helpers
# -----------------------------


def make_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    return z, y, x


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3x3 rotation matrix."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Enforce a proper right-handed rotation.
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    # Avoid sign bias from QR.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q @ np.diag(signs)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q.astype(np.float32)


def add_ellipsoid(
    chi: np.ndarray,
    center: Iterable[float],
    radii: Iterable[float],
    rotation: np.ndarray | None = None,
) -> None:
    """Add a rotated ellipsoidal void to chi, in-place.

    chi convention: 1=void, 0=solid.
    center/radii are in voxel coordinates ordered as z,y,x.
    """
    n = chi.shape[0]
    z, y, x = make_grid(n)
    coords = np.stack([z - center[0], y - center[1], x - center[2]], axis=0).reshape(3, -1)
    if rotation is not None:
        coords = rotation.T @ coords
    rz, ry, rx = np.maximum(np.asarray(list(radii), dtype=np.float32), 1.0)
    val = (coords[0] / rz) ** 2 + (coords[1] / ry) ** 2 + (coords[2] / rx) ** 2
    mask = val.reshape(n, n, n) <= 1.0
    chi[mask] = 1


def add_sphere_or_ellipsoid_at_random(
    chi: np.ndarray,
    rng: np.random.Generator,
    center_mode: str = "interior",
    elongated: bool = False,
    radius_scale: tuple[float, float] = (2.0, 5.0),
) -> None:
    n = chi.shape[0]

    if center_mode == "interior":
        margin = 3
        center = rng.uniform(margin, n - 1 - margin, size=3)
    elif center_mode == "boundary_near":
        center = rng.uniform(3, n - 4, size=3)
        axis = int(rng.integers(0, 3))
        side = int(rng.integers(0, 2))
        center[axis] = rng.uniform(0.5, 4.0) if side == 0 else rng.uniform(n - 5.0, n - 1.5)
    elif center_mode == "boundary_connected":
        center = rng.uniform(4, n - 5, size=3)
        axis = int(rng.integers(0, 3))
        side = int(rng.integers(0, 2))
        # Place center slightly outside or very near the boundary so the ellipsoid is truncated.
        center[axis] = rng.uniform(-2.0, 2.0) if side == 0 else rng.uniform(n - 3.0, n + 1.0)
    else:
        raise ValueError(f"Unknown center_mode={center_mode}")

    if elongated:
        # One long radius and two short radii.
        short1 = rng.uniform(radius_scale[0], radius_scale[1])
        short2 = rng.uniform(radius_scale[0], radius_scale[1])
        long = rng.uniform(7.0, 15.0)
        radii = np.array([short1, short2, long], dtype=np.float32)
        rng.shuffle(radii)
    else:
        r = rng.uniform(radius_scale[0], radius_scale[1], size=3)
        radii = r.astype(np.float32)

    add_ellipsoid(chi, center=center, radii=radii, rotation=random_rotation(rng))


def porosity(chi: np.ndarray) -> float:
    return float(np.mean(chi.astype(np.float32)))


def enforce_porosity_range(
    chi: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    fill_mode: str = "ellipsoid",
    max_iter: int = 80,
) -> np.ndarray:
    """Add small voids until low <= phi <= high when possible.

    If phi > high, the structure is kept because for OOD patterns exact porosity control is
    less important than topology. The generator parameters below already target reasonable ranges.
    """
    for _ in range(max_iter):
        phi = porosity(chi)
        if phi >= low:
            break
        if fill_mode == "boundary":
            mode = "boundary_near" if rng.random() < 0.7 else "boundary_connected"
            add_sphere_or_ellipsoid_at_random(chi, rng, center_mode=mode, elongated=rng.random() < 0.4)
        elif fill_mode == "elongated":
            add_sphere_or_ellipsoid_at_random(chi, rng, center_mode="interior", elongated=True)
        else:
            add_sphere_or_ellipsoid_at_random(chi, rng, center_mode="interior", elongated=rng.random() < 0.25)
    return chi


# -----------------------------
# Bristol-like synthetic geometries
# -----------------------------


def geom_isolated_ellipsoids(n: int, rng: np.random.Generator, phi_range: tuple[float, float]) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    num = int(rng.integers(1, 4))
    for _ in range(num):
        add_sphere_or_ellipsoid_at_random(chi, rng, "interior", elongated=rng.random() < 0.25, radius_scale=(2.0, 4.5))
    return enforce_porosity_range(chi, rng, *phi_range, fill_mode="ellipsoid")


def geom_cluster(n: int, rng: np.random.Generator, phi_range: tuple[float, float]) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    base_center = rng.uniform(7, n - 8, size=3)
    num = int(rng.integers(4, 10))
    for _ in range(num):
        center = base_center + rng.normal(scale=rng.uniform(1.5, 4.0), size=3)
        center = np.clip(center, 1.0, n - 2.0)
        radii = rng.uniform(1.6, 4.2, size=3)
        if rng.random() < 0.25:
            radii[int(rng.integers(0, 3))] *= rng.uniform(1.5, 2.5)
        add_ellipsoid(chi, center=center, radii=radii, rotation=random_rotation(rng))
    return enforce_porosity_range(chi, rng, *phi_range, fill_mode="ellipsoid")


def geom_elongated_voids(n: int, rng: np.random.Generator, phi_range: tuple[float, float]) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    num = int(rng.integers(1, 4))
    for _ in range(num):
        add_sphere_or_ellipsoid_at_random(chi, rng, "interior", elongated=True, radius_scale=(1.4, 3.0))
    return enforce_porosity_range(chi, rng, *phi_range, fill_mode="elongated")


def geom_boundary_voids(n: int, rng: np.random.Generator, phi_range: tuple[float, float]) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    num = int(rng.integers(1, 5))
    for _ in range(num):
        mode = "boundary_connected" if rng.random() < 0.55 else "boundary_near"
        add_sphere_or_ellipsoid_at_random(chi, rng, mode, elongated=rng.random() < 0.45, radius_scale=(2.0, 5.0))
    return enforce_porosity_range(chi, rng, *phi_range, fill_mode="boundary")


# -----------------------------
# Strong OOD geometries
# -----------------------------


def draw_disk_2d(img: np.ndarray, y: int, x: int, radius: int) -> None:
    n = img.shape[0]
    yy, xx = np.ogrid[:n, :n]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    img[mask] = 1


def draw_line_2d(img: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], thickness: int = 1) -> None:
    y0, x0 = p0
    y1, x1 = p1
    length = max(abs(y1 - y0), abs(x1 - x0), 1)
    steps = int(np.ceil(length * 2)) + 1
    for t in np.linspace(0.0, 1.0, steps):
        y = int(round((1 - t) * y0 + t * y1))
        x = int(round((1 - t) * x0 + t * x1))
        draw_disk_2d(img, y, x, thickness)


def extrude_2d(mask2d: np.ndarray, n: int, rng: np.random.Generator, full: bool = True, z_thickness: int | None = None) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    if full:
        chi[:, mask2d > 0] = 1
    else:
        if z_thickness is None:
            z_thickness = int(rng.integers(6, 17))
        z0 = int(rng.integers(0, max(1, n - z_thickness + 1)))
        chi[z0 : z0 + z_thickness, mask2d > 0] = 1
    return chi


def geom_square_rings(n: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n, n), dtype=np.uint8)
    thickness = int(rng.integers(1, 3))
    # concentric square void rings, similar to the user's example
    for k in range(3, n // 2 - 2, 5):
        y0, y1 = k, n - k - 1
        x0, x1 = k, n - k - 1
        mask[y0 : y0 + thickness, x0:x1 + 1] = 1
        mask[y1 - thickness + 1 : y1 + 1, x0:x1 + 1] = 1
        mask[y0:y1 + 1, x0 : x0 + thickness] = 1
        mask[y0:y1 + 1, x1 - thickness + 1 : x1 + 1] = 1
    return extrude_2d(mask, n, rng, full=(rng.random() < 0.65))


def geom_checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    block = int(rng.integers(3, 6))
    mask = np.zeros((n, n), dtype=np.uint8)
    for y in range(0, n, block):
        for x in range(0, n, block):
            if ((y // block) + (x // block)) % 2 == 0:
                # keep it thinner than a full checkerboard so the void fraction is not excessive
                if rng.random() < 0.55:
                    mask[y : min(n, y + block), x : min(n, x + block)] = 1
    return extrude_2d(mask, n, rng, full=(rng.random() < 0.5), z_thickness=int(rng.integers(8, 20)))


def geom_tree(n: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n, n), dtype=np.uint8)
    thickness = int(rng.integers(1, 3))
    root = (n - 2, n // 2 + rng.integers(-3, 4))
    trunk_top = (int(rng.integers(6, 12)), root[1] + rng.integers(-2, 3))
    draw_line_2d(mask, root, trunk_top, thickness)
    # main branches
    for angle_sign in [-1, 1]:
        for level in [0.25, 0.45, 0.65]:
            y = root[0] * (1 - level) + trunk_top[0] * level
            x = root[1] * (1 - level) + trunk_top[1] * level
            length = rng.uniform(8, 16)
            y2 = max(1, y - rng.uniform(5, 11))
            x2 = np.clip(x + angle_sign * length, 1, n - 2)
            draw_line_2d(mask, (y, x), (y2, x2), thickness)
    return extrude_2d(mask, n, rng, full=(rng.random() < 0.6), z_thickness=int(rng.integers(10, 25)))


def geom_cross_channels(n: int, rng: np.random.Generator) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    thickness = int(rng.integers(1, 3))
    # A few through-channels in different directions.
    for axis in rng.choice([0, 1, 2], size=int(rng.integers(2, 5)), replace=True):
        a = int(rng.integers(5, n - 5))
        b = int(rng.integers(5, n - 5))
        sl = slice(max(0, a - thickness), min(n, a + thickness + 1))
        sl2 = slice(max(0, b - thickness), min(n, b + thickness + 1))
        if axis == 0:  # along z, fixed y/x tube square
            chi[:, sl, sl2] = 1
        elif axis == 1:  # along y
            chi[sl, :, sl2] = 1
        else:  # along x
            chi[sl, sl2, :] = 1
    return chi


def geom_lattice(n: int, rng: np.random.Generator) -> np.ndarray:
    chi = np.zeros((n, n, n), dtype=np.uint8)
    period = int(rng.integers(7, 11))
    thickness = 1
    offset = int(rng.integers(1, period))
    positions = list(range(offset, n, period))
    for p in positions:
        chi[max(0, p - thickness):min(n, p + thickness + 1), :, :] = 1
        if rng.random() < 0.8:
            chi[:, max(0, p - thickness):min(n, p + thickness + 1), :] = 1
        if rng.random() < 0.8:
            chi[:, :, max(0, p - thickness):min(n, p + thickness + 1)] = 1
    return chi


def geom_random_maze(n: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n, n), dtype=np.uint8)
    thickness = int(rng.integers(1, 3))
    # Random rectilinear path.
    y = int(rng.integers(3, n - 3))
    x = int(rng.integers(3, n - 3))
    for _ in range(int(rng.integers(7, 14))):
        if rng.random() < 0.5:
            x2 = int(rng.integers(2, n - 2))
            draw_line_2d(mask, (y, x), (y, x2), thickness)
            x = x2
        else:
            y2 = int(rng.integers(2, n - 2))
            draw_line_2d(mask, (y, x), (y2, x), thickness)
            y = y2
    return extrude_2d(mask, n, rng, full=(rng.random() < 0.55), z_thickness=int(rng.integers(8, 24)))


# -----------------------------
# Dataset assembly
# -----------------------------


BRISTOL_LIKE_GENERATORS = {
    "isolated_ellipsoid": geom_isolated_ellipsoids,
    "cluster": geom_cluster,
    "elongated_void": geom_elongated_voids,
    "boundary_void": geom_boundary_voids,
}

STRONG_OOD_GENERATORS = {
    "square_rings_extruded": geom_square_rings,
    "checkerboard_extruded": geom_checkerboard,
    "tree_extruded": geom_tree,
    "cross_channels": geom_cross_channels,
    "lattice": geom_lattice,
    "random_maze_extruded": geom_random_maze,
}


def porosity_bin_id(phi: float) -> int:
    if phi < 0.01:
        return 0
    if phi < 0.02:
        return 1
    return 2


def make_overview_png(chi_list: list[np.ndarray], geometry_type: list[str], geometry_family: list[str], out_png: Path, max_items: int = 48) -> None:
    if plt is None:
        return
    n_items = min(len(chi_list), max_items)
    if n_items <= 0:
        return
    ncols = 8
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 1.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n_items):
        chi = chi_list[i]
        z = chi.shape[0] // 2
        ax = axes.ravel()[i]
        ax.imshow(chi[z], cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{i}\n{geometry_family[i]}\n{geometry_type[i]}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_dataset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n = int(args.crop_size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chis: list[np.ndarray] = []
    splits: list[str] = []
    phis: list[float] = []
    origins: list[list[int]] = []
    bin_ids: list[int] = []
    geom_types: list[str] = []
    geom_families: list[str] = []

    def add_sample(chi: np.ndarray, family: str, gtype: str) -> None:
        chi = (chi > 0).astype(np.uint8)
        phi = porosity(chi)
        if phi < args.min_porosity:
            return
        if args.max_porosity > 0 and phi > args.max_porosity and family == "bristol_like":
            # For Bristol-like geometries we keep the intended porosity envelope tighter.
            return
        chis.append(chi)
        splits.append("test")
        phis.append(phi)
        origins.append([0, 0, 0])
        bin_ids.append(porosity_bin_id(phi))
        geom_types.append(gtype)
        geom_families.append(family)

    phi_range = (float(args.bristol_phi_min), float(args.bristol_phi_max))

    for gtype, gen in BRISTOL_LIKE_GENERATORS.items():
        attempts = 0
        target = int(args.num_bristol_like_per_type)
        while sum(1 for gt in geom_types if gt == gtype) < target and attempts < target * 20:
            attempts += 1
            chi = gen(n, rng, phi_range)
            add_sample(chi, "bristol_like", gtype)

    for gtype, gen in STRONG_OOD_GENERATORS.items():
        attempts = 0
        target = int(args.num_ood_per_type)
        while sum(1 for gt in geom_types if gt == gtype) < target and attempts < target * 20:
            attempts += 1
            chi = gen(n, rng)
            # Filter only very extreme all-void-like cases.
            phi = porosity(chi)
            if phi < args.min_porosity or phi > args.ood_max_porosity:
                continue
            add_sample(chi, "strong_ood", gtype)

    if not chis:
        raise RuntimeError("No synthetic samples were generated. Relax porosity filters.")

    chi_arr = np.stack(chis).astype(np.uint8)
    split_arr = np.asarray(splits, dtype="U16")
    porosity_arr = np.asarray(phis, dtype=np.float32)
    origins_arr = np.asarray(origins, dtype=np.int32)
    bin_arr = np.asarray(bin_ids, dtype=np.int32)
    geom_type_arr = np.asarray(geom_types, dtype="U64")
    geom_family_arr = np.asarray(geom_families, dtype="U32")

    np.savez_compressed(
        out_path,
        chi_void=chi_arr,
        split=split_arr,
        porosity=porosity_arr,
        origins=origins_arr,
        porosity_bin_id=bin_arr,
        geometry_type=geom_type_arr,
        geometry_family=geom_family_arr,
        crop_size=np.asarray([n], dtype=np.int32),
        stride=np.asarray([n], dtype=np.int32),
        dataset_kind=np.asarray(["synthetic_geometry_ood_test"], dtype="U64"),
        chi_void_convention=np.asarray(["1=void,0=solid"], dtype="U32"),
        note=np.asarray(["Generated for zero-shot AENO testing after Bristol training."], dtype="U128"),
    )

    index_csv = out_path.with_suffix(".index.csv")
    with index_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rve_id", "split", "geometry_family", "geometry_type", "porosity", "porosity_bin_id", "z0", "y0", "x0"])
        for i, (sp, fam, typ, phi, bid, org) in enumerate(zip(splits, geom_families, geom_types, phis, bin_ids, origins)):
            writer.writerow([i, sp, fam, typ, f"{phi:.8f}", bid, org[0], org[1], org[2]])

    if args.overview_png:
        make_overview_png(chis, geom_types, geom_families, Path(args.overview_png), max_items=args.overview_max)

    print(f"Saved: {out_path}")
    print(f"Saved index: {index_csv}")
    print(f"N = {len(chis)}, shape = {chi_arr.shape}, porosity range = {porosity_arr.min():.4f}..{porosity_arr.max():.4f}, mean = {porosity_arr.mean():.4f}")
    print("Counts by geometry type:")
    for gt in sorted(set(geom_types)):
        idx = [i for i, t in enumerate(geom_types) if t == gt]
        vals = porosity_arr[idx]
        print(f"  {gt:24s} n={len(idx):4d}, phi_mean={vals.mean():.4f}, phi_min={vals.min():.4f}, phi_max={vals.max():.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic Bristol-like and strong-OOD 3D void geometries for AENO testing.")
    p.add_argument("--out", type=str, default="data/synthetic_void_rve32_test.npz")
    p.add_argument("--crop_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_bristol_like_per_type", type=int, default=25)
    p.add_argument("--num_ood_per_type", type=int, default=15)
    p.add_argument("--bristol_phi_min", type=float, default=0.005, help="Bristol-like minimum porosity, default 0.5%")
    p.add_argument("--bristol_phi_max", type=float, default=0.08, help="Bristol-like maximum porosity, default 8%")
    p.add_argument("--min_porosity", type=float, default=0.001)
    p.add_argument("--max_porosity", type=float, default=0.12, help="Hard cap for Bristol-like generated samples; set <=0 to disable")
    p.add_argument("--ood_max_porosity", type=float, default=0.45, help="Filter very extreme strong-OOD cases above this void fraction")
    p.add_argument("--overview_png", type=str, default="data/synthetic_void_rve32_overview.png")
    p.add_argument("--overview_max", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
