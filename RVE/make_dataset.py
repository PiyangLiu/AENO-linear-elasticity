import argparse
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm


POROSITY_BINS = [0.0, 0.01, 0.02, np.inf]
POROSITY_BIN_NAMES = ["phi_lt_1pct", "phi_1_2pct", "phi_ge_2pct"]


def natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def find_tif_files(tif_dir: Path):
    files = list(tif_dir.glob("*.tif")) + list(tif_dir.glob("*.tiff"))
    files = sorted(files, key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {tif_dir}")
    return files


def load_tif_stack(tif_dir: Path, z_start=None, z_end=None):
    files = find_tif_files(tif_dir)
    if z_start is not None or z_end is not None:
        files = files[z_start:z_end]
    arrs = []
    for f in tqdm(files, desc="Reading TIF slices"):
        arrs.append(tiff.imread(str(f)))
    vol = np.stack(arrs, axis=0).astype(np.float32)  # [Z, Y, X]
    finite = np.isfinite(vol)
    if not finite.all():
        vol[~finite] = np.nanmedian(vol)
    vmin, vmax = float(vol.min()), float(vol.max())
    vol = (vol - vmin) / (vmax - vmin + 1e-12)
    return vol


def global_void_threshold(volume, target_porosity):
    if not (0.0 < target_porosity < 1.0):
        raise ValueError("target_porosity must be between 0 and 1")
    return float(np.quantile(volume, target_porosity))


def crop_origins(shape, crop_size, stride):
    zmax, ymax, xmax = shape
    origins = []
    for z in range(0, zmax - crop_size + 1, stride):
        for y in range(0, ymax - crop_size + 1, stride):
            for x in range(0, xmax - crop_size + 1, stride):
                origins.append((z, y, x))
    return origins


def porosity_bin_name(phi: float) -> str:
    if phi < 0.01:
        return "phi_lt_1pct"
    if phi < 0.02:
        return "phi_1_2pct"
    return "phi_ge_2pct"


def porosity_bin_id(phi: float) -> int:
    if phi < 0.01:
        return 0
    if phi < 0.02:
        return 1
    return 2


def assign_spatial_splits(origins, crop_size, shape, axis="z", train_frac=0.70, val_frac=0.10):
    """
    Strict spatial split.

    A crop is assigned only if its full voxel support lies inside a slab. Crops crossing
    the train/val/test boundaries are discarded. This avoids direct overlap across
    splits even when stride < crop_size.
    """
    axis_to_i = {"z": 0, "y": 1, "x": 2}
    ai = axis_to_i[axis]
    n_axis = int(shape[ai])
    train_cut = int(round(train_frac * n_axis))
    val_cut = int(round((train_frac + val_frac) * n_axis))
    split = []
    keep = []
    for o in origins:
        a0 = int(o[ai])
        a1 = a0 + crop_size
        if a1 <= train_cut:
            split.append("train")
            keep.append(True)
        elif a0 >= train_cut and a1 <= val_cut:
            split.append("val")
            keep.append(True)
        elif a0 >= val_cut:
            split.append("test")
            keep.append(True)
        else:
            # Boundary-crossing crop: discard to avoid spatial leakage.
            split.append("gap")
            keep.append(False)
    return np.asarray(split, dtype="U8"), np.asarray(keep, dtype=bool), {
        "axis": axis,
        "n_axis": n_axis,
        "train_cut_voxel": train_cut,
        "val_cut_voxel": val_cut,
        "policy": "full crop support must lie inside split slab; boundary-crossing crops are discarded",
    }



def assign_spatial_block_splits(origins, crop_size, shape, axis="z", train_frac=0.70, val_frac=0.10, gap_layers=0):
    """
    Strict non-overlapping split by available crop-start layers.

    This avoids the empty-validation problem that can occur when voxel slabs are
    narrower than crop_size, e.g. 240 voxels with 32^3 crops and a 70/10/20 slab
    split. Crops are assigned by their origin layer along the selected axis.
    With stride >= crop_size, train/val/test crops do not overlap. Optional
    gap_layers discards crop-start layers between train/val and val/test.
    """
    axis_to_i = {"z": 0, "y": 1, "x": 2}
    ai = axis_to_i[axis]
    layer_values = sorted({int(o[ai]) for o in origins})
    n_layers = len(layer_values)
    if n_layers < 3:
        raise RuntimeError(
            f"Need at least 3 crop-start layers for train/val/test split along {axis}; got {n_layers}. "
            "Reduce crop_size/stride or use a larger volume."
        )
    if gap_layers < 0:
        raise ValueError("gap_layers must be >= 0")

    # Reserve at least one val layer and one test layer.
    n_val = max(1, int(round(val_frac * n_layers)))
    n_train = int(round(train_frac * n_layers))
    n_train = max(1, min(n_train, n_layers - n_val - 1 - 2 * gap_layers))
    if n_train < 1:
        raise RuntimeError(
            f"Not enough crop-start layers ({n_layers}) for gap_layers={gap_layers}; "
            "reduce --gap_layers or reduce --stride."
        )
    train_layers = set(layer_values[:n_train])
    val_start = n_train + gap_layers
    val_end = val_start + n_val
    if val_end + gap_layers >= n_layers:
        # Shrink val if needed, keeping at least one test layer.
        n_val = max(1, n_layers - val_start - gap_layers - 1)
        val_end = val_start + n_val
    val_layers = set(layer_values[val_start:val_end])
    test_start = val_end + gap_layers
    test_layers = set(layer_values[test_start:])
    if len(val_layers) == 0 or len(test_layers) == 0:
        raise RuntimeError(
            f"Empty val/test split with n_layers={n_layers}, n_train={n_train}, n_val={n_val}, gap_layers={gap_layers}."
        )

    split = []
    keep = []
    for o in origins:
        a0 = int(o[ai])
        if a0 in train_layers:
            split.append("train")
            keep.append(True)
        elif a0 in val_layers:
            split.append("val")
            keep.append(True)
        elif a0 in test_layers:
            split.append("test")
            keep.append(True)
        else:
            split.append("gap")
            keep.append(False)
    return np.asarray(split, dtype="U8"), np.asarray(keep, dtype=bool), {
        "axis": axis,
        "n_axis": int(shape[ai]),
        "crop_start_layers": layer_values,
        "n_layers": int(n_layers),
        "train_layers": sorted(train_layers),
        "val_layers": sorted(val_layers),
        "test_layers": sorted(test_layers),
        "gap_layers": int(gap_layers),
        "policy": "split by non-overlapping crop-start layers; optional gap layers are discarded",
    }

def assign_random_splits(origins, seed=0):
    # Retained only as a diagnostic baseline. Not recommended for microstructure generalization claims.
    rng = random.Random(seed)
    n = len(origins)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.10 * n)
    split = np.empty(n, dtype="U8")
    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train:n_train + n_val])
    for i in range(n):
        if i in train_idx:
            split[i] = "train"
        elif i in val_idx:
            split[i] = "val"
        else:
            split[i] = "test"
    return split, np.ones(n, dtype=bool), {"policy": "random crop split; use only for debugging, not for microstructure generalization claims"}


def cap_rves_stratified(chis, origins, porosities, splits, max_rves, seed=0):
    if max_rves is None or max_rves <= 0 or len(chis) <= max_rves:
        return np.arange(len(chis), dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    # Preserve train/val/test roughly and keep porosity-bin diversity.
    groups = []
    for sp in ["train", "val", "test"]:
        for bid in [0, 1, 2]:
            idx = [i for i, (s, p) in enumerate(zip(splits, porosities)) if s == sp and porosity_bin_id(float(p)) == bid]
            if idx:
                groups.append(idx)
    total = sum(len(g) for g in groups)
    for g in groups:
        quota = max(1, int(round(max_rves * len(g) / total)))
        quota = min(quota, len(g))
        selected.extend(rng.choice(g, size=quota, replace=False).tolist())
    if len(selected) > max_rves:
        selected = rng.choice(selected, size=max_rves, replace=False).tolist()
    selected = sorted(set(selected), key=lambda i: tuple(origins[i]))
    return np.asarray(selected, dtype=int)


def main():
    p = argparse.ArgumentParser(description="Build spatially separated void-containing RVE dataset from Bristol XCT TIF stack.")
    p.add_argument("--tif_dir", type=str, required=True, help="Folder containing reconstructed TIF slices, e.g. 25micron_60min")
    p.add_argument("--out", type=str, required=True, help="Output .npz dataset path")
    p.add_argument("--crop_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=32, help="Use stride >= crop_size for non-overlapping RVE crops. Default gives non-overlapping 32^3 RVEs.")
    p.add_argument("--target_porosity", type=float, default=0.02, help="Global low-greyscale quantile used as void threshold")
    p.add_argument("--min_porosity", type=float, default=0.001, help="Discard nearly void-free RVE crops below this porosity")
    p.add_argument("--max_porosity", type=float, default=0.20, help="Discard suspicious crops above this porosity")
    p.add_argument("--store_gray", action="store_true", help="Store normalized grayscale XCT RVE crops for visualization/debugging. Training ignores this key.")
    p.add_argument("--max_rves", type=int, default=0, help="Optional cap after filtering and splitting. 0 keeps all. Cap is split/bin-stratified.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="spatial_z_blocks", choices=["spatial_z_blocks", "spatial_y_blocks", "spatial_x_blocks", "spatial_z_strict", "spatial_y_strict", "spatial_x_strict", "random"])
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--gap_layers", type=int, default=0, help="For *_blocks split, discard this many crop-start layers between train/val and val/test.")
    p.add_argument("--z_start", type=int, default=None)
    p.add_argument("--z_end", type=int, default=None)
    args = p.parse_args()

    if args.stride < args.crop_size:
        warnings.warn(
            "stride < crop_size creates overlapping RVE crops. The strict spatial split still discards boundary-crossing crops, "
            "but neighboring crops inside the same split remain correlated. For a clean first demonstration, use stride=crop_size."
        )

    tif_dir = Path(args.tif_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    volume = load_tif_stack(tif_dir, args.z_start, args.z_end)
    threshold = global_void_threshold(volume, args.target_porosity)
    void_volume = volume <= threshold

    origins_all = crop_origins(volume.shape, args.crop_size, args.stride)
    if not origins_all:
        raise RuntimeError(f"No crops of size {args.crop_size} fit volume shape {volume.shape}")

    chis = []
    grays = []
    origins = []
    porosities = []
    for z, y, x in tqdm(origins_all, desc="Cropping RVEs"):
        gray = volume[z:z+args.crop_size, y:y+args.crop_size, x:x+args.crop_size]
        chi = void_volume[z:z+args.crop_size, y:y+args.crop_size, x:x+args.crop_size]
        phi = float(chi.mean())
        if args.min_porosity <= phi <= args.max_porosity:
            chis.append(chi.astype(np.uint8))
            if args.store_gray:
                grays.append(gray.astype(np.float32))
            origins.append((z, y, x))
            porosities.append(phi)

    if not chis:
        raise RuntimeError("No RVE crops passed porosity filtering. Relax --min_porosity/--max_porosity or check threshold.")

    if args.split_mode == "random":
        splits, keep, split_meta = assign_random_splits(origins, args.seed)
    else:
        axis = args.split_mode.split("_")[1]
        if args.split_mode.endswith("_blocks"):
            splits, keep, split_meta = assign_spatial_block_splits(
                origins, args.crop_size, volume.shape, axis=axis,
                train_frac=args.train_frac, val_frac=args.val_frac, gap_layers=args.gap_layers
            )
        else:
            splits, keep, split_meta = assign_spatial_splits(
                origins, args.crop_size, volume.shape, axis=axis,
                train_frac=args.train_frac, val_frac=args.val_frac
            )

    chis = [c for c, k in zip(chis, keep) if k]
    if args.store_gray:
        grays = [g for g, k in zip(grays, keep) if k]
    origins = [o for o, k in zip(origins, keep) if k]
    porosities = [p0 for p0, k in zip(porosities, keep) if k]
    splits = splits[keep]

    if not chis:
        raise RuntimeError("No RVE crops remain after strict spatial splitting. Reduce crop_size or adjust train/val fractions.")

    # Spatial order is kept for auditability; optional cap is stratified by split and porosity bin.
    keep_idx = cap_rves_stratified(chis, origins, porosities, splits, args.max_rves, seed=args.seed)
    chis = np.stack([chis[i] for i in keep_idx], axis=0)
    if args.store_gray:
        grays = np.stack([grays[i] for i in keep_idx], axis=0).astype(np.float32)
    origins = np.asarray([origins[i] for i in keep_idx], dtype=np.int32)
    porosities = np.asarray([porosities[i] for i in keep_idx], dtype=np.float32)
    splits = np.asarray([splits[i] for i in keep_idx], dtype="U8")
    bin_ids = np.asarray([porosity_bin_id(float(p0)) for p0 in porosities], dtype=np.int32)
    bin_names = np.asarray([porosity_bin_name(float(p0)) for p0 in porosities], dtype="U16")

    save_dict = dict(
        chi_void=chis,
        origins=origins,
        porosity=porosities,
        porosity_bin_id=bin_ids,
        porosity_bin=bin_names,
        split=splits,
        threshold=np.array([threshold], dtype=np.float32),
        crop_size=np.array([args.crop_size], dtype=np.int32),
        stride=np.array([args.stride], dtype=np.int32),
    )
    if args.store_gray:
        save_dict["gray"] = grays
    np.savez_compressed(out, **save_dict)

    index_df = pd.DataFrame({
        "rve_id": np.arange(chis.shape[0], dtype=int),
        "split": splits,
        "z0": origins[:, 0],
        "y0": origins[:, 1],
        "x0": origins[:, 2],
        "porosity": porosities,
        "porosity_bin": bin_names,
        "porosity_bin_id": bin_ids,
    })
    index_df.to_csv(out.with_suffix(".index.csv"), index=False)

    split_counts = index_df.groupby("split").size().to_dict()
    bin_counts = index_df.groupby(["split", "porosity_bin"]).size().unstack(fill_value=0).to_dict(orient="index")
    meta = {
        "tif_dir": str(tif_dir),
        "volume_shape_zyx": list(map(int, volume.shape)),
        "crop_size": args.crop_size,
        "stride": args.stride,
        "target_porosity": args.target_porosity,
        "store_gray": bool(args.store_gray),
        "global_threshold": threshold,
        "num_rves": int(chis.shape[0]),
        "porosity_min_mean_max": [float(porosities.min()), float(porosities.mean()), float(porosities.max())],
        "split_mode": args.split_mode,
        "split_meta": split_meta,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "porosity_bin_counts_by_split": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in bin_counts.items()},
        "microstructure_generalization_claim": "AENO is evaluated on spatially held-out, non-overlapping RVE crops from the same XCT volume, not on independent XCT scans.",
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    saved_npz = out if out.suffix == ".npz" else Path(str(out) + ".npz")
    print(json.dumps(meta, indent=2))
    print(f"Saved dataset: {saved_npz}")
    print(f"Saved index: {out.with_suffix('.index.csv')}")


if __name__ == "__main__":
    main()
