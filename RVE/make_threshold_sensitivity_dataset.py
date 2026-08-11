#!/usr/bin/env python3
"""Re-segment fixed Bristol crop origins for threshold-sensitivity runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_dataset import load_tif_stack, global_void_threshold, porosity_bin_id, porosity_bin_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_dir", required=True)
    parser.add_argument("--base_data", required=True, help="Baseline NPZ supplying fixed crop origins and split labels.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold_quantile", type=float, required=True)
    parser.add_argument("--store_gray", action="store_true")
    parser.add_argument("--z_start", type=int, default=None)
    parser.add_argument("--z_end", type=int, default=None)
    args = parser.parse_args()

    base = np.load(args.base_data, allow_pickle=False)
    origins = np.asarray(base["origins"], dtype=np.int32)
    splits = np.asarray(base["split"])
    crop_size = int(np.asarray(base["crop_size"]).reshape(-1)[0])
    stride = int(np.asarray(base["stride"]).reshape(-1)[0])

    volume = load_tif_stack(Path(args.tif_dir), args.z_start, args.z_end)
    threshold = global_void_threshold(volume, args.threshold_quantile)
    void_volume = volume <= threshold

    chis = []
    grays = []
    for z, y, x in origins:
        gray = volume[z : z + crop_size, y : y + crop_size, x : x + crop_size]
        if gray.shape != (crop_size, crop_size, crop_size):
            raise ValueError(f"Baseline origin {(z, y, x)} does not fit volume shape {volume.shape}")
        chis.append(void_volume[z : z + crop_size, y : y + crop_size, x : x + crop_size].astype(np.uint8))
        if args.store_gray:
            grays.append(gray.astype(np.float32))

    chis = np.stack(chis)
    porosities = chis.mean(axis=(1, 2, 3)).astype(np.float32)
    bin_ids = np.asarray([porosity_bin_id(float(v)) for v in porosities], dtype=np.int32)
    bin_names = np.asarray([porosity_bin_name(float(v)) for v in porosities], dtype="U16")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi_void": chis,
        "origins": origins,
        "porosity": porosities,
        "porosity_bin_id": bin_ids,
        "porosity_bin": bin_names,
        "split": splits,
        "threshold": np.asarray([threshold], dtype=np.float32),
        "crop_size": np.asarray([crop_size], dtype=np.int32),
        "stride": np.asarray([stride], dtype=np.int32),
    }
    if args.store_gray:
        payload["gray"] = np.stack(grays).astype(np.float32)
    np.savez_compressed(out, **payload)

    index = pd.DataFrame({
        "rve_id": np.arange(len(chis)),
        "split": splits,
        "z0": origins[:, 0],
        "y0": origins[:, 1],
        "x0": origins[:, 2],
        "porosity": porosities,
        "porosity_bin": bin_names,
        "porosity_bin_id": bin_ids,
    })
    index.to_csv(out.with_suffix(".index.csv"), index=False)
    metadata = {
        "source_volume": str(Path(args.tif_dir)),
        "base_data": str(Path(args.base_data)),
        "threshold_quantile": args.threshold_quantile,
        "global_threshold": threshold,
        "fixed_origins_and_splits": True,
        "num_rves": int(len(chis)),
        "split_counts": {str(k): int(v) for k, v in index.groupby("split").size().items()},
        "porosity_min_mean_max": [float(porosities.min()), float(porosities.mean()), float(porosities.max())],
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
