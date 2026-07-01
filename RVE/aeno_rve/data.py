from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import compute_void_features


class RVENPZDataset(Dataset):
    def __init__(
        self,
        npz_path,
        split="train",
        Es_range=(0.5, 2.0),
        nu_range=(0.25, 0.40),
        seed=0,
        fixed_material=None,
        precompute_features=True,
        high_nu_prob=0.0,
        high_nu_min=0.35,
    ):
        data = np.load(npz_path, allow_pickle=True)
        chi = data["chi_void"].astype(np.float32)
        gray = data["gray"].astype(np.float32) if "gray" in data else None
        splits = data["split"].astype(str)
        porosity = data["porosity"].astype(np.float32)
        origins = data["origins"].astype(np.int32)
        if "porosity_bin_id" in data:
            porosity_bin_id = data["porosity_bin_id"].astype(np.int32)
        else:
            porosity_bin_id = np.where(porosity < 0.01, 0, np.where(porosity < 0.02, 1, 2)).astype(np.int32)
        mask = splits == split
        if mask.sum() == 0:
            raise RuntimeError(f"No samples found for split={split}")
        self.chi = chi[mask]
        self.gray = gray[mask] if gray is not None else None
        self.porosity = porosity[mask]
        self.porosity_bin_id = porosity_bin_id[mask]
        self.origins = origins[mask]
        self.indices = np.nonzero(mask)[0].astype(np.int32)
        self.Es_min, self.Es_max = map(float, Es_range)
        self.nu_min, self.nu_max = map(float, nu_range)
        self.rng = np.random.default_rng(seed)
        self.fixed_material = fixed_material
        self.high_nu_prob = float(high_nu_prob)
        self.high_nu_min = float(high_nu_min)

        self.sdf_void = None
        self.interface = None
        self.boundary_distance = None
        if precompute_features:
            sdfs, interfaces, bds = [], [], []
            for c in self.chi:
                feats = compute_void_features(c, normalize_by=max(c.shape))
                sdfs.append(feats["sdf_void"])
                interfaces.append(feats["interface"])
                bds.append(feats["boundary_distance"])
            self.sdf_void = np.stack(sdfs).astype(np.float32)
            self.interface = np.stack(interfaces).astype(np.float32)
            self.boundary_distance = np.stack(bds).astype(np.float32)

    def __len__(self):
        return len(self.chi)

    def _sample_material(self):
        Es = self.rng.uniform(self.Es_min, self.Es_max)
        # Mixture sampling improves the worst case at high nu without changing the test target.
        if self.high_nu_prob > 0 and self.rng.random() < self.high_nu_prob:
            lo = min(max(self.high_nu_min, self.nu_min), self.nu_max)
            nu = self.rng.uniform(lo, self.nu_max)
        else:
            nu = self.rng.uniform(self.nu_min, self.nu_max)
        return Es, nu

    def __getitem__(self, i):
        if self.fixed_material is None:
            Es, nu = self._sample_material()
        else:
            Es, nu = self.fixed_material
        out = {
            "chi_void": torch.from_numpy(self.chi[i]),
            "Es": torch.tensor(Es, dtype=torch.float32),
            "nu_s": torch.tensor(nu, dtype=torch.float32),
            "porosity": torch.tensor(self.porosity[i], dtype=torch.float32),
            "porosity_bin_id": torch.tensor(int(self.porosity_bin_id[i]), dtype=torch.long),
            "rve_id": torch.tensor(int(self.indices[i]), dtype=torch.long),
            "origin": torch.from_numpy(self.origins[i]),
        }
        if self.gray is not None:
            out["gray"] = torch.from_numpy(self.gray[i])
        if self.sdf_void is not None:
            out["sdf_void"] = torch.from_numpy(self.sdf_void[i])
            out["interface"] = torch.from_numpy(self.interface[i])
            out["boundary_distance"] = torch.from_numpy(self.boundary_distance[i])
        return out
