#!/usr/bin/env python3
"""Quantify morphology-error associations for the synthetic CT-RVE tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata, spearmanr

try:
    from skimage.measure import euler_number as skimage_euler_number
except ImportError:  # Optional dependency; the NumPy cubical-complex fallback is used.
    skimage_euler_number = None


ERROR_COLUMNS = {
    "stress_error": "rel_l2_sig",
    "effective_modulus_error": "rel_err_Eeff",
    "Ksigma95_error": "rel_err_Ksigma95",
}


def cubical_euler_number(mask):
    """Euler characteristic of the union of occupied unit cubes, V-E+F-C."""
    cubes = np.asarray(mask, dtype=bool)
    d, h, w = cubes.shape

    vertices = np.zeros((d + 1, h + 1, w + 1), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                vertices[dz : dz + d, dy : dy + h, dx : dx + w] |= cubes

    edges_x = np.zeros((d + 1, h + 1, w), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            edges_x[dz : dz + d, dy : dy + h, :] |= cubes
    edges_y = np.zeros((d + 1, h, w + 1), dtype=bool)
    for dz in (0, 1):
        for dx in (0, 1):
            edges_y[dz : dz + d, :, dx : dx + w] |= cubes
    edges_z = np.zeros((d, h + 1, w + 1), dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            edges_z[:, dy : dy + h, dx : dx + w] |= cubes

    faces_x = np.zeros((d, h, w + 1), dtype=bool)
    faces_x[:, :, :w] |= cubes
    faces_x[:, :, 1:] |= cubes
    faces_y = np.zeros((d, h + 1, w), dtype=bool)
    faces_y[:, :h, :] |= cubes
    faces_y[:, 1:, :] |= cubes
    faces_z = np.zeros((d + 1, h, w), dtype=bool)
    faces_z[:d, :, :] |= cubes
    faces_z[1:, :, :] |= cubes

    n_vertices = int(vertices.sum())
    n_edges = int(edges_x.sum() + edges_y.sum() + edges_z.sum())
    n_faces = int(faces_x.sum() + faces_y.sum() + faces_z.sum())
    n_cubes = int(cubes.sum())
    return n_vertices - n_edges + n_faces - n_cubes


def morphology_euler_number(mask):
    if skimage_euler_number is not None:
        return int(skimage_euler_number(mask, connectivity=3))
    return int(cubical_euler_number(mask))


def percolates(labels, axis):
    first = np.unique(np.take(labels, 0, axis=axis))
    last = np.unique(np.take(labels, -1, axis=axis))
    return float(len((set(first) & set(last)) - {0}) > 0)


def largest_axis_span(labels, axis):
    best = 0
    for label_id in range(1, int(labels.max()) + 1):
        coords = np.where(labels == label_id)[axis]
        if coords.size:
            best = max(best, int(coords.max() - coords.min() + 1))
    return float(best / labels.shape[axis])


def descriptors(chi):
    void = np.asarray(chi > 0.5, dtype=bool)
    solid = ~void
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_components = ndimage.label(void, structure=structure)
    interface_count = (
        np.count_nonzero(void[1:, :, :] != void[:-1, :, :])
        + np.count_nonzero(void[:, 1:, :] != void[:, :-1, :])
        + np.count_nonzero(void[:, :, 1:] != void[:, :, :-1])
    )
    possible_interfaces = (
        (void.shape[0] - 1) * void.shape[1] * void.shape[2]
        + void.shape[0] * (void.shape[1] - 1) * void.shape[2]
        + void.shape[0] * void.shape[1] * (void.shape[2] - 1)
    )
    # Array order is (z,y,x); loading is along x, hence axis=2.
    solid_fraction_by_x = solid.mean(axis=(0, 1))
    return {
        "porosity": float(void.mean()),
        "void_component_count": int(n_components),
        "void_euler_characteristic": morphology_euler_number(void),
        "void_percolates_x": percolates(labels, axis=2),
        "largest_void_x_span": largest_axis_span(labels, axis=2),
        "minimum_solid_cross_section_x": float(solid_fraction_by_x.min()),
        "solid_cross_section_cv_x": float(solid_fraction_by_x.std() / (solid_fraction_by_x.mean() + 1e-30)),
        "interface_density": float(interface_count / max(possible_interfaces, 1)),
    }


def residualized_rank(values, controls):
    y = rankdata(np.asarray(values, dtype=np.float64))
    X = np.asarray(controls, dtype=np.float64)
    X = np.column_stack([np.ones(len(y)), X])
    fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return y - fitted


def family_design(families):
    dummies = pd.get_dummies(pd.Series(families), drop_first=True, dtype=float)
    return dummies.to_numpy(dtype=np.float64)


def association(x, y, porosity, families, descriptor_name):
    raw_rho = float(spearmanr(x, y).statistic)
    family = family_design(families)
    controls = family if descriptor_name == "porosity" else np.column_stack([rankdata(porosity), family])
    rx = residualized_rank(x, controls)
    ry = residualized_rank(y, controls)
    partial_rho = float(np.corrcoef(rx, ry)[0, 1])
    return raw_rho, partial_rho


def stratified_bootstrap(x, y, porosity, families, descriptor_name, n_boot, rng):
    families = np.asarray(families)
    groups = [np.where(families == name)[0] for name in np.unique(families)]
    values = []
    for _ in range(n_boot):
        index = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        try:
            values.append(association(x[index], y[index], porosity[index], families[index], descriptor_name)[1])
        except (ValueError, np.linalg.LinAlgError):
            continue
    values = np.asarray([v for v in values if np.isfinite(v)])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratified_permutation_p(x, y, porosity, families, descriptor_name, observed, n_perm, rng):
    families = np.asarray(families)
    groups = [np.where(families == name)[0] for name in np.unique(families)]
    exceed = 0
    for _ in range(n_perm):
        permuted = np.asarray(x).copy()
        for group in groups:
            permuted[group] = rng.permutation(permuted[group])
        value = association(permuted, y, porosity, families, descriptor_name)[1]
        exceed += int(abs(value) >= abs(observed))
    return float((exceed + 1) / (n_perm + 1))


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * len(p) / (rank + 1))
        q[idx] = running
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    chis = data["chi_void"]
    family_key = "geometry_family" if "geometry_family" in data.files else "geom_family"
    type_key = "geometry_type" if "geometry_type" in data.files else "geom_type"
    families = data[family_key].astype(str)
    types = data[type_key].astype(str)
    rows = []
    for i, chi in enumerate(chis):
        row = {"rve_id": i, "morphology_family": families[i], "morphology_type": types[i]}
        row.update(descriptors(chi))
        rows.append(row)
    descriptor_frame = pd.DataFrame(rows)

    metrics = pd.read_csv(args.metrics)
    merged = descriptor_frame.merge(metrics, on="rve_id", how="inner", validate="one_to_one")
    if len(merged) != len(descriptor_frame):
        raise ValueError(f"Descriptor/metric merge retained {len(merged)} of {len(descriptor_frame)} synthetic samples")

    descriptor_columns = [c for c in descriptor_frame.columns if c not in {"rve_id", "morphology_family", "morphology_type"}]
    porosity = merged["porosity"].to_numpy(dtype=float)
    family_values = merged["morphology_family"].to_numpy()
    rng = np.random.default_rng(args.seed)
    associations = []
    for error_name, error_column in ERROR_COLUMNS.items():
        y = merged[error_column].to_numpy(dtype=float)
        for descriptor_name in descriptor_columns:
            x = merged[descriptor_name].to_numpy(dtype=float)
            if np.unique(x).size < 2:
                continue
            raw_rho, partial_rho = association(x, y, porosity, family_values, descriptor_name)
            ci_low, ci_high = stratified_bootstrap(
                x, y, porosity, family_values, descriptor_name, args.bootstrap, rng
            )
            p_value = stratified_permutation_p(
                x, y, porosity, family_values, descriptor_name, partial_rho, args.permutations, rng
            )
            associations.append({
                "error": error_name,
                "descriptor": descriptor_name,
                "spearman_rho": raw_rho,
                "partial_spearman_rho": partial_rho,
                "partial_ci_low": ci_low,
                "partial_ci_high": ci_high,
                "family_stratified_permutation_p": p_value,
                "controls": "morphology family" if descriptor_name == "porosity" else "porosity and morphology family",
                "n": len(merged),
            })

    association_frame = pd.DataFrame(associations)
    association_frame["bh_q"] = benjamini_hochberg(association_frame["family_stratified_permutation_p"])
    association_frame = association_frame.sort_values(["error", "partial_spearman_rho"], key=lambda s: s.abs() if s.name == "partial_spearman_rho" else s, ascending=[True, False])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "synthetic_morphology_descriptors_and_errors.csv", index=False)
    association_frame.to_csv(out_dir / "S6_morphology_error_associations.csv", index=False)
    stress = association_frame[association_frame["error"] == "stress_error"]
    selected = stress.iloc[stress["partial_spearman_rho"].abs().argmax()].to_dict()
    with open(out_dir / "main_text_descriptor_association.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(association_frame.to_string(index=False))
    print("\nSelected stress-error association:\n" + json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
