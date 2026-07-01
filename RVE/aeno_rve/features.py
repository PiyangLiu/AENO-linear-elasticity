from __future__ import annotations

import numpy as np


def _safe_edt(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform with a finite fallback for all-true/all-false masks."""
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for V1 microstructure features. Install with `pip install scipy`.") from exc
    mask = mask.astype(bool)
    if mask.all():
        # All pixels are inside the phase. No interface exists in this crop.
        return np.full(mask.shape, max(mask.shape), dtype=np.float32)
    if (~mask).all():
        return np.zeros(mask.shape, dtype=np.float32)
    return distance_transform_edt(mask).astype(np.float32)


def compute_void_features(chi_void: np.ndarray, normalize_by: float | None = None) -> dict[str, np.ndarray]:
    """Compute interface-aware morphology features for a void/solid RVE.

    Parameters
    ----------
    chi_void:
        Array [D,H,W], 1 for void and 0 for solid.
    normalize_by:
        Length scale used to normalize distances. Defaults to max(D,H,W).

    Returns
    -------
    dict with:
        sdf_void: positive in solid, negative in void, approximately distance to void interface.
        interface: 1 near void-solid interface, 0 away from interface.
        boundary_distance: normalized distance to the exterior domain boundary.
    """
    try:
        from scipy.ndimage import maximum_filter, minimum_filter
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for V1 microstructure features. Install with `pip install scipy`.") from exc

    chi = (chi_void > 0.5)
    D, H, W = chi.shape
    L = float(normalize_by if normalize_by is not None else max(D, H, W))

    # Signed distance: positive in solid, negative in void.
    dist_solid_to_void = _safe_edt(~chi)  # distance for solid voxels to nearest void
    dist_void_to_solid = _safe_edt(chi)   # distance for void voxels to nearest solid
    sdf = (dist_solid_to_void - dist_void_to_solid) / max(L, 1.0)
    sdf = np.clip(sdf, -1.0, 1.0).astype(np.float32)

    # Morphological interface indicator. This is intentionally local and cheap.
    chi_f = chi.astype(np.float32)
    interface = maximum_filter(chi_f, size=3, mode="nearest") - minimum_filter(chi_f, size=3, mode="nearest")
    interface = np.clip(interface, 0.0, 1.0).astype(np.float32)

    # Distance to external boundary, normalized to roughly [0,0.5].
    z = np.linspace(0.0, 1.0, D, dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    bd = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy, zz, 1.0 - zz]).astype(np.float32)

    return {"sdf_void": sdf, "interface": interface, "boundary_distance": bd}
