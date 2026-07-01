from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch


COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def lame_parameters(E: torch.Tensor, nu: torch.Tensor):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def stress_from_strain(eps: dict, E: torch.Tensor, nu: torch.Tensor):
    """Isotropic stress from tensorial small strain. E, nu broadcast to [B,D,H,W]."""
    lam, mu = lame_parameters(E, nu)
    tr = eps["xx"] + eps["yy"] + eps["zz"]
    sig = {
        "xx": lam * tr + 2.0 * mu * eps["xx"],
        "yy": lam * tr + 2.0 * mu * eps["yy"],
        "zz": lam * tr + 2.0 * mu * eps["zz"],
        "xy": 2.0 * mu * eps["xy"],
        "xz": 2.0 * mu * eps["xz"],
        "yz": 2.0 * mu * eps["yz"],
    }
    return sig


def energy_density(eps: dict, sig: dict):
    return 0.5 * (
        sig["xx"] * eps["xx"] + sig["yy"] * eps["yy"] + sig["zz"] * eps["zz"]
        + 2.0 * sig["xy"] * eps["xy"] + 2.0 * sig["xz"] * eps["xz"] + 2.0 * sig["yz"] * eps["yz"]
    )


def von_mises(sig: dict):
    sxx, syy, szz = sig["xx"], sig["yy"], sig["zz"]
    sxy, sxz, syz = sig["xy"], sig["xz"], sig["yz"]
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    return torch.sqrt(torch.clamp(vm2, min=0.0) + 1e-20)


def build_material_fields(chi_void: torch.Tensor, Es: torch.Tensor, nu_s: torch.Tensor, alpha_void=1e-4):
    """Build element-centered E(x), nu(x) for a solid-void RVE.

    chi_void: [B,N,N,N], 1=void, 0=solid.
    Es, nu_s: [B] or [B,1].
    """
    B = chi_void.shape[0]
    view = (B, 1, 1, 1)
    Es = Es.reshape(view)
    nu_s = nu_s.reshape(view)
    E = Es * ((1.0 - chi_void) + alpha_void * chi_void)
    nu_void = torch.full_like(nu_s, 0.20)
    nu = nu_s * (1.0 - chi_void) + nu_void * chi_void
    return E, nu


def _hex8_local_signs(device, dtype):
    return torch.tensor([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ], device=device, dtype=dtype)


def _shape_derivatives_hex8(xi, eta, zeta, device, dtype):
    s = _hex8_local_signs(device, dtype)
    sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
    xi = torch.as_tensor(xi, device=device, dtype=dtype)
    eta = torch.as_tensor(eta, device=device, dtype=dtype)
    zeta = torch.as_tensor(zeta, device=device, dtype=dtype)
    dN_dxi = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
    dN_deta = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
    dN_dzeta = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return torch.stack([dN_dxi, dN_deta, dN_dzeta], dim=1)  # [8,3], local xi,eta,zeta


def _B_matrix_from_dN_dx(dN_dx: torch.Tensor) -> torch.Tensor:
    """Build tensorial-strain B matrix from physical derivatives [8,3]."""
    B = torch.zeros((6, 24), dtype=dN_dx.dtype, device=dN_dx.device)
    for a in range(8):
        dNx, dNy, dNz = dN_dx[a, 0], dN_dx[a, 1], dN_dx[a, 2]
        c = 3 * a
        B[0, c + 0] = dNx
        B[1, c + 1] = dNy
        B[2, c + 2] = dNz
        B[3, c + 0] = 0.5 * dNy
        B[3, c + 1] = 0.5 * dNx
        B[4, c + 0] = 0.5 * dNz
        B[4, c + 2] = 0.5 * dNx
        B[5, c + 1] = 0.5 * dNz
        B[5, c + 2] = 0.5 * dNy
    return B


def hex8_B_center(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Center-point Hex8 B matrix for a unit cube split into n elements per axis."""
    h = 1.0 / float(n)
    dN = _shape_derivatives_hex8(0.0, 0.0, 0.0, device, dtype)
    dN = dN.clone()
    dN[:, 0] *= 2.0 / h
    dN[:, 1] *= 2.0 / h
    dN[:, 2] *= 2.0 / h
    return _B_matrix_from_dN_dx(dN)


def hex8_B_gauss(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """2x2x2 Gauss Hex8 B matrices [8,6,24] for a unit cube split into n elements."""
    h = 1.0 / float(n)
    gp = 1.0 / (3.0 ** 0.5)
    mats = []
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                dN = _shape_derivatives_hex8(xi, eta, zeta, device, dtype).clone()
                dN[:, 0] *= 2.0 / h
                dN[:, 1] *= 2.0 / h
                dN[:, 2] *= 2.0 / h
                mats.append(_B_matrix_from_dN_dx(dN))
    return torch.stack(mats, dim=0)


def isotropic_D_matrix(E: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Batch isotropic elasticity matrix [B,6,6] for tensorial shear strains."""
    E = E.reshape(-1)
    nu = nu.reshape(-1)
    Bsz = E.shape[0]
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = torch.zeros((Bsz, 6, 6), device=E.device, dtype=E.dtype)
    D[:, :3, :3] = lam[:, None, None]
    D[:, 0, 0] += 2.0 * mu
    D[:, 1, 1] += 2.0 * mu
    D[:, 2, 2] += 2.0 * mu
    D[:, 3, 3] = 2.0 * mu
    D[:, 4, 4] = 2.0 * mu
    D[:, 5, 5] = 2.0 * mu
    return D


def hex8_stiffness_batch(E: torch.Tensor, nu: torch.Tensor, n: int) -> torch.Tensor:
    """2x2x2 Gauss-integrated element stiffness matrices [B,24,24]."""
    D = isotropic_D_matrix(E, nu)
    Bg = hex8_B_gauss(n, device=E.device, dtype=E.dtype)
    detJ = (1.0 / float(n)) ** 3 / 8.0
    Ke = torch.zeros((D.shape[0], 24, 24), device=E.device, dtype=E.dtype)
    for g in range(Bg.shape[0]):
        Bmat = Bg[g]
        Ke = Ke + torch.einsum("mi,bmn,nj->bij", Bmat, D, Bmat) * detJ
    return Ke


def element_dof_stack(u_nodal: torch.Tensor) -> torch.Tensor:
    """Gather Hex8 nodal displacement dofs as [B,24,N,N,N].

    u_nodal has shape [B,3,N+1,N+1,N+1], axes are z,y,x and components x,y,z.
    Node ordering matches aeno_rve.fem.element_nodes.
    """
    if u_nodal.ndim != 5 or u_nodal.shape[1] != 3:
        raise ValueError(f"u_nodal must be [B,3,N+1,N+1,N+1], got {tuple(u_nodal.shape)}")
    n = u_nodal.shape[-1] - 1
    nodes = [
        u_nodal[:, :, 0:n,   0:n,   0:n  ],
        u_nodal[:, :, 0:n,   0:n,   1:n+1],
        u_nodal[:, :, 0:n,   1:n+1, 1:n+1],
        u_nodal[:, :, 0:n,   1:n+1, 0:n  ],
        u_nodal[:, :, 1:n+1, 0:n,   0:n  ],
        u_nodal[:, :, 1:n+1, 0:n,   1:n+1],
        u_nodal[:, :, 1:n+1, 1:n+1, 1:n+1],
        u_nodal[:, :, 1:n+1, 1:n+1, 0:n  ],
    ]
    # [B,3,8,N,N,N] -> [B,8,3,N,N,N] -> node-major dof order.
    return torch.stack(nodes, dim=2).permute(0, 2, 1, 3, 4, 5).reshape(u_nodal.shape[0], 24, n, n, n)


def u_center_from_nodal(u_nodal: torch.Tensor) -> torch.Tensor:
    """Element-center displacement by averaging the eight Hex8 nodal values."""
    n = u_nodal.shape[-1] - 1
    nodes = [
        u_nodal[:, :, 0:n,   0:n,   0:n  ],
        u_nodal[:, :, 0:n,   0:n,   1:n+1],
        u_nodal[:, :, 0:n,   1:n+1, 1:n+1],
        u_nodal[:, :, 0:n,   1:n+1, 0:n  ],
        u_nodal[:, :, 1:n+1, 0:n,   0:n  ],
        u_nodal[:, :, 1:n+1, 0:n,   1:n+1],
        u_nodal[:, :, 1:n+1, 1:n+1, 1:n+1],
        u_nodal[:, :, 1:n+1, 1:n+1, 0:n  ],
    ]
    return torch.stack(nodes, dim=0).mean(dim=0)


def strain_from_nodal_hex8(u_nodal: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Element-centered small strain using the same center Hex8 B matrix as FEM."""
    n = u_nodal.shape[-1] - 1
    ue = element_dof_stack(u_nodal)
    Bc = hex8_B_center(n, device=u_nodal.device, dtype=u_nodal.dtype)
    eps_vec = torch.einsum("ij,bjzyx->bizyx", Bc, ue)  # [B,6,N,N,N]
    return {k: eps_vec[:, i] for i, k in enumerate(COMPONENTS)}


def assemble_element_vectors(fe: torch.Tensor, nnode: int) -> torch.Tensor:
    """Assemble element nodal vectors fe[B,24,N,N,N] into global nodal residual."""
    B = fe.shape[0]
    n = nnode - 1
    fev = fe.reshape(B, 8, 3, n, n, n)
    r = torch.zeros((B, 3, nnode, nnode, nnode), device=fe.device, dtype=fe.dtype)
    sl = [
        (slice(0, n),   slice(0, n),   slice(0, n)),
        (slice(0, n),   slice(0, n),   slice(1, nnode)),
        (slice(0, n),   slice(1, nnode), slice(1, nnode)),
        (slice(0, n),   slice(1, nnode), slice(0, n)),
        (slice(1, nnode), slice(0, n),   slice(0, n)),
        (slice(1, nnode), slice(0, n),   slice(1, nnode)),
        (slice(1, nnode), slice(1, nnode), slice(1, nnode)),
        (slice(1, nnode), slice(1, nnode), slice(0, n)),
    ]
    for a, (zs, ys, xs) in enumerate(sl):
        r[:, :, zs, ys, xs] = r[:, :, zs, ys, xs] + fev[:, a]
    return r


def assemble_element_scalar_to_nodes(w_elem: torch.Tensor, nnode: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average an element scalar field to nodes by simple adjacency accumulation."""
    B = w_elem.shape[0]
    n = nnode - 1
    accum = torch.zeros((B, 1, nnode, nnode, nnode), device=w_elem.device, dtype=w_elem.dtype)
    count = torch.zeros_like(accum)
    w = w_elem[:, None]
    sl = [
        (slice(0, n),   slice(0, n),   slice(0, n)),
        (slice(0, n),   slice(0, n),   slice(1, nnode)),
        (slice(0, n),   slice(1, nnode), slice(1, nnode)),
        (slice(0, n),   slice(1, nnode), slice(0, n)),
        (slice(1, nnode), slice(0, n),   slice(0, n)),
        (slice(1, nnode), slice(0, n),   slice(1, nnode)),
        (slice(1, nnode), slice(1, nnode), slice(1, nnode)),
        (slice(1, nnode), slice(1, nnode), slice(0, n)),
    ]
    one = torch.ones_like(w)
    for zs, ys, xs in sl:
        accum[:, :, zs, ys, xs] = accum[:, :, zs, ys, xs] + w
        count[:, :, zs, ys, xs] = count[:, :, zs, ys, xs] + one
    return accum / torch.clamp(count, min=1.0), count


def element_internal_force_and_energy(
    u_nodal: torch.Tensor,
    chi_void: torch.Tensor,
    Es: torch.Tensor,
    nu_s: torch.Tensor,
    alpha_void: float = 1e-4,
):
    """Return element internal forces and total strain energy.

    The force is fe = Ke(chi,E,nu) ue with Ke from the same 2x2x2 Gauss Hex8
    integration used by the FEM assembler. The energy is 0.5 * ue^T fe summed
    over all elements in the unit cube.
    """
    n = chi_void.shape[-1]
    ue = element_dof_stack(u_nodal)
    Bsz = chi_void.shape[0]
    Esv = Es.reshape(Bsz)
    nuv = nu_s.reshape(Bsz)
    Ke_solid = hex8_stiffness_batch(Esv, nuv, n)
    Ke_void = hex8_stiffness_batch(alpha_void * Esv, torch.full_like(nuv, 0.20), n)
    fs = torch.einsum("bij,bjzyx->bizyx", Ke_solid, ue)
    fv = torch.einsum("bij,bjzyx->bizyx", Ke_void, ue)
    q = chi_void[:, None]
    fe = (1.0 - q) * fs + q * fv
    energy_per = 0.5 * (ue * fe).sum(dim=1).sum(dim=(1, 2, 3))
    return fe, energy_per


def fem_energy_residual_objective(
    u_nodal: torch.Tensor,
    chi_void: torch.Tensor,
    Es: torch.Tensor,
    nu_s: torch.Tensor,
    eps0: float,
    alpha_void: float = 1e-4,
    interface: torch.Tensor | None = None,
    interface_alpha: float = 2.0,
    residual_scale: str = "force",
):
    """Energy and free-DOF equilibrium residual terms for label-free training.

    Returns a dict with:
      energy_per: total strain energy per sample, [B]
      residual_per: normalized weighted residual MSE per sample, [B]
      residual_raw: unnormalized weighted residual MSE per sample, [B]
    """
    fe, energy_per = element_internal_force_and_energy(u_nodal, chi_void, Es, nu_s, alpha_void=alpha_void)
    nnode = u_nodal.shape[-1]
    n = nnode - 1
    r = assemble_element_vectors(fe, nnode)
    r_free = r[:, :, 1:n, 1:n, 1:n]

    if interface is not None and interface_alpha > 0.0:
        w_elem = 1.0 + float(interface_alpha) * torch.clamp(interface, 0.0, 1.0)
        w_node, _ = assemble_element_scalar_to_nodes(w_elem, nnode)
        w_free = w_node[:, :, 1:n, 1:n, 1:n]
        residual_raw = (w_free * r_free.square()).sum(dim=(1, 2, 3, 4)) / (3.0 * w_free.sum(dim=(1, 2, 3, 4)) + 1e-30)
    else:
        residual_raw = r_free.square().mean(dim=(1, 2, 3, 4))

    if residual_scale == "force":
        # Internal nodal force scale ~ stress * face area ~ E * eps0 * h^2.
        scale = (Es.reshape(-1) * float(eps0) / float(n * n)).square() + 1e-30
        residual_per = residual_raw / scale
    elif residual_scale == "energy":
        # Alternative: scale by mean elastic energy density squared. Mostly for diagnostics.
        scale = (energy_per / max(n ** 3, 1)).square() + 1e-30
        residual_per = residual_raw / scale
    elif residual_scale == "none":
        residual_per = residual_raw
    else:
        raise ValueError(f"Unknown residual_scale={residual_scale!r}; use force, energy or none")

    return {"energy_per": energy_per, "residual_per": residual_per, "residual_raw": residual_raw, "residual": r}


def compute_fields(u_nodal, chi_void, Es, nu_s, eps0, spacing=None, alpha_void=1e-4):
    """Recover element-centered fields from nodal displacement using Hex8 B matrix.

    Parameters
    ----------
    u_nodal:
        [B,3,N+1,N+1,N+1] nodal displacement. A legacy [B,3,N,N,N]
        cell-centered field is intentionally not accepted in V2, because strain
        recovery must be FEM-consistent.
    chi_void:
        [B,N,N,N] element void indicator/fraction.
    spacing:
        Kept only for backward-compatible call signatures; ignored in V2.
    """
    n = chi_void.shape[-1]
    if u_nodal.shape[-1] != n + 1:
        raise ValueError(
            f"V2 expects nodal displacement [B,3,N+1,N+1,N+1] for N={n}; got {tuple(u_nodal.shape)}. "
            "Use a V2 checkpoint/model."
        )
    E, nu = build_material_fields(chi_void, Es, nu_s, alpha_void=alpha_void)
    eps = strain_from_nodal_hex8(u_nodal)
    sig = stress_from_strain(eps, E, nu)
    ed = energy_density(eps, sig)
    vm = von_mises(sig)
    eff = sig["xx"].mean(dim=(1, 2, 3)) / eps0
    mean_vm = vm.mean(dim=(1, 2, 3)) + 1e-12
    p95 = torch.quantile(vm.flatten(1), 0.95, dim=1)
    k95 = p95 / mean_vm
    u_center = u_center_from_nodal(u_nodal)
    return {
        "E": E,
        "nu": nu,
        "eps": eps,
        "sig": sig,
        "energy_density": ed,
        "von_mises": vm,
        "Eeff": eff,
        "Ksigma95": k95,
        "u_center": u_center,
    }
