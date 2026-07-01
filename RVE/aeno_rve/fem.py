from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class FEMResult:
    u_center: np.ndarray          # [3, N, N, N]
    u_nodal: np.ndarray           # [N+1, N+1, N+1, 3]
    eps: Dict[str, np.ndarray]    # each [N, N, N]
    sig: Dict[str, np.ndarray]    # each [N, N, N]
    von_mises: np.ndarray         # [N, N, N]
    Eeff: float
    Ksigma95: float
    solve_info: Dict[str, float]


def isotropic_D(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix using tensorial shear strains.

    Strain vector order: [xx, yy, zz, xy, xz, yz], where xy/xz/yz are tensor
    shear components, not engineering shear gamma. Therefore the shear diagonal
    entries are 2*mu.
    """
    E = float(E)
    nu = float(nu)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[:3, :3] = lam
    D[0, 0] += 2.0 * mu
    D[1, 1] += 2.0 * mu
    D[2, 2] += 2.0 * mu
    D[3, 3] = 2.0 * mu
    D[4, 4] = 2.0 * mu
    D[5, 5] = 2.0 * mu
    return D


_HEX8_LOCAL = np.asarray([
    [-1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0],
], dtype=np.float64)


def shape_derivatives_hex8(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return dN/d(xi,eta,zeta) as [8,3] for a trilinear hex element."""
    sx = _HEX8_LOCAL[:, 0]
    sy = _HEX8_LOCAL[:, 1]
    sz = _HEX8_LOCAL[:, 2]
    dN_dxi = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
    dN_deta = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
    dN_dzeta = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return np.stack([dN_dxi, dN_deta, dN_dzeta], axis=1)


def B_matrix_hex8(dN_dx: np.ndarray) -> np.ndarray:
    """Build tensorial-strain B matrix from physical derivatives [8,3]."""
    B = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        dNx, dNy, dNz = dN_dx[a]
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


def hex8_B_center(hx: float, hy: float, hz: float) -> np.ndarray:
    dN_dlocal = shape_derivatives_hex8(0.0, 0.0, 0.0)
    # local coordinates map to physical coordinates: x = hx/2 * xi + x0.
    dN_dx = dN_dlocal.copy()
    dN_dx[:, 0] *= 2.0 / hx
    dN_dx[:, 1] *= 2.0 / hy
    dN_dx[:, 2] *= 2.0 / hz
    return B_matrix_hex8(dN_dx)


def hex8_stiffness(E: float, nu: float, hx: float, hy: float, hz: float) -> np.ndarray:
    """2x2x2 Gauss-integrated stiffness matrix for one Hex8 element."""
    D = isotropic_D(E, nu)
    gp = 1.0 / np.sqrt(3.0)
    detJ = hx * hy * hz / 8.0
    Ke = np.zeros((24, 24), dtype=np.float64)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                dN_dlocal = shape_derivatives_hex8(xi, eta, zeta)
                dN_dx = dN_dlocal.copy()
                dN_dx[:, 0] *= 2.0 / hx
                dN_dx[:, 1] *= 2.0 / hy
                dN_dx[:, 2] *= 2.0 / hz
                B = B_matrix_hex8(dN_dx)
                Ke += B.T @ D @ B * detJ
    return Ke


def node_id(z: int, y: int, x: int, nnode: int) -> int:
    return (z * nnode + y) * nnode + x


def element_nodes(z: int, y: int, x: int, nnode: int) -> Tuple[int, ...]:
    return (
        node_id(z,     y,     x,     nnode),
        node_id(z,     y,     x + 1, nnode),
        node_id(z,     y + 1, x + 1, nnode),
        node_id(z,     y + 1, x,     nnode),
        node_id(z + 1, y,     x,     nnode),
        node_id(z + 1, y,     x + 1, nnode),
        node_id(z + 1, y + 1, x + 1, nnode),
        node_id(z + 1, y + 1, x,     nnode),
    )


def element_dofs(nodes: Tuple[int, ...]) -> np.ndarray:
    dofs = np.empty(24, dtype=np.int64)
    for a, nid in enumerate(nodes):
        dofs[3 * a + 0] = 3 * nid + 0
        dofs[3 * a + 1] = 3 * nid + 1
        dofs[3 * a + 2] = 3 * nid + 2
    return dofs


def von_mises_np(sig_vec: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, sxz, syz = [sig_vec[..., i] for i in range(6)]
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    return np.sqrt(np.maximum(vm2, 0.0) + 1e-20)


def block_average_3d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average last three axes by an integer factor."""
    factor = int(factor)
    if factor == 1:
        return arr.copy()
    if factor <= 0:
        raise ValueError("factor must be positive")
    leading = arr.shape[:-3]
    d, h, w = arr.shape[-3:]
    d2, h2, w2 = d // factor, h // factor, w // factor
    trimmed = arr[..., :d2 * factor, :h2 * factor, :w2 * factor]
    new_shape = leading + (d2, factor, h2, factor, w2, factor)
    return trimmed.reshape(new_shape).mean(axis=(-5, -3, -1))


def solve_fem_kubc(
    chi_void: np.ndarray,
    Es: float,
    nu_s: float,
    eps0: float = 0.01,
    alpha_void: float = 1e-4,
    tol: float = 1e-8,
    maxiter: int = 2000,
    verbose: bool = False,
) -> FEMResult:
    """Solve a voxelized solid-void RVE with Hex8 FEM and KUBC.

    Parameters
    ----------
    chi_void:
        Array [N,N,N]. Values may be binary or continuous void fractions in [0,1].
        Element stiffness is interpolated as E=Es*((1-chi)+alpha_void*chi),
        nu=nu_s*(1-chi)+0.20*chi.
    Es, nu_s:
        Solid-phase elastic parameters.
    eps0:
        Applied macroscopic strain in x direction. Boundary displacement is
        u=[eps0*x,0,0] on the entire outer surface.
    """
    t0 = time.perf_counter()
    chi = np.asarray(chi_void, dtype=np.float64)
    if chi.ndim != 3 or chi.shape[0] != chi.shape[1] or chi.shape[0] != chi.shape[2]:
        raise ValueError(f"chi_void must be cubic [N,N,N], got {chi.shape}")
    chi = np.clip(chi, 0.0, 1.0)
    n = chi.shape[0]
    nnode = n + 1
    nnodes = nnode ** 3
    ndof = 3 * nnodes
    nelem = n ** 3
    hx = hy = hz = 1.0 / n

    if verbose:
        print(f"[FEM] n={n}, elements={nelem:,}, nodes={nnodes:,}, dofs={ndof:,}")

    # For binary data this exactly uses two matrices; for continuous downsampled
    # void fractions we linearly mix element stiffnesses. This is equivalent to a
    # local Voigt-type element stiffness interpolation.
    Ke_solid = hex8_stiffness(Es, nu_s, hx, hy, hz)
    Ke_void = hex8_stiffness(alpha_void * Es, 0.20, hx, hy, hz)

    nentry = nelem * 24 * 24
    rows = np.empty(nentry, dtype=np.int64)
    cols = np.empty(nentry, dtype=np.int64)
    vals = np.empty(nentry, dtype=np.float64)

    ptr = 0
    for z in range(n):
        for y in range(n):
            for x in range(n):
                q = chi[z, y, x]
                Ke = (1.0 - q) * Ke_solid + q * Ke_void
                dofs = element_dofs(element_nodes(z, y, x, nnode))
                rr = np.repeat(dofs, 24)
                cc = np.tile(dofs, 24)
                rows[ptr:ptr + 576] = rr
                cols[ptr:ptr + 576] = cc
                vals[ptr:ptr + 576] = Ke.reshape(-1)
                ptr += 576

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    del rows, cols, vals
    t_assemble = time.perf_counter() - t0

    fixed = np.zeros(ndof, dtype=bool)
    u_all = np.zeros(ndof, dtype=np.float64)
    for z in range(nnode):
        for y in range(nnode):
            for x in range(nnode):
                on_boundary = (x == 0 or x == n or y == 0 or y == n or z == 0 or z == n)
                if on_boundary:
                    nid = node_id(z, y, x, nnode)
                    xd = x / n
                    fixed[3 * nid + 0] = True
                    fixed[3 * nid + 1] = True
                    fixed[3 * nid + 2] = True
                    u_all[3 * nid + 0] = eps0 * xd
                    u_all[3 * nid + 1] = 0.0
                    u_all[3 * nid + 2] = 0.0

    free = ~fixed
    Kff = K[free][:, free].tocsr()
    rhs = -K[free][:, fixed] @ u_all[fixed]

    diag = Kff.diagonal().copy()
    diag[np.abs(diag) < 1e-30] = 1.0
    M = spla.LinearOperator(Kff.shape, matvec=lambda x: x / diag)

    t1 = time.perf_counter()
    try:
        u_free, info = spla.cg(Kff, rhs, rtol=tol, atol=0.0, maxiter=maxiter, M=M)
    except TypeError:
        u_free, info = spla.cg(Kff, rhs, tol=tol, maxiter=maxiter, M=M)
    t_solve = time.perf_counter() - t1
    if info != 0 and verbose:
        print(f"[FEM] CG ended with info={info}. Positive means maxiter reached.")
    u_all[free] = u_free
    u_nodal = u_all.reshape(nnodes, 3).reshape(nnode, nnode, nnode, 3)

    # Element-centered fields.
    Bc = hex8_B_center(hx, hy, hz)
    D_solid = isotropic_D(Es, nu_s)
    D_void = isotropic_D(alpha_void * Es, 0.20)
    eps_arr = np.zeros((n, n, n, 6), dtype=np.float64)
    sig_arr = np.zeros((n, n, n, 6), dtype=np.float64)
    u_center = np.zeros((3, n, n, n), dtype=np.float64)

    for z in range(n):
        for y in range(n):
            for x in range(n):
                nodes = element_nodes(z, y, x, nnode)
                dofs = element_dofs(nodes)
                ue = u_all[dofs]
                eps_e = Bc @ ue
                q = chi[z, y, x]
                D_e = (1.0 - q) * D_solid + q * D_void
                sig_e = D_e @ eps_e
                eps_arr[z, y, x, :] = eps_e
                sig_arr[z, y, x, :] = sig_e
                node_vals = np.asarray([u_all[3 * nid:3 * nid + 3] for nid in nodes])
                u_center[:, z, y, x] = node_vals.mean(axis=0)

    vm = von_mises_np(sig_arr)
    Eeff = float(sig_arr[..., 0].mean() / eps0)
    Ksigma95 = float(np.quantile(vm.reshape(-1), 0.95) / (vm.mean() + 1e-12))
    solve_info = {
        "n": float(n),
        "num_elements": float(nelem),
        "num_nodes": float(nnodes),
        "num_dofs": float(ndof),
        "num_free_dofs": float(free.sum()),
        "cg_info": float(info),
        "assembly_time_s": float(t_assemble),
        "solve_time_s": float(t_solve),
        "total_time_s": float(time.perf_counter() - t0),
    }
    eps = {"xx": eps_arr[..., 0], "yy": eps_arr[..., 1], "zz": eps_arr[..., 2], "xy": eps_arr[..., 3], "xz": eps_arr[..., 4], "yz": eps_arr[..., 5]}
    sig = {"xx": sig_arr[..., 0], "yy": sig_arr[..., 1], "zz": sig_arr[..., 2], "xy": sig_arr[..., 3], "xz": sig_arr[..., 4], "yz": sig_arr[..., 5]}
    return FEMResult(u_center=u_center, u_nodal=u_nodal, eps=eps, sig=sig, von_mises=vm, Eeff=Eeff, Ksigma95=Ksigma95, solve_info=solve_info)
