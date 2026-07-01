import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)


Nx, Ny, Nz = 40, 40, 40
Lx, Ly, Lz = 100.0, 100.0, 100.0
nu, rho, alpha, Pp_val = 0.25, 2500.0, 0.8, 35e6
U0_val, V0_val, W0_val = 0.05, -0.05, 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def E_np(x, y, z): return 30e9 + 10e9 * np.sin(2 * np.pi * x / Lx) * np.cos(2 * np.pi * z / Lz)

def E_th(x, y, z): return 30e9 + 10e9 * torch.sin(2 * torch.pi * x / Lx) * torch.cos(2 * torch.pi * z / Lz)

def get_mms_exact_u_np(x, y, z):
    ux = -0.2 * (x / Lx) + U0_val * np.sin(np.pi * x / Lx) * np.cos(np.pi * y / Ly) * np.cos(np.pi * z / Lz)
    uy = -0.1 * (y / Ly) + V0_val * np.cos(np.pi * x / Lx) * np.sin(np.pi * y / Ly) * np.cos(np.pi * z / Lz)
    uz = -0.1 * (z / Lz) + W0_val * np.cos(np.pi * x / Lx) * np.cos(np.pi * y / Ly) * np.sin(np.pi * z / Lz)
    return ux, uy, uz

def U_base_np(x, y, z):
    return -0.2 * (x / Lx), -0.1 * (y / Ly), -0.1 * (z / Lz)

def U_base_th(x, y, z):
    return -0.2 * (x / Lx), -0.1 * (y / Ly), -0.1 * (z / Lz)


def precompute_mms_body_force(x, y, z):
    hx, hy, hz = torch.pi * x / Lx, torch.pi * y / Ly, torch.pi * z / Lz
    ux = -0.2 * (x / Lx) + U0_val * torch.sin(hx) * torch.cos(hy) * torch.cos(hz)
    uy = -0.1 * (y / Ly) + V0_val * torch.cos(hx) * torch.sin(hy) * torch.cos(hz)
    uz = -0.1 * (z / Lz) + W0_val * torch.cos(hx) * torch.cos(hy) * torch.sin(hz)

    exx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    eyy = torch.autograd.grad(uy.sum(), y, create_graph=True)[0]
    ezz = torch.autograd.grad(uz.sum(), z, create_graph=True)[0]
    exy = 0.5 * (torch.autograd.grad(ux.sum(), y, create_graph=True)[0] +
                 torch.autograd.grad(uy.sum(), x, create_graph=True)[0])
    eyz = 0.5 * (torch.autograd.grad(uy.sum(), z, create_graph=True)[0] +
                 torch.autograd.grad(uz.sum(), y, create_graph=True)[0])
    exz = 0.5 * (torch.autograd.grad(ux.sum(), z, create_graph=True)[0] +
                 torch.autograd.grad(uz.sum(), x, create_graph=True)[0])

    vol = exx + eyy + ezz
    E, lam = E_th(x, y, z), E_th(x, y, z) * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    sxx = lam * vol + 2 * mu * exx - alpha * Pp_val
    syy = lam * vol + 2 * mu * eyy - alpha * Pp_val
    szz = lam * vol + 2 * mu * ezz - alpha * Pp_val
    sxy, syz, sxz = 2 * mu * exy, 2 * mu * eyz, 2 * mu * exz

    bx = -(torch.autograd.grad(sxx.sum(), x, retain_graph=True)[0] +
           torch.autograd.grad(sxy.sum(), y, retain_graph=True)[0] +
           torch.autograd.grad(sxz.sum(), z, retain_graph=True)[0])
    by = -(torch.autograd.grad(sxy.sum(), x, retain_graph=True)[0] +
           torch.autograd.grad(syy.sum(), y, retain_graph=True)[0] +
           torch.autograd.grad(syz.sum(), z, retain_graph=True)[0])
    bz = -(torch.autograd.grad(sxz.sum(), x, retain_graph=True)[0] +
           torch.autograd.grad(syz.sum(), y, retain_graph=True)[0] + torch.autograd.grad(szz.sum(), z)[0])
    return bx, by, bz

def get_b_exact_batch(x_np, y_np, z_np):
    bx_all, by_all, bz_all = [], [], []
    chunk_size = 50000
    
    with torch.enable_grad():
        for i in range(0, len(x_np), chunk_size):
            x = torch.tensor(x_np[i:i + chunk_size], dtype=torch.float64, device=device, requires_grad=True)
            y = torch.tensor(y_np[i:i + chunk_size], dtype=torch.float64, device=device, requires_grad=True)
            z = torch.tensor(z_np[i:i + chunk_size], dtype=torch.float64, device=device, requires_grad=True)
            
            bx, by, bz = precompute_mms_body_force(x, y, z)
            
            bx_all.append(bx.detach().cpu().numpy())
            by_all.append(by.detach().cpu().numpy())
            bz_all.append(bz.detach().cpu().numpy())
            
    return np.concatenate(bx_all), np.concatenate(by_all), np.concatenate(bz_all)


def Run_FEM_Mega_MMS():
    print("=" * 90)

    start_time = time.time()

    x, y, z = np.linspace(0, Lx, Nx + 1), np.linspace(0, Ly, Ny + 1), np.linspace(0, Lz, Nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    Total_DOF = len(nodes) * 3

    i, j, k = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    n0 = i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
    elements = np.vstack([
        n0.ravel(), (n0 + (Ny + 1) * (Nz + 1)).ravel(), (n0 + (Ny + 1) * (Nz + 1) + (Nz + 1)).ravel(), (n0 + (Nz + 1)).ravel(),
        (n0 + 1).ravel(), (n0 + (Ny + 1) * (Nz + 1) + 1).ravel(), (n0 + (Ny + 1) * (Nz + 1) + (Nz + 1) + 1).ravel(), (n0 + (Nz + 1) + 1).ravel()
    ]).T
    N_elements = len(elements)

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    c_D = 1.0 / ((1 + nu) * (1 - 2 * nu))
    D_base = c_D * np.array([[1 - nu, nu, nu, 0, 0, 0], [nu, 1 - nu, nu, 0, 0, 0], [nu, nu, 1 - nu, 0, 0, 0],
                             [0, 0, 0, 0.5 - nu, 0, 0], [0, 0, 0, 0, 0.5 - nu, 0], [0, 0, 0, 0, 0, 0.5 - nu]])
    coords_ref = np.array([[0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0], [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]])
    xi_i, eta_i, zeta_i = np.array([-1, 1, 1, -1, -1, 1, 1, -1]), np.array([-1, -1, 1, 1, -1, -1, 1, 1]), np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    gp = 1.0 / np.sqrt(3.0)
    gp_list = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp), (-gp, -gp, gp), (gp, -gp, gp), (gp, gp, gp), (-gp, gp, gp)]

    K_elem_all = np.zeros((N_elements, 24, 24))
    Fe_all = np.zeros((N_elements, 24))
    elem_nodes = nodes[elements]


    for gp_idx in range(8):
        xi, eta, zeta = gp_list[gp_idx]
        dN = 0.125 * np.vstack([xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta), eta_i * (1 + xi_i * xi) * (1 + zeta_i * zeta), zeta_i * (1 + xi_i * xi) * (1 + eta_i * eta)])
        J = dN @ coords_ref; detJ = np.linalg.det(J); invJ = np.linalg.inv(J); dN_dxyz = invJ @ dN
        B = np.zeros((6, 24))
        for _i in range(8):
            B[0, _i*3], B[1, _i*3+1], B[2, _i*3+2] = dN_dxyz[0, _i], dN_dxyz[1, _i], dN_dxyz[2, _i]
            B[3, _i*3], B[3, _i*3+1] = dN_dxyz[1, _i], dN_dxyz[0, _i]
            B[4, _i*3+1], B[4, _i*3+2] = dN_dxyz[2, _i], dN_dxyz[1, _i]
            B[5, _i*3], B[5, _i*3+2] = dN_dxyz[2, _i], dN_dxyz[0, _i]

        N_shape = 0.125 * (1 + xi_i * xi) * (1 + eta_i * eta) * (1 + zeta_i * zeta)
        N_mat = np.zeros((3, 24))
        for _i in range(8): N_mat[0, _i*3], N_mat[1, _i*3+1], N_mat[2, _i*3+2] = N_shape[_i], N_shape[_i], N_shape[_i]

        xyz_gp = np.sum(elem_nodes * N_shape[None, :, None], axis=1)
        E_gp = E_np(xyz_gp[:, 0], xyz_gp[:, 1], xyz_gp[:, 2])
        bx_gp, by_gp, bz_gp = get_b_exact_batch(xyz_gp[:, 0], xyz_gp[:, 1], xyz_gp[:, 2])

        K_base_gp = B.T @ D_base @ B * detJ
        K_elem_all += K_base_gp[None, :, :] * E_gp[:, None, None]

        b_vec = np.stack([bx_gp, by_gp, bz_gp], axis=1)
        Fe_all += (b_vec @ N_mat) * detJ
        Fe_all += B.T @ (alpha * Pp_val * np.array([1, 1, 1, 0, 0, 0])) * detJ

    dof_map = np.zeros((N_elements, 24), dtype=np.int32)
    for i in range(8): dof_map[:, i * 3:i * 3 + 3] = elements[:, i:i + 1] * 3 + np.array([0, 1, 2])

    I, J, V = np.repeat(dof_map[:, :, None], 24, axis=2).ravel(), np.repeat(dof_map[:, None, :], 24, axis=1).ravel(), K_elem_all.ravel()
    K_sparse = sp.coo_matrix((V, (I, J)), shape=(Total_DOF, Total_DOF)).tocsr()
    F_global = np.bincount(dof_map.ravel(), weights=Fe_all.ravel(), minlength=Total_DOF)
    del I, J, V, K_elem_all, dof_map; gc.collect()

    tol = 1e-5; bc_dict = {}
    for n in np.where(nodes[:, 0] < tol)[0]: bc_dict[n*3+0] = 0.0
    for n in np.where(nodes[:, 0] > Lx - tol)[0]: bc_dict[n*3+0] = -0.2
    for n in np.where(nodes[:, 1] < tol)[0]: bc_dict[n*3+1] = 0.0
    for n in np.where(nodes[:, 1] > Ly - tol)[0]: bc_dict[n*3+1] = -0.1
    for n in np.where(nodes[:, 2] < tol)[0]: bc_dict[n*3+2] = 0.0
    for n in np.where(nodes[:, 2] > Lz - tol)[0]: bc_dict[n*3+2] = -0.1

    bc_dofs = np.array(list(bc_dict.keys()), dtype=np.int32)
    bc_vals = np.array(list(bc_dict.values()), dtype=np.float64)
    is_free = np.ones(Total_DOF, dtype=bool); is_free[bc_dofs] = False
    free_dofs = np.where(is_free)[0]

    U_bc_full = np.zeros(Total_DOF); U_bc_full[bc_dofs] = bc_vals
    F_eff = F_global[free_dofs] - K_sparse[free_dofs, :].dot(U_bc_full)
    K_free = K_sparse[free_dofs, :][:, free_dofs]
    M_free = sp.diags(1.0 / K_free.diagonal())


    U_guess = np.zeros(Total_DOF)
    U_guess[0::3], U_guess[1::3], U_guess[2::3] = -0.2 * (nodes[:, 0] / Lx), -0.1 * (nodes[:, 1] / Ly), -0.1 * (nodes[:, 2] / Lz)
    U_free, _ = spla.cg(K_free, F_eff, M=M_free, tol=1e-8, maxiter=8000, x0=U_guess[free_dofs])
    U_global = np.zeros(Total_DOF)
    U_global[bc_dofs] = bc_vals
    U_global[free_dofs] = U_free
    
    Pi_fem = 0.5 * np.dot(U_global, K_sparse.dot(U_global)) - np.dot(U_global, F_global)
    print(f"     {time.time() - start_time:.2f} s\n")
    return nodes, elements, U_global, Pi_fem


class SmoothPE(nn.Module):

    def forward(self, x):
        return torch.cat([
            x, 
            torch.sin(torch.pi * x), 
            torch.cos(torch.pi * x)
        ], dim=-1)

class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gamma = nn.Sequential(nn.Linear(cond_dim, out_dim), nn.Tanh())
        self.beta = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        h = self.norm(self.linear(x))
        return F.silu((1.0 + 0.25 * self.gamma(cond)) * h + self.beta(cond))

class OperatorBAVENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = SmoothPE()
        in_dim = 9
        cond_dim = 1

        self.block1 = FiLMBlock(in_dim, 128, cond_dim)
        self.block2 = FiLMBlock(128, 128, cond_dim)
        self.block3 = FiLMBlock(128, 128, cond_dim)

        self.head = nn.Linear(128, 3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x_norm):

        xn, yn, zn = x_norm[:, 0:1], x_norm[:, 1:2], x_norm[:, 2:3]
        x_phys, y_phys, z_phys = xn * Lx, yn * Ly, zn * Lz


        E_true_GPa = E_th(x_phys, y_phys, z_phys) / 1e9
        E_cond = E_true_GPa / 45.0


        x_in = self.pe(x_norm)
        h = self.block1(x_in, E_cond)
        h = self.block2(h, E_cond)
        h = self.block3(h, E_cond)
        corr = self.head(h)


        u_base, v_base, w_base = U_base_th(x_phys, y_phys, z_phys)

        u = u_base + 4.0 * xn * (1.0 - xn) * corr[:, 0:1]
        v = v_base + 4.0 * yn * (1.0 - yn) * corr[:, 1:2]
        w = w_base + 4.0 * zn * (1.0 - zn) * corr[:, 2:3]
        return torch.cat([u, v, w], dim=1)

def Train_BAVENet():

    start_time = time.time()
    model = OperatorBAVENet().to(device)
    E_scale = 1e9

    N_pts = 65536
    chunk_size = 16384


    soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=42)
    pts_norm_pool = soboleng.draw(N_pts).to(torch.float64).to(device)

    with torch.no_grad():
        pts_phys_pool = pts_norm_pool * torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)
        E_pool = E_th(pts_phys_pool[:, 0:1], pts_phys_pool[:, 1:2], pts_phys_pool[:, 2:3])
        lam_pool = (E_pool * nu) / ((1 + nu) * (1 - 2 * nu)) / E_scale
        mu_pool = E_pool / (2 * (1 + nu)) / E_scale
        
        pts_phys_np = pts_phys_pool.cpu().numpy()
        bx_np, by_np, bz_np = get_b_exact_batch(pts_phys_np[:, 0], pts_phys_np[:, 1], pts_phys_np[:, 2])
        bx_pool = torch.tensor(bx_np / E_scale, device=device).unsqueeze(1)
        by_pool = torch.tensor(by_np / E_scale, device=device).unsqueeze(1)
        bz_pool = torch.tensor(bz_np / E_scale, device=device).unsqueeze(1)

    def evaluate_exact_full_batch_gradient(optimizer=None, return_raw_energy=False):
        if optimizer is not None: optimizer.zero_grad()
        total_scaled_loss = 0.0
        total_raw_energy = 0.0

        for i in range(0, N_pts, chunk_size):
            end_idx = min(i + chunk_size, N_pts)
            pts_chunk = pts_norm_pool[i:end_idx].detach().clone().requires_grad_(True)

            U_pred = model(pts_chunk)
            u, v, w = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]

            du = torch.autograd.grad(u.sum(), pts_chunk, create_graph=True)[0]
            dv = torch.autograd.grad(v.sum(), pts_chunk, create_graph=True)[0]
            dw = torch.autograd.grad(w.sum(), pts_chunk, create_graph=True)[0]

            exx, eyy, ezz = du[:, 0:1] / Lx, dv[:, 1:2] / Ly, dw[:, 2:3] / Lz
            exy = 0.5 * (du[:, 1:2] / Ly + dv[:, 0:1] / Lx)
            eyz = 0.5 * (dv[:, 2:3] / Lz + dw[:, 1:2] / Ly)
            ezx = 0.5 * (dw[:, 0:1] / Lx + du[:, 2:3] / Lz)
            vol_strain = exx + eyy + ezz

            lam_chunk, mu_chunk = lam_pool[i:end_idx], mu_pool[i:end_idx]
            bx_chunk, by_chunk, bz_chunk = bx_pool[i:end_idx], by_pool[i:end_idx], bz_pool[i:end_idx]

            W_strain = 0.5 * lam_chunk * vol_strain ** 2 + mu_chunk * (
                        exx ** 2 + eyy ** 2 + ezz ** 2 + 2 * exy ** 2 + 2 * eyz ** 2 + 2 * ezx ** 2)
            W_pore = - (alpha * Pp_val / E_scale) * vol_strain
            W_body = - (bx_chunk * u + by_chunk * v + bz_chunk * w)
            
            raw_energy = torch.mean(W_strain + W_pore + W_body)
            loss_chunk = raw_energy * 100000.0

            weight = (end_idx - i) / float(N_pts)
            weighted_loss = loss_chunk * weight

            if optimizer is not None: weighted_loss.backward()
            total_scaled_loss += weighted_loss.item()
            total_raw_energy += (raw_energy.item() * weight * E_scale * (Lx*Ly*Lz))

        return total_scaled_loss if not return_raw_energy else total_raw_energy


    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=1200, eta_min=1e-5)

    for epoch in range(1200):
        loss_val = evaluate_exact_full_batch_gradient(optimizer_adam)
        optimizer_adam.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0: print(f"       Adam Epoch {epoch + 1:>4} | Scaled Energy: {loss_val:.6f}")


    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=200, tolerance_grad=1e-5, tolerance_change=1e-7, line_search_fn='strong_wolfe')
    lbfgs_iter = 0
    def closure():
        nonlocal lbfgs_iter
        loss_val = evaluate_exact_full_batch_gradient(optimizer_lbfgs)
        lbfgs_iter += 1
        if lbfgs_iter % 50 == 0: print(f"       L-BFGS Iter {lbfgs_iter:>4} | Scaled Energy: {loss_val:.6f}")
        return torch.tensor(loss_val, device=device)

    optimizer_lbfgs.step(closure)
    
    final_pi_pinn = evaluate_exact_full_batch_gradient(optimizer=None, return_raw_energy=True)
    print(f"    : {time.time() - start_time:.2f} s\n")
    return model, final_pi_pinn


def get_exact_strain_stress_autograd(pts_phys_np):
    pts = torch.tensor(pts_phys_np, dtype=torch.float64, device=device, requires_grad=True)
    x, y, z = pts[:,0:1], pts[:,1:2], pts[:,2:3]
    hx, hy, hz = torch.pi * x / Lx, torch.pi * y / Ly, torch.pi * z / Lz
    
    ux = -0.2 * (x / Lx) + U0_val * torch.sin(hx) * torch.cos(hy) * torch.cos(hz)
    uy = -0.1 * (y / Ly) + V0_val * torch.cos(hx) * torch.sin(hy) * torch.cos(hz)
    uz = -0.1 * (z / Lz) + W0_val * torch.cos(hx) * torch.cos(hy) * torch.sin(hz)
    
    du = torch.autograd.grad(ux.sum(), pts, retain_graph=True, create_graph=False)[0]
    dv = torch.autograd.grad(uy.sum(), pts, retain_graph=True, create_graph=False)[0]
    dw = torch.autograd.grad(uz.sum(), pts, retain_graph=False, create_graph=False)[0]
    
    exx, eyy, ezz = du[:, 0], dv[:, 1], dw[:, 2]
    exy = 0.5 * (du[:, 1] + dv[:, 0])
    eyz = 0.5 * (dv[:, 2] + dw[:, 1])
    ezx = 0.5 * (dw[:, 0] + du[:, 2])
    
    vol = exx + eyy + ezz
    E = E_th(x, y, z).squeeze(-1)
    lam = E * nu / ((1+nu)*(1-2*nu))
    mu = E / (2*(1+nu))
    
    sxx = lam * vol + 2*mu*exx - alpha * Pp_val
    syy = lam * vol + 2*mu*eyy - alpha * Pp_val
    szz = lam * vol + 2*mu*ezz - alpha * Pp_val
    sxy, syz, sxz = 2*mu*exy, 2*mu*eyz, 2*mu*ezx
    
    strain = torch.stack([exx, eyy, ezz, exy, eyz, ezx], dim=1)
    stress = torch.stack([sxx, syy, szz, sxy, syz, sxz], dim=1)
    
    return strain.detach().cpu().numpy(), stress.detach().cpu().numpy()

def Evaluate_And_Print_Tables(nodes, U_exact, pinn_model, Pi_fem, Pi_pinn):

    U_pinn = np.zeros(len(nodes) * 3)
    batch_size = 32000
    with torch.no_grad():
        for i in range(0, len(nodes), batch_size):
            end = min(i + batch_size, len(nodes))
            batch_nodes = torch.tensor(nodes[i:end], dtype=torch.float64, device=device)
            scale = torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)
            U_pinn[i * 3:end * 3] = pinn_model(batch_nodes / scale).cpu().numpy().flatten()

    tol = 1e-5
    idx_x_bnd = np.where((nodes[:, 0] < tol) | (nodes[:, 0] > Lx - tol))[0]
    idx_y_bnd = np.where((nodes[:, 1] < tol) | (nodes[:, 1] > Ly - tol))[0]
    idx_z_bnd = np.where((nodes[:, 2] < tol) | (nodes[:, 2] > Lz - tol))[0]

    def get_bnd_err(idx, axis):
        pred = U_pinn[idx * 3 + axis]
        true = U_base_np(nodes[idx, 0], nodes[idx, 1], nodes[idx, 2])[axis]
        norm_true = np.linalg.norm(true)
        err = np.linalg.norm(pred - true)
        return err / (norm_true + 1e-12) if norm_true > 0 else err, np.max(np.abs(pred - true))

    err_gx, max_gx = get_bnd_err(idx_x_bnd, 0)
    err_gy, max_gy = get_bnd_err(idx_y_bnd, 1)
    err_gz, max_gz = get_bnd_err(idx_z_bnd, 2)
    max_g_all = max([max_gx, max_gy, max_gz])

    print("-" * 60)

    print("-" * 60)
    print(f" : {err_gx:.2e}")
    print(f" : {err_gy:.2e}")
    print(f" : {err_gz:.2e}")
    print(f" : {max_g_all:.2e} m")
    
    u_l2_err = np.linalg.norm(U_exact - U_pinn) / (np.linalg.norm(U_exact) + 1e-12)
    
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    xc = np.linspace(dx/2, Lx - dx/2, Nx)
    yc = np.linspace(dy/2, Ly - dy/2, Ny)
    zc = np.linspace(dz/2, Lz - dz/2, Nz)
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
    pts_c = np.vstack([Xc.ravel(), Yc.ravel(), Zc.ravel()]).T

    ex_strain_c, ex_stress_c = get_exact_strain_stress_autograd(pts_c)

    pinn_strain_c = np.zeros_like(ex_strain_c)
    pinn_stress_c = np.zeros_like(ex_stress_c)
    with torch.enable_grad():
        for i in range(0, len(pts_c), batch_size):
            end = min(i + batch_size, len(pts_c))
            pts_chunk = torch.tensor(pts_c[i:end], dtype=torch.float64, device=device)
            scale = torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)
            pts_norm = (pts_chunk / scale).requires_grad_(True)
            
            U_pred = pinn_model(pts_norm)
            u_p, v_p, w_p = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]
            
            du = torch.autograd.grad(u_p.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
            dv = torch.autograd.grad(v_p.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
            dw = torch.autograd.grad(w_p.sum(), pts_norm, retain_graph=False, create_graph=False)[0]
            
            exx, eyy, ezz = du[:, 0]/Lx, dv[:, 1]/Ly, dw[:, 2]/Lz
            exy = 0.5 * (du[:, 1]/Ly + dv[:, 0]/Lx)
            eyz = 0.5 * (dv[:, 2]/Lz + dw[:, 1]/Ly)
            ezx = 0.5 * (dw[:, 0]/Lx + du[:, 2]/Lz)
            
            vol = exx + eyy + ezz
            E_t = E_th(pts_chunk[:,0:1], pts_chunk[:,1:2], pts_chunk[:,2:3]).squeeze(-1)
            lam_t = E_t * nu / ((1+nu)*(1-2*nu))
            mu_t = E_t / (2*(1+nu))
            
            sxx = lam_t * vol + 2*mu_t*exx - alpha * Pp_val
            syy = lam_t * vol + 2*mu_t*eyy - alpha * Pp_val
            szz = lam_t * vol + 2*mu_t*ezz - alpha * Pp_val
            sxy, syz, sxz = 2*mu_t*exy, 2*mu_t*eyz, 2*mu_t*ezx
            
            pinn_strain_c[i:end] = torch.stack([exx, eyy, ezz, exy, eyz, ezx], dim=1).detach().cpu().numpy()
            pinn_stress_c[i:end] = torch.stack([sxx, syy, szz, sxy, syz, sxz], dim=1).detach().cpu().numpy()

    def frob_norm(tensor_array):
        return np.sqrt(tensor_array[:,0]**2 + tensor_array[:,1]**2 + tensor_array[:,2]**2 + 
                       2*tensor_array[:,3]**2 + 2*tensor_array[:,4]**2 + 2*tensor_array[:,5]**2)

    eps_l2_err = np.linalg.norm(frob_norm(pinn_strain_c - ex_strain_c)) / (np.linalg.norm(frob_norm(ex_strain_c)) + 1e-12)
    sig_l2_err = np.linalg.norm(frob_norm(pinn_stress_c - ex_stress_c)) / (np.linalg.norm(frob_norm(ex_stress_c)) + 1e-12)   
    energy_err = abs(Pi_pinn - Pi_fem) / abs(Pi_fem)


    return U_pinn

def Export_And_Visualize_1D_Curves(nodes, U_fem, pinn_model, device):

    N_hr = 250
    t_hr = np.linspace(0, Lx, N_hr)
    val_25 = 25.0
    idx_25 = int(round(val_25 / (Lx / Nx)))

    # --- FEM 数据提取 ---
    U_x_3d = U_fem[0::3].reshape(Nx + 1, Ny + 1, Nz + 1)
    U_y_3d = U_fem[1::3].reshape(Nx + 1, Ny + 1, Nz + 1)
    U_z_3d = U_fem[2::3].reshape(Nx + 1, Ny + 1, Nz + 1)

    exx_fem_3d = np.gradient(U_x_3d, Lx / Nx, axis=0)
    eyy_fem_3d = np.gradient(U_y_3d, Ly / Ny, axis=1)
    ezz_fem_3d = np.gradient(U_z_3d, Lz / Nz, axis=2)
    vol_fem_3d = exx_fem_3d + eyy_fem_3d + ezz_fem_3d

    x_fem = np.linspace(0, Lx, Nx + 1)
    y_fem = np.linspace(0, Ly, Ny + 1)
    z_fem = np.linspace(0, Lz, Nz + 1)
    X_g, Y_g, Z_g = np.meshgrid(x_fem, y_fem, z_fem, indexing='ij')

    E_g = E_np(X_g, Y_g, Z_g)
    lam_g = (E_g * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_g = E_g / (2 * (1 + nu))
    sxx_fem_3d = lam_g * vol_fem_3d + 2 * mu_g * exx_fem_3d - alpha * Pp_val

    Ux_fem_line = U_x_3d[:, idx_25, idx_25] * 1000.0
    Uy_fem_line = U_y_3d[idx_25, :, idx_25] * 1000.0
    Uz_fem_line = U_z_3d[idx_25, idx_25, :] * 1000.0
    sxx_fem_line = sxx_fem_3d[:, idx_25, idx_25] / 1e6

    # --- DEM 与 Exact 提取 ---
    def get_hr_data(axis_idx):
        if axis_idx == 0:
            x_arr, y_arr, z_arr = t_hr, np.full_like(t_hr, val_25), np.full_like(t_hr, val_25)
        elif axis_idx == 1:
            x_arr, y_arr, z_arr = np.full_like(t_hr, val_25), t_hr, np.full_like(t_hr, val_25)
        else:
            x_arr, y_arr, z_arr = np.full_like(t_hr, val_25), np.full_like(t_hr, val_25), t_hr

        pts_phys = np.vstack([x_arr, y_arr, z_arr]).T
        pts_norm = (torch.tensor(pts_phys, dtype=torch.float64, device=device) / torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)).requires_grad_(True)
        U_pred = pinn_model(pts_norm)
        u, v, w = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]

        du = torch.autograd.grad(u.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
        dv = torch.autograd.grad(v.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
        dw = torch.autograd.grad(w.sum(), pts_norm, retain_graph=False, create_graph=False)[0]

        exx_dem = (du[:, 0] / Lx).cpu().numpy().flatten()
        eyy_dem = (dv[:, 1] / Ly).cpu().numpy().flatten()
        ezz_dem = (dw[:, 2] / Lz).cpu().numpy().flatten()
        vol_dem = exx_dem + eyy_dem + ezz_dem

        E_val = E_np(x_arr, y_arr, z_arr)
        lam_val = (E_val * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_val = E_val / (2 * (1 + nu))

        sxx_dem = (lam_val * vol_dem + 2 * mu_val * exx_dem - alpha * Pp_val) / 1e6
        u_dem, v_dem, w_dem = u.detach().cpu().numpy().flatten() * 1000, v.detach().cpu().numpy().flatten() * 1000, w.detach().cpu().numpy().flatten() * 1000

        ux_ex, uy_ex, uz_ex = get_mms_exact_u_np(x_arr, y_arr, z_arr)
        ux_ex *= 1000; uy_ex *= 1000; uz_ex *= 1000

        exx_ex = -0.2 / Lx + U0_val * (np.pi / Lx) * np.cos(np.pi * x_arr / Lx) * np.cos(np.pi * y_arr / Ly) * np.cos(np.pi * z_arr / Lz)
        eyy_ex = -0.1 / Ly + V0_val * (np.pi / Ly) * np.cos(np.pi * x_arr / Lx) * np.cos(np.pi * y_arr / Ly) * np.cos(np.pi * z_arr / Lz)
        ezz_ex = -0.1 / Lz + W0_val * (np.pi / Lz) * np.cos(np.pi * x_arr / Lx) * np.cos(np.pi * y_arr / Ly) * np.cos(np.pi * z_arr / Lz)

        vol_ex = exx_ex + eyy_ex + ezz_ex
        sxx_ex = (lam_val * vol_ex + 2 * mu_val * exx_ex - alpha * Pp_val) / 1e6

        return u_dem, v_dem, w_dem, sxx_dem, ux_ex, uy_ex, uz_ex, sxx_ex

    u_dem_x, _, _, sxx_dem_x, u_ex_x, _, _, sxx_ex_x = get_hr_data(0)
    _, v_dem_y, _, _, _, v_ex_y, _, _ = get_hr_data(1)
    _, _, w_dem_z, _, _, _, w_ex_z, _ = get_hr_data(2)


    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_single(ax, x_hr, exact, dem, x_fem, fem, title_str, xlabel_str, ylabel_str):
        ax.plot(x_hr, exact, '-k', linewidth=3, label='Exact Analytical')
        ax.plot(x_hr, dem, '--', color='#2980B9', linewidth=2.5, label='AENO')
        ax.plot(x_fem[::2], fem[::2], 'or', markerfacecolor='none', markersize=8, markeredgewidth=1.8, label='FEM Reference')
        ax.set_xlabel(xlabel_str, fontsize=18)
        ax.set_ylabel(ylabel_str, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=18, loc='best', frameon=True, edgecolor='gray')
        

    plot_single(axes[0, 0], t_hr, u_ex_x, u_dem_x, x_fem, Ux_fem_line, '', r'$X$-coordinate (m)', r'Displacement $U_x$ (mm)')
    plot_single(axes[0, 1], t_hr, v_ex_y, v_dem_y, x_fem, Uy_fem_line, '', r'$Y$-coordinate (m)', r'Displacement $U_y$ (mm)')
    plot_single(axes[1, 0], t_hr, w_ex_z, w_dem_z, x_fem, Uz_fem_line, '', r'$Z$-coordinate (m)', r'Displacement $U_z$ (mm)')
    plot_single(axes[1, 1], t_hr, sxx_ex_x, sxx_dem_x, x_fem, sxx_fem_line, '', r'$X$-coordinate (m)', r'Normal Stress $\sigma_{xx}$ (MPa)')

    plt.tight_layout()
    plt.savefig('Fig_MMS_1D_Profiles_Clean.png', dpi=600, bbox_inches='tight')



# =============================================================================
# [核心出图模块 2] 2D 九宫格全空间等高线矩阵 (纯净学术版)
# =============================================================================
def Export_And_Visualize_Verification_2D(nodes, U_exact, U_fem, pinn_model, U_pinn, err_fem, err_dem):


    z_eval = 25.0
    slice_idx = np.where(np.abs(nodes[:, 2] - z_eval) < 1e-4)[0]
    nodes_slice = nodes[slice_idx]

    X_grid = nodes_slice[:, 0].reshape(Nx + 1, Ny + 1)
    Y_grid = nodes_slice[:, 1].reshape(Nx + 1, Ny + 1)

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    z_mid = int(round(z_eval / dz))
    E_grid = E_np(X_grid, Y_grid, np.full_like(X_grid, z_eval))
    lam_grid = (E_grid * nu) / ((1 + nu) * (1 - 2 * nu))
    mu_grid = E_grid / (2 * (1 + nu))

    # [1] 原生解析物理真值
    U_x_exact_slice = U_exact[slice_idx * 3 + 0].reshape(Nx + 1, Ny + 1) * 1000.0
    exx_exact_slice = -0.2 / Lx + U0_val * (np.pi / Lx) * np.cos(np.pi * X_grid / Lx) * np.cos(np.pi * Y_grid / Ly) * np.cos(np.pi * z_eval / Lz)
    eyy_exact_slice = -0.1 / Ly + V0_val * (np.pi / Ly) * np.cos(np.pi * X_grid / Lx) * np.cos(np.pi * Y_grid / Ly) * np.cos(np.pi * z_eval / Lz)
    ezz_exact_slice = -0.1 / Lz + W0_val * (np.pi / Lz) * np.cos(np.pi * X_grid / Lx) * np.cos(np.pi * Y_grid / Ly) * np.cos(np.pi * z_eval / Lz)
    vol_strain_exact_slice = exx_exact_slice + eyy_exact_slice + ezz_exact_slice
    sxx_exact_slice = (lam_grid * vol_strain_exact_slice + 2 * mu_grid * exx_exact_slice - alpha * Pp_val) / 1e6

    # [2] DEM 场萃取
    U_x_pinn_slice = U_pinn[slice_idx * 3 + 0].reshape(Nx + 1, Ny + 1) * 1000.0
    pts_phys = np.vstack([X_grid.ravel(), Y_grid.ravel(), np.full_like(X_grid.ravel(), z_eval)]).T
    scale_tensor = torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)
    pts_norm = (torch.tensor(pts_phys, dtype=torch.float64, device=device) / scale_tensor).requires_grad_(True)

    U_pred = pinn_model(pts_norm)
    u_p, v_p, w_p = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]
    du = torch.autograd.grad(u_p.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
    dv = torch.autograd.grad(v_p.sum(), pts_norm, retain_graph=True, create_graph=False)[0]
    dw = torch.autograd.grad(w_p.sum(), pts_norm, retain_graph=False, create_graph=False)[0]

    exx_pinn = (du[:, 0] / Lx).cpu().numpy().reshape(Nx + 1, Ny + 1)
    eyy_pinn = (dv[:, 1] / Ly).cpu().numpy().reshape(Nx + 1, Ny + 1)
    ezz_pinn = (dw[:, 2] / Lz).cpu().numpy().reshape(Nx + 1, Ny + 1)
    vol_strain_pinn = exx_pinn + eyy_pinn + ezz_pinn
    sxx_pinn_slice = (lam_grid * vol_strain_pinn + 2 * mu_grid * exx_pinn - alpha * Pp_val) / 1e6

    import matplotlib.ticker as ticker
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    def plot_row(row_idx, exact_data, dem_data, cmap_name):
        err_data = np.abs(exact_data - dem_data)
        vmin = min(exact_data.min(), dem_data.min())
        vmax = max(exact_data.max(), dem_data.max())
        if np.isclose(vmin, vmax): vmin, vmax = vmin - 1e-5, vmax + 1e-5
        
        levels = np.linspace(vmin, vmax, 41)
        ticks = np.linspace(vmin, vmax, 5)

        # Exact and DEM
        for j, data in enumerate([exact_data, dem_data]):
            ax = axes[row_idx, j]
            ax.set_aspect('equal')
            c = ax.contourf(X_grid, Y_grid, data, levels=levels, cmap=cmap_name, extend='both')
            cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
            cb.ax.tick_params(labelsize=16)

        # Absolute Error
        ax_err = axes[row_idx, 2]
        ax_err.set_aspect('equal')
        err_max = np.percentile(err_data, 99.5) if np.percentile(err_data, 99.5) > 1e-8 else 1e-8
        levels_err = np.linspace(0, err_max, 41)
        ticks_err = np.linspace(0, err_max, 5)
        
        c_err = ax_err.contourf(X_grid, Y_grid, err_data, levels=levels_err, cmap='Reds', extend='max')
        cb_err = fig.colorbar(c_err, ax=ax_err, fraction=0.046, pad=0.04, ticks=ticks_err)
        cb_err.ax.tick_params(labelsize=16)
        
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 2))
        cb_err.ax.yaxis.set_major_formatter(formatter)

        # Axis ticks formatting
        for j in range(3):
            axes[row_idx, j].tick_params(axis='both', which='major', labelsize=16)
            if row_idx != 2:
                axes[row_idx, j].set_xticklabels([])
            if j != 0:
                axes[row_idx, j].set_yticklabels([])

    plot_row(0, U_x_exact_slice, U_x_pinn_slice, "jet")
    plot_row(1, exx_exact_slice * 1e6, exx_pinn * 1e6, "viridis")
    plot_row(2, sxx_exact_slice, sxx_pinn_slice, "coolwarm")

    plt.subplots_adjust(wspace=0.35, hspace=0.15)
    plt.savefig('Fig_MMS_2D_Contours_Clean.png', dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    nodes, elements, U_fem, Pi_fem = Run_FEM_Mega_MMS()

    ux_ex, uy_ex, uz_ex = get_mms_exact_u_np(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    U_exact = np.zeros(len(nodes) * 3)
    U_exact[0::3], U_exact[1::3], U_exact[2::3] = ux_ex, uy_ex, uz_ex

    pinn_model, Pi_pinn = Train_BAVENet()

    U_pinn = Evaluate_And_Print_Tables(nodes, U_exact, pinn_model, Pi_fem, Pi_pinn)
    

    Export_And_Visualize_1D_Curves(nodes, U_fem, pinn_model, device)
    Export_And_Visualize_Verification_2D(nodes, U_exact, U_fem, pinn_model, U_pinn, 0.0, 0.0)