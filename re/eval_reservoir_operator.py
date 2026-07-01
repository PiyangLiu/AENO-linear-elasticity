import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from scipy.interpolate import RegularGridInterpolator

from train_reservoir_operator import ReservoirOperator, Lx, Ly, Lz, nu_const, rho_const, gravity, E_SCALE

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


TEST_C_SCALE = 18.0    
TEST_C_SHIFT = 10.0    
TEST_DU_X = -1.4       
TEST_DV_Y = -0.2       

Nx, Ny, Nz = 40, 40, 10

def Run_FEM_Benchmark(V_seis_grid_np):

    start_time = time.time()
    
    x, y, z = np.linspace(0, Lx, Nx+1), np.linspace(0, Ly, Ny+1), np.linspace(0, Lz, Nz+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    Total_DOF = len(nodes) * 3

    i, j, k = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    n0 = i * (Ny+1) * (Nz+1) + j * (Nz+1) + k
    elements = np.vstack([n0.ravel(), (n0+(Ny+1)*(Nz+1)).ravel(), (n0+(Ny+1)*(Nz+1)+(Nz+1)).ravel(), (n0+(Nz+1)).ravel(),
                          (n0+1).ravel(), (n0+(Ny+1)*(Nz+1)+1).ravel(), (n0+(Ny+1)*(Nz+1)+(Nz+1)+1).ravel(), (n0+(Nz+1)+1).ravel()]).T

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    c = 1.0 / ((1 + nu_const) * (1 - 2 * nu_const))
    D_base = c * np.array([[1-nu_const, nu_const, nu_const, 0, 0, 0], [nu_const, 1-nu_const, nu_const, 0, 0, 0], [nu_const, nu_const, 1-nu_const, 0, 0, 0],
                           [0, 0, 0, 0.5-nu_const, 0, 0], [0, 0, 0, 0, 0.5-nu_const, 0], [0, 0, 0, 0, 0, 0.5-nu_const]])

    K_all, F_elem_all = np.zeros((len(elements), 24, 24)), np.zeros((len(elements), 24))
    elem_nodes = nodes[elements]
    coords_ref = np.array([[0,0,0], [dx,0,0], [dx,dy,0], [0,dy,0], [0,0,dz], [dx,0,dz], [dx,dy,dz], [0,dy,dz]])
    xi_i, eta_i, zeta_i = np.array([-1,1,1,-1,-1,1,1,-1]), np.array([-1,-1,1,1,-1,-1,1,1]), np.array([-1,-1,-1,-1,1,1,1,1])
    gp = 1.0 / np.sqrt(3.0)

    lin_ax = np.linspace(0, 1, 32)
    interp = RegularGridInterpolator((lin_ax, lin_ax, lin_ax), V_seis_grid_np[0, 0], bounds_error=False, fill_value=None)

    for xi, eta, zeta in [(-gp,-gp,-gp), (gp,-gp,-gp), (gp,gp,-gp), (-gp,gp,-gp), (-gp,-gp,gp), (gp,-gp,gp), (gp,gp,gp), (-gp,gp,gp)]:
        N_shape = 0.125 * (1 + xi_i * xi) * (1 + eta_i * eta) * (1 + zeta_i * zeta)
        gp_coords = np.sum(elem_nodes * N_shape[None, :, None], axis=1)

        pts_norm_np = gp_coords / np.array([Lx, Ly, Lz])
        E_true_gp = (TEST_C_SCALE * interp(pts_norm_np[:, [2, 1, 0]]) + TEST_C_SHIFT) * 1e9

        dN = 0.125 * np.vstack([xi_i*(1+eta_i*eta)*(1+zeta_i*zeta), eta_i*(1+xi_i*xi)*(1+zeta_i*zeta), zeta_i*(1+xi_i*xi)*(1+eta_i*eta)])
        dN_dxyz = np.linalg.inv(dN @ coords_ref) @ dN

        B = np.zeros((6, 24))
        for _i in range(8):
            B[0, _i*3], B[1, _i*3+1], B[2, _i*3+2] = dN_dxyz[0, _i], dN_dxyz[1, _i], dN_dxyz[2, _i]
            B[3, _i*3], B[3, _i*3+1] = dN_dxyz[1, _i], dN_dxyz[0, _i]; B[4, _i*3+1], B[4, _i*3+2] = dN_dxyz[2, _i], dN_dxyz[1, _i]
            B[5, _i*3], B[5, _i*3+2] = dN_dxyz[2, _i], dN_dxyz[0, _i]

        detJ = np.linalg.det(dN @ coords_ref)
        K_all += np.einsum('ik,nkj->nij', B.T, np.einsum('nkl,lj->nkj', D_base[None,:,:] * E_true_gp[:,None,None], B)) * detJ
        
        N_mat = np.zeros((3, 24))
        for _i in range(8): N_mat[0, _i*3], N_mat[1, _i*3+1], N_mat[2, _i*3+2] = N_shape[_i], N_shape[_i], N_shape[_i]
        F_elem_all += (N_mat.T @ np.array([0, 0, -rho_const * gravity]) * detJ)[None, :]

    dof_map = np.zeros((len(elements), 24), dtype=np.int32)
    for i in range(8): dof_map[:, i*3:i*3+3] = elements[:, i:i+1] * 3 + np.array([0, 1, 2])
    I, J, V = np.repeat(dof_map[:, :, None], 24, axis=2).ravel(), np.repeat(dof_map[:, None, :], 24, axis=1).ravel(), K_all.ravel()
    K_sparse = sp.coo_matrix((V, (I, J)), shape=(Total_DOF, Total_DOF)).tocsr()
    F_global = np.bincount(dof_map.ravel(), weights=F_elem_all.ravel(), minlength=Total_DOF)

    bc_dict = {}
    tol = 1e-5
    for n in np.where(nodes[:, 0] < tol)[0]: bc_dict[n*3+0] = 0.0
    for n in np.where(nodes[:, 0] > Lx - tol)[0]: bc_dict[n*3+0] = TEST_DU_X
    for n in np.where(nodes[:, 1] < tol)[0]: bc_dict[n*3+1] = 0.0
    for n in np.where(nodes[:, 1] > Ly - tol)[0]: bc_dict[n*3+1] = TEST_DV_Y
    for n in np.where(nodes[:, 2] < tol)[0]: bc_dict[n*3+2] = 0.0 

    bc_dofs = np.array(list(bc_dict.keys()), dtype=np.int32)
    bc_vals = np.array(list(bc_dict.values()), dtype=np.float64)
    is_free = np.ones(Total_DOF, dtype=bool); is_free[bc_dofs] = False
    free_dofs = np.where(is_free)[0]

    U_bc_full = np.zeros(Total_DOF); U_bc_full[bc_dofs] = bc_vals
    F_eff = F_global[free_dofs] - K_sparse[free_dofs, :].dot(U_bc_full)
    
    K_free = K_sparse[free_dofs, :][:, free_dofs]
    U_free, _ = spla.cg(K_free, F_eff, M=sp.diags(1.0/K_free.diagonal()), rtol=1e-6)
    
    U_global = np.zeros(Total_DOF); U_global[bc_dofs], U_global[free_dofs] = bc_vals, U_free
    fem_time = time.time() - start_time
    print(f" {fem_time:.2f} 秒\n")
    return nodes, U_global, fem_time

def Evaluate_Operator(nodes, U_fem, fem_time, V_seis_grid, device):

    start_time = time.time()
    
    model = ReservoirOperator().to(device)
    model.load_state_dict(torch.load('reservoir_operator_smooth.pth', map_location=device, weights_only=True))
    model.eval()
    
    params = torch.tensor([[TEST_C_SCALE, TEST_C_SHIFT, TEST_DU_X, TEST_DV_Y]], dtype=torch.float64, device=device)
    p_min = torch.tensor([10.0, 2.0, -2.0, -2.0], device=device)
    p_max = torch.tensor([30.0, 15.0, 0.0, 0.0], device=device)
    params_norm = 2.0 * (params - p_min) / (p_max - p_min) - 1.0
    

    U_op = np.zeros(len(nodes) * 3)
    batch_size = 32000
    with torch.no_grad():
        for i in range(0, len(nodes), batch_size):
            end = min(i + batch_size, len(nodes))
            pts_phys = torch.tensor(nodes[i:end], dtype=torch.float64, device=device)
            pts_norm = (pts_phys / torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)).unsqueeze(0)
            
            U_pred, _ = model.forward_disp(pts_norm, params, params_norm, V_seis_grid)
            U_op[i*3:end*3] = U_pred.squeeze(0).cpu().numpy().flatten()
            

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    xc, yc, zc = np.linspace(dx/2, Lx-dx/2, Nx), np.linspace(dy/2, Ly-dy/2, Ny), np.linspace(dz/2, Lz-dz/2, Nz)
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
    pts_c = np.vstack([Xc.ravel(), Yc.ravel(), Zc.ravel()]).T
    
    sxx_op_c = np.zeros(len(pts_c))
    exx_op_c = np.zeros(len(pts_c))
    
    with torch.no_grad():
        for i in range(0, len(pts_c), batch_size):
            end = min(i + batch_size, len(pts_c))
            pts_phys = torch.tensor(pts_c[i:end], dtype=torch.float64, device=device)
            pts_norm = (pts_phys / torch.tensor([Lx, Ly, Lz], dtype=torch.float64, device=device)).unsqueeze(0)
            

            eps_pred = model.forward_strain_fd(pts_norm, params, params_norm, V_seis_grid, h_fd=5.0e-3)
            _, E_true_GPa = model.forward_disp(pts_norm, params, params_norm, V_seis_grid)
            

            exx_op_c[i:end] = eps_pred[0, :, 0].cpu().numpy()
            
            lam = (E_true_GPa * nu_const) / ((1.0 + nu_const) * (1.0 - 2.0 * nu_const))
            mu = E_true_GPa / (2.0 * (1.0 + nu_const))
            tr = eps_pred[0,:,0] + eps_pred[0,:,1] + eps_pred[0,:,2]
            

            sxx = lam[0,:,0] * tr + 2.0 * mu[0,:,0] * eps_pred[0,:,0]
            sxx_op_c[i:end] = (sxx * 1000.0).cpu().numpy() # 转化为 MPa
            
    sxx_op_c = sxx_op_c.reshape(Nx, Ny, Nz)
    exx_op_c = exx_op_c.reshape(Nx, Ny, Nz)
    
    op_time = time.time() - start_time
    print(f"   {op_time:.4f} 秒. 提速比: {fem_time/op_time:.0f} 倍！\n")
    

    U_x_3d = U_fem[0::3].reshape(Nx+1, Ny+1, Nz+1)
    U_y_3d = U_fem[1::3].reshape(Nx+1, Ny+1, Nz+1)
    U_z_3d = U_fem[2::3].reshape(Nx+1, Ny+1, Nz+1)
    

    exx_fem = 0.25/dx * ((U_x_3d[1:,:-1,:-1]-U_x_3d[:-1,:-1,:-1])+(U_x_3d[1:,1:,:-1]-U_x_3d[:-1,1:,:-1])+(U_x_3d[1:,:-1,1:]-U_x_3d[:-1,:-1,1:])+(U_x_3d[1:,1:,1:]-U_x_3d[:-1,1:,1:]))
    eyy_fem = 0.25/dy * ((U_y_3d[:-1,1:,:-1]-U_y_3d[:-1,:-1,:-1])+(U_y_3d[1:,1:,:-1]-U_y_3d[1:,:-1,:-1])+(U_y_3d[:-1,1:,1:]-U_y_3d[:-1,:-1,1:])+(U_y_3d[1:,1:,1:]-U_y_3d[1:,:-1,1:]))
    ezz_fem = 0.25/dz * ((U_z_3d[:-1,:-1,1:]-U_z_3d[:-1,:-1,:-1])+(U_z_3d[1:,:-1,1:]-U_z_3d[1:,:-1,:-1])+(U_z_3d[:-1,1:,1:]-U_z_3d[:-1,1:,:-1])+(U_z_3d[1:,1:,1:]-U_z_3d[1:,1:,:-1]))
    

    lin_ax = np.linspace(0, 1, 32)
    interp = RegularGridInterpolator((lin_ax, lin_ax, lin_ax), V_seis_grid.cpu().numpy()[0,0], bounds_error=False, fill_value=None)
    pts_norm_np = pts_c / np.array([Lx, Ly, Lz])
    E_fem_GPa = (TEST_C_SCALE * interp(pts_norm_np[:, [2, 1, 0]]) + TEST_C_SHIFT).reshape(Nx, Ny, Nz)
    
    lam_f = (E_fem_GPa * nu_const) / ((1.0 + nu_const) * (1.0 - 2.0 * nu_const))
    mu_f = E_fem_GPa / (2.0 * (1.0 + nu_const))
    sxx_fem_c = (lam_f * (exx_fem + eyy_fem + ezz_fem) + 2.0 * mu_f * exx_fem) * 1000.0

    u_l2_err = np.linalg.norm(U_fem - U_op) / (np.linalg.norm(U_fem) + 1e-12)
    e_l2_err = np.linalg.norm(exx_op_c - exx_fem) / (np.linalg.norm(exx_fem) + 1e-12)
    s_l2_err = np.linalg.norm(sxx_op_c - sxx_fem_c) / (np.linalg.norm(sxx_fem_c) + 1e-12)
    

    print(f" (U) : {u_l2_err * 100:.3f} %")
    print(f"  (Exx) : {e_l2_err * 100:.3f} %")
    print(f" (Sxx): {s_l2_err * 100:.3f} %")
    
    z_mid = Nz // 2
    

    X_node_slice = nodes[:, 0].reshape(Nx+1, Ny+1, Nz+1)[:, :, z_mid]
    Y_node_slice = nodes[:, 1].reshape(Nx+1, Ny+1, Nz+1)[:, :, z_mid]

    X_slice, Y_slice = Xc[:, :, z_mid], Yc[:, :, z_mid]
    

    U_x_fem_slice = U_x_3d[:, :, z_mid] * 1000.0
    U_x_op_slice = U_op[0::3].reshape(Nx+1, Ny+1, Nz+1)[:, :, z_mid] * 1000.0
    

    exx_fem_slice = exx_fem[:, :, z_mid] * 1e6
    exx_op_slice = exx_op_c[:, :, z_mid] * 1e6
    

    sxx_fem_slice = sxx_fem_c[:, :, z_mid]
    sxx_op_slice = sxx_op_c[:, :, z_mid]

    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    def plot_row(row_idx, grid_X, grid_Y, fem_data, pinn_data, cmap_name, ylabel):
        err_data = np.abs(fem_data - pinn_data)
        vmin = min(fem_data.min(), pinn_data.min())
        vmax = max(fem_data.max(), pinn_data.max())
        if np.isclose(vmin, vmax): vmin, vmax = vmin - 1e-5, vmax + 1e-5
        
        strict_levels = np.linspace(vmin, vmax, 41)
        main_ticks = np.linspace(vmin, vmax, 5)
        
        err_max = np.percentile(err_data, 99.5) if np.percentile(err_data, 99.5) > 1e-8 else 1e-8
        err_levels = np.linspace(0, err_max, 41)
        err_ticks = np.linspace(0, err_max, 5)
        
        fmt = ticker.FormatStrFormatter('%.3g')

        # FEM 
        ax1 = axes[row_idx, 0]
        c1 = ax1.contourf(grid_X, grid_Y, fem_data, levels=strict_levels, cmap=cmap_name, extend='both')
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        cb1 = fig.colorbar(c1, ax=ax1, ticks=main_ticks, format=fmt, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=14)

        # Operator
        ax2 = axes[row_idx, 1]
        c2 = ax2.contourf(grid_X, grid_Y, pinn_data, levels=strict_levels, cmap=cmap_name, extend='both')
        ax2.set_xticks([]); ax2.set_yticks([])
        cb2 = fig.colorbar(c2, ax=ax2, ticks=main_ticks, format=fmt, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=14)

        # Error
        ax3 = axes[row_idx, 2]
        c3 = ax3.contourf(grid_X, grid_Y, err_data, levels=err_levels, cmap='Reds', extend='max')
        ax3.set_xticks([]); ax3.set_yticks([])
        cb3 = fig.colorbar(c3, ax=ax3, ticks=err_ticks, format=fmt, fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=14)


    plot_row(0, X_node_slice, Y_node_slice, U_x_fem_slice, U_x_op_slice, "jet", "Disp X (mm)")
    

    plot_row(1, X_slice, Y_slice, exx_fem_slice, exx_op_slice, "viridis", "Strain XX (με)")
    

    plot_row(2, X_slice, Y_slice, sxx_fem_slice, sxx_op_slice, "coolwarm", "Stress XX (MPa)")

    # 设置列标题
    axes[0, 0].set_title("FEM Truth (Reference)", fontsize=18, pad=15)
    axes[0, 1].set_title(f"Operator Predict\nInference: {op_time:.3f}s", fontsize=18, pad=15)
    axes[0, 2].set_title("Absolute Error", fontsize=18, pad=15)

    plt.subplots_adjust(wspace=0.35, hspace=0.1)
    plt.savefig('Reservoir_Inversion_AllFields_Smooth.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V_seis_grid = torch.load('seismic_prior.pt', map_location=device, weights_only=True)
    nodes, U_fem, fem_time = Run_FEM_Benchmark(V_seis_grid.cpu().numpy())
    Evaluate_Operator(nodes, U_fem, fem_time, V_seis_grid, device)