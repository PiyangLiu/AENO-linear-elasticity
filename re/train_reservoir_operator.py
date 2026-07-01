import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
import warnings
import os

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


Lx, Ly, Lz = 2000.0, 2000.0, 200.0  
nu_const, rho_const, gravity = 0.25, 2500.0, 9.81
E_SCALE = 1.0e9

def generate_seismic_data(grid_size=32):

    z_lin = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    y_lin = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    x_lin = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    Z, Y, X = torch.meshgrid(z_lin, y_lin, x_lin, indexing='ij') 
    
    # 稍微降低地层自身波动的频率，避免给网络引入不必要的强迫震荡
    layering = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * Z)
    lens = torch.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.15)
    V_seis = 0.6 * layering + 0.4 * lens
    V_seis = torch.clamp(V_seis, 0.0, 1.0)
    
    tensor_5d = V_seis.unsqueeze(0).unsqueeze(0)
    torch.save(tensor_5d, 'seismic_prior.pt') 
    return tensor_5d


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

class ReservoirOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = SmoothPE()
        self.param_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 64))
        
        in_dim = 9
        cond_dim = 64 + 1     
        
        self.block1 = FiLMBlock(in_dim, 128, cond_dim)
        self.block2 = FiLMBlock(128, 128, cond_dim)
        self.block3 = FiLMBlock(128, 128, cond_dim)
        
        self.head = nn.Linear(128, 3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward_disp(self, pts_norm, params, params_norm, V_seis_grid):
        B, N, _ = pts_norm.shape
        c_scale, c_shift = params[:, 0:1].unsqueeze(1), params[:, 1:2].unsqueeze(1)
        du_x, dv_y = params[:, 2:3].unsqueeze(1), params[:, 3:4].unsqueeze(1)
        
        grid_coords = (pts_norm * 2.0 - 1.0).unsqueeze(1).unsqueeze(1).float() 
        V_seis_local = F.grid_sample(V_seis_grid.expand(B,-1,-1,-1,-1).float(), 
                                     grid_coords, align_corners=True, padding_mode='border')
        V_seis_local = V_seis_local.squeeze(2).squeeze(2).transpose(1, 2).to(torch.float64)
        
        E_true_GPa = c_scale * V_seis_local + c_shift
        E_cond = E_true_GPa / 45.0 
        
        glob_feat = self.param_encoder(params_norm).unsqueeze(1).expand(-1, N, -1)
        cond = torch.cat([glob_feat, E_cond], dim=-1)
        
        x_in = self.pe(pts_norm)
        h = self.block1(x_in, cond)
        h = self.block2(h, cond)
        h = self.block3(h, cond)
        corr = self.head(h)
        
        xh, yh, zh = pts_norm[..., 0:1], pts_norm[..., 1:2], pts_norm[..., 2:3]
        u = xh * du_x + 4.0 * xh * (1.0 - xh) * corr[..., 0:1]
        v = yh * dv_y + 4.0 * yh * (1.0 - yh) * corr[..., 1:2]
        w = zh * corr[..., 2:3]
        return torch.cat([u, v, w], dim=-1), E_true_GPa

    def forward_strain_fd(self, pts_norm, params, params_norm, V_seis_grid, h_fd=5.0e-3):

        eye = torch.eye(3, device=pts_norm.device, dtype=pts_norm.dtype).view(1, 1, 3, 3)
        coords_pm = pts_norm.unsqueeze(2) + h_fd * torch.cat([eye, -eye], dim=2) 
        
        disp_list = []
        for k in range(6):
            uk, _ = self.forward_disp(coords_pm[:, :, k, :], params, params_norm, V_seis_grid)
            disp_list.append(uk)
        uxp, uyp, uzp, uxm, uym, uzm = disp_list
        
        idx, idy, idz = 1.0/(2.0*h_fd*Lx), 1.0/(2.0*h_fd*Ly), 1.0/(2.0*h_fd*Lz)
        exx, eyy, ezz = (uxp[...,0]-uxm[...,0])*idx, (uyp[...,1]-uym[...,1])*idy, (uzp[...,2]-uzm[...,2])*idz
        exy = 0.5 * ((uyp[...,0]-uym[...,0])*idy + (uxp[...,1]-uxm[...,1])*idx)
        eyz = 0.5 * ((uzp[...,1]-uzm[...,1])*idz + (uyp[...,2]-uym[...,2])*idy)
        ezx = 0.5 * ((uxp[...,2]-uxm[...,2])*idx + (uzp[...,0]-uzm[...,0])*idz)
        return torch.stack([exx, eyy, ezz, exy, eyz, ezx], dim=-1)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    
    if not os.path.exists('seismic_prior.pt'): generate_seismic_data()
    V_seis_grid = torch.load('seismic_prior.pt', map_location=device, weights_only=True).to(device)
    
    model = ReservoirOperator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    epochs = 400
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    batch_size, N_pts = 16, 8192
    

    soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=42)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        
        c_scale = torch.rand(batch_size, device=device) * 20.0 + 10.0   
        c_shift = torch.rand(batch_size, device=device) * 13.0 + 2.0    
        du_x = -torch.rand(batch_size, device=device) * 2.0             
        dv_y = -torch.rand(batch_size, device=device) * 2.0             
        
        params = torch.stack([c_scale, c_shift, du_x, dv_y], dim=1)
        p_min = torch.tensor([10.0, 2.0, -2.0, -2.0], device=device)
        p_max = torch.tensor([30.0, 15.0, 0.0, 0.0], device=device)
        params_norm = 2.0 * (params - p_min) / (p_max - p_min) - 1.0 
        

        pts_norm = soboleng.draw(batch_size * N_pts).to(device).to(torch.float64).view(batch_size, N_pts, 3)
        
        u_pred, _ = model.forward_disp(pts_norm, params, params_norm, V_seis_grid)

        eps_pred = model.forward_strain_fd(pts_norm, params, params_norm, V_seis_grid, h_fd=5.0e-3)
        _, E_true_GPa = model.forward_disp(pts_norm, params, params_norm, V_seis_grid)
        E_true_GPa = E_true_GPa.squeeze(-1)
        
        lam = (E_true_GPa * nu_const) / ((1.0 + nu_const) * (1.0 - 2.0 * nu_const)) 
        mu = E_true_GPa / (2.0 * (1.0 + nu_const)) 
        
        tr = eps_pred[...,0] + eps_pred[...,1] + eps_pred[...,2]
        strain_sq = eps_pred[...,0]**2 + eps_pred[...,1]**2 + eps_pred[...,2]**2 + 2.0*(eps_pred[...,3]**2 + eps_pred[...,4]**2 + eps_pred[...,5]**2)
        W_strain = 0.5 * lam * (tr**2) + mu * strain_sq
        
        W_ext = - (rho_const * gravity / E_SCALE) * u_pred[..., 2]
        
        loss = (W_strain - W_ext).mean() * 100000.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch+1) % 50 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs} |  {W_strain.mean().item():.6f}")

    torch.save(model.state_dict(), 'reservoir_operator_smooth.pth')
    print(f" {time.time()-start_time:.1f} 秒。")

if __name__ == '__main__':
    train()