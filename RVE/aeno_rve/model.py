from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SE3D(nn.Module):
    """Lightweight squeeze-excitation for strengthening the local-prior encoder."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool3d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResidualConvBlock3D(nn.Module):
    def __init__(self, cin, cout, use_se=True):
        super().__init__()
        groups = max(1, min(8, cout // 4))
        self.proj = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
        # Replication padding reduces artificial edge stripes relative to zero padding.
        self.conv1 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(cin, cout, 3, padding=0))
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=cout)
        self.conv2 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(cout, cout, 3, padding=0))
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=cout)
        self.se = SE3D(cout) if use_se else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.se(h)
        return F.gelu(h + self.proj(x))


class UNet3D(nn.Module):
    """Residual 3D U-Net used as the nodal interior correction operator."""
    def __init__(self, in_ch=10, out_ch=3, base=16):
        super().__init__()
        self.enc1 = ResidualConvBlock3D(in_ch, base)
        self.enc2 = ResidualConvBlock3D(base, base * 2)
        self.enc3 = ResidualConvBlock3D(base * 2, base * 4)
        self.bot = ResidualConvBlock3D(base * 4, base * 4)
        self.dec3 = ResidualConvBlock3D(base * 8, base * 2)
        self.dec2 = ResidualConvBlock3D(base * 4, base)
        self.dec1 = ResidualConvBlock3D(base * 2, base)
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))
        b = self.bot(F.avg_pool3d(e3, 2))
        d3 = F.interpolate(b, size=e3.shape[-3:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def coordinate_grid(batch_size, npoints, device, dtype=torch.float32):
    """Return node/point coordinates [B,3,D,H,W], channel order x,y,z."""
    z = torch.linspace(0.0, 1.0, int(npoints), device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, int(npoints), device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, int(npoints), device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    return grid


def bubble_function(npoints, device, dtype=torch.float32, power=1.0):
    grid = coordinate_grid(1, npoints, device, dtype=dtype)[0]
    x, y, z = grid[0], grid[1], grid[2]
    b = x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z)
    b = b / (b.max() + 1e-12)
    if abs(power - 1.0) > 1e-12:
        b = torch.clamp(b, min=0.0) ** power
    return b[None, None]


def _to_nodal_channel(v: torch.Tensor, nnode: int) -> torch.Tensor:
    """Interpolate an element-centered scalar channel [B,D,H,W] to nodal resolution."""
    if v.ndim == 4:
        v = v[:, None]
    return F.interpolate(v, size=(nnode, nnode, nnode), mode="trilinear", align_corners=True)


class AdmissibleAENO(nn.Module):
    """AENO with hard KUBC and nodal displacement output.

    V2 changes relative to V1b:
    - The network predicts displacement corrections on the (N+1)^3 Hex8 node grid.
    - The returned field is nodal displacement u[B,3,N+1,N+1,N+1].
    - Element-centered strain/stress are recovered outside the model with the same
      Hex8 center B-matrix used by the FEM reference.

    The generalization target remains microstructure-only for the current Bristol
    RVE case; E_s and nu_s are still accepted as inputs for compatibility but may
    be fixed to a single value during training/evaluation.
    """
    def __init__(self, n=32, base=16, eps0=0.01, Es_range=(1.0, 1.0), nu_range=(0.33, 0.33), bubble_power=1.0):
        super().__init__()
        self.n = int(n)                 # number of Hex8 elements per axis
        self.nnode = self.n + 1         # number of displacement nodes per axis
        self.eps0 = float(eps0)
        self.Es_min, self.Es_max = map(float, Es_range)
        self.nu_min, self.nu_max = map(float, nu_range)
        self.bubble_power = float(bubble_power)
        # input: chi_void, local_phi_3, local_phi_7, sdf_void, interface,
        # boundary_distance, x, y, z, nu_norm. E_s is deliberately excluded.
        self.net = UNet3D(in_ch=10, out_ch=3, base=base)

    @staticmethod
    def local_porosity_channels(chi):
        chi1 = chi[:, None]
        phi3 = F.avg_pool3d(chi1, kernel_size=3, stride=1, padding=1)
        phi7 = F.avg_pool3d(chi1, kernel_size=7, stride=1, padding=3)
        return phi3, phi7

    @staticmethod
    def fallback_interface(chi_void):
        chi1 = chi_void[:, None]
        mx = F.max_pool3d(chi1, kernel_size=3, stride=1, padding=1)
        mn = -F.max_pool3d(-chi1, kernel_size=3, stride=1, padding=1)
        return torch.clamp(mx - mn, 0.0, 1.0)

    def fallback_boundary_distance(self, batch_size, device, dtype):
        coords = coordinate_grid(batch_size, self.n, device, dtype=dtype)
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        return torch.minimum(torch.minimum(torch.minimum(x, 1.0 - x), torch.minimum(y, 1.0 - y)), torch.minimum(z, 1.0 - z))

    def make_input(self, chi_void, Es, nu_s, sdf_void=None, interface=None, boundary_distance=None):
        B, D, H, W = chi_void.shape
        assert D == H == W == self.n, f"Expected cubic element crop {self.n}, got {chi_void.shape}"
        device = chi_void.device
        dtype = chi_void.dtype
        nnode = self.nnode

        coords = coordinate_grid(B, nnode, device, dtype=dtype)
        nun = (nu_s - self.nu_min) / (self.nu_max - self.nu_min + 1e-12)
        nu_ch = nun.reshape(B, 1, 1, 1, 1).expand(B, 1, nnode, nnode, nnode)

        phi3, phi7 = self.local_porosity_channels(chi_void)
        if sdf_void is None:
            sdf_void = 1.0 - 2.0 * chi_void
        if interface is None:
            interface = self.fallback_interface(chi_void).squeeze(1)
        if boundary_distance is None:
            boundary_distance = self.fallback_boundary_distance(B, device, dtype).squeeze(1)

        # Element-centered microstructure channels are interpolated to the nodal grid
        # so that the operator output and the FEM unknowns live on the same nodes.
        chi_n = _to_nodal_channel(chi_void, nnode)
        phi3_n = F.interpolate(phi3, size=(nnode, nnode, nnode), mode="trilinear", align_corners=True)
        phi7_n = F.interpolate(phi7, size=(nnode, nnode, nnode), mode="trilinear", align_corners=True)
        sdf_n = _to_nodal_channel(sdf_void, nnode)
        int_n = _to_nodal_channel(interface, nnode)
        bd_n = _to_nodal_channel(boundary_distance, nnode)

        return torch.cat([chi_n, phi3_n, phi7_n, sdf_n, int_n, bd_n, coords, nu_ch], dim=1)

    def forward(self, chi_void, Es, nu_s, sdf_void=None, interface=None, boundary_distance=None):
        B = chi_void.shape[0]
        device = chi_void.device
        dtype = chi_void.dtype
        inp = self.make_input(
            chi_void, Es, nu_s,
            sdf_void=sdf_void,
            interface=interface,
            boundary_distance=boundary_distance,
        )
        corr = self.net(inp)
        coords = coordinate_grid(B, self.nnode, device, dtype=dtype)
        x = coords[:, 0:1]
        u_aff = torch.cat([self.eps0 * x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
        b = bubble_function(self.nnode, device, dtype=dtype, power=self.bubble_power)
        u = u_aff + self.eps0 * b * corr
        return u
