"""Local diffusion decoder: optional 3D-conv stem (arXiv:2604.08358) +
transformer with soft local attention, denoising the physical error e
conditioned on the syndrome s. Predicting e (not the logical class, as in
2509.22347 / 2604.24640) is what makes a local recovery map possible.

Soft local attention: attention weights are scaled by exp(-dist/xi) with a
TRAINABLE per-head decay length xi (lattice units), implemented as an additive
logit bias -dist/xi fed to F.scaled_dot_product_attention. The learned xi is
the empirical locality scale (Markov length). An optional hard cutoff
(window_radius, lattice units) adds -inf beyond the radius in the same bias.
No neighbor gathers are materialized anywhere -- the bias is one (h, T, T)
tensor broadcast over the batch, so memory is batch-independent in the bias
and the fused SDPA kernels handle the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_TOKEN = 2  # value embedding index for a masked error bit (0, 1, or MASK)


@dataclass
class ModelConfig:
    d_model: int = 64            # scaling analysis: 64 already sufficient
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 256
    conv_channels: int = 32
    conv_layers: int = 2
    dropout: float = 0.0
    use_conv_stem: bool = True
    xi_init: float = 2.0         # initial attention decay length (LATTICE units)
    window_radius: Optional[float] = None  # optional hard cutoff (lattice units)
    sigma_init: float = 2.0      # legacy field (pre-SDPA checkpoints); unused


def _lattice_coords(coords: np.ndarray) -> np.ndarray:
    """Raw stim coords -> lattice units: x,y spacing is 2 on the detector grid,
    t spacing is 1. NaN (undetectable mechanisms) -> origin."""
    c = coords.copy()
    c[np.isnan(c).any(axis=1)] = 0.0
    c[:, 0] /= 2.0
    c[:, 1] /= 2.0
    return c.astype(np.float32)


class SpacetimeConvStem(nn.Module):
    """Scatter detector embeddings onto the (X,Y,T) voxel grid, run 3D convs
    (arXiv:2604.08358 style), gather back. conv_layers sets the local RF;
    keep it small to force the attention (and hence xi) to carry the range.
    """

    def __init__(self, cfg: ModelConfig, det_grid_idx: np.ndarray, grid_shape):
        super().__init__()
        self.cfg = cfg
        self.grid_shape = grid_shape  # (X, Y, T)
        self.register_buffer("gidx", torch.as_tensor(det_grid_idx, dtype=torch.long))
        ch = cfg.conv_channels
        layers = [nn.Conv3d(cfg.d_model, ch, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(cfg.conv_layers - 1):
            layers += [nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.GELU()]
        layers += [nn.Conv3d(ch, cfg.d_model, kernel_size=3, padding=1)]
        self.conv = nn.Sequential(*layers)

    def forward(self, det_feats: torch.Tensor) -> torch.Tensor:
        B, n_det, dm = det_feats.shape
        X, Y, T = self.grid_shape
        grid = det_feats.new_zeros(B, dm, X, Y, T)
        gx, gy, gt = self.gidx[:, 0], self.gidx[:, 1], self.gidx[:, 2]
        grid[:, :, gx, gy, gt] = det_feats.transpose(1, 2)
        grid = grid + self.conv(grid)            # residual spacetime conv
        return grid[:, :, gx, gy, gt].transpose(1, 2)


class SoftLocalAttention(nn.Module):
    """MHA via F.scaled_dot_product_attention with an additive distance bias.

    Bias per head: -dist/xi  (== scaling attention weights by exp(-dist/xi)
    before normalization), xi = softplus(raw_xi) trainable, in lattice units.
    Optional hard window: -inf beyond window_radius (same bias tensor, so the
    kernel never materializes anything batch-dependent beyond q,k,v).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        xi0 = max(cfg.xi_init, 1e-3)
        # softplus^-1; for large x it's ~x (avoids expm1 overflow -> inf param)
        inv = xi0 if xi0 > 20 else float(np.log(np.expm1(xi0)))
        self.raw_xi = nn.Parameter(torch.full((cfg.n_heads,), inv))
        self.dropout_p = cfg.dropout
        self.window_radius = cfg.window_radius

    def xi(self) -> torch.Tensor:
        return F.softplus(self.raw_xi)  # (h,), lattice units

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model); dist: (T, T) pairwise distances, lattice units
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (B, h, T, dk)
        xi = self.xi().view(self.h, 1, 1)                 # (h, 1, 1)
        bias = -(dist.unsqueeze(0) / xi)                  # (h, T, T)
        if self.window_radius is not None:
            bias = bias.masked_fill(dist.unsqueeze(0) > self.window_radius,
                                    float("-inf"))
        ctx = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.unsqueeze(0),         # broadcast over B
            dropout_p=self.dropout_p if self.training else 0.0)
        ctx = ctx.transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.out(ctx)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SoftLocalAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x, dist):
        x = x + self.attn(self.ln1(x), dist)
        x = x + self.ff(self.ln2(x))
        return x


class LocalDiffusionDecoder(nn.Module):
    """Masked-diffusion denoiser over the physical error e, conditioned on s."""

    def __init__(self, cfg: ModelConfig, code):
        super().__init__()
        self.cfg = cfg
        self.n_det = code.n_det
        self.n_err = code.n_err

        # pairwise distances in LATTICE units (xi reads off directly)
        det_c = _lattice_coords(code.det_coords)
        err_c = _lattice_coords(code.err_coords)
        all_c = np.concatenate([det_c, err_c], axis=0)          # (T, 3)
        dist = np.sqrt(((all_c[:, None, :] - all_c[None, :, :]) ** 2).sum(-1))
        self.register_buffer("dist", torch.as_tensor(dist, dtype=torch.float32))

        # detector voxel grid index for the conv stem + geometry
        gidx = np.round(det_c).astype(int)
        gidx -= gidx.min(axis=0, keepdims=True)
        grid_shape = tuple((gidx.max(axis=0) + 1).tolist())
        self.det_grid_shape = grid_shape

        dm = cfg.d_model
        self.syn_val = nn.Embedding(2, dm)
        self.err_val = nn.Embedding(3, dm)
        self.type_emb = nn.Embedding(2, dm)       # 0 = detector, 1 = error
        self.coord_proj = nn.Linear(3, dm)
        # normalize coords to ~unit scale for the embedding only
        span = max(float(np.ptp(all_c)), 1.0)
        self.register_buffer("coords",
                             torch.as_tensor(all_c / span, dtype=torch.float32))

        self.stem = (SpacetimeConvStem(cfg, gidx, grid_shape)
                     if cfg.use_conv_stem else None)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, 1)

    def forward(self, s: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """s: (B, n_det) {0,1}; e_t: (B, n_err) {0,1,MASK}. -> logits (B, n_err)."""
        dm = self.cfg.d_model
        det = self.syn_val(s.long()) + self.type_emb.weight[0].view(1, 1, dm)
        det = det + self.coord_proj(self.coords[: self.n_det]).unsqueeze(0)
        if self.stem is not None:
            det = det + self.stem(det)
        err = self.err_val(e_t.long()) + self.type_emb.weight[1].view(1, 1, dm)
        err = err + self.coord_proj(self.coords[self.n_det:]).unsqueeze(0)

        x = torch.cat([det, err], dim=1)          # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x, self.dist)
        x = self.ln_f(x)
        return self.head(x[:, self.n_det:, :]).squeeze(-1)

    def locality_radii(self):
        """Learned per-head xi (lattice units) for every layer."""
        return {f"layer{i}": blk.attn.xi().detach().cpu().numpy()
                for i, blk in enumerate(self.blocks)}
