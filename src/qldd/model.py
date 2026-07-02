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
import torch.utils.checkpoint
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
    xi_space_init: float = 2.0   # initial SPATIAL decay length (lattice units)
    xi_time_init: float = 2.0    # initial TEMPORAL decay length (rounds)
    xi_min: float = 1.0          # FLOOR: xi = xi_min + softplus(raw). Stops the
                                 # degenerate "attend to nothing" collapse (xi->0)
                                 # when no conv stem regularizes context. A token
                                 # always reaches its lattice neighbors.
    causal_time: bool = False    # online constraint: attend only to t_key <= t_query
    window_radius: Optional[float] = None  # optional hard cutoff (lattice units)
    xi_init: float = 2.0         # legacy isotropic init; unused
    grad_checkpoint: bool = True # recompute blocks in backward: one block's attn
                                 # matrices live at a time instead of n_layers.
                                 # Needed because a grad-requiring attn_mask
                                 # (trainable xi) forces SDPA onto the math
                                 # backend, which materializes (B,h,T,T).
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
        self.causal = bool(getattr(cfg, "causal_time", False))
        self.register_buffer("gidx", torch.as_tensor(det_grid_idx, dtype=torch.long))
        ch = cfg.conv_channels
        # Symmetric (offline): pad=1 on all axes -> kernel sees t-1,t,t+1.
        # Causal (online): T-padding 0 in the conv; we F.pad the PAST side by 2
        # before each conv, so the kernel sees t-2,t-1,t and NEVER t+1. This
        # preserves the streaming guarantee while keeping the local feature
        # extractor the offline model relies on.
        tpad = 0 if self.causal else 1
        convs = [nn.Conv3d(cfg.d_model, ch, kernel_size=3, padding=(1, 1, tpad))]
        for _ in range(cfg.conv_layers - 1):
            convs.append(nn.Conv3d(ch, ch, kernel_size=3, padding=(1, 1, tpad)))
        convs.append(nn.Conv3d(ch, cfg.d_model, kernel_size=3, padding=(1, 1, tpad)))
        self.convs = nn.ModuleList(convs)
        self.act = nn.GELU()

    def _run_convs(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            if self.causal:
                # (B,C,X,Y,T): F.pad pads last dim first -> (T_left, T_right, ...)
                x = torch.nn.functional.pad(x, (2, 0, 0, 0, 0, 0))
            x = conv(x)
            if i < n - 1:
                x = self.act(x)
        return x

    def forward(self, det_feats: torch.Tensor) -> torch.Tensor:
        B, n_det, dm = det_feats.shape
        X, Y, T = self.grid_shape
        grid = det_feats.new_zeros(B, dm, X, Y, T)
        gx, gy, gt = self.gidx[:, 0], self.gidx[:, 1], self.gidx[:, 2]
        grid[:, :, gx, gy, gt] = det_feats.transpose(1, 2)
        grid = grid + self._run_convs(grid)      # residual spacetime conv
        return grid[:, :, gx, gy, gt].transpose(1, 2)


class SoftLocalAttention(nn.Module):
    """MHA via F.scaled_dot_product_attention with an additive distance bias.

    Anisotropic soft locality: bias per head is -(d_xy/xi_s + |dt|/xi_t), i.e.
    attention weights scale as exp(-d_xy/xi_s) * exp(-|dt|/xi_t) with SEPARATE
    trainable spatial and temporal decay lengths (no reason they are equal --
    xi_s is the spatial Markov length in lattice units, xi_t the temporal one
    in rounds). Optional hard window (spatial) and causal-time mask (online
    constraint: keys strictly in the future get -inf) live in the same bias
    tensor, so nothing batch-dependent is materialized beyond q,k,v.
    """

    @staticmethod
    def _inv_softplus(x0: float) -> float:
        x0 = max(x0, 1e-3)
        return x0 if x0 > 20 else float(np.log(np.expm1(x0)))

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.xi_min = cfg.xi_min
        self.raw_xi_s = nn.Parameter(torch.full(
            (cfg.n_heads,), self._inv_softplus(max(cfg.xi_space_init - cfg.xi_min, 1e-3))))
        self.raw_xi_t = nn.Parameter(torch.full(
            (cfg.n_heads,), self._inv_softplus(max(cfg.xi_time_init - cfg.xi_min, 1e-3))))
        self.dropout_p = cfg.dropout
        self.window_radius = cfg.window_radius
        self.causal_time = cfg.causal_time

    def xi_space(self) -> torch.Tensor:
        return self.xi_min + F.softplus(self.raw_xi_s)  # (h,), lattice units, >= xi_min

    def xi_time(self) -> torch.Tensor:
        return self.xi_min + F.softplus(self.raw_xi_t)  # (h,), rounds, >= xi_min

    def forward(self, x: torch.Tensor, geom: dict) -> torch.Tensor:
        # x: (B, T, d_model); geom: dist_s/dist_t (T,T), dt_signed (T,T)
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (B, h, T, dk)
        xs = self.xi_space().view(self.h, 1, 1)
        xt = self.xi_time().view(self.h, 1, 1)
        bias = -(geom["dist_s"].unsqueeze(0) / xs) \
               - (geom["dist_t"].unsqueeze(0) / xt)       # (h, T, T)
        if self.window_radius is not None:
            bias = bias.masked_fill(
                geom["dist_s"].unsqueeze(0) > self.window_radius, float("-inf"))
        if self.causal_time:
            # key strictly in the future of the query -> blocked
            bias = bias.masked_fill(
                geom["dt_signed"].unsqueeze(0) > 0, float("-inf"))
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

    def forward(self, x, geom):
        x = x + self.attn(self.ln1(x), geom)
        x = x + self.ff(self.ln2(x))
        return x


class LocalDiffusionDecoder(nn.Module):
    """Masked-diffusion denoiser over the physical error e, conditioned on s."""

    def __init__(self, cfg: ModelConfig, code):
        super().__init__()
        self.cfg = cfg
        self.n_det = code.n_det
        self.n_err = code.n_err

        # pairwise distances in LATTICE units (xi_s/xi_t read off directly)
        det_c = _lattice_coords(code.det_coords)
        err_c = _lattice_coords(code.err_coords)
        all_c = np.concatenate([det_c, err_c], axis=0)          # (T, 3)
        dist_s = np.sqrt(((all_c[:, None, :2] - all_c[None, :, :2]) ** 2).sum(-1))
        dt_signed = all_c[None, :, 2] - all_c[:, None, 2]       # t_key - t_query
        self.register_buffer("dist_s", torch.as_tensor(dist_s, dtype=torch.float32))
        self.register_buffer("dist_t", torch.as_tensor(np.abs(dt_signed),
                                                       dtype=torch.float32))
        self.register_buffer("dt_signed", torch.as_tensor(dt_signed,
                                                          dtype=torch.float32))

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
        geom = {"dist_s": self.dist_s, "dist_t": self.dist_t,
                "dt_signed": self.dt_signed}
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, geom, use_reentrant=False)
            else:
                x = blk(x, geom)
        x = self.ln_f(x)
        return self.head(x[:, self.n_det:, :]).squeeze(-1)

    def locality_radii(self):
        """Learned per-head (xi_space, xi_time) for every layer."""
        return {f"layer{i}": {
                    "xi_space": blk.attn.xi_space().detach().cpu().numpy(),
                    "xi_time": blk.attn.xi_time().detach().cpu().numpy()}
                for i, blk in enumerate(self.blocks)}
