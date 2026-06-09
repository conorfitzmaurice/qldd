"""Local diffusion decoder: 3D-conv stem (arXiv:2604.08358) + local-attention
transformer with a trainable per-head locality radius sigma, denoising the
physical error e conditioned on the syndrome s. Predicting e (not the logical
class, as in 2509.22347 / 2604.24640) is what makes a local recovery map
possible; sigma vs code distance is the locality measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_TOKEN = 2  # value embedding index for a masked error bit (0, 1, or MASK)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    conv_channels: int = 32
    conv_layers: int = 2
    dropout: float = 0.0
    use_conv_stem: bool = True
    sigma_init: float = 2.0      # initial locality radius (in normalized coords)
    window_radius: float = None  # hard attention window (lattice units); None = dense O(T^2)


def _normalize_coords(coords: np.ndarray):
    # map raw spacetime coords to unit scale; NaN -> 0
    c = coords.copy()
    nan = np.isnan(c).any(axis=1)
    c[nan] = 0.0
    lo = c.min(axis=0, keepdims=True)
    hi = c.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    cn = (c - lo) / span
    return cn.astype(np.float32), nan


class SpacetimeConvStem(nn.Module):
    """Scatter detector embeddings onto the (X,Y,T) voxel grid, run 3D convs
    (arXiv:2604.08358 style), gather back. conv_layers sets the local RF;
    keep it small to force the attention (and hence sigma) to carry the range.
    """

    def __init__(self, cfg: ModelConfig, det_grid_idx: np.ndarray, grid_shape):
        super().__init__()
        self.cfg = cfg
        self.grid_shape = grid_shape  # (X, Y, T)
        # integer grid index (n_det, 3) for scatter/gather
        self.register_buffer("gidx", torch.as_tensor(det_grid_idx, dtype=torch.long))
        ch = cfg.conv_channels
        layers = [nn.Conv3d(cfg.d_model, ch, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(cfg.conv_layers - 1):
            layers += [nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.GELU()]
        layers += [nn.Conv3d(ch, cfg.d_model, kernel_size=3, padding=1)]
        self.conv = nn.Sequential(*layers)

    def forward(self, det_feats: torch.Tensor) -> torch.Tensor:
        # det_feats: (B, n_det, d_model)
        B, n_det, dm = det_feats.shape
        X, Y, T = self.grid_shape
        grid = det_feats.new_zeros(B, dm, X, Y, T)
        gx, gy, gt = self.gidx[:, 0], self.gidx[:, 1], self.gidx[:, 2]
        # scatter: place each detector's feature at its voxel
        grid[:, :, gx, gy, gt] = det_feats.transpose(1, 2)
        grid = grid + self.conv(grid)            # residual spacetime conv
        gathered = grid[:, :, gx, gy, gt].transpose(1, 2)  # (B, n_det, d_model)
        return gathered


class LocalMultiHeadAttention(nn.Module):
    """MHA with a per-head trainable distance bias -D^2/(2 sigma^2).

    Dense mode is O(T^2) regardless of sigma (heavy at d=7, T~1.5k); windowed
    mode is O(T*K). sigma measures locality, it doesn't buy speed in dense mode.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        # raw_sigma -> sigma = softplus(raw_sigma); one per head (the locality radius)
        inv = float(np.log(np.expm1(max(cfg.sigma_init, 1e-3))))
        self.raw_sigma = nn.Parameter(torch.full((cfg.n_heads,), inv))
        self.drop = nn.Dropout(cfg.dropout)

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma)  # (h,)

    def forward(self, x: torch.Tensor, geom) -> torch.Tensor:
        if geom["mode"] == "dense":
            return self._forward_dense(x, geom["d2"])
        return self._forward_windowed(x, geom["nbr_idx"], geom["nbr_mask"],
                                      geom["nbr_d2"])

    def _forward_dense(self, x: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model);  d2: (T, T) squared spacetime distances. O(T^2).
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, T, dk)
        scores = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)  # (B, h, T, T)
        sig = self.sigma().view(1, self.h, 1, 1)
        bias = -d2.view(1, 1, T, T) / (2.0 * sig ** 2 + 1e-9)
        attn = torch.softmax(scores + bias, dim=-1)
        attn = self.drop(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.out(ctx)

    def _forward_windowed(self, x, nbr_idx, nbr_mask, nbr_d2) -> torch.Tensor:
        # Sparse local attention: each token attends only to its <=K neighbors
        # within the spacetime window. Memory/compute O(T*K), K << T.
        B, T, _ = x.shape
        K = nbr_idx.shape[1]
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, T, dk)
        k_nbr = k[:, :, nbr_idx, :]                    # (B, h, T, K, dk)
        v_nbr = v[:, :, nbr_idx, :]                    # (B, h, T, K, dk)
        scores = (q.unsqueeze(3) * k_nbr).sum(-1) / (self.dk ** 0.5)  # (B,h,T,K)
        sig = self.sigma().view(1, self.h, 1, 1)
        bias = -nbr_d2.view(1, 1, T, K) / (2.0 * sig ** 2 + 1e-9)
        scores = scores + bias
        scores = scores.masked_fill(~nbr_mask.view(1, 1, T, K), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = (attn.unsqueeze(-1) * v_nbr).sum(3)      # (B, h, T, dk)
        ctx = ctx.transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.out(ctx)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LocalMultiHeadAttention(cfg)
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
        T = self.n_det + self.n_err

        # pairwise squared spacetime distances
        det_c, _ = _normalize_coords(code.det_coords)
        err_c, _ = _normalize_coords(code.err_coords)
        all_c = np.concatenate([det_c, err_c], axis=0)          # (T, 3)
        d2 = ((all_c[:, None, :] - all_c[None, :, :]) ** 2).sum(-1)
        self.register_buffer("d2", torch.as_tensor(d2, dtype=torch.float32))

        # optional sparse-attention neighbor structure
        self.windowed = cfg.window_radius is not None
        if self.windowed:
            dc = code.det_coords
            gw = int(max(np.ptp(np.round(dc[:, 0] / 2).astype(int)),
                         np.ptp(np.round(dc[:, 1] / 2).astype(int))) + 1)
            span = max(gw - 1, 1)
            r_norm2 = (cfg.window_radius / span) ** 2
            within = d2 <= r_norm2 + 1e-9                        # (T, T) bool
            Kmax = int(within.sum(1).max())
            T = within.shape[0]
            nbr_idx = np.zeros((T, Kmax), dtype=np.int64)
            nbr_mask = np.zeros((T, Kmax), dtype=bool)
            nbr_d2 = np.zeros((T, Kmax), dtype=np.float32)
            for i in range(T):
                js = np.nonzero(within[i])[0]
                nbr_idx[i, :len(js)] = js
                nbr_mask[i, :len(js)] = True
                nbr_d2[i, :len(js)] = d2[i, js]
            self.register_buffer("nbr_idx", torch.as_tensor(nbr_idx))
            self.register_buffer("nbr_mask", torch.as_tensor(nbr_mask))
            self.register_buffer("nbr_d2", torch.as_tensor(nbr_d2, dtype=torch.float32))
            self.window_K = Kmax

        # detector voxel grid index for the conv stem
        dc = code.det_coords
        gx = np.round(dc[:, 0] / 2).astype(int)
        gy = np.round(dc[:, 1] / 2).astype(int)
        gt = np.round(dc[:, 2]).astype(int)
        gidx = np.stack([gx, gy, gt], axis=1)
        gidx -= gidx.min(axis=0, keepdims=True)
        grid_shape = tuple((gidx.max(axis=0) + 1).tolist())
        self.det_grid_shape = grid_shape   # stored regardless of conv stem (geometry)

        dm = cfg.d_model
        # value embeddings: syndrome {0,1}; error {0,1,MASK}
        self.syn_val = nn.Embedding(2, dm)
        self.err_val = nn.Embedding(3, dm)
        self.type_emb = nn.Embedding(2, dm)       # 0 = detector, 1 = error
        self.coord_proj = nn.Linear(3, dm)        # continuous coordinate features
        self.register_buffer("coords", torch.as_tensor(all_c, dtype=torch.float32))

        self.stem = (SpacetimeConvStem(cfg, gidx, grid_shape)
                     if cfg.use_conv_stem else None)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, 1)              # per-error-token logit for e_j=1

    def forward(self, s: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """s: (B, n_det) in {0,1}; e_t: (B, n_err) in {0,1,2(=MASK)}.
        Returns logits (B, n_err) for e_j = 1."""
        B = s.shape[0]
        dm = self.cfg.d_model
        det = self.syn_val(s.long()) + self.type_emb.weight[0].view(1, 1, dm)
        det = det + self.coord_proj(self.coords[: self.n_det]).unsqueeze(0)
        if self.stem is not None:
            det = det + self.stem(det)
        err = self.err_val(e_t.long()) + self.type_emb.weight[1].view(1, 1, dm)
        err = err + self.coord_proj(self.coords[self.n_det:]).unsqueeze(0)

        x = torch.cat([det, err], dim=1)          # (B, T, d_model)
        if self.windowed:
            geom = {"mode": "windowed", "nbr_idx": self.nbr_idx,
                    "nbr_mask": self.nbr_mask, "nbr_d2": self.nbr_d2}
        else:
            geom = {"mode": "dense", "d2": self.d2}
        for blk in self.blocks:
            x = blk(x, geom)
        x = self.ln_f(x)
        err_out = x[:, self.n_det:, :]            # (B, n_err, d_model)
        return self.head(err_out).squeeze(-1)     # (B, n_err)

    def locality_radii(self):
        """Current per-head sigma for every layer -- the learned locality scale."""
        return {f"layer{i}": blk.attn.sigma().detach().cpu().numpy()
                for i, blk in enumerate(self.blocks)}
