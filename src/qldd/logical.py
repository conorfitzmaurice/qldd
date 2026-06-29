"""MWPM-free maximum-likelihood logical decoder.

Per Frank: an offline decoder's training pipeline is independent of MWPM. The
Stim DEM IS the dataset -- sample (syndrome, logical-flip) pairs directly and
learn P(logical | syndrome) end to end. MWPM appears nowhere in training; it is
only a baseline at eval time.

Target: the true logical observable flip l = L e (n_obs bits). The model reads
the spacetime syndrome and outputs logical-flip logits; argmax (thresh 0.5) is
the Bayes-optimal logical class. This can beat MWPM because MWPM returns the
minimum-WEIGHT correction, not the maximum-LIKELIHOOD logical class (it ignores
degeneracy / coset mass); a net trained on (s, l) learns the posterior directly.

Cheap vs the chain model: the token sequence is the DETECTORS only (n_det), not
detectors + 1216 error mechanisms, so attention is small and no grad
checkpointing is needed.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _lattice_coords, SpacetimeConvStem, Block, ModelConfig


class LogicalDecoder(nn.Module):
    """Syndrome (n_det) -> logical-flip logits (n_obs). conv stem + soft-local
    attention transformer over detector tokens, attention-pooled to n_obs."""

    def __init__(self, cfg: ModelConfig, code):
        super().__init__()
        self.cfg = cfg
        self.n_det = code.n_det
        self.n_obs = code.L.shape[0]

        det_c = _lattice_coords(code.det_coords)
        dist_s = np.sqrt(((det_c[:, None, :2] - det_c[None, :, :2]) ** 2).sum(-1))
        dt = det_c[None, :, 2] - det_c[:, None, 2]
        self.register_buffer("dist_s", torch.as_tensor(dist_s, dtype=torch.float32))
        self.register_buffer("dist_t", torch.as_tensor(np.abs(dt), dtype=torch.float32))
        self.register_buffer("dt_signed", torch.as_tensor(dt, dtype=torch.float32))

        gidx = np.round(det_c).astype(int)
        gidx -= gidx.min(axis=0, keepdims=True)
        grid_shape = tuple((gidx.max(axis=0) + 1).tolist())
        self.det_grid_shape = grid_shape

        dm = cfg.d_model
        self.syn_val = nn.Embedding(2, dm)
        self.coord_proj = nn.Linear(3, dm)
        span = max(float(np.ptp(det_c)), 1.0)
        self.register_buffer("coords", torch.as_tensor(det_c / span, dtype=torch.float32))
        self.stem = (SpacetimeConvStem(cfg, gidx, grid_shape)
                     if cfg.use_conv_stem else None)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(dm)
        self.pool_q = nn.Parameter(torch.randn(self.n_obs, dm) * 0.02)
        self.head = nn.Linear(dm, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        dm = self.cfg.d_model
        x = self.syn_val(s.long()) + self.coord_proj(self.coords).unsqueeze(0)
        if self.stem is not None:
            x = x + self.stem(x)
        geom = {"dist_s": self.dist_s, "dist_t": self.dist_t,
                "dt_signed": self.dt_signed}
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, geom, use_reentrant=False)
            else:
                x = blk(x, geom)
        x = self.ln_f(x)
        q = self.pool_q.unsqueeze(0).expand(x.shape[0], -1, -1)
        attn = torch.softmax(q @ x.transpose(1, 2) / dm ** 0.5, dim=-1)
        pooled = attn @ x
        return self.head(pooled).squeeze(-1)            # (B, n_obs)


def logical_loss(model, s_t, l_t, pos_weight=None):
    """BCE on the true logical flip. No MWPM anywhere."""
    logits = model(s_t)
    return F.binary_cross_entropy_with_logits(
        logits, l_t.float(), pos_weight=pos_weight)


@torch.no_grad()
def logical_evaluate(model, code, s_np, l_np, device="cpu", batch=4096,
                     matching=None, thresholds=(0.5,)):
    """LER of the model's argmax logical prediction. If a matching is passed, also
    report MWPM LER on the SAME shots as a comparison (not used by the model)."""
    model.eval()
    N = s_np.shape[0]
    probs = np.zeros((N, model.n_obs), dtype=np.float32)
    for i in range(0, N, batch):
        st = torch.as_tensor(s_np[i:i+batch], dtype=torch.long, device=device)
        probs[i:i+batch] = torch.sigmoid(model(st)).cpu().numpy()
    L = l_np.astype(np.uint8)
    out = {}
    best = None
    for th in thresholds:
        pred = (probs > th).astype(np.uint8)
        fail = np.any(pred != L, axis=1)
        rec = {"thresh": float(th), "model_ler": float(fail.mean())}
        out[f"{th:.2f}"] = rec
        if best is None or rec["model_ler"] < best["model_ler"]:
            best = rec
    res = {"model_ler": best["model_ler"], "best_thresh": best["thresh"],
           "sweep": out}
    if matching is not None:
        o0 = np.asarray(matching.decode_batch(s_np), dtype=np.uint8)
        if o0.ndim == 1:
            o0 = o0[:, None]
        res["mwpm_ler"] = float(np.any(o0 != L, axis=1).mean())
        res["beats_mwpm"] = bool(res["model_ler"] < res["mwpm_ler"])
    return res
