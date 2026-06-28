"""e0-anchored (observable-frame) decoding.

The offline error-chain model underperforms MWPM at d>=7 because per-bit
commitment yields globally-incoherent chains -> wrong coset. This reformulation
removes chain prediction entirely:

  1. MWPM decodes the syndrome -> base correction with observable flip o0 = M(s).
  2. A network reads the FULL syndrome and predicts a binary residual r in {0,1}^n_obs:
     "does MWPM land in the wrong logical coset on this shot?"
  3. Final logical prediction: o_hat = o0 XOR r_hat.

The target r = l XOR o0 is exactly MWPM's per-shot logical error pattern, so the
network only has to learn the (sparse) cases MWPM gets wrong. It cannot produce a
syndrome-inconsistent result (it never touches the error chain); the only failure
mode is mispredicting the residual, which is a clean supervised binary problem.
Provably lower-bounds at MWPM iff the model predicts r=0 (i.e. trust MWPM).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _lattice_coords, SpacetimeConvStem, Block, ModelConfig


class AnchoredResidualDecoder(nn.Module):
    """Reads the spacetime syndrome, predicts the logical residual vs MWPM.

    Reuses the syndrome encoder (conv stem + soft-local-attention transformer)
    from the chain model, but tokens are the DETECTORS only (n_det), and the
    head pools to n_obs residual logits. No error-mechanism tokens, no masking.
    """

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
        self.pool_q = nn.Parameter(torch.randn(self.n_obs, dm) * 0.02)  # learned obs queries
        self.head = nn.Linear(dm, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, n_det) {0,1} -> residual logits (B, n_obs)."""
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
        x = self.ln_f(x)                                  # (B, n_det, dm)
        # attention pool: learned per-observable query over detector tokens
        q = self.pool_q.unsqueeze(0).expand(x.shape[0], -1, -1)   # (B, n_obs, dm)
        attn = torch.softmax(q @ x.transpose(1, 2) / dm ** 0.5, dim=-1)  # (B,n_obs,n_det)
        pooled = attn @ x                                 # (B, n_obs, dm)
        return self.head(pooled).squeeze(-1)              # (B, n_obs)


def anchored_targets(code, s_np, e_np, l_np, matching):
    """Compute MWPM base obs o0 and residual target r = l XOR o0 (per shot)."""
    o0 = np.asarray(matching.decode_batch(s_np), dtype=np.uint8)
    if o0.ndim == 1:
        o0 = o0[:, None]
    r = (l_np.astype(np.uint8) ^ o0) & 1
    return o0, r


def anchored_loss(model, s_t, r_t, pos_weight=None):
    """BCE on the residual. r is sparse (~MWPM LER), so pos_weight helps."""
    logits = model(s_t)
    return F.binary_cross_entropy_with_logits(
        logits, r_t.float(),
        pos_weight=pos_weight if pos_weight is not None else None)


@torch.no_grad()
def anchored_evaluate(model, code, s_np, l_np, matching, device="cpu",
                      batch=2048, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """LER of o_hat = o0 XOR (residual_prob > thresh) vs MWPM, swept over decision
    thresholds. Higher thresh = trust MWPM more (provably -> MWPM at thresh->1).
    Reports the BEST threshold (max net_fix) plus the full sweep. The threshold is
    a free eval-time knob, no retraining."""
    model.eval()
    o0 = np.asarray(matching.decode_batch(s_np), dtype=np.uint8)
    if o0.ndim == 1:
        o0 = o0[:, None]
    N = s_np.shape[0]
    probs = np.zeros(o0.shape, dtype=np.float32)
    for i in range(0, N, batch):
        st = torch.as_tensor(s_np[i:i+batch], dtype=torch.long, device=device)
        probs[i:i+batch] = torch.sigmoid(model(st)).cpu().numpy()
    mwpm_fail = np.any(o0 != l_np.astype(np.uint8), axis=1)
    mwpm_ler = float(mwpm_fail.mean())
    sweep = {}
    best = {"thresh": 1.0, "anchored_ler": mwpm_ler, "net_fix": 0,
            "n_mwpm_only": 0, "n_anc_only": 0, "residual_pred_rate": 0.0}
    for th in thresholds:
        rhat = (probs > th).astype(np.uint8)
        o_hat = (o0 ^ rhat) & 1
        af = np.any(o_hat != l_np.astype(np.uint8), axis=1)
        both = mwpm_fail.astype(int) - af.astype(int)
        nf = int((both == 1).sum() - (both == -1).sum())
        rec = {"thresh": float(th), "anchored_ler": float(af.mean()),
               "net_fix": nf, "n_mwpm_only": int((both == 1).sum()),
               "n_anc_only": int((both == -1).sum()),
               "residual_pred_rate": float(rhat.mean())}
        sweep[f"{th:.2f}"] = rec
        if nf > best["net_fix"]:
            best = rec
    return {"mwpm_ler": mwpm_ler, "best": best, "sweep": sweep,
            "anchored_ler": best["anchored_ler"], "net_fix": best["net_fix"],
            "residual_pred_rate": best["residual_pred_rate"]}
