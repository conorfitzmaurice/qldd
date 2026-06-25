"""Masked discrete diffusion over the physical error e, conditioned on s.
Masked-diffusion formulation per arXiv:2509.22347 (Eq. A37-A39) but applied to
e instead of the logical error. Train: BCE on masked bits. Decode: iteratively
unmask the most-confident bits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import MASK_TOKEN


def mask_errors(e: torch.Tensor, frac: torch.Tensor, generator=None):
    """Forward process: mask a frac-fraction of bits per row.
    Returns (e_t, mask) with MASK_TOKEN at masked positions."""
    B, n = e.shape
    k = torch.round(frac * n).long().clamp(0, n)            # (B,)
    u = torch.rand(B, n, device=e.device, generator=generator)
    ranks = u.argsort(dim=1).argsort(dim=1)                 # (B,n) ranks in [0,n)
    mask = ranks < k.unsqueeze(1)
    e_t = e.clone().long()
    e_t[mask] = MASK_TOKEN
    return e_t, mask


def diffusion_loss(model, s: torch.Tensor, e: torch.Tensor,
                   weight_by_t: bool = False) -> torch.Tensor:
    """One training step's loss. s: (B, n_det) {0,1}; e: (B, n_err) {0,1}."""
    B, n = e.shape
    # t uniform in {1..n}, frac = t/n
    t = torch.randint(1, n + 1, (B,), device=e.device).float()
    frac = t / n
    e_t, mask = mask_errors(e, frac)
    logits = model(s, e_t)                         # (B, n_err)
    # BCE only on masked positions
    loss_per = F.binary_cross_entropy_with_logits(
        logits, e.float(), reduction="none")       # (B, n_err)
    loss_per = loss_per * mask.float()
    denom = mask.float().sum(dim=1).clamp_min(1.0)
    per_sample = loss_per.sum(dim=1) / denom
    if weight_by_t:
        per_sample = per_sample * (n / t)          # the 1/t-style reweighting
    return per_sample.mean()


@torch.no_grad()
def decode(model, s: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
    """Iterative-unmasking inference. Returns e_guess: (B, n_err) {0,1}.
    n_steps trades speed for accuracy (1 = unmask everything at once)."""
    model.eval()
    B = s.shape[0]
    n = model.n_err
    if n_steps is None:
        n_steps = n
    n_steps = max(1, min(n_steps, n))
    e_t = torch.full((B, n), MASK_TOKEN, dtype=torch.long, device=s.device)
    remaining = torch.full((B,), n, dtype=torch.long, device=s.device)

    for step in range(n_steps):
        logits = model(s, e_t)                     # (B, n_err)
        prob1 = torch.sigmoid(logits)
        pred = (prob1 > 0.5).long()
        conf = torch.maximum(prob1, 1 - prob1)     # confidence in [0.5, 1]
        is_masked = (e_t == MASK_TOKEN)
        conf = conf.masked_fill(~is_masked, -1.0)  # only consider masked slots
        k = int(np.ceil(n / n_steps))              # unmask k per step
        idx = torch.topk(conf, min(k, n), dim=1).indices       # (B, k)
        sel = torch.zeros_like(e_t, dtype=torch.bool)
        sel.scatter_(1, idx, True)
        sel &= is_masked                           # only commit genuinely-masked
        e_t = torch.where(sel, pred, e_t)
    # any still-masked (shouldn't happen) -> threshold
    still = (e_t == MASK_TOKEN)
    if still.any():
        logits = model(s, e_t)
        e_t[still] = (torch.sigmoid(logits)[still] > 0.5).long()
    return e_t.to(torch.uint8)


@torch.no_grad()
def decode_constrained(model, s: torch.Tensor, code, n_steps: Optional[int] = None,
                       project_every: int = 1, final_project: bool = True
                       ) -> torch.Tensor:
    """Syndrome-constrained iterative unmasking.

    The plain `decode` learns per-bit marginals but never enforces H e = s, so
    at d>=7 the committed chain almost never clears the syndrome. Here, after
    unmasking a batch of bits, we periodically PROJECT the current commitment
    back toward syndrome-consistency by greedily flipping committed bits that
    reduce the residual detector weight. Later unmasking steps then condition on
    a (more) consistent partial state instead of compounding inconsistency.

    H, s as torch on the model device; greedy flip step is vectorized over the
    batch. final_project guarantees H e = s on exit (cleared = 1 by construction)
    when the residual is reachable by single-bit flips.
    """
    model.eval()
    dev = s.device
    B = s.shape[0]
    n = model.n_err
    if n_steps is None:
        n_steps = n
    n_steps = max(1, min(n_steps, n))
    H = torch.as_tensor(code.H, dtype=torch.uint8, device=dev)        # (D, n)
    s_u = s.to(torch.uint8)

    Hf = H.float()
    def residual(e_bits):
        He = torch.remainder((e_bits.float() @ Hf.T).round().long(), 2)
        return (s_u.int() ^ He.int()) & 1

    def greedy_clear(e_bits, committed, iters):
        """Clear residual by walking error chains. For each lit detector, flip a
        committed bit incident to it that doesn't create more new lit detectors
        than it clears -- this escapes the single-flip local minima that trap a
        pure net-weight-descent greedy (where clearing needs a flip that is
        locally weight-neutral). Vectorized over the batch."""
        for _ in range(iters):
            r = residual(e_bits)
            w = r.sum(dim=1)
            if int(w.max()) == 0:
                break
            rf = r.float()
            lit = (rf @ Hf)
            unlit = ((1.0 - rf) @ Hf)
            avail = committed & (lit > 0)               # committed bits touching a lit det
            # score: clear as many lit as possible while lighting few new ones
            score = (lit - unlit).float()
            score = score.masked_fill(~avail, -1e9)
            best = score.argmax(dim=1)
            # accept if the chosen bit touches >=1 lit detector (guaranteed
            # progress on that detector even if net weight is flat)
            ok = avail.gather(1, best.unsqueeze(1)).squeeze(1)
            flip = torch.zeros_like(e_bits, dtype=torch.bool)
            flip.scatter_(1, best.unsqueeze(1), True)
            flip &= ok.unsqueeze(1)
            if not bool(flip.any()):
                break
            e_bits = torch.where(flip, 1 - e_bits, e_bits)
        return e_bits

    e_t = torch.full((B, n), MASK_TOKEN, dtype=torch.long, device=dev)
    for step in range(n_steps):
        logits = model(s, e_t)
        prob1 = torch.sigmoid(logits)
        pred = (prob1 > 0.5).long()
        conf = torch.maximum(prob1, 1 - prob1)
        is_masked = (e_t == MASK_TOKEN)
        conf = conf.masked_fill(~is_masked, -1.0)
        k = int(np.ceil(n / n_steps))
        idx = torch.topk(conf, min(k, n), dim=1).indices
        sel = torch.zeros_like(e_t, dtype=torch.bool)
        sel.scatter_(1, idx, True)
        sel &= is_masked
        e_t = torch.where(sel, pred, e_t)
        # project committed bits toward consistency so later steps condition on it
        if project_every and (step % project_every == 0):
            committed = (e_t != MASK_TOKEN)
            e_bits = e_t.clamp(max=1).to(torch.uint8)
            e_bits = greedy_clear(e_bits, committed, iters=4)
            e_t = torch.where(committed, e_bits.long(), e_t)

    still = (e_t == MASK_TOKEN)
    if still.any():
        logits = model(s, e_t)
        e_t[still] = (torch.sigmoid(logits)[still] > 0.5).long()
    e_bits = e_t.to(torch.uint8)
    if final_project:
        committed = torch.ones_like(e_bits, dtype=torch.bool)
        e_bits = greedy_clear(e_bits, committed, iters=n)  # full clear pass
    return e_bits


@torch.no_grad()
def evaluate_ler(model, code, s_np, e_np, l_np, n_steps=None, device="cpu",
                 batch=512) -> dict:
    """LER of the diffusion decoder + residual diagnostics; comparable to the
    MWPM baseline numbers."""
    from .baseline import residual_is_logical
    model.eval()
    H = torch.as_tensor(code.H, dtype=torch.float32, device=device)
    L = torch.as_tensor(code.L, dtype=torch.float32, device=device)
    N = s_np.shape[0]
    n_fail = 0
    n_clear = 0
    n_stab = 0
    eg_all = np.zeros_like(e_np)
    for i in range(0, N, batch):
        s = torch.as_tensor(s_np[i:i+batch], dtype=torch.long, device=device)
        eg = decode(model, s, n_steps=n_steps)     # (b, n_err) uint8
        eg_all[i:i+batch] = eg.cpu().numpy()
    clears, logical = residual_is_logical(code, e_np, eg_all)
    # residual spacetime defects: detectors still lit after the correction,
    # i.e. weight of s XOR H e_guess. clears == (defects == 0).
    res_syn = (s_np.astype(np.uint8) ^ ((code.H @ eg_all.T) % 2).T.astype(np.uint8))
    defects = res_syn.sum(axis=1).astype(int)
    n_fail = int(logical.sum())
    n_clear = int(clears.sum())
    n_stab = int((clears & ~logical).sum())
    # observable-based LER (matches MWPM's only failure mode)
    ler = n_fail / N
    # strict LER: not clearing the syndrome also counts as failure
    strict_fail = logical | (~clears)
    ler_strict = float(strict_fail.mean())
    stderr = float(np.sqrt(max(ler * (1 - ler), 1e-12) / N))
    return {
        "ler": ler, "ler_strict": ler_strict, "stderr": stderr,
        "n_fail": n_fail, "shots": N,
        "syndrome_cleared_frac": n_clear / N,
        "pure_stabilizer_residual_frac": n_stab / N,
        "residual_defects_mean": float(defects.mean()),
        "residual_defects_p90": float(np.percentile(defects, 90)),
        "residual_defects_max": int(defects.max()),
        "residual_defects_mean_given_unclean": (
            float(defects[defects > 0].mean()) if (defects > 0).any() else 0.0),
    }
