"""Effective-range analysis.

The locality measurement is confounded if the conv stem's receptive field
already spans the code, so report total_range = max(conv_RF, xi) against the
grid width. xi (the soft-attention decay length) is already in lattice units.
Clean locality runs need the conv stem off.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import LocalDiffusionDecoder


def conv_receptive_field_voxels(model: LocalDiffusionDecoder) -> dict:
    """Empirical RF of the conv stem per spatial axis, via the gradient support
    of a single output voxel (bias-independent)."""
    if model.stem is None:
        return {"conv_stem": False}
    stem = model.stem
    X, Y, T = stem.grid_shape
    dm = model.cfg.d_model
    dev = next(stem.conv.parameters()).device
    grid = torch.zeros(1, dm, X, Y, T, requires_grad=True, device=dev)
    out = stem.conv(grid)                           # (1, dm, X, Y, T)
    cx, cy, ct = X // 2, Y // 2, T // 2
    out[:, :, cx, cy, ct].sum().backward()
    g = grid.grad.abs().sum(dim=(0, 1)).cpu().numpy()  # (X, Y, T) support
    def diameter(mask_axis):
        nz = np.nonzero(mask_axis)[0]
        return int(nz.max() - nz.min() + 1) if len(nz) else 0
    rf_x = diameter(g.sum(axis=(1, 2)) > 0)
    rf_y = diameter(g.sum(axis=(0, 2)) > 0)
    rf_t = diameter(g.sum(axis=(0, 1)) > 0)
    n_conv = sum(1 for m in stem.conv if isinstance(m, torch.nn.Conv3d))
    analytic = 2 * n_conv + 1
    return {"conv_stem": True, "n_conv_layers": n_conv,
            "rf_voxels": {"x": rf_x, "y": rf_y, "t": rf_t},
            "rf_analytic": analytic, "grid_shape": (X, Y, T)}


def grid_spatial_width(model: LocalDiffusionDecoder) -> int:
    X, Y, _ = model.det_grid_shape     # geometry, available even with conv stem off
    return int(max(X, Y))


@torch.no_grad()
def attention_range_lattice(model: LocalDiffusionDecoder) -> dict:
    """xi per layer (lattice units): the exp(-d/xi) decay length, per head.
    If a hard window_radius is set, the effective range is capped by it."""
    cap = model.cfg.window_radius
    out = {}
    for i, blk in enumerate(model.blocks):
        xs = blk.attn.xi_space().detach().cpu().numpy()
        xt = blk.attn.xi_time().detach().cpu().numpy()
        eff = np.minimum(xs, cap) if cap is not None else xs
        out[f"layer{i}"] = {
            "xi_space_mean": float(xs.mean()), "xi_space_min": float(xs.min()),
            "xi_time_mean": float(xt.mean()), "xi_time_min": float(xt.min()),
            "sigma_lattice_mean": float(eff.mean()),   # spatial range; key kept for aggregators
        }
    return out


def locality_report(model: LocalDiffusionDecoder, code) -> dict:
    rf = conv_receptive_field_voxels(model)
    width = grid_spatial_width(model)
    att = attention_range_lattice(model)
    # binding attention range = smallest layer effective xi
    sig_latt = [v["sigma_lattice_mean"] for v in att.values()
                if v["sigma_lattice_mean"] is not None]
    att_range = float(np.min(sig_latt)) if sig_latt else None

    conv_global = False
    conv_rf = None
    if rf.get("conv_stem"):
        conv_rf = max(rf["rf_voxels"]["x"], rf["rf_voxels"]["y"])
        conv_global = conv_rf >= width      # RF saturates the code spatially

    total_range = None
    if conv_rf is not None and att_range is not None:
        total_range = max(conv_rf, att_range)
    elif att_range is not None:
        total_range = att_range

    meaningful = (not conv_global) and (width is not None) and (width >= 3)
    return {
        "distance": code.distance,
        "grid_width_voxels": width,
        "conv_rf_voxels": conv_rf,
        "conv_is_global": conv_global,
        "attention_range_lattice": att_range,
        "total_effective_range_lattice": total_range,
        "locality_test_meaningful": bool(meaningful),
        "note": ("conv RF saturates the code; rerun with use_conv_stem=false"
                 if conv_global else
                 "conv RF is sub-global; total range reflects learned locality"),
        "attention_detail": att,
        "conv_detail": rf,
    }
