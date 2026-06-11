"""Training harness. Data is sampled fresh from Stim every step; checkpoints
every ckpt_every steps plus a max_minutes wall-clock guard so requeued jobs
resume cleanly. Run: python -m qldd.train --config configs/d3_mig.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .data import make_code_data, sample
from .model import LocalDiffusionDecoder, ModelConfig
from .diffusion import diffusion_loss, evaluate_ler
from .baseline import build_matching, logical_error_rate
from .analysis import locality_report


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * prog))


def save_ckpt(path, model, opt, step, history, mcfg, cfg):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "step": step, "history": history,
        "model_cfg": asdict(mcfg), "cfg": cfg,
    }, tmp)
    os.replace(tmp, path)  # atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = ("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
              else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    matching = build_matching(code)
    mwpm = logical_error_rate(code, cfg.get("eval_shots", 20000), matching, seed=12345)

    mcfg = ModelConfig(**cfg.get("model", {}))
    model = LocalDiffusionDecoder(mcfg, code).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 1e-4))

    total = cfg["steps"]
    start_step = 0
    history = []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("model_cfg") != asdict(mcfg):
            print("[resume] checkpoint model_cfg differs from config; starting fresh",
                  flush=True)
        else:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_step = ck["step"] + 1; history = ck["history"]
            print(f"[resume] from step {start_step}", flush=True)

    print(f"[setup] device={device} d={code.distance} n_err={code.n_err} "
          f"params={sum(p.numel() for p in model.parameters())} "
          f"MWPM_LER={mwpm['ler']:.4f}", flush=True)

    bs = cfg["batch_size"]
    t_start = time.time()
    max_minutes = cfg.get("max_minutes", 1e9)

    for step in range(start_step, total):
        lr = cosine_lr(step, total, cfg["lr"], cfg.get("min_lr", cfg["lr"] * 0.05),
                       cfg.get("warmup", 200))
        for g in opt.param_groups:
            g["lr"] = lr
        e, s, _ = sample(code, bs)
        st = torch.as_tensor(s, dtype=torch.long, device=device)
        ee = torch.as_tensor(e, dtype=torch.long, device=device)
        loss = diffusion_loss(model, st, ee, weight_by_t=cfg.get("weight_by_t", False))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        opt.step()

        if step % cfg.get("log_every", 200) == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}", flush=True)

        if step > 0 and step % cfg.get("eval_every", 2000) == 0:
            e, s, l = sample(code, cfg.get("eval_shots", 20000), seed=999)
            rep = evaluate_ler(model, code, s, e, l,
                               n_steps=cfg.get("infer_steps"), device=device,
                               batch=cfg.get("eval_batch", 128))
            rads = model.locality_radii()
            xi_s = float(np.mean([np.mean(v["xi_space"]) for v in rads.values()]))
            xi_t = float(np.mean([np.mean(v["xi_time"]) for v in rads.values()]))
            loc = locality_report(model, code)
            rec = {"step": step, "loss": loss.item(),
                   "diff_ler": rep["ler"], "diff_ler_strict": rep["ler_strict"],
                   "mwpm_ler": mwpm["ler"],
                   "cleared": rep["syndrome_cleared_frac"],
                   "stab_residual": rep["pure_stabilizer_residual_frac"],
                   "defects_mean": rep["residual_defects_mean"],
                   "defects_p90": rep["residual_defects_p90"],
                   "defects_mean_unclean": rep["residual_defects_mean_given_unclean"],
                   "xi_space_mean": xi_s, "xi_time_mean": xi_t,
                   "total_range_lattice": loc["total_effective_range_lattice"],
                   "grid_width": loc["grid_width_voxels"],
                   "conv_is_global": loc["conv_is_global"],
                   "locality_meaningful": loc["locality_test_meaningful"]}
            history.append(rec)
            print(f"[eval] step {step}  diff_LER {rep['ler']:.4f}  "
                  f"MWPM {mwpm['ler']:.4f}  cleared {rep['syndrome_cleared_frac']:.3f}  "
                  f"xi_s~{xi_s:.2f} xi_t~{xi_t:.2f} "
                  f"defects~{rep['residual_defects_mean']:.2f} "
                  f"range~{loc['total_effective_range_lattice']:.2f}/{loc['grid_width_voxels']}"
                  f"{'  [CONFOUNDED]' if loc['conv_is_global'] else ''}", flush=True)
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

        if step % cfg.get("ckpt_every", 2000) == 0 and step > start_step:
            save_ckpt(ckpt_path, model, opt, step, history, mcfg, cfg)

        if (time.time() - t_start) / 60.0 > max_minutes:
            print(f"[walltime] guard hit at step {step}; checkpointing & exiting "
                  f"for requeue.", flush=True)
            save_ckpt(ckpt_path, model, opt, step, history, mcfg, cfg)
            return

    save_ckpt(ckpt_path, model, opt, total - 1, history, mcfg, cfg)
    open(os.path.join(run_dir, "DONE"), "w").close()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
