"""Train the MWPM-free logical decoder. Dataset = Stim DEM samples (s, l); the
model learns P(logical | syndrome) by BCE. MWPM is computed ONLY at eval as a
baseline, never in training.

Usage: python scripts/train_logical.py --distance 7 --p 0.03 --steps 80000 \
    --batch 1024 --run runs/d7_logical
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from qldd.data import make_code_data, sample
from qldd.model import ModelConfig
from qldd.logical import LogicalDecoder, logical_loss, logical_evaluate
from qldd.baseline import build_matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--p", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_shots", type=int, default=100000)
    ap.add_argument("--causal", action="store_true",
                    help="online decoder: causal-time attention, no future leakage")
    ap.add_argument("--run", default="runs/d7_logical")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run, exist_ok=True)

    code = make_code_data(distance=args.distance,
                          rounds=args.rounds or args.distance, p=args.p)
    matching = build_matching(code)               # eval baseline ONLY
    # token seq is detectors only -> small attention, no grad checkpoint needed
    # ONLINE: causal-time attention + CAUSAL conv stem (past-only temporal
    # padding, sees t-2..t, never t+1). The stem is essential -- without it the
    # causal model sat at chance (0.455 ~ marginal) at d=7; raw attention could
    # not form the local syndrome features from scratch.
    cfg = ModelConfig(d_model=args.d_model, n_heads=args.n_heads,
                      n_layers=args.n_layers, d_ff=4 * args.d_model,
                      use_conv_stem=True, conv_layers=1, xi_min=0.5,
                      xi_space_init=3.0, xi_time_init=3.0,
                      causal_time=args.causal,
                      grad_checkpoint=False)
    model = LogicalDecoder(cfg, code).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    # marginal logical-flip rate (for optional class balancing)
    _, ps, pl = sample(code, 50000, seed=1)
    rate = max(float(pl.mean()), 1e-4)
    pos_weight = (torch.tensor([(1 - rate) / rate], device=dev)
                  if rate < 0.4 else None)
    mode = "ONLINE (causal)" if args.causal else "offline"
    print(f"[setup] {mode} d={args.distance} n_det={code.n_det} n_obs={code.n_obs} "
          f"P(l=1)~{rate:.4f} pos_weight={'none' if pos_weight is None else round(pos_weight.item(),1)} "
          f"conv_stem={cfg.use_conv_stem} params={sum(p.numel() for p in model.parameters()):,} dev={dev}",
          flush=True)

    hist = []
    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        _, s, l = sample(code, args.batch)
        st = torch.as_tensor(s, dtype=torch.long, device=dev)
        lt = torch.as_tensor(l, device=dev)
        loss = logical_loss(model, st, lt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)
        if step > 0 and step % args.eval_every == 0:
            _, s, l = sample(code, args.eval_shots, seed=999)
            rep = logical_evaluate(model, code, s, l, device=dev, batch=8192,
                                   matching=matching,
                                   thresholds=(0.4, 0.5, 0.6))
            rep["step"] = step
            hist.append(rep)
            torch.save({"model": model.state_dict(), "model_cfg": cfg.__dict__,
                        "cfg": {"distance": args.distance,
                                "rounds": args.rounds or args.distance, "p": args.p},
                        "step": step, "history": hist},
                       os.path.join(args.run, "ckpt.pt"))
            json.dump(hist, open(os.path.join(args.run, "history.json"), "w"), indent=2)
            v = "BEATS MWPM" if rep.get("beats_mwpm") else "ties/worse"
            print(f"  [eval] step {step}  model {rep['model_ler']:.5f} "
                  f"@thr{rep['best_thresh']:.1f}  MWPM {rep.get('mwpm_ler', float('nan')):.5f}"
                  f"  <- {v}", flush=True)
        if (time.time() - t0) / 60 > 1320:
            print("[walltime] guard; checkpoint saved, exiting", flush=True)
            break
    print("done")


if __name__ == "__main__":
    main()
