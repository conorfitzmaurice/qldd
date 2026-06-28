"""Train the e0-anchored residual decoder: predict whether MWPM gets the coset
wrong, from the syndrome. Provably >= MWPM at the trust-MWPM (r=0) solution, so
the question is whether it learns to FIX MWPM's coset errors net-positive.

Usage: python scripts/train_anchored.py --distance 7 --p 0.03 --steps 40000
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from qldd.data import make_code_data, sample
from qldd.model import ModelConfig
from qldd.anchored import (AnchoredResidualDecoder, anchored_targets,
                           anchored_loss, anchored_evaluate)
from qldd.baseline import build_matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--p", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_shots", type=int, default=50000)
    ap.add_argument("--run", default="runs/d7_anchored")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run, exist_ok=True)

    code = make_code_data(distance=args.distance,
                          rounds=args.rounds or args.distance, p=args.p)
    matching = build_matching(code)
    cfg = ModelConfig(d_model=args.d_model, n_heads=args.n_heads,
                      n_layers=args.n_layers, d_ff=4 * args.d_model,
                      use_conv_stem=True, conv_layers=1, xi_min=0.5,
                      xi_space_init=3.0, xi_time_init=3.0)
    model = AnchoredResidualDecoder(cfg, code).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    # residual is sparse (~MWPM LER); upweight positives so the model doesn't
    # collapse to the trivial r=0 (which only matches MWPM, never beats it).
    pe, ps, pl = sample(code, 20000, seed=1)
    _, r0 = anchored_targets(code, ps, pe, pl, matching)
    pos_rate = max(float(r0.mean()), 1e-4)
    # sqrt of the inverse-frequency ratio: enough to escape the trivial
    # r=0 basin without making the model flip-happy (full ratio overshoots).
    pos_weight = torch.tensor([np.sqrt((1 - pos_rate) / pos_rate)], device=dev)
    print(f"[setup] d={args.distance} n_det={code.n_det} n_obs={code.n_obs} "
          f"MWPM_LER~{pos_rate:.4f} pos_weight={pos_weight.item():.1f} "
          f"params={sum(p.numel() for p in model.parameters()):,} dev={dev}")

    hist = []
    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        e, s, l = sample(code, args.batch)
        _, r = anchored_targets(code, s, e, l, matching)
        st = torch.as_tensor(s, dtype=torch.long, device=dev)
        rt = torch.as_tensor(r, device=dev)
        loss = anchored_loss(model, st, rt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)
        if step > 0 and step % args.eval_every == 0:
            e, s, l = sample(code, args.eval_shots, seed=999)
            rep = anchored_evaluate(model, code, s, l, matching, device=dev, batch=4096)
            rep["step"] = step
            hist.append(rep)
            torch.save({"model": model.state_dict(),
                        "model_cfg": cfg.__dict__,
                        "cfg": {"distance": args.distance,
                                "rounds": args.rounds or args.distance,
                                "p": args.p},
                        "step": step, "history": hist},
                       os.path.join(args.run, "ckpt.pt"))
            json.dump(hist, open(os.path.join(args.run, "history.json"), "w"), indent=2)
            b = rep["best"]
            verdict = "BEATS" if b["anchored_ler"] < rep["mwpm_ler"] else "ties/worse"
            print(f"  [eval] step {step}  anchored {b['anchored_ler']:.5f} "
                  f"@thr{b['thresh']:.2f}  MWPM {rep['mwpm_ler']:.5f}  "
                  f"net_fix {b['net_fix']:+d}  pred_rate {b['residual_pred_rate']:.4f}  "
                  f"<- {verdict}", flush=True)
        if (time.time() - t0) / 60 > 1320:
            print("[walltime] guard; checkpoint saved, exiting", flush=True)
            break
    print("done")


if __name__ == "__main__":
    main()
