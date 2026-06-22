"""Diagnose a d>=5 checkpoint's clearing failure: is the iterative unmasker
starving (cleared rises with more infer_steps) or has the model not learned
consistent chains (cleared flat)? And are residual defects spatially clustered
(few hard spots -> fixable) or spread (global incoherence)?

Loads a CURRENT-arch checkpoint (xi floor / aniso), runs evaluate_ler at a
sweep of infer_steps on a FIXED shot set, and reports cleared / strict LER /
defect stats + a per-detector residual heatmap summary.

Usage:
  python scripts/infer_diag.py --run runs/d7_local --shots 20000 \
      --steps 24 48 96 192 --batch 256
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import decode
from qldd.baseline import residual_is_logical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d7_local")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--steps", type=int, nargs="+", default=[24, 48, 96, 192])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(os.path.join(args.run, "ckpt.pt"),
                    map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    step = ck.get("step", "?")
    print(f"loaded {args.run} @ step {step}  d={code.distance} p={code.p} "
          f"shots={args.shots}")

    e, s, l = sample(code, args.shots, seed=12345)   # fixed set across all steps
    H = code.H.astype(np.uint8)

    # accumulate per-detector lit-frequency at the largest step for clustering
    per_det_lit = None
    print(f"{'steps':>6} {'cleared':>8} {'strict':>8} {'obs_LER':>8} "
          f"{'def_mean':>8} {'def|uncl':>9} {'def_p90':>7} {'def_max':>7}")
    rows = []
    with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        for n in args.steps:
            eg = np.zeros_like(e)
            for i in range(0, args.shots, args.batch):
                st = torch.as_tensor(s[i:i+args.batch], dtype=torch.long, device=dev)
                eg[i:i+args.batch] = decode(model, st, n_steps=n).cpu().numpy()
            clears, logical = residual_is_logical(code, e, eg)
            res = (s.astype(np.uint8) ^ ((H @ eg.T) % 2).T.astype(np.uint8))
            defects = res.sum(axis=1)
            row = dict(steps=n, cleared=float(clears.mean()),
                       strict=float((logical | ~clears).mean()),
                       obs_ler=float(logical.mean()),
                       def_mean=float(defects.mean()),
                       def_unclean=float(defects[defects>0].mean()) if (defects>0).any() else 0.0,
                       def_p90=float(np.percentile(defects, 90)),
                       def_max=int(defects.max()))
            rows.append(row)
            print(f"{n:>6} {row['cleared']:>8.4f} {row['strict']:>8.4f} "
                  f"{row['obs_ler']:>8.4f} {row['def_mean']:>8.3f} "
                  f"{row['def_unclean']:>9.3f} {row['def_p90']:>7.1f} {row['def_max']:>7d}")
            if n == args.steps[-1]:
                per_det_lit = res.mean(axis=0)   # fraction of shots each detector stays lit

    # clustering metric: how concentrated is the residual on the detector graph?
    pdl = per_det_lit
    order = np.argsort(pdl)[::-1]
    top = pdl[order]
    total = top.sum()
    if total > 0:
        frac_in_top10pct = top[:max(1, len(top)//10)].sum() / total
    else:
        frac_in_top10pct = 0.0
    print(f"\nresidual concentration: top 10% of detectors carry "
          f"{frac_in_top10pct:.1%} of all residual lit-weight")
    print(f"  (>~0.5 => clustered hard spots [inference-fixable]; "
          f"~0.1 => spread/global [capacity or training])")
    print(f"per-detector lit-frac: max {pdl.max():.3f} mean {pdl.mean():.3f} "
          f"median {np.median(pdl):.3f}")

    if args.out:
        json.dump({"run": args.run, "step": step, "rows": rows,
                   "residual_top10pct_share": float(frac_in_top10pct),
                   "per_det_lit": pdl.tolist()}, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
