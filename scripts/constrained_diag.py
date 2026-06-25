"""Compare inference schemes on a trained checkpoint:
  (A) plain decode (per-bit unmasking, no constraint)
  (B) final projection only (greedy clear after decode -- the Stage-1 recipe)
  (C) in-loop projection (project committed bits every step + final)

Reports cleared / strict LER / observable LER / defect stats for each. The
question: does enforcing H e = s during/after generation rescue clearing at
d=7, and does the resulting chain land in the right logical coset (strict LER)?

Usage: python scripts/constrained_diag.py --run runs/d7_local --shots 3000 --steps 48
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import decode, decode_constrained, decode_match_projected
from qldd.baseline import residual_is_logical, build_matching, logical_error_rate


def stats(code, e_true, e_guess, s):
    clears, logical = residual_is_logical(code, e_true, e_guess)
    H = code.H.astype(np.uint8)
    res = (s.astype(np.uint8) ^ ((H @ e_guess.T) % 2).T.astype(np.uint8))
    d = res.sum(axis=1)
    return dict(cleared=float(clears.mean()),
                strict=float((logical | ~clears).mean()),
                obs=float(logical.mean()),
                dmean=float(d.mean()),
                dmax=int(d.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d7_local")
    ap.add_argument("--shots", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    matching = build_matching(code)
    mwpm = logical_error_rate(code, args.shots, matching, seed=7)["ler"]
    e, s, l = sample(code, args.shots, seed=4321)
    print(f"{args.run} @ step {ck.get('step','?')}  d={code.distance} "
          f"p={code.p}  MWPM {mwpm:.4f}  shots={args.shots} steps={args.steps}")

    def run(tag, fn):
        t0 = time.time()
        eg = np.zeros_like(e)
        for i in range(0, args.shots, args.batch):
            st = torch.as_tensor(s[i:i+args.batch], dtype=torch.long, device=dev)
            eg[i:i+args.batch] = fn(st).cpu().numpy()
        r = stats(code, e, eg, s)
        print(f"  {tag:<22} cleared {r['cleared']:.4f}  strict {r['strict']:.4f}  "
              f"obs {r['obs']:.4f}  def {r['dmean']:.2f} (max {r['dmax']})  "
              f"[{time.time()-t0:.0f}s]")

    run("(A) plain",
        lambda st: decode(model, st, n_steps=args.steps))
    run("(B) final-proj only",
        lambda st: decode_constrained(model, st, code, n_steps=args.steps,
                                      project_every=0, final_project=True))
    run("(C) in-loop proj",
        lambda st: decode_constrained(model, st, code, n_steps=args.steps,
                                      project_every=1, final_project=True))

    # (D) MWPM-on-residual: minimum-weight completion (observable space)
    t0 = time.time()
    obs_pred, cleared = decode_match_projected(model, s, code, matching,
                                               n_steps=args.steps, device=dev,
                                               batch=args.batch)
    strict = float(np.any(obs_pred != l, axis=1).mean())
    print(f"  {'(D) MWPM-resid proj':<22} cleared {cleared.mean():.4f}  "
          f"strict {strict:.4f}  obs {strict:.4f}  def 0.00 (max 0)  "
          f"[{time.time()-t0:.0f}s]   <- vs MWPM {mwpm:.4f}")


if __name__ == "__main__":
    main()
