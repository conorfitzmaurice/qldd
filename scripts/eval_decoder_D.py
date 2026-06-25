"""Definitive evaluation of decode scheme (D) = model + MWPM-on-residual, at high
shot count. (D) provably lower-bounds at MWPM; this measures whether a trained
checkpoint pushes strictly BELOW MWPM at d=7 (i.e. the model adds coset info).

Reports (D) strict LER, plain MWPM on the same shots, and the paired difference
with a binomial stderr, so 'beats MWPM' is a statistically grounded statement.

Usage: python scripts/eval_decoder_D.py --run runs/d7_conv --shots 200000 --steps 48
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import decode_match_projected
from qldd.baseline import build_matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d7_conv")
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    matching = build_matching(code)

    e, s, l = sample(code, args.shots, seed=20260625)
    # plain MWPM on the same shots (paired)
    pred = np.asarray(matching.decode_batch(s), dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    mwpm_fail = np.any(pred != l, axis=1)

    t0 = time.time()
    with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        obs_pred, cleared = decode_match_projected(
            model, s, code, matching, n_steps=args.steps, device=dev,
            batch=args.batch)
    d_fail = np.any(obs_pred != l, axis=1)

    mwpm_ler = mwpm_fail.mean()
    d_ler = d_fail.mean()
    # paired difference: shots where they differ
    both = mwpm_fail.astype(int) - d_fail.astype(int)
    n_mwpm_only = int((both == 1).sum())   # MWPM failed, D succeeded
    n_d_only = int((both == -1).sum())     # D failed, MWPM succeeded
    se = np.sqrt(max(d_ler * (1 - d_ler), 1e-12) / args.shots)

    print(f"=== decode (D) vs MWPM @ d={code.distance} p={code.p}, "
          f"step {ck.get('step','?')}, {args.shots} shots ===")
    print(f"  MWPM           LER {mwpm_ler:.5f}")
    print(f"  (D) model+MWPM LER {d_ler:.5f}  (stderr {se:.5f})  cleared {cleared.mean():.4f}")
    print(f"  difference     {d_ler - mwpm_ler:+.5f}  "
          f"({'D BEATS MWPM' if d_ler < mwpm_ler else 'MWPM better or tied'})")
    print(f"  paired: D fixed {n_mwpm_only} MWPM-failures; D broke {n_d_only} MWPM-successes "
          f"(net {n_mwpm_only - n_d_only:+d})")
    print(f"  [{time.time()-t0:.0f}s, {dev}]")

    if args.out:
        json.dump({"run": args.run, "step": ck.get("step"),
                   "shots": args.shots, "mwpm_ler": float(mwpm_ler),
                   "d_ler": float(d_ler), "d_stderr": float(se),
                   "diff": float(d_ler - mwpm_ler),
                   "n_mwpm_only": n_mwpm_only, "n_d_only": n_d_only,
                   "cleared": float(cleared.mean())}, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
