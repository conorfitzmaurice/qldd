"""LER vs p sweep for a trained checkpoint (inference only; fp16/TF32 fast path).

Tests generalization: the Stage-1 model was trained at a single p; here we
decode 1e6 fresh shots at each p in [p_min, p_max] (log grid) and compare
against MWPM on the SAME noise instances. Reports raw diffusion LER, LER after
greedy syndrome projection (guarantees cleared=1), cleared fraction, and
residual-defect stats. Writes a JSON table and a log-log plot.

Usage:
  python scripts/ler_vs_p.py --run runs/stage1_global --legacy \
      --shots 1000000 --pmin 0.003 --pmax 0.03 --npoints 8
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from qldd.data import make_code_data, sample
from qldd.diffusion import decode
from qldd.baseline import build_matching, residual_is_logical


def project(code, s, eg):
    """Greedy syndrome projection (same as inference_rescue): flip the single
    error bit with the best lit-vs-unlit gain until no improving flip."""
    H = code.H.astype(np.uint8)
    eg = eg.copy()
    res = (s ^ ((H @ eg.T) % 2).T).astype(np.uint8)
    for i in np.nonzero(res.any(axis=1))[0]:
        r = res[i]
        for _ in range(code.n_err):
            if not r.any():
                break
            lit = H[r.astype(bool)].sum(axis=0).astype(int)
            unlit = H[~r.astype(bool)].sum(axis=0).astype(int)
            gain = lit - unlit
            j = int(np.argmax(gain))
            if gain[j] <= 0:
                break
            eg[i, j] ^= 1
            r = r ^ H[:, j]
        res[i] = r
    return eg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stage1_global")
    ap.add_argument("--legacy", action="store_true",
                    help="checkpoint uses the pre-SDPA architecture")
    ap.add_argument("--shots", type=int, default=1_000_000)
    ap.add_argument("--pmin", type=float, default=0.003)
    ap.add_argument("--pmax", type=float, default=0.03)
    ap.add_argument("--npoints", type=int, default=8)
    ap.add_argument("--infer-steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", default="ler_vs_p")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ck = torch.load(os.path.join(args.run, "ckpt.pt"),
                    map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    if args.legacy:
        from qldd.legacy_model import LocalDiffusionDecoder, ModelConfig
    else:
        from qldd.model import LocalDiffusionDecoder, ModelConfig
    mcfg = ModelConfig(**ck["model_cfg"])

    ps = np.geomspace(args.pmin, args.pmax, args.npoints)
    rows = []
    for p in ps:
        t0 = time.time()
        code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                              p=float(p), q=cfg.get("q"))
        model = LocalDiffusionDecoder(mcfg, code).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()

        e, s, l = sample(code, args.shots, seed=int(1e6 * p))
        eg = np.zeros_like(e)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for i in range(0, args.shots, args.batch):
                st = torch.as_tensor(s[i:i + args.batch], dtype=torch.long,
                                     device=dev)
                eg[i:i + args.batch] = decode(
                    model, st, n_steps=args.infer_steps).cpu().numpy()

        clears, logical = residual_is_logical(code, e, eg)
        res = (s ^ ((code.H @ eg.T) % 2).T).astype(np.uint8)
        defects = res.sum(axis=1)
        egp = project(code, s, eg)
        _, logical_p = residual_is_logical(code, e, egp)

        m = build_matching(code)
        pred = np.asarray(m.decode_batch(s), dtype=np.uint8)
        mwpm_fail = np.any(pred != l, axis=1)

        row = {
            "p": float(p),
            "ler_raw": float(logical.mean()),
            "ler_strict": float((logical | ~clears).mean()),
            "ler_projected": float(logical_p.mean()),
            "ler_mwpm": float(mwpm_fail.mean()),
            "cleared": float(clears.mean()),
            "defects_mean": float(defects.mean()),
            "defects_mean_unclean": (float(defects[defects > 0].mean())
                                     if (defects > 0).any() else 0.0),
            "shots": args.shots,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"p={p:.4f}  raw {row['ler_raw']:.2e}  proj {row['ler_projected']:.2e}  "
              f"MWPM {row['ler_mwpm']:.2e}  cleared {row['cleared']:.4f}  "
              f"defects {row['defects_mean']:.3f}  [{row['seconds']}s]", flush=True)

    with open(args.out + ".json", "w") as f:
        json.dump(rows, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = [r["p"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.loglog(P, [r["ler_mwpm"] for r in rows], "k:o", label="MWPM")
    a1.loglog(P, [r["ler_raw"] for r in rows], "-o", label="diffusion (raw)")
    a1.loglog(P, [r["ler_projected"] for r in rows], "-o",
              label="diffusion + projection")
    a1.set_xlabel("p"); a1.set_ylabel("logical error rate")
    a1.set_title(f"LER vs p (trained at p={cfg['p']}, {args.shots:.0e} shots/pt)")
    a1.legend(); a1.grid(alpha=0.3, which="both")
    a2.semilogx(P, [r["cleared"] for r in rows], "-o", label="P(cleared)")
    a2b = a2.twinx()
    a2b.semilogx(P, [r["defects_mean_unclean"] for r in rows], "r--s",
                 label="defects | unclean")
    a2.set_xlabel("p"); a2.set_ylabel("P(syndrome cleared)"); a2.set_ylim(0, 1.02)
    a2b.set_ylabel("mean defects given unclean", color="r")
    a2.set_title("clearing behavior vs p"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
    print(f"wrote {args.out}.json / {args.out}.png")


if __name__ == "__main__":
    main()
