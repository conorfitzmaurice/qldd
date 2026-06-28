"""LER vs physical error rate for decode scheme (D) = model + MWPM-on-residual,
on a trained checkpoint. Tests how the d=7 gap to MWPM varies with p: uniform,
or only near threshold? The model is trained at one p but generalizes (shown at
d=3). MWPM is rebuilt per-p (its weights depend on p) for a fair paired compare.

Usage: python scripts/sweep_decoder_D.py --run runs/d7_cap --shots 20000 \
    --pmin 0.005 --pmax 0.03 --npoints 7 --steps 48
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
    ap.add_argument("--run", default="runs/d7_cap")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--pmin", type=float, default=0.005)
    ap.add_argument("--pmax", type=float, default=0.03)
    ap.add_argument("--npoints", type=int, default=7)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default="d7_sweep_D")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]
    mcfg = ModelConfig(**ck["model_cfg"])
    ps = np.geomspace(args.pmin, args.pmax, args.npoints)
    rows = []
    print(f"sweep (D) vs MWPM | {args.run} step {ck.get('step','?')} "
          f"(trained p={cfg['p']}) | {args.shots} shots/pt")
    print(f"{'p':>8} {'D_LER':>9} {'MWPM':>9} {'diff':>9} {'stderr':>8} "
          f"{'net_fix':>8} {'cleared':>8}")
    for p in ps:
        t0 = time.time()
        code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                              p=float(p), q=cfg.get("q"))
        model = LocalDiffusionDecoder(mcfg, code).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
        matching = build_matching(code)               # MWPM weights depend on p
        e, s, l = sample(code, args.shots, seed=int(7e6 * p) + 1)
        pred = np.asarray(matching.decode_batch(s), dtype=np.uint8)
        if pred.ndim == 1:
            pred = pred[:, None]
        mwpm_fail = np.any(pred != l, axis=1)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            obs_pred, cleared = decode_match_projected(
                model, s, code, matching, n_steps=args.steps, device=dev,
                batch=args.batch)
        d_fail = np.any(obs_pred != l, axis=1)
        d_ler = float(d_fail.mean()); mw = float(mwpm_fail.mean())
        se = float(np.sqrt(max(d_ler * (1 - d_ler), 1e-12) / args.shots))
        diff = both = (mwpm_fail.astype(int) - d_fail.astype(int))
        net = int((both == 1).sum() - (both == -1).sum())
        row = dict(p=float(p), d_ler=d_ler, mwpm_ler=mw, diff=d_ler - mw,
                   stderr=se, net_fix=net, cleared=float(cleared.mean()),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{p:>8.4f} {d_ler:>9.5f} {mw:>9.5f} {d_ler-mw:>+9.5f} "
              f"{se:>8.5f} {net:>+8d} {cleared.mean():>8.4f}", flush=True)

    json.dump(rows, open(args.out + ".json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = [r["p"] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        a1.loglog(P, [r["mwpm_ler"] for r in rows], "k:o", label="MWPM")
        a1.loglog(P, [r["d_ler"] for r in rows], "-o", label="model + MWPM-resid (D)")
        a1.set_xlabel("physical error rate p"); a1.set_ylabel("logical error rate")
        a1.set_title(f"d={cfg['distance']} LER vs p (trained at p={cfg['p']})")
        a1.legend(); a1.grid(alpha=0.3, which="both")
        a2.semilogx(P, [r["diff"] for r in rows], "-o", color="crimson")
        a2.axhline(0, ls="--", c="k", alpha=0.6)
        a2.set_xlabel("physical error rate p")
        a2.set_ylabel("(D) LER  -  MWPM LER")
        a2.set_title("gap to MWPM (<0 = model wins)"); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
        print("wrote", args.out + ".json /", args.out + ".png")
    except Exception as ex:
        print("plot skipped:", ex, "| json written:", args.out + ".json")


if __name__ == "__main__":
    main()
