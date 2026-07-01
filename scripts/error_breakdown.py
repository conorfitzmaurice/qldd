"""Failure decomposition + residual-syndrome statistics for a chain-model
checkpoint (src/qldd/model.py LocalDiffusionDecoder).

Two failure types (per Frank):
  Type A  UNRESOLVED : decoder output does not clear the syndrome (H e != s).
  Type B  WRONG SECTOR: syndrome cleared (H e = s) but wrong logical coset (L e != l).
MWPM only ever makes Type-B errors (it always clears); a neural chain decoder can
make both. The split says whether to fix CLEARING (A) or COSET SELECTION (B).

Also: histogram of residual-syndrome weight |s ^ H e| (number of unsatisfied
detectors) over shots -- reveals whether unresolved failures are a few stubborn
defects (fixable) or globally incoherent (structural).

Usage: python scripts/error_breakdown.py --run runs/d5_a100 --shots 50000 --steps 16
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import decode
from qldd.baseline import build_matching, logical_error_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d5_a100")
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    mcfg = ModelConfig(**ck["model_cfg"])
    model = LocalDiffusionDecoder(mcfg, code).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    steps = args.steps or cfg.get("infer_steps", 16)

    e, s, l = sample(code, args.shots, seed=4242)
    H = code.H.astype(np.uint8); L = code.L.astype(np.uint8)

    eg = np.zeros_like(e)
    for i in range(0, args.shots, args.batch):
        st = torch.as_tensor(s[i:i+args.batch], dtype=torch.long, device=dev)
        eg[i:i+args.batch] = decode(model, st, n_steps=steps).cpu().numpy()

    # residual syndrome = s XOR H e_guess  (unsatisfied detectors)
    res_syn = (s.astype(np.uint8) ^ ((H @ eg.T) % 2).T.astype(np.uint8))
    res_weight = res_syn.sum(axis=1)                       # (N,) defects per shot
    cleared = (res_weight == 0)

    # logical correctness (only meaningful where cleared; but compute L e vs l)
    obs_pred = (L @ eg.T % 2).T.astype(np.uint8)
    logical_wrong = np.any(obs_pred != l.astype(np.uint8), axis=1)

    typeA = ~cleared                                       # unresolved
    typeB = cleared & logical_wrong                        # cleared, wrong sector
    success = cleared & ~logical_wrong

    mwpm = logical_error_rate(code, args.shots, build_matching(code), seed=4242)["ler"]

    N = args.shots
    print(f"=== failure breakdown | {args.run} step {ck.get('step','?')} "
          f"d={code.distance} p={code.p} | {N} shots, {steps} infer steps ===")
    print(f"  success (clear + right sector) : {success.mean():.4f}")
    print(f"  Type A  UNRESOLVED (H e != s)  : {typeA.mean():.4f}")
    print(f"  Type B  WRONG SECTOR (cleared) : {typeB.mean():.4f}")
    print(f"  total logical error rate       : {(typeA|typeB).mean():.4f}")
    print(f"  [MWPM baseline LER             : {mwpm:.4f}]")
    print(f"  clear rate                     : {cleared.mean():.4f}")
    # residual weight stats
    print(f"  residual-syndrome weight: mean {res_weight.mean():.2f}  "
          f"median {np.median(res_weight):.0f}  p90 {np.percentile(res_weight,90):.0f}  "
          f"max {res_weight.max()}")
    # histogram (counts by residual weight)
    maxw = int(res_weight.max())
    hist = np.bincount(res_weight, minlength=maxw+1)
    print("  residual-weight histogram (weight: count):")
    for w in range(0, min(maxw, 20) + 1):
        if hist[w] > 0:
            bar = "#" * int(60 * hist[w] / hist.max())
            print(f"    {w:3d}: {hist[w]:7d} {bar}")
    if maxw > 20:
        print(f"    >20: {hist[21:].sum():7d}")

    if args.out:
        json.dump({"run": args.run, "step": ck.get("step"), "shots": N,
                   "steps": steps, "success": float(success.mean()),
                   "typeA_unresolved": float(typeA.mean()),
                   "typeB_wrong_sector": float(typeB.mean()),
                   "ler": float((typeA|typeB).mean()), "mwpm_ler": float(mwpm),
                   "clear_rate": float(cleared.mean()),
                   "res_weight_mean": float(res_weight.mean()),
                   "res_weight_hist": hist.tolist()}, open(args.out, "w"), indent=2)
        # also a clean PNG histogram
        _plot_hist(res_weight, code, ck.get("step"), args.out)
        print("wrote", args.out)


def _plot_hist(res_weight, code, step, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    maxw = int(res_weight.max())
    ax.hist(res_weight, bins=np.arange(-0.5, maxw + 1.5, 1),
            color="#1a3e6e", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Residual-syndrome weight  $|s \\oplus H\\hat e|$  (unsatisfied detectors)")
    ax.set_ylabel("Shots")
    ax.set_title(f"$d={code.distance}$ residual syndrome, step {step}", fontsize=11, pad=8)
    ax.axvline(0, color="#7a1f1f", ls=":", lw=1.0)
    ax.text(0.3, ax.get_ylim()[1]*0.5, "cleared\n(weight 0)", fontsize=9,
            color="#7a1f1f", va="top")
    fig.tight_layout()
    png = out.replace(".json", "") + "_hist.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
