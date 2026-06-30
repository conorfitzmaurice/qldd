"""LER vs physical error rate for the MWPM-free logical decoder, on a trained
checkpoint. The model is trained at one p; this tests generalization across the
noise regime and whether its margin over MWPM holds/widens (as it did at d=3).
MWPM is rebuilt per p (its weights are p-dependent) as the paired baseline; the
model itself never uses MWPM.

Writes JSON + a journal-style PNG/PDF.

Usage: python scripts/sweep_logical.py --run runs/d7_logical --shots 200000 \
    --pmin 0.005 --pmax 0.03 --npoints 8
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import ModelConfig
from qldd.logical import LogicalDecoder, logical_evaluate
from qldd.baseline import build_matching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d7_logical")
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--pmin", type=float, default=0.005)
    ap.add_argument("--pmax", type=float, default=0.03)
    ap.add_argument("--npoints", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]; mcfg = ModelConfig(**ck["model_cfg"])
    out = args.out or os.path.join(args.run, "sweep_logical")
    ps = np.geomspace(args.pmin, args.pmax, args.npoints)
    rows = []
    print(f"sweep logical decoder vs MWPM | {args.run} step {ck.get('step','?')} "
          f"(trained p={cfg['p']}) | {args.shots} shots/pt")
    print(f"{'p':>8} {'model':>9} {'MWPM':>9} {'ratio':>7} {'thr':>5} {'sec':>6}")
    for p in ps:
        t0 = time.time()
        code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                              p=float(p), q=cfg.get("q"))
        model = LogicalDecoder(mcfg, code).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
        matching = build_matching(code)
        _, s, l = sample(code, args.shots, seed=int(8e6 * p) + 3)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            rep = logical_evaluate(model, code, s, l, device=dev, batch=args.batch,
                                   matching=matching, thresholds=(0.4, 0.5, 0.6))
        ml, mw = rep["model_ler"], rep["mwpm_ler"]
        se = np.sqrt(max(ml*(1-ml), 1e-12)/args.shots)
        mwse = np.sqrt(max(mw*(1-mw), 1e-12)/args.shots)
        row = dict(p=float(p), model_ler=ml, mwpm_ler=mw, model_se=float(se),
                   mwpm_se=float(mwse), best_thresh=rep["best_thresh"],
                   ratio=ml/mw if mw > 0 else float("nan"),
                   seconds=round(time.time()-t0, 1))
        rows.append(row)
        print(f"{p:>8.4f} {ml:>9.5f} {mw:>9.5f} {ml/mw:>7.3f} "
              f"{rep['best_thresh']:>5.1f} {row['seconds']:>6.0f}", flush=True)

    json.dump(rows, open(out + ".json", "w"), indent=2)
    _plot(rows, cfg, out)
    print("wrote", out + ".json /", out + ".png/.pdf")


def _plot(rows, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "legend.frameon": True, "legend.framealpha": 1.0,
        "legend.edgecolor": "0.3", "legend.fancybox": False})
    P = np.array([r["p"] for r in rows])
    m = np.array([r["model_ler"] for r in rows]); mse = np.array([r["model_se"] for r in rows])
    w = np.array([r["mwpm_ler"] for r in rows]); wse = np.array([r["mwpm_se"] for r in rows])
    d = cfg["distance"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    a1.errorbar(P, w, yerr=wse, fmt="s--", ms=4, lw=1.1, color="0.25",
                mfc="white", mec="0.25", capsize=2, label="MWPM")
    a1.errorbar(P, m, yerr=mse, fmt="o-", ms=4, lw=1.3, color="#1a3e6e",
                mfc="white", mec="#1a3e6e", capsize=2, label="Neural decoder")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Physical error rate $p$"); a1.set_ylabel("Logical error rate")
    a1.legend(loc="upper left", fontsize=10, handlelength=1.8)
    a1.set_title(f"$d={d}$ rotated surface code", fontsize=11, pad=8)

    a2.errorbar(P, m/w, yerr=mse/w, fmt="o-", ms=4, lw=1.3, color="#7a1f1f",
                mfc="white", mec="#7a1f1f", capsize=2)
    a2.axhline(1.0, ls=":", lw=1.0, color="0.4")
    a2.set_xscale("log")
    a2.set_xlabel("Physical error rate $p$")
    a2.set_ylabel("Neural / MWPM  logical error ratio")
    a2.set_title("Relative performance", fontsize=11, pad=8)
    a2.text(0.04, 0.06, "below 1: neural decoder lower", transform=a2.transAxes,
            fontsize=9, color="0.35")
    for ax in (a1, a2):
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
