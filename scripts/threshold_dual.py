"""Dual threshold check: run MWPM threshold for BOTH noise models to (a) validate
the machinery against the known code-capacity value (~0.10) and (b) show the
phenomenological model we actually use (multi-round + measurement noise) has its
own, lower threshold (~0.03).

  code-capacity : rounds=1, q=0 (perfect measurement), data depolarization only
  phenomenological: rounds=d, q=p (noisy measurement), data depolarization

A clean crossing at the literature value for each certifies the DEM/baseline.

Usage: python scripts/threshold_dual.py --distances 3 5 7 --shots 200000
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qldd.data import make_code_data
from qldd.baseline import build_matching, logical_error_rate


def sweep(distances, ps, shots, rounds_mode, q_mode, seed0=1):
    out = {}
    for d in distances:
        lers = []
        for p in ps:
            rounds = 1 if rounds_mode == "cc" else d
            q = 0.0 if q_mode == "perfect" else float(p)
            code = make_code_data(distance=d, rounds=rounds, p=float(p), q=q)
            r = logical_error_rate(code, shots, build_matching(code),
                                   seed=seed0 + int(1e5 * p) + d)["ler"]
            lers.append(r)
        out[str(d)] = lers
        print(f"  d={d}: " + " ".join(f"{p:.3f}:{l:.4f}" for p, l in zip(ps, lers)),
              flush=True)
    return out


def crossing(distances, ps, ler):
    """Estimate p* where the d-ordering flips (curves cross)."""
    ps = np.array(ps)
    ds = sorted(int(x) for x in distances)
    lo, hi = ds[0], ds[-1]
    diff = np.array(ler[str(hi)]) - np.array(ler[str(lo)])   # <0 below p*, >0 above
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) > 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    # linear interp on the difference through zero
    x0, x1 = ps[i], ps[i + 1]; y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--out", default="runs/threshold_dual")
    args = ap.parse_args()

    # code-capacity: threshold ~0.10, sweep around it
    cc_ps = [0.09, 0.11, 0.12, 0.13, 0.135, 0.14, 0.15, 0.17]
    # phenomenological: threshold ~0.03, sweep around it
    ph_ps = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]

    print("=== CODE-CAPACITY (rounds=1, perfect meas., DEPOLARIZING) — expect p*~0.13-0.14 ===")
    cc = sweep(args.distances, cc_ps, args.shots, "cc", "perfect")
    cc_star = crossing(args.distances, cc_ps, cc)
    print(f"  code-capacity p* estimate: {cc_star}")

    print("=== PHENOMENOLOGICAL (rounds=d, noisy measurement) — expect p*~0.03 ===")
    ph = sweep(args.distances, ph_ps, args.shots, "phenom", "noisy")
    ph_star = crossing(args.distances, ph_ps, ph)
    print(f"  phenomenological p* estimate: {ph_star}")

    data = {"distances": args.distances,
            "code_capacity": {"ps": cc_ps, "ler": cc, "p_star": cc_star},
            "phenomenological": {"ps": ph_ps, "ler": ph, "p_star": ph_star}}
    json.dump(data, open(args.out + ".json", "w"), indent=2)
    _plot(data, args.out)
    print("wrote", args.out + ".json /.png/.pdf")


def _plot(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "legend.frameon": True, "legend.edgecolor": "0.3", "legend.fancybox": False})
    ds = data["distances"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ds)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, title, exp in [
        (axes[0], "code_capacity", "Code capacity (1 round, ideal readout, depolarizing)", 0.135),
        (axes[1], "phenomenological", "Phenomenological ($d$ rounds, noisy readout)", None)]:
        blk = data[key]; ps = np.array(blk["ps"])
        for d, c in zip(ds, colors):
            ax.plot(ps, blk["ler"][str(d)], "o-", ms=4, lw=1.2, color=c,
                    mfc="white", mec=c, label=f"$d={d}$")
        pstar = blk["p_star"]
        if pstar:
            ax.axvline(pstar, ls=":", lw=1.0, color="0.4")
            ax.text(pstar, ax.get_ylim()[0], f"  $p^*\\approx{pstar:.3f}$",
                    fontsize=9, color="0.35", va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate $p$")
        ax.set_ylabel("Logical error rate (MWPM)")
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.legend(fontsize=9.5, loc="lower right")
        ax.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
