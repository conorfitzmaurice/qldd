"""MWPM threshold verification: LER vs p for several distances on one figure.

Correctness check on the noise model / DEM plumbing, independent of any neural
net. If MWPM is set up right, the d curves CROSS at one point (the threshold p*):
below p* larger d gives LOWER LER; above p* larger d gives HIGHER LER. A clean
crossing validates that every "beats MWPM" comparison uses a sound baseline.

Usage: python scripts/mwpm_threshold_check.py --distances 3 5 7 9 \
    --pmin 0.02 --pmax 0.10 --npoints 9 --shots 50000
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qldd.baseline import threshold_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--pmax", type=float, default=0.10)
    ap.add_argument("--npoints", type=int, default=9)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--out", default="mwpm_threshold")
    args = ap.parse_args()

    ps = np.linspace(args.pmin, args.pmax, args.npoints)
    print(f"MWPM threshold sweep: d={args.distances}, {args.shots} shots/pt")
    sweep = threshold_sweep(distances=args.distances, ps=ps, shots=args.shots)
    print("p:      " + "  ".join(f"{p:6.3f}" for p in ps))
    for d in args.distances:
        print(f"d={d}:  " + "  ".join(f"{x:6.4f}" for x in sweep['ler'][d]))

    # locate approximate crossing (where higher-d overtakes lower-d)
    d_sorted = sorted(args.distances)
    ler = sweep["ler"]
    crossings = []
    for i in range(len(d_sorted) - 1):
        a, b = ler[d_sorted[i]], ler[d_sorted[i + 1]]
        diff = b - a                       # >0 below thr (b lower->neg?) sign flips at p*
        sign = np.sign(diff)
        idx = np.where(np.diff(sign) != 0)[0]
        for j in idx:
            crossings.append(0.5 * (ps[j] + ps[j + 1]))
    p_star = float(np.median(crossings)) if crossings else None
    print(f"estimated threshold p* ~ {p_star}" if p_star else "no clean crossing found")

    json.dump({"distances": args.distances, "ps": ps.tolist(),
               "ler": {str(d): ler[d].tolist() for d in args.distances},
               "p_star_est": p_star}, open(args.out + ".json", "w"), indent=2)

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
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    greys = plt.cm.viridis(np.linspace(0.15, 0.8, len(d_sorted)))
    for d, c in zip(d_sorted, greys):
        y = ler[d]; se = np.sqrt(np.clip(y*(1-y), 1e-12, None)/args.shots)
        ax.errorbar(ps, y, yerr=se, fmt="o-", ms=4, lw=1.3, color=c,
                    mfc="white", mec=c, capsize=2, label=f"$d={d}$")
    ax.set_yscale("log")
    if p_star:
        ax.axvline(p_star, ls=":", lw=1.0, color="0.4")
        y0, y1 = ax.get_ylim()
        ax.text(p_star, y0*(y1/y0)**0.03, f"  $p^*\\approx{p_star:.3f}$",
                fontsize=9, color="0.35", va="bottom")
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate (MWPM)")
    ax.set_title("Surface-code threshold (MWPM baseline check)", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=10, handlelength=1.8)
    ax.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".json/.png/.pdf")


if __name__ == "__main__":
    main()
