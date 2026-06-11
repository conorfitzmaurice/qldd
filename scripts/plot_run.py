"""Plot training diagnostics from one or more runs' history.json.

Panels: (1) P(syndrome fully cleared) vs step  [the bet: this is bad],
(2) mean residual spacetime defects vs step (with mean-given-unclean),
(3) LER vs step against the MWPM line, (4) learned xi_space / xi_time vs step.

Usage: python scripts/plot_run.py runs/d7_local runs/d7_online -o d7_plots.png
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default="run_plots.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax_clear, ax_def), (ax_ler, ax_xi) = axes
    for run in args.runs:
        hpath = os.path.join(run, "history.json")
        if not os.path.exists(hpath):
            print(f"skip {run}: no history.json"); continue
        h = json.load(open(hpath))
        name = os.path.basename(run.rstrip("/"))
        steps = [r["step"] for r in h]
        ax_clear.plot(steps, [r.get("cleared") for r in h], label=name)
        ax_def.plot(steps, [r.get("defects_mean") for r in h], label=name)
        dmu = [r.get("defects_mean_unclean") for r in h]
        if any(v is not None for v in dmu):
            ax_def.plot(steps, dmu, ls="--", alpha=0.6,
                        label=f"{name} (given unclean)")
        ax_ler.plot(steps, [r.get("diff_ler") for r in h], label=name)
        if h and h[0].get("mwpm_ler") is not None:
            ax_ler.axhline(h[0]["mwpm_ler"], ls=":", c="k", alpha=0.7)
        if h and h[0].get("xi_space_mean") is not None:
            ax_xi.plot(steps, [r.get("xi_space_mean") for r in h],
                       label=f"{name} xi_space")
            ax_xi.plot(steps, [r.get("xi_time_mean") for r in h], ls="--",
                       label=f"{name} xi_time")

    ax_clear.set_title("P(syndrome fully cleared)"); ax_clear.set_ylim(0, 1)
    ax_def.set_title("residual spacetime defects (mean)")
    ax_ler.set_title("logical error rate (dotted = MWPM)"); ax_ler.set_yscale("log")
    ax_xi.set_title("learned locality scales (lattice units / rounds)")
    for ax in axes.flat:
        ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
