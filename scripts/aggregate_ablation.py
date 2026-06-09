"""Aggregate locality-ablation runs into one table.
Usage: python scripts/aggregate_ablation.py --glob 'runs_ablation/d7_*'
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from qldd.data import make_code_data
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.analysis import locality_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()
    rows = []
    for run in sorted(glob.glob(args.glob)):
        ckpt = os.path.join(run, "ckpt.pt")
        hist = os.path.join(run, "history.json")
        if not os.path.exists(ckpt):
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                              p=cfg["p"], q=cfg.get("q"))
        model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code)
        model.load_state_dict(ck["model"]); model.eval()
        loc = locality_report(model, code)
        last = {}
        if os.path.exists(hist):
            h = json.load(open(hist))
            if h:
                last = h[-1]
        rows.append({
            "run": os.path.basename(run),
            "L": ck["model_cfg"].get("conv_layers") if ck["model_cfg"].get("use_conv_stem") else 0,
            "sigma_init": ck["model_cfg"].get("sigma_init"),
            "range": loc["total_effective_range_lattice"],
            "grid_w": loc["grid_width_voxels"],
            "diff_ler": last.get("diff_ler"),
            "mwpm_ler": last.get("mwpm_ler"),
            "meaningful": loc["locality_test_meaningful"],
        })

    hdr = f"{'run':<18}{'L':>3}{'s_init':>8}{'range':>7}{'grid':>6}{'diff_LER':>10}{'MWPM':>9}{'locality?':>10}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        dl = f"{r['diff_ler']:.4f}" if r["diff_ler"] is not None else "  -  "
        ml = f"{r['mwpm_ler']:.4f}" if r["mwpm_ler"] is not None else "  -  "
        rng = f"{r['range']:.1f}" if r["range"] is not None else " - "
        print(f"{r['run']:<18}{r['L']:>3}{r['sigma_init']:>8}{rng:>7}"
              f"{r['grid_w']:>6}{dl:>10}{ml:>9}{str(r['meaningful']):>10}")
    print("\nOnly rows with locality?=True count as locality evidence.")
    with open("ablation_table.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
