"""Report diffusion LER vs MWPM LER and learned sigma per checkpoint.
Usage: python scripts/threshold_gate.py --runs runs/d3_mig runs/d5_a100 ...
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import evaluate_ler
from qldd.baseline import build_matching, logical_error_rate


def load_model(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"),
                    map_location=device, weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, code, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--infer-steps", type=int, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'d':>3} {'p':>6} {'diff_LER':>10} {'MWPM_LER':>10} "
          f"{'rel_gap':>8} {'cleared':>8} {'sigma':>8}")
    rows = []
    for run in args.runs:
        model, code, cfg = load_model(run, device)
        e, s, l = sample(code, args.shots, seed=2024)
        diff = evaluate_ler(model, code, s, e, l,
                            n_steps=args.infer_steps, device=device)
        mwpm = logical_error_rate(code, args.shots, build_matching(code), seed=2024)
        sig = float(np.mean([np.mean(v) for v in model.locality_radii().values()]))
        rel = (diff["ler"] - mwpm["ler"]) / max(mwpm["ler"], 1e-9)
        print(f"{code.distance:>3} {code.p:>6.3f} {diff['ler']:>10.5f} "
              f"{mwpm['ler']:>10.5f} {rel:>7.1%} "
              f"{diff['syndrome_cleared_frac']:>8.3f} {sig:>8.2f}")
        rows.append({"d": code.distance, "p": code.p, "diff_ler": diff["ler"],
                     "mwpm_ler": mwpm["ler"], "rel_gap": rel, "sigma": sig,
                     "cleared": diff["syndrome_cleared_frac"]})

    print("\nGate: rel_gap should stay within ~5-10% at all distances.")
    with open("gate_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
