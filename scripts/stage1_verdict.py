"""Stage-1 go/no-go: does diffusion-over-e reach MWPM at d=3 in the global
limit? Usage: python scripts/stage1_verdict.py [--run ...] [--shots ...]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import evaluate_ler
from qldd.baseline import build_matching, logical_error_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stage1_global")
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=0.10,
                    help="relative LER gap to MWPM allowed for a GO (default 10%)")
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.run, "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    e, s, l = sample(code, args.shots, seed=31337)
    diff = evaluate_ler(model, code, s, e, l,
                        n_steps=cfg.get("infer_steps"), device=device,
                        batch=cfg.get("eval_batch", 256))
    mwpm = logical_error_rate(code, args.shots, build_matching(code), seed=31337)["ler"]

    rel = (diff["ler"] - mwpm) / max(mwpm, 1e-9)
    print(f"distance       : {code.distance}  (p={code.p})")
    print(f"trained steps  : {ck['step'] + 1}")
    print(f"MWPM LER       : {mwpm:.5f}")
    print(f"diffusion LER  : {diff['ler']:.5f}  (observable)")
    print(f"diffusion LER  : {diff['ler_strict']:.5f}  (strict: non-clear = fail)")
    print(f"syndrome clear : {diff['syndrome_cleared_frac']:.4f}")
    print(f"rel. gap vs MWPM: {rel:+.1%}  (tolerance {args.tol:.0%})")

    go = (rel <= args.tol) and (diff["syndrome_cleared_frac"] > 0.99)
    print("\nVERDICT:", "GO -> proceed to the locality study" if go else
          "NO-GO -> does not reach MWPM globally")
    print("(GO = LER within tolerance of MWPM and >99% of syndromes cleared.)")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
