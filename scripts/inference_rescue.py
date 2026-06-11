"""Re-evaluate a trained checkpoint with stronger inference: more unmasking
steps, plus a greedy syndrome-projection cleanup (flip the single error bit
that kills the most remaining lit detectors, repeat). No retraining."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.diffusion import decode
from qldd.baseline import build_matching, logical_error_rate, residual_is_logical

def project(code, s, eg):
    """Greedy cleanup: while detectors are lit, flip the error bit that reduces
    the residual syndrome weight the most (ties -> lowest index)."""
    H = code.H.astype(np.uint8)
    eg = eg.copy()
    res = (s ^ (H @ eg.T % 2).T).astype(np.uint8)          # residual syndrome
    for i in np.nonzero(res.any(axis=1))[0]:
        r = res[i]
        for _ in range(code.n_err):                         # safety bound
            if not r.any(): break
            # gain_j = (# lit dets bit j touches) - (# unlit dets it would light)
            lit = H[r.astype(bool)].sum(axis=0).astype(int)
            unlit = H[~r.astype(bool)].sum(axis=0).astype(int)
            gain = lit - unlit
            j = int(np.argmax(gain))
            if gain[j] <= 0: break                          # no improving flip
            eg[i, j] ^= 1
            r = r ^ H[:, j]
        res[i] = r
    return eg
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stage1_global")
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    code = make_code_data(distance=cfg["distance"], rounds=cfg.get("rounds"),
                          p=cfg["p"], q=cfg.get("q"))
    model = LocalDiffusionDecoder(ModelConfig(**ck["model_cfg"]), code).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mwpm = logical_error_rate(code, args.shots, build_matching(code), seed=777)["ler"]
    e, s, l = sample(code, args.shots, seed=777)
    print(f"d={code.distance} p={code.p}  MWPM LER = {mwpm:.5f}  shots={args.shots}")
    print(f"{'steps':>6} {'proj':>5} {'LER':>8} {'strict':>8} {'cleared':>8} {'vs MWPM':>8}")
    for n in args.steps:
        eg = np.zeros_like(e)
        for i in range(0, args.shots, 2048):
            st = torch.as_tensor(s[i:i+2048], dtype=torch.long, device=dev)
            eg[i:i+2048] = decode(model, st, n_steps=n).cpu().numpy()
        for use_proj in (False, True):
            g = project(code, s, eg) if use_proj else eg
            clears, logical = residual_is_logical(code, e, g)
            ler = logical.mean(); strict = (logical | ~clears).mean()
            print(f"{n:>6} {str(use_proj):>5} {ler:>8.5f} {strict:>8.5f} "
                  f"{clears.mean():>8.4f} {(ler-mwpm)/mwpm:>+7.1%}")

if __name__ == "__main__":
    main()
