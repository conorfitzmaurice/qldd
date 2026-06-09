"""Generate the (conv_layers x sigma_init) ablation config grid + launcher.
Run at d>=7; smaller grids make the conv global even at L=1.
Usage: python scripts/make_ablation.py --distance 7 --p 0.03
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml

CONV_LAYERS = [0, 1, 2, 4]          # 0 => use_conv_stem: false
SIGMA_INIT = [0.3, 0.6, 1.0, 100.0] # last = ~global attention


def base_cfg(distance, p):
    return dict(
        distance=distance, rounds=distance, p=p,
        steps=120000, batch_size=96, lr=2.0e-3, min_lr=8.0e-5, warmup=800,
        weight_decay=1.0e-4, grad_clip=1.0, eval_every=3000, eval_shots=80000,
        infer_steps=24, eval_batch=96, log_every=200, ckpt_every=3000, max_minutes=1320, seed=0,
        model=dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                   conv_channels=32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.03)
    ap.add_argument("--outdir", default="runs_ablation")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg_dir = os.path.join(args.outdir, "configs"); os.makedirs(cfg_dir, exist_ok=True)

    launch = ["#!/bin/bash", "# edit module/conda/mail lines in slurm/train_a100.slurm first", ""]
    for L in CONV_LAYERS:
        for s in SIGMA_INIT:
            name = f"d{args.distance}_L{L}_s{s}"
            cfg = base_cfg(args.distance, args.p)
            cfg["run_dir"] = os.path.join(args.outdir, name)
            cfg["model"]["use_conv_stem"] = (L > 0)
            cfg["model"]["conv_layers"] = max(L, 1)
            cfg["model"]["sigma_init"] = s
            path = os.path.join(cfg_dir, name + ".yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            launch.append(
                f"sbatch --export=CONFIG={path},NAME={name} slurm/train_a100.slurm")
    sh = os.path.join(args.outdir, f"launch_d{args.distance}.sh")
    with open(sh, "w") as f:
        f.write("\n".join(launch) + "\n")
    print(f"wrote {len(CONV_LAYERS)*len(SIGMA_INIT)} configs to {cfg_dir}")
    print(f"launcher: {sh}")


if __name__ == "__main__":
    main()
