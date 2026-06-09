# qldd

Local masked-diffusion decoder for the rotated surface code under
phenomenological noise. It predicts the physical error chain `e` rather than
the logical class, using attention with a trainable per-head locality radius
sigma. Reading sigma off after training (and how it scales with code distance)
is the locality measurement.

## Background

Three related decoders:

- arXiv:2604.08358 (Gu et al.): pure 3D-conv over the spacetime lattice,
  L ~ d layers, predicts the logical observable.
- arXiv:2509.22347 (Liu/Gong/Clark): masked diffusion, predicts the logical
  error `l = Le`, BB code, global factored attention.
- arXiv:2604.24640 (DiffQEC): discrete-diffusion posterior, also global,
  also logical-class.

All three predict the logical class, which is global by construction, so none
of them can be local. Predicting `e` is what admits a local decoder. The conv
stem reuses Gu et al.'s 3D spacetime convolution for local features, and the
transformer on top biases attention by `-d^2/(2 sigma^2)` with sigma trainable
per head (large sigma recovers global attention). Surface code rather than BB
because it's geometrically local in 2+1D (arXiv:2404.07251, 2508.06614,
2511.01976). Global attention also costs O(d^4), which is the motivation for
the Mamba decoder in arXiv:2510.22724; bounded-range local attention is the
cheaper regime anyway.

One caveat to keep in mind: clearing the syndrome is necessary but not
sufficient. The residual `e_true XOR e_guess` is either a stabilizer
(harmless) or a logical (failure). `evaluate_ler` reports the observable-based
LER (comparable to MWPM), a strict LER where a non-clearing guess also counts
as a failure, and the pure-stabilizer-residual fraction.

## Layout

```
src/qldd/
  data.py       Stim surface code -> (e, s, l), H, L, priors, geometry
  baseline.py   PyMatching MWPM baseline, LER, residual classifier, threshold sweep
  model.py      conv stem + local-attention transformer, trainable sigma
  diffusion.py  masked diffusion over e: masking, loss, iterative-unmask decode, LER
  analysis.py   conv receptive field, sigma in lattice units, locality_report
  train.py      config-driven harness, checkpoint + requeue
configs/        per-run configs (d3_mig, d5_a100, d7_a100, stage1_global, d7_local)
slurm/          Della MIG + A100 scripts with auto-requeue
scripts/        sanity_check, threshold_gate, stage1_verdict, make/aggregate_ablation
tests/          contract + physics regression tests
```

Data contract: `s = H e (mod 2)` and `l = L e (mod 2)` in DEM fault-mechanism
space, checked bit-exactly by `verify_contract`.

## Quick start

```bash
pip install -r requirements.txt
python scripts/sanity_check.py    # s=He/l=Le + MWPM threshold crossing
python -m qldd.train --config configs/d3_mig.yaml
python scripts/threshold_gate.py --runs runs/d3_mig runs/d5_a100 runs/d7_a100
```

## Running on Della

MIG partition (10 GB, fast queue) for d=3 and debugging; one A100 for d=5/7.
Write active output to `/scratch/gpfs/$USER`. Jobs stay under 24h since the
harness checkpoints at ~22h and the SLURM scripts requeue themselves.

```bash
mkdir -p logs runs
sbatch slurm/stage1.slurm    # global-limit train + go/no-go verdict
sbatch slurm/train_d3_mig.slurm
sbatch --export=CONFIG=configs/d5_a100.yaml,NAME=d5_a100 slurm/train_a100.slurm
sbatch --export=CONFIG=configs/d7_a100.yaml,NAME=d7_a100 slurm/train_a100.slurm
```

Edit `--mail-user` and the `module load` / `conda activate` lines first.
Interactive debug: `salloc --nodes=1 --ntasks=1 --time=60:00 --gres=gpu:1
--partition=mig`.

## Plan

Stage 1: confirm diffusion-over-`e` reaches MWPM at d=3 in the global limit
(large `sigma_init`, `configs/stage1_global.yaml`). If it can't match MWPM
globally there's no point tuning locality.

Stage 2 is the locality study, and it has a confound to watch: with conv depth
L ~ d the conv receptive field saturates the code (at d=3 it's global even at
L=1), so a small sigma proves nothing on its own. The right quantity is the
total effective range max(conv_RF, attention_range(sigma)) compared to the
grid width. In practice that means the clean test runs with the conv stem off
(`configs/d7_local.yaml`) and needs d >= 7, since smaller grids leave no room
for a sub-global receptive field. `locality_report` flags confounded configs
and `train.py` logs the total range at every eval. `make_ablation.py` and
`aggregate_ablation.py` sweep the conv_layers x sigma_init grid.

Stage 3: `threshold_gate.py` reports diffusion LER vs MWPM and the effective
range per distance. The gate is tracking MWPM within a few percent at d=5/7.
The main question is whether the total range stays O(1) as d grows (7, 9, 11).
Growth ~d would also be a clean result.

Attention modes (`model.window_radius`):

- dense (`null`): materializes `(B, heads, T, T)`, O(T^2). Fine at d=3, heavy
  at d=7 where T ~ 1.5k.
- windowed (`R` in lattice units): each token attends to its <=K neighbors
  within R, O(T*K). Matches dense to 3e-7 at large R; at d=7, R=1.5 gives
  K=84 (about 5% of T), so d=7 fits one A100. Used by `d7_local.yaml`.

The windowed kernel still gathers `(B,h,T,K,dk)`, so a fused local-attention
kernel is the remaining efficiency item.

## Status

Done: data pipeline checked bit-exact (d=3/5/7), MWPM baseline with threshold
crossing at p* ~ 0.04, model + diffusion train/infer, sigma trainable and
logged, locality analysis with the conv-RF confound check, windowed attention
kernel, strict LER, checkpoint/requeue chaining.

Todo: full GPU training to convergence on Della (stages 1-3), fused local
attention kernel, pure-conv logical-head baseline for ablation, pure-X
data-noise variant (`data_noise_channel`).

Everything so far is CPU-validated at d=3 (loss converges, decode runs, sigma
trains). The decoder is not yet trained to convergence and does not yet beat
MWPM; that needs the GPU runs above.
