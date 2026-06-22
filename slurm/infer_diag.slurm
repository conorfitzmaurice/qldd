#!/bin/bash
#SBATCH --job-name=qldd-diag
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python scripts/infer_diag.py \
    --run "${RUN:-runs/d7_local}" --shots 5000 --steps 24 48 96 \
    --batch 256 --out "${RUN:-runs/d7_local}/infer_diag.json"
