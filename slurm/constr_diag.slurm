#!/bin/bash
#SBATCH --job-name=qldd-cdiag
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python scripts/constrained_diag.py \
    --run "${RUN:-runs/d7_local}" --shots 3000 --steps 48 --batch 256
