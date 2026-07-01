#!/bin/bash
#SBATCH --job-name=qldd-thr2
#SBATCH --partition=gpu-ee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python scripts/threshold_dual.py \
    --distances 3 5 7 --shots "${SHOTS:-200000}" --out runs/threshold_dual
