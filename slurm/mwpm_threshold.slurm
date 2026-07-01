#!/bin/bash
#SBATCH --job-name=qldd-thresh
#SBATCH --partition=gpu-ee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python scripts/mwpm_threshold_check.py \
    --distances 3 5 7 9 --pmin 0.02 --pmax 0.10 --npoints 9 \
    --shots "${SHOTS:-200000}" --out runs/mwpm_threshold
