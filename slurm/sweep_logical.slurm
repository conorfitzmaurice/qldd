#!/bin/bash
#SBATCH --job-name=qldd-swlog
#SBATCH --partition=gpu-ee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun ~/.conda/envs/qldd/bin/python scripts/sweep_logical.py \
    --run "${RUN:-runs/d7_logical}" --shots "${SHOTS:-200000}" \
    --pmin 0.005 --pmax 0.03 --npoints 8 --batch 8192
