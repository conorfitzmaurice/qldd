#!/bin/bash
#SBATCH --job-name=qldd-online
#SBATCH --partition=gpu-ee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=cf2185@princeton.edu
#SBATCH --requeue
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun ~/.conda/envs/qldd/bin/python scripts/train_logical.py \
    --distance "${DIST:-7}" --p "${P:-0.03}" --steps "${STEPS:-80000}" \
    --batch "${BATCH:-1024}" --d_model "${DMODEL:-128}" --causal \
    --run "${RUN:-runs/d7_online}"
