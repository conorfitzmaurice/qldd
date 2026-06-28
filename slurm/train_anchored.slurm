#!/bin/bash
#SBATCH --job-name=qldd-anch
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=cf2185@princeton.edu
#SBATCH --requeue
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun ~/.conda/envs/qldd/bin/python scripts/train_anchored.py \
    --distance "${DIST:-7}" --p "${P:-0.03}" --steps "${STEPS:-60000}" \
    --batch "${BATCH:-256}" --run "${RUN:-runs/d7_anchored}"
