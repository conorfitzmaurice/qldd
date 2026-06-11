#!/bin/bash
#SBATCH --job-name=qldd-mig
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1               # MIG requires exactly 1
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cf2185@princeton.edu
#SBATCH --requeue

# Usage: sbatch --export=ALL,CONFIG=configs/d7_local.yaml,NAME=d7_local slurm/train_mig.slurm
CONFIG="${CONFIG:-configs/d7_local.yaml}"
NAME="${NAME:-d7_local}"

module purge
module load anaconda3/2025.12
conda activate qldd

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
export QLDD_SCRATCH="/scratch/gpfs/$USER/qldd"
mkdir -p "$QLDD_SCRATCH" 2>/dev/null || echo "(scratch unavailable, continuing)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python -m qldd.train --config "$CONFIG"

if [ -f "runs/${NAME}/DONE" ]; then
    echo "Run complete."
elif grep -q '\[walltime\]' "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null; then
    echo "Walltime guard hit; requeuing..."
    sbatch --export=ALL,CONFIG="$CONFIG",NAME="$NAME" "$0"
else
    echo "Exited abnormally -- NOT requeuing. Check errors above."
fi
