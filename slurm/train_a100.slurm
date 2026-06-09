#!/bin/bash
#SBATCH --job-name=qldd-a100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # for Stim sampling / host work
#SBATCH --gres=gpu:1                    # no 80GB constraint, schedules faster
#SBATCH --mem=64G
#SBATCH --time=23:30:00                 # < 24h; harness checkpoints at ~22h
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=CHANGEME@princeton.edu
#SBATCH --requeue

# Usage: sbatch --export=CONFIG=configs/d5_a100.yaml,NAME=d5_a100 slurm/train_a100.slurm
CONFIG="${CONFIG:-configs/d5_a100.yaml}"
NAME="${NAME:-d5_a100}"

module purge
module load anaconda3/2024.6
conda activate qldd

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
export QLDD_SCRATCH="/scratch/gpfs/$USER/qldd"; mkdir -p "$QLDD_SCRATCH"

srun python -m qldd.train --config "$CONFIG"

if [ ! -f "runs/${NAME}/DONE" ] && grep -q '\[walltime\]' "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null; then
    echo "Requeuing to continue..."
    sbatch --export=CONFIG="$CONFIG",NAME="$NAME" "$0"
fi
