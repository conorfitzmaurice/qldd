#!/bin/bash
#SBATCH --job-name=qldd-stage1
#SBATCH --partition=mig                 # d=3 global limit fits the 10 GB MIG slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00                 # < 24h; harness checkpoints at ~22h
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=CHANGEME@princeton.edu
#SBATCH --requeue

# train the global-limit model, then print the go/no-go verdict;
# resumes automatically from runs/stage1_global/ckpt.pt
module purge
module load anaconda3/2024.6
conda activate qldd

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
export QLDD_SCRATCH="/scratch/gpfs/$USER/qldd"; mkdir -p "$QLDD_SCRATCH" 2>/dev/null || echo "(scratch unavailable, continuing)"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun python -m qldd.train --config configs/stage1_global.yaml

if [ -f runs/stage1_global/DONE ]; then
    echo "stage-1 verdict:"
    python scripts/stage1_verdict.py --run runs/stage1_global
elif grep -q '\[walltime\]' "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"; then
    echo "Walltime guard hit; requeuing to continue from checkpoint..."
    sbatch "$0"
else
    echo "Exited abnormally -- NOT requeuing. Check errors above."
fi
