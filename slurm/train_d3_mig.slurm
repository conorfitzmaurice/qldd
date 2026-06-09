#!/bin/bash
#SBATCH --job-name=qldd-d3-mig
#SBATCH --partition=mig                 # 10 GB A100 slice; fast queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1               # MIG: CPU mem fixed at 32 GB
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00                 # < 24h; harness checkpoints at ~22h
#SBATCH --signal=B:USR1@300             # warn 5 min before kill (optional hook)
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=CHANGEME@princeton.edu
#SBATCH --requeue                       # allow automatic requeue to chain runs

# adjust module/conda names to the local Della setup
module purge
module load anaconda3/2024.6
conda activate qldd

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

# active output goes to fast scratch, not /projects
export QLDD_SCRATCH="/scratch/gpfs/$USER/qldd"
mkdir -p "$QLDD_SCRATCH"

srun python -m qldd.train --config configs/d3_mig.yaml

# auto-chain: if not finished, resubmit to continue from the checkpoint
if [ ! -f runs/d3_mig/DONE ] && grep -q '\[walltime\]' "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null; then
    echo "Requeuing to continue training from checkpoint..."
    sbatch "$0"
fi
