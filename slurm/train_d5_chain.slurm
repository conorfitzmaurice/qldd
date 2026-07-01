#!/bin/bash
#SBATCH --job-name=qldd-d5
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
srun ~/.conda/envs/qldd/bin/python -m qldd.train --config configs/d5_a100.yaml
