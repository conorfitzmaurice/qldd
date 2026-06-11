#!/bin/bash
#SBATCH --job-name=qldd-lersweep
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=cf2185@princeton.edu

module purge
module load anaconda3/2025.12
conda activate qldd
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun ~/.conda/envs/qldd/bin/python scripts/ler_vs_p.py --run runs/stage1_global \
    --legacy --shots 1000000 --pmin 0.003 --pmax 0.03 --npoints 8
