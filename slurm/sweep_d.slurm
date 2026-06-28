#!/bin/bash
#SBATCH --job-name=qldd-sweepd
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
module purge; module load anaconda3/2025.12; conda activate qldd
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
srun ~/.conda/envs/qldd/bin/python scripts/sweep_decoder_D.py \
    --run "${RUN:-runs/d7_cap}" --shots "${SHOTS:-20000}" \
    --pmin 0.005 --pmax 0.03 --npoints 7 --steps 48 --batch 1024 \
    --out "${RUN:-runs/d7_cap}/sweep_D"
