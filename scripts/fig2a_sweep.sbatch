#!/bin/bash
#Fig 2A sweep: one model per (seed, phi), phi in {15, 30, ..., 180}. Array index = 12*seed + (phi/15 - 1).
#Usage (from repo root): sbatch --array=0-23 scripts/fig2a_sweep.sbatch
#SBATCH --job-name=fig2a
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm_logs/%x_%a.out

REPO=${SLURM_SUBMIT_DIR}
SEED=$((SLURM_ARRAY_TASK_ID / 12))
ANGLE=$((15 * (SLURM_ARRAY_TASK_ID % 12 + 1)))
LOG2_TASKS=${LOG2_TASKS:-11}
NOISE_STD=${NOISE_STD:-0}

nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
"$REPO/.venv/bin/python" -u run_linreg.py --angle "$ANGLE" --seed "$SEED" \
    --log2_tasks "$LOG2_TASKS" --noise_std "$NOISE_STD" --run_dir "$REPO/runs/fig2a/"
