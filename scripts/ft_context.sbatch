#!/bin/bash
#Zero-context vs variable-context fine-tuning of the phi=150 models on delta=90 tasks (src/run_ft_context.py).
#Usage (from repo root): sbatch scripts/ft_context.sbatch
#SBATCH --job-name=ftctx
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|H100|A100|A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_logs/%x.out

REPO=${SLURM_SUBMIT_DIR}
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
"$REPO/.venv/bin/python" -u run_ft_context.py \
    --pretrained "$REPO/runs/fig2a/seed0_noise0.0_tasks11_angle150" "$REPO/runs/fig2a/seed1_noise0.0_tasks11_angle150" \
    --out "$REPO/runs/adaptation/ft_context_phi150_delta90.npz"
