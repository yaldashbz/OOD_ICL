#!/bin/bash
#OLS / ridge baselines on the same fine-tuning examples as scripts/adaptation.sbatch (same array index mapping).
#The data are replayed from the adaptation run's seeded CUDA generator, which only reproduces on the same GPU model
#(the adaptation jobs ran on H100 80GB HBM3); run_baselines.py refuses to save if the replay does not match.
#Usage (from repo root): sbatch --array=0-7 scripts/baselines.sbatch
#SBATCH --job-name=baselines
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm_logs/%x_%a.out

REPO=${SLURM_SUBMIT_DIR}
PHIS=(30 150)
NOISES=(0.0 0.25)
PHI=${PHIS[$((SLURM_ARRAY_TASK_ID % 2))]}
SEED=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
NOISE_VAR=${NOISES[$((SLURM_ARRAY_TASK_ID / 4))]}

export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
"$REPO/.venv/bin/python" -u run_baselines.py \
    --adaptation "$REPO/runs/fig2a/seed${SEED}_noise0.0_tasks11_angle${PHI}/adaptation_noise${NOISE_VAR}.npz"
