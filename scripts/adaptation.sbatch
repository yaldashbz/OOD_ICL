#!/bin/bash
#ICL vs FT vs FT+ICL on downstream tasks at angle delta, for the specialized (phi=30) and generalized (phi=150)
#models from the Fig 2A sweep. Array index = 4*noise_idx + 2*seed + phi_idx, noise_var in {0, 0.25}.
#Usage (from repo root): sbatch --array=0-7 scripts/adaptation.sbatch
#SBATCH --job-name=adapt
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|H100|A100|A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm_logs/%x_%a.out

REPO=${SLURM_SUBMIT_DIR}
PHIS=(30 150)
NOISES=(0.0 0.25)
PHI=${PHIS[$((SLURM_ARRAY_TASK_ID % 2))]}
SEED=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
NOISE_VAR=${NOISES[$((SLURM_ARRAY_TASK_ID / 4))]}

#torch's eager bmm backward JIT-compiles Triton kernels; a shared NFS cache races between jobs on one node
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
"$REPO/.venv/bin/python" -u run_adaptation.py --lrs 1e-4 3e-4 1e-3 3e-3 --noise_var "$NOISE_VAR" \
    --pretrained "$REPO/runs/fig2a/seed${SEED}_noise0.0_tasks11_angle${PHI}"
