#!/bin/bash
#ICL vs FT on fixed targets cos(delta) v + sin(delta) e_j at delta=60 for all perpendicular directions j=1..9
#(e_2 = A, e_3 = B in the anisotropic cap), with datasets paired across directions.
#Usage (from repo root): sbatch scripts/directions.sbatch <pretrained run dir>
#SBATCH --job-name=dirs
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|H100|A100|A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_logs/%x_%j.out

REPO=${SLURM_SUBMIT_DIR}
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
"$REPO/.venv/bin/python" -u run_adaptation.py --pretrained "$1" --deltas 60 --directions 1 2 3 4 5 6 7 8 9 \
    --tasks_val 8 --tasks_test 24 --ns 16 32 64 128 --lrs 1e-4 3e-4 1e-3 3e-3 --n_ctx 32
