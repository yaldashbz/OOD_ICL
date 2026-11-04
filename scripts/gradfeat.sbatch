#!/bin/bash
#Gradient features of the pretrained checkpoints (run_grad_features.py) for each run_adaptation.py --directions output
#given as an argument. The data are replayed from the adaptation run's seeded CUDA generator, which only reproduces on
#the same GPU model (the direction jobs ran on H100 80GB HBM3); run_grad_features.py refuses to save otherwise.
#Usage (from repo root): sbatch scripts/gradfeat.sbatch runs/fig2a/seed*_angle60/adaptation_dirs_*.npz runs/aniso/*/adaptation_dirs_*.npz
#SBATCH --job-name=gradfeat
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm_logs/%x_%j.out

REPO=${SLURM_SUBMIT_DIR}
FILES=()
for f in "$@"; do
    FILES+=("$(realpath "$f")")
done
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
for f in "${FILES[@]}"; do
    "$REPO/.venv/bin/python" -u run_grad_features.py --adaptation "$f" || exit 1
done
