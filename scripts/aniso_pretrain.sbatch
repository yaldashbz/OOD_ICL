#!/bin/bash
#Pretrain phi=60 models on an anisotropic cap (perpendicular scales e_2: 1.0, e_3: 0.2, e_4..e_10: 0.5), seed = array
#index, into runs/aniso/. ANISO=0 trains the isotropic cap instead, into runs/fig2a/ next to the Fig 2A sweep.
#Usage (from repo root): sbatch --array=0-2 scripts/aniso_pretrain.sbatch;  ANISO=0 sbatch --array=2 scripts/aniso_pretrain.sbatch
#SBATCH --job-name=pretrain60
#SBATCH --partition=gpu,gpu100
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|H100|A100|A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm_logs/%x_%a.out

REPO=${SLURM_SUBMIT_DIR}
ANISO=${ANISO:-1}
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/src"
if [ "$ANISO" = "1" ]; then
    EXTRA=(--task_scales 1.0 0.2 0.5 0.5 0.5 0.5 0.5 0.5 0.5 --run_dir "$REPO/runs/aniso/")
else
    EXTRA=(--run_dir "$REPO/runs/fig2a/")
fi
"$REPO/.venv/bin/python" -u run_linreg.py --angle 60 --seed "$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
