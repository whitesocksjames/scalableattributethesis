#!/bin/bash -l
#SBATCH --job-name=scalable_cli_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/scalable_cli_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/scalable_cli_smoke_%j.err

set -euo pipefail
module load cuda/11.8.0

export PATH="$WORK/software/private/conda/envs/unicorn-me-py38/bin:$PATH"
export PYTHONPATH="$HOME/Scalable-Attribute-Thesis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

ROOT="$WORK/scalable_attribute_thesis"
RUN="$ROOT/experiments/scalable_attribute_cli_smoke/$SLURM_JOB_ID"
BASE="$ROOT/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/2k128/epoch_last.pth"
DATA="$ROOT/datasets/RWTT/processed/train_h5/h5/100000/RWT116"
mkdir -p "$RUN"

python "$HOME/Scalable-Attribute-Thesis/scripts/scalable_attribute/train.py" \
  --data-root "$DATA" \
  --base-checkpoint "$BASE" \
  --output-dir "$RUN/train" \
  --base-lambda 2048 \
  --rd-lambda 1000 \
  --lr 1e-4 \
  --epochs 1 \
  --max-steps 1 \
  --save-every 1 \
  --val-every 1

python "$HOME/Scalable-Attribute-Thesis/scripts/scalable_attribute/evaluate.py" \
  --data-root "$DATA" \
  --base-checkpoint "$BASE" \
  --enhancement-checkpoint "$RUN/train/checkpoints/last.pth" \
  --output-dir "$RUN/eval" \
  --base-lambda 2048 \
  --max-samples 1 \
  --gpcc-binary "$HOME/Scalable-Attribute-Thesis/third_party/tmc3_v21"

find "$RUN" -maxdepth 3 -type f -printf '%P\n' | sort
