#!/bin/bash -l
#SBATCH --job-name=scalable_attr_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/scalable_attr_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/scalable_attr_smoke_%j.err

set -euo pipefail
module load cuda/11.8.0

export PATH="$WORK/software/private/conda/envs/unicorn-me-py38/bin:$PATH"
export PYTHONPATH="$HOME/Scalable-Attribute-Thesis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SCALABLE_DATA_ROOT="$WORK/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000"
export SCALABLE_BASE_CHECKPOINT="$WORK/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/2k128/epoch_last.pth"
export SCALABLE_SMOKE_OUTPUT="$WORK/scalable_attribute_thesis/logs/scalable_attribute_smoke_${SLURM_JOB_ID}.json"
export SCALABLE_BASE_LAMBDA=2048
export SCALABLE_RD_LAMBDA=1000

RUNTIME="$WORK/scalable_attribute_thesis/experiments/scalable_attribute_smoke/runtime"
mkdir -p "$RUNTIME"
if [[ ! -e "$RUNTIME/tmc3_v21" ]]; then
  ln -s "$HOME/Scalable-Attribute-Thesis/third_party/tmc3_v21" "$RUNTIME/tmc3_v21"
fi
cd "$RUNTIME"

python "$HOME/Scalable-Attribute-Thesis/scripts/scalable_attribute/smoke_test.py"
