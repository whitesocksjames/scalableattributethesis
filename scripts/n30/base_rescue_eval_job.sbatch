#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --job-name=base_rescue_eval

set -euo pipefail
: "${RESCUE_TRAIN_DIR:?required}"
: "${RESCUE_STEPS_CSV:?required}"
: "${RESCUE_EVAL_ROOT:?required}"
: "${RESCUE_SOURCE_ROOT:?required}"
: "${RESCUE_DATA_ROOT:?required}"
: "${RESCUE_FILE_LIST:?required}"
: "${RESCUE_RELEASED_CHECKPOINT:?required}"
: "${RESCUE_CHECKPOINT_PROFILE:?required}"
: "${RESCUE_LAMBDA:?required}"
: "${RESCUE_GPCC_BINARY:?required}"

source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=/data/run01/scz0ade/Tanzeyu/.cache/torch_extensions
export LD_LIBRARY_PATH=/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PYTHONPATH="${RESCUE_SOURCE_ROOT}"
IFS=, read -r -a RESCUE_STEPS <<< "${RESCUE_STEPS_CSV}"
cd "${RESCUE_SOURCE_ROOT}"
exec python scripts/scalable_attribute/canonical/evaluate_base_rescue_arm.py \
  --train-dir "${RESCUE_TRAIN_DIR}" \
  --steps "${RESCUE_STEPS[@]}" \
  --eval-root "${RESCUE_EVAL_ROOT}" \
  --source-root "${RESCUE_SOURCE_ROOT}" \
  --data-root "${RESCUE_DATA_ROOT}" \
  --file-list "${RESCUE_FILE_LIST}" \
  --released-checkpoint "${RESCUE_RELEASED_CHECKPOINT}" \
  --checkpoint-profile "${RESCUE_CHECKPOINT_PROFILE}" \
  --conditioning-lambda "${RESCUE_LAMBDA}" \
  --gpcc-binary "${RESCUE_GPCC_BINARY}"
