#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --job-name=trace4k2k

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /REDACTED/N30_PROJECT_ROOT/envs/unicorn-me-py38
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR=/REDACTED/N30_PROJECT_ROOT/.cache/torch_extensions
export LD_LIBRARY_PATH=/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/REDACTED/N30_PROJECT_ROOT/code/Scalable-Attribute-Thesis
export OMP_NUM_THREADS=6

ROOT=/REDACTED/N30_PROJECT_ROOT/experiments/fixed_model_internal_trace_20260903
SOURCE=/REDACTED/N30_PROJECT_ROOT/code/Scalable-Attribute-Thesis

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python "${SOURCE}/scripts/diagnostics/fixed_model_internal_trace.py" \
  --external-manifest "${TRACE_MANIFEST:-${ROOT}/fixed_model_internal_trace_external_n30.tsv}" \
  --config "${TRACE_CONFIG:-${SOURCE}/scripts/diagnostics/fixed_model_internal_trace_n30.json}" \
  --gpcc-binary "${SOURCE}/third_party/tmc3_v21" \
  --output-dir "${ROOT}/${TRACE_RUN_NAME}" \
  --max-blocks "${TRACE_MAX_BLOCKS:-0}"
