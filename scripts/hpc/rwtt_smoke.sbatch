#!/bin/bash -l
#SBATCH --job-name=rwtt_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_smoke_%j.err

unset SLURM_EXPORT_ENV
set -euo pipefail

REPO_DIR=/home/hpc/iwnt/iwnt193h/Scalable-Attribute-Thesis
WORKSPACE_DIR=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis
RWTT_DIR=${WORKSPACE_DIR}/datasets/RWTT
PYTHON_BIN=/home/woody/iwnt/iwnt193h/software/private/conda/envs/unicorn-me-py38/bin/python
SMOKE_DIR=${RWTT_DIR}/processed/smoke/job_${SLURM_JOB_ID}

export PYTHONPATH=${REPO_DIR}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load cuda/11.8.0

mkdir -p "${WORKSPACE_DIR}/logs" "${SMOKE_DIR}"

echo "===== RWTT RWT4 SMOKE START ====="
date
hostname
echo "Output: ${SMOKE_DIR}"

"${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
  --process mesh2pc_color \
  --input_rootdir "${RWTT_DIR}/extracted/smoke" \
  --output_rootdir "${SMOKE_DIR}/dense" \
  --input_format obj \
  --input_length 1 \
  --output_length 1 \
  --num_points 500000 \
  --resolution 1023

"${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
  --process partition_color \
  --input_rootdir "${SMOKE_DIR}/dense/h5" \
  --output_rootdir "${SMOKE_DIR}/train_h5" \
  --input_format h5 \
  --input_length 1 \
  --max_points 100000

"${PYTHON_BIN}" - "${SMOKE_DIR}" <<'PY'
import sys
from pathlib import Path

import h5py

smoke_dir = Path(sys.argv[1])
dense_files = sorted((smoke_dir / "dense" / "h5").rglob("*.h5"))
part_files = sorted((smoke_dir / "train_h5" / "h5").rglob("*.h5"))

if len(dense_files) != 1:
    raise RuntimeError(f"Expected one dense HDF5, found {len(dense_files)}")
if len(part_files) <= 1:
    raise RuntimeError(
        f"Smoke did not exercise recursive KD-tree partitioning: {len(part_files)} block(s)"
    )

with h5py.File(dense_files[0], "r") as handle:
    sampled_unique = len(handle["coords"])
    coords_dtype = handle["coords"].dtype
    feats_dtype = handle["feats"].dtype
    rgb_min = handle["feats"][:].min()
    rgb_max = handle["feats"][:].max()

partition_sizes = []
for path in part_files:
    with h5py.File(path, "r") as handle:
        partition_sizes.append(len(handle["coords"]))

if max(partition_sizes) > 100000:
    raise RuntimeError(f"Partition exceeds 100000 points: {max(partition_sizes)}")

print("===== RWTT SMOKE SUMMARY =====")
print("Sampled points: 500000")
print(f"Unique voxels: {sampled_unique}")
print(f"Output partitions: {len(partition_sizes)}")
print(f"Partition sizes: {partition_sizes}")
print(f"Max partition size: {max(partition_sizes)}")
print(f"Coordinates dtype: {coords_dtype}")
print(f"Features dtype: {feats_dtype}")
print(f"RGB range: [{rgb_min}, {rgb_max}]")
print("RWTT RWT4 COMPLETE SMOKE: PASS")
PY

date
