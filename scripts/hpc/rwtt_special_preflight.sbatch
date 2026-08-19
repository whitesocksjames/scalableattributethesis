#!/bin/bash -l
#SBATCH --job-name=rwtt_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_preflight_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_preflight_%A_%a.err

unset SLURM_EXPORT_ENV
set -euo pipefail

REPO_DIR=/home/hpc/iwnt/iwnt193h/Scalable-Attribute-Thesis
WORKSPACE_DIR=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis
RWTT_DIR=${WORKSPACE_DIR}/datasets/RWTT
EXTRACT_DIR=${RWTT_DIR}/extracted/full
DENSE_DIR=${RWTT_DIR}/processed/dense
TRAIN_DIR=${RWTT_DIR}/processed/train_h5
PYTHON_BIN=/home/woody/iwnt/iwnt193h/software/private/conda/envs/unicorn-me-py38/bin/python

export PYTHONPATH=${REPO_DIR}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load cuda/11.8.0

mkdir -p "${WORKSPACE_DIR}/logs" "${DENSE_DIR}" "${TRAIN_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL_NAME=RWT11
    SOURCE_DIR=${EXTRACT_DIR}/RWT11
    SOURCE_OBJ=${SOURCE_DIR}/cruxici.obj
    TEMP_ROOT=$(mktemp -d)
    TEMP_MODEL_DIR=${TEMP_ROOT}/RWT11
    mkdir -p "${TEMP_MODEL_DIR}"

    for source_file in "${SOURCE_DIR}"/*; do
      if [[ "$(basename "${source_file}")" != "cruxici.obj" ]]; then
        ln -s "${source_file}" "${TEMP_MODEL_DIR}/$(basename "${source_file}")"
      fi
    done

    "${PYTHON_BIN}" - "${SOURCE_OBJ}" "${TEMP_MODEL_DIR}/cruxici.obj" <<'PY'
import sys

source_path, output_path = sys.argv[1:]
replacement_count = 0

with open(source_path, "r", errors="replace") as source, open(output_path, "w") as output:
    for line in source:
        if line.startswith("vn "):
            fields = line.split()
            for index in range(1, min(4, len(fields))):
                token = fields[index].upper()
                if "#IND" in token or "#INF" in token:
                    fields[index] = "0"
                    replacement_count += 1
            line = " ".join(fields) + "\n"
        output.write(line)

print(f"Temporary invalid-normal replacements: {replacement_count}")
if replacement_count == 0:
    raise RuntimeError("Expected invalid RWT11 normals were not found")
PY

    INPUT_ROOT=${TEMP_ROOT}
    INPUT_OBJ=${TEMP_MODEL_DIR}/cruxici.obj
    DENSE_H5=${DENSE_DIR}/h5/RWT11/cruxici.h5
    ;;
  1)
    MODEL_NAME=RWT112
    INPUT_ROOT=${EXTRACT_DIR}
    INPUT_OBJ=${EXTRACT_DIR}/RWT112/model.obj
    DENSE_H5=${DENSE_DIR}/h5/RWT112/model.h5
    ;;
  *)
    echo "Unexpected preflight index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "===== RWTT SPECIAL PREFLIGHT START ====="
date
hostname
echo "Model: ${MODEL_NAME}"
echo "OBJ: ${INPUT_OBJ}"

"${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
  --process mesh2pc_color \
  --input_rootdir "${INPUT_ROOT}" \
  --input_file "${INPUT_OBJ}" \
  --output_rootdir "${DENSE_DIR}" \
  --num_points 10000000 \
  --resolution 1023

"${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
  --process partition_color \
  --input_rootdir "${DENSE_DIR}/h5" \
  --input_file "${DENSE_H5}" \
  --output_rootdir "${TRAIN_DIR}" \
  --max_points 100000

"${PYTHON_BIN}" - "${DENSE_H5}" "${TRAIN_DIR}/h5/100000/${MODEL_NAME}" <<'PY'
import sys
from pathlib import Path

import h5py

dense_path = Path(sys.argv[1])
part_dir = Path(sys.argv[2])

with h5py.File(dense_path, "r") as handle:
    dense_count = len(handle["coords"])
    coords_dtype = handle["coords"].dtype
    feats_dtype = handle["feats"].dtype
    rgb_min = handle["feats"][:].min()
    rgb_max = handle["feats"][:].max()

part_files = sorted(part_dir.glob("*.h5"))
part_sizes = []
for path in part_files:
    with h5py.File(path, "r") as handle:
        part_sizes.append(len(handle["coords"]))

if not part_sizes or sum(part_sizes) != dense_count:
    raise RuntimeError("Partition total does not match dense HDF5")
if max(part_sizes) > 100000:
    raise RuntimeError(f"Partition exceeds 100000: {max(part_sizes)}")

print("===== SPECIAL PREFLIGHT SUMMARY =====")
print(f"Dense points: {dense_count}")
print(f"Partitions: {len(part_sizes)}")
print(f"Max partition size: {max(part_sizes)}")
print(f"Coordinates dtype: {coords_dtype}")
print(f"Features dtype: {feats_dtype}")
print(f"RGB range: [{rgb_min}, {rgb_max}]")
print("SPECIAL PREFLIGHT: PASS")
PY

date
