#!/bin/bash -l
#SBATCH --job-name=rwtt_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_full_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_full_%A_%a.err

unset SLURM_EXPORT_ENV
set -euo pipefail

REPO_DIR=/home/hpc/iwnt/iwnt193h/Scalable-Attribute-Thesis
WORKSPACE_DIR=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis
RWTT_DIR=${WORKSPACE_DIR}/datasets/RWTT
EXTRACT_DIR=${RWTT_DIR}/extracted/full
PROCESSED_DIR=${RWTT_DIR}/processed
DENSE_DIR=${PROCESSED_DIR}/dense
TRAIN_DIR=${PROCESSED_DIR}/train_h5
MANIFEST=${RWTT_DIR}/manifests/rwtt_obj_files.txt
EXCLUDED_MANIFEST=${RWTT_DIR}/manifests/excluded_models.txt
PYTHON_BIN=/home/woody/iwnt/iwnt193h/software/private/conda/envs/unicorn-me-py38/bin/python

export PYTHONPATH=${REPO_DIR}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load cuda/11.8.0

OBJ_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
if [[ -z "${OBJ_FILE}" ]]; then
  echo "No OBJ entry for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

RELATIVE_OBJ=${OBJ_FILE#${EXTRACT_DIR}/}
RELATIVE_STEM=${RELATIVE_OBJ%.*}

if [[ -f "${EXCLUDED_MANIFEST}" ]] && grep -qF "${RELATIVE_OBJ} |" "${EXCLUDED_MANIFEST}"; then
  echo "Excluded model: ${RELATIVE_OBJ}"
  exit 0
fi

DENSE_H5=${DENSE_DIR}/h5/${RELATIVE_STEM}.h5
PART_DIR=${TRAIN_DIR}/h5/100000/$(dirname "${RELATIVE_STEM}")
PART_STEM=$(basename "${RELATIVE_STEM}")

mkdir -p "${WORKSPACE_DIR}/logs" "${DENSE_DIR}" "${TRAIN_DIR}"

echo "===== RWTT MODEL START ====="
date
hostname
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "OBJ: ${OBJ_FILE}"
echo "Dense HDF5: ${DENSE_H5}"

STATUS=$("${PYTHON_BIN}" - "${DENSE_H5}" "${PART_DIR}" "${PART_STEM}" <<'PY'
import sys
from pathlib import Path

import h5py

dense_path = Path(sys.argv[1])
part_dir = Path(sys.argv[2])
part_stem = sys.argv[3]

def h5_length(path):
    try:
        with h5py.File(path, "r") as handle:
            coords = handle["coords"]
            feats = handle["feats"]
            if len(coords) == 0 or len(coords) != len(feats):
                return None
            return len(coords)
    except (OSError, KeyError):
        return None

dense_length = h5_length(dense_path)
if dense_length is None:
    print("PROCESS")
    raise SystemExit

part_files = sorted(part_dir.glob(f"{part_stem}_P*.h5"))
part_lengths = [h5_length(path) for path in part_files]
if (
    part_files
    and all(length is not None and length <= 100000 for length in part_lengths)
    and sum(part_lengths) == dense_length
):
    print("COMPLETE")
else:
    print("DENSE_ONLY")
PY
)

echo "Existing output status: ${STATUS}"
if [[ "${STATUS}" == "COMPLETE" ]]; then
  echo "Valid output already exists; skipping."
  exit 0
fi

if [[ "${STATUS}" == "PROCESS" ]]; then
  "${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
    --process mesh2pc_color \
    --input_rootdir "${EXTRACT_DIR}" \
    --input_file "${OBJ_FILE}" \
    --output_rootdir "${DENSE_DIR}" \
    --num_points 10000000 \
    --resolution 1023
fi

"${PYTHON_BIN}" "${REPO_DIR}/scripts/data/prepare_rwtt.py" \
  --process partition_color \
  --input_rootdir "${DENSE_DIR}/h5" \
  --input_file "${DENSE_H5}" \
  --output_rootdir "${TRAIN_DIR}" \
  --max_points 100000

"${PYTHON_BIN}" - "${DENSE_H5}" "${PART_DIR}" "${PART_STEM}" <<'PY'
import sys
from pathlib import Path

import h5py

dense_path = Path(sys.argv[1])
part_dir = Path(sys.argv[2])
part_stem = sys.argv[3]

with h5py.File(dense_path, "r") as handle:
    dense_length = len(handle["coords"])

part_files = sorted(part_dir.glob(f"{part_stem}_P*.h5"))
part_lengths = []
for path in part_files:
    with h5py.File(path, "r") as handle:
        if len(handle["coords"]) != len(handle["feats"]):
            raise RuntimeError(f"coords/feats mismatch: {path}")
        part_lengths.append(len(handle["coords"]))

if not part_lengths or sum(part_lengths) != dense_length:
    raise RuntimeError("Partition total does not match dense HDF5")
if max(part_lengths) > 100000:
    raise RuntimeError(f"Partition exceeds 100000: {max(part_lengths)}")

print(f"Dense points: {dense_length}")
print(f"Partitions: {len(part_lengths)}")
print(f"Max partition size: {max(part_lengths)}")
print("RWTT MODEL PREPROCESSING: PASS")
PY

date
