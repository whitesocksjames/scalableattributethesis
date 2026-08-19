#!/bin/bash -l
#SBATCH --job-name=rwtt_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_extract_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/rwtt_extract_%j.err

unset SLURM_EXPORT_ENV
set -euo pipefail

WORKSPACE_DIR=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis
RWTT_DIR=${WORKSPACE_DIR}/datasets/RWTT
RAW_DIR=${RWTT_DIR}/raw
EXTRACT_DIR=${RWTT_DIR}/extracted/full

mkdir -p "${WORKSPACE_DIR}/logs" "${EXTRACT_DIR}"

echo "===== RWTT EXTRACTION START ====="
date
hostname

archive_count=0
for archive in "${RAW_DIR}"/RealWorldTexturedThings_chunk*.zip; do
  echo "Extracting: ${archive}"
  unzip -q -n "${archive}" -d "${EXTRACT_DIR}"
  archive_count=$((archive_count + 1))
done

echo "Archives extracted: ${archive_count}"
echo "Top-level directories: $(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "OBJ files: $(find "${EXTRACT_DIR}" -type f -iname '*.obj' | wc -l)"
echo "MTL files: $(find "${EXTRACT_DIR}" -type f -iname '*.mtl' | wc -l)"
echo "PNG files: $(find "${EXTRACT_DIR}" -type f -iname '*.png' | wc -l)"

echo "===== EXTRACTED LAYOUT SAMPLE ====="
find "${EXTRACT_DIR}" -type f \( -iname '*.obj' -o -iname '*.mtl' -o -iname '*.png' \) \
  -printf '%P\n' | sort | head -60

echo "RWTT EXTRACTION: PASS"
date
