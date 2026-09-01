#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=mvub10_smoke

set -eu
root=/data/run01/scz0ade/Tanzeyu
src="$root/code/Scalable-Attribute-Thesis"
tarfile="$root/data/MVUB/mvub10_h5.tar"
out="$root/experiments/mvub10_transfer_smoke"
stage="/dev/shm/Tanzeyu_${SLURM_JOB_ID}"
test ! -e "$stage"
mkdir -p "$stage" "$out/slurm"
cleanup() {
  case "$stage" in
    /dev/shm/Tanzeyu_${SLURM_JOB_ID}) rm -rf -- "$stage" ;;
    *) echo "Unsafe stage cleanup target: $stage" >&2; exit 2 ;;
  esac
}
trap cleanup EXIT

tar -xf "$tarfile" -C "$stage"
data="$stage/MVUB10_H5"
manifest="$stage/five_subject_smoke.txt"
: > "$manifest"
for subject in Andrew10 David10 Phil10 Ricardo10 Sarah10; do
  file=$(find "$data/$subject" -maxdepth 1 -type f -name '*.h5' | sort | head -n 1)
  test -n "$file"
  printf '%s\n' "$file" >> "$manifest"
done

source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$root/envs/unicorn-me-py38"
export PYTHONNOUSERSITE=1 PYTHONPATH="$src" OMP_NUM_THREADS=6
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export LD_LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
cd "$src"
python scripts/scalable_attribute/canonical/probe_human_data.py \
  --data-root "$data" --file-list "$manifest" \
  --released-checkpoint "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/32k8k/epoch_last.pth" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --enhancement-checkpoint "$root/experiments/canonical_enhancement_r01/full_pass_lr_compare/lr5e5/checkpoints/step_3525.pth" \
  --conditioning-lambda 32768 \
  --output "$out/probe_summary.json"

cp "$manifest" "$out/selected_h5.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$out/job_id.txt"
