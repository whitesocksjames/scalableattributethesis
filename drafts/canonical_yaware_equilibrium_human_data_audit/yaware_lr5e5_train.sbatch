#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=yaw_lr5e5

set -eu
arm="$1"
weights="$2"
root=/data/run01/scz0ade/Tanzeyu
src="$root/code/Scalable-Attribute-Thesis"
out="$root/experiments/canonical_yaware_lr5e5_r01/$arm/train"
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$root/envs/unicorn-me-py38"
export PYTHONNOUSERSITE=1 PYTHONPATH="$src" OMP_NUM_THREADS=6
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export LD_LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
cd "$src"
python scripts/scalable_attribute/canonical/train_enhancement.py \
  --data-root "$root/data/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000" \
  --train-file-list "$root/data/scalable_attribute_thesis/datasets/RWTT/splits/model_95_5_seed0/train_h5.txt" \
  --released-checkpoint "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/32k8k/epoch_last.pth" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --resume-checkpoint "$root/experiments/canonical_enhancement_r01/full_pass_lr_compare/lr5e5/checkpoints/step_1763.pth" \
  --allow-resume-distortion-weight-change \
  --conditioning-lambda 32768 --rd-lambda 32768 \
  --distortion-weights "$weights" --lr 5e-5 --batch-size 4 \
  --max-steps 3525 --save-steps 2263 2763 3263 3525 \
  --seed 0 --output-dir "$out"
