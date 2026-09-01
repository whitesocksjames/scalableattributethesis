#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=yaw5e5_eval

set -eu
root=/data/run01/scz0ade/Tanzeyu
src="$root/code/Scalable-Attribute-Thesis"
new="$root/experiments/canonical_yaware_lr5e5_r01"
old="$root/experiments/canonical_yaware_r01"
out="$new/subset28_trajectory"
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$root/envs/unicorn-me-py38"
export PYTHONNOUSERSITE=1 PYTHONPATH="$src" OMP_NUM_THREADS=6
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export LD_LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
cd "$src"
python scripts/scalable_attribute/canonical/evaluate_enhancement_subset.py \
  --data-root "$root/data/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000" \
  --file-list "$old/yaware_subset_28h5.txt" \
  --released-checkpoint "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/32k8k/epoch_last.pth" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --conditioning-lambda 32768 --gpcc-binary "$src/third_party/tmc3_v21" \
  --checkpoint D411_lr5e5_p500="$new/D411/train/checkpoints/step_2263.pth" \
  --checkpoint D411_lr5e5_p1000="$new/D411/train/checkpoints/step_2763.pth" \
  --checkpoint D411_lr5e5_p1500="$new/D411/train/checkpoints/step_3263.pth" \
  --checkpoint D411_lr5e5_p1762="$new/D411/train/checkpoints/step_3525.pth" \
  --checkpoint D611_lr5e5_p500="$new/D611/train/checkpoints/step_2263.pth" \
  --checkpoint D611_lr5e5_p1000="$new/D611/train/checkpoints/step_2763.pth" \
  --checkpoint D611_lr5e5_p1500="$new/D611/train/checkpoints/step_3263.pth" \
  --checkpoint D611_lr5e5_p1762="$new/D611/train/checkpoints/step_3525.pth" \
  --output-dir "$out"
