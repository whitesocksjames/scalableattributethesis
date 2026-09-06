#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=d111_28lite

set -euo pipefail
root=/data/run01/scz0ade/Tanzeyu
src="$root/code/joint_endpoint_0c4885f"
out="$root/experiments/joint32k_tafa_selectedbase_d111_0c4885f/comparison_eval"

source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$root/envs/unicorn-me-py38"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$src"
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export LD_LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6

cd "$src"
python scripts/scalable_attribute/canonical/evaluate_scalable_formal.py \
  --data-root "$root/data/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000" \
  --file-list "$root/experiments/canonical_multirate_base_v1/rwtt_28lite/rwtt_28lite_manifest.txt" \
  --released-checkpoint "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/32k8k/epoch_last.pth" \
  --gpcc-binary "$src/third_party/tmc3_v21" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --enhancement-checkpoint "$root/experiments/canonical_enhancement_r01/full_pass_lr_compare/lr5e5/checkpoints/step_1763.pth" \
  --conditioning-lambda 32768 \
  --checkpoint-profile 32k8k \
  --output-dir "$out/rwtt_28lite/d111_step1763" \
  --require-exact
