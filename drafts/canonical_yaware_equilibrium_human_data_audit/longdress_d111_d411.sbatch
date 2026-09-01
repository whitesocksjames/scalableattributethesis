#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=yaw_ld_14

set -eu
root=/data/run01/scz0ade/Tanzeyu
src="$root/code/Scalable-Attribute-Thesis"
out="$root/experiments/canonical_yaware_equilibrium_r01/longdress/D111_D411"
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$root/envs/unicorn-me-py38"
export PYTHONNOUSERSITE=1 PYTHONPATH="$src" OMP_NUM_THREADS=6
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export LD_LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
cd "$src"
python scripts/scalable_attribute/canonical/evaluate_8ivfb_sequence.py \
  --sequence longdress --frame 1300 \
  --input-ply "$root/data/8iVFB/longdress/longdress_vox10_1300.ply" \
  --released-checkpoint-root "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt" \
  --gpcc-binary "$src/third_party/tmc3_v21" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --enhancement-step1763 "$root/experiments/canonical_yaware_equilibrium_r01/D111/train/checkpoints/step_3263.pth" \
  --enhancement-step3525 "$root/experiments/canonical_yaware_equilibrium_r01/D411/train/checkpoints/step_3263.pth" \
  --conditioning-lambda 32768 --output-dir "$out/result"
