#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --job-name=j32k_8i

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

case "$SLURM_ARRAY_TASK_ID" in
  0) sequence=longdress; frame=1300 ;;
  1) sequence=loot; frame=1200 ;;
  2) sequence=redandblack; frame=1550 ;;
  3) sequence=soldier; frame=0690 ;;
  *) exit 64 ;;
esac

cd "$src"
python scripts/scalable_attribute/canonical/evaluate_8ivfb_sequence.py \
  --sequence "$sequence" \
  --frame "$frame" \
  --input-ply "$root/data/8iVFB/$sequence/${sequence}_vox10_${frame}.ply" \
  --released-checkpoint-root "$root/data/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt" \
  --base-synthesis-checkpoint "$root/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth" \
  --scalable-checkpoint "Joint32K_step3525=$root/experiments/joint32k_tafa_selectedbase_d111_0c4885f/input/step_3525.pth" \
  --gpcc-binary "$src/third_party/tmc3_v21" \
  --conditioning-lambda 32768 \
  --output-dir "$out/8ivfb/$sequence"
