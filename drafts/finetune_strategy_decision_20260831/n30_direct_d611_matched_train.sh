#!/bin/bash
set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38

gxx_runtime="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${gxx_runtime}:/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/data/apps/openblas/0.3.22/lib:${LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data/run01/scz0ade/Tanzeyu/.cache/torch_extensions
export MAX_JOBS=2

source_root=/data/run01/scz0ade/Tanzeyu/scratch/direct_d611_matched_source_20260831
output_root=/data/run01/scz0ade/Tanzeyu/experiments/direct_d611_matched_budget_v1
data_root=/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis
export PYTHONPATH="${source_root}:${PYTHONPATH:-}"

cd "${source_root}"
python scripts/scalable_attribute/canonical/train_enhancement_mixed.py \
  --arm-name M0_matched \
  --rwtt-root "${data_root}/datasets/RWTT/processed/train_h5/h5/100000" \
  --rwtt-file-list "${data_root}/datasets/RWTT/splits/model_95_5_seed0/train_h5.txt" \
  --mvub-root "${output_root}/unused_mvub" \
  --mvub-file-list "${output_root}/unused_mvub.txt" \
  --andrew-file-list "${output_root}/unused_andrew.txt" \
  --released-checkpoint "${data_root}/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/32k8k/epoch_last.pth" \
  --base-synthesis-checkpoint /data/run01/scz0ade/Tanzeyu/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth \
  --resume-checkpoint /data/run01/scz0ade/Tanzeyu/experiments/direct_d611_mixing_v1/train/M0/checkpoints/step_1762.pth \
  --conditioning-lambda 32768 \
  --rd-lambda 32768 \
  --distortion-weights 6,1,1 \
  --lr 5e-5 \
  --physical-batch-size 4 \
  --effective-batch-size 4 \
  --max-steps 3525 \
  --mvub-updates 0 \
  --save-steps 2000 2250 2500 2750 3000 3250 3525 \
  --validation-steps \
  --seed 0 \
  --output-dir "${output_root}/train"
