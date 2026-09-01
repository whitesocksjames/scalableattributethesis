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
export PYTHONPATH="${source_root}:${PYTHONPATH:-}"
train_root="${output_root}/train/checkpoints"
old_m0=/data/run01/scz0ade/Tanzeyu/experiments/direct_d611_mixing_v1/train/M0/checkpoints/step_1762.pth
two_stage=/data/run01/scz0ade/Tanzeyu/experiments/canonical_yaware_lr5e5_r01/D611/train/checkpoints/step_3525.pth
data_root=/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis
released_root="${data_root}/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt"
base_checkpoint=/data/run01/scz0ade/Tanzeyu/experiments/canonical_base_r01/full_pass_from_step2000/checkpoints/step_5525.pth
gpcc="${source_root}/third_party/tmc3_v21"

cd "${source_root}"

checkpoint_args=(
  --enhancement-checkpoint "Direct_s1762=${old_m0}"
  --enhancement-checkpoint "Direct_s2000=${train_root}/step_2000.pth"
  --enhancement-checkpoint "Direct_s2250=${train_root}/step_2250.pth"
  --enhancement-checkpoint "Direct_s2500=${train_root}/step_2500.pth"
  --enhancement-checkpoint "Direct_s2750=${train_root}/step_2750.pth"
  --enhancement-checkpoint "Direct_s3000=${train_root}/step_3000.pth"
  --enhancement-checkpoint "Direct_s3250=${train_root}/step_3250.pth"
  --enhancement-checkpoint "Direct_s3525=${train_root}/step_3525.pth"
  --enhancement-checkpoint "TwoStage_s3525=${two_stage}"
)

for spec in \
  "longdress:1300:/data/run01/scz0ade/Tanzeyu/data/8iVFB/longdress/longdress_vox10_1300.ply" \
  "loot:1200:/data/run01/scz0ade/Tanzeyu/data/8iVFB/loot/loot_vox10_1200.ply" \
  "redandblack:1550:/data/run01/scz0ade/Tanzeyu/data/8iVFB/redandblack/redandblack_vox10_1550.ply" \
  "soldier:0690:/data/run01/scz0ade/Tanzeyu/data/8iVFB/soldier/soldier_vox10_0690.ply"
do
  IFS=: read -r sequence frame input_ply <<<"${spec}"
  python scripts/scalable_attribute/canonical/evaluate_8ivfb_sequence.py \
    --sequence "${sequence}" \
    --frame "${frame}" \
    --input-ply "${input_ply}" \
    --released-checkpoint-root "${released_root}" \
    --base-synthesis-checkpoint "${base_checkpoint}" \
    "${checkpoint_args[@]}" \
    --gpcc-binary "${gpcc}" \
    --conditioning-lambda 32768 \
    --output-dir "${output_root}/8ivfb/${sequence}"
done

python scripts/scalable_attribute/canonical/analyze_direct_d611_trajectory.py \
  --screen-root "${output_root}/8ivfb" \
  --checkpoint-root "${train_root}" \
  --initial-checkpoint "${old_m0}" \
  --output-dir "${output_root}/analysis"

while IFS== read -r label checkpoint
do
  python scripts/scalable_attribute/canonical/evaluate_scalable_formal.py \
    --data-root "${data_root}/datasets/RWTT/processed/train_h5/h5/100000" \
    --file-list "${data_root}/datasets/RWTT/splits/model_95_5_seed0/val_h5.txt" \
    --released-checkpoint "${released_root}/32k8k/epoch_last.pth" \
    --gpcc-binary "${gpcc}" \
    --base-synthesis-checkpoint "${base_checkpoint}" \
    --enhancement-checkpoint "${checkpoint}" \
    --conditioning-lambda 32768 \
    --output-dir "${output_root}/rwtt_full28/${label}" \
    --require-exact
done < "${output_root}/analysis/shortlist.txt"

python scripts/scalable_attribute/canonical/analyze_direct_d611_trajectory.py \
  --screen-root "${output_root}/8ivfb" \
  --checkpoint-root "${train_root}" \
  --initial-checkpoint "${old_m0}" \
  --output-dir "${output_root}/analysis" \
  --full28-root "${output_root}/rwtt_full28" \
  --reference-full28 /data/run01/scz0ade/Tanzeyu/experiments/canonical_yaware_lr5e5_r01/formal/D611_step3525_full28/endpoint_summary.csv
