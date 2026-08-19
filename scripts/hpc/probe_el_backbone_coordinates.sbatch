#!/bin/bash -l
#SBATCH --job-name=el_coord_gate
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/el_coord_gate_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/logs/el_coord_gate_%j.err

set -euo pipefail
module load cuda/11.8.0

export PATH="$WORK/software/private/conda/envs/unicorn-me-py38/bin:$PATH"
export PYTHONPATH="$HOME/Scalable-Attribute-Thesis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export UNICORN_PROBE_DATA_ROOT="$WORK/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000"
export UNICORN_PROBE_CHECKPOINT="$WORK/scalable_attribute_thesis/checkpoints/unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/lossy_attribute/rwtt/2k128/epoch_last.pth"
export UNICORN_PROBE_OUTPUT="$WORK/scalable_attribute_thesis/logs/el_coordinate_gate_${SLURM_JOB_ID}.json"
export UNICORN_PROBE_LMB=2048

"$WORK/software/private/conda/envs/unicorn-me-py38/bin/python" \
  "$HOME/Scalable-Attribute-Thesis/scripts/probes/probe_el_backbone_coordinates.py" \
  --color_format yuv --normalize 1 --stage 1 --scale 5 --Vmode 1
