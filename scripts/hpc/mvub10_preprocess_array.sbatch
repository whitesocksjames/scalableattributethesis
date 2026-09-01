#!/bin/bash -l
#SBATCH --job-name=mvub10_h5
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/MVUB/logs/preprocess_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/MVUB/logs/preprocess_%A_%a.err

unset SLURM_EXPORT_ENV
set -euo pipefail

repo=/home/hpc/iwnt/iwnt193h/Scalable-Attribute-Thesis
root=/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/MVUB
python=/home/woody/iwnt/iwnt193h/software/private/conda/envs/unicorn-me-py38/bin/python
subjects=(Andrew10 David10 Phil10 Ricardo10 Sarah10)
directories=(andrew10 david10 phil10 ricardo10 sarah10)
subject=${subjects[$SLURM_ARRAY_TASK_ID]}
directory=${directories[$SLURM_ARRAY_TASK_ID]}

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$repo"
"$python" scripts/data/prepare_mvub10.py \
  --subject "$subject" \
  --input-dir "$root/extracted/full/$directory/ply" \
  --output-root "$root/processed/MVUB10_H5" \
  --max-points 100000
