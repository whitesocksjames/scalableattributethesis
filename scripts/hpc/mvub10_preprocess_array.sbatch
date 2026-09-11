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
#SBATCH --output=/REDACTED/FAU_PROJECT_ROOT/datasets/MVUB/logs/preprocess_%A_%a.out
#SBATCH --error=/REDACTED/FAU_PROJECT_ROOT/datasets/MVUB/logs/preprocess_%A_%a.err

unset SLURM_EXPORT_ENV
set -euo pipefail

repo=/REDACTED/HPC_SHARED_ROOT/HPC_ACCOUNT/Scalable-Attribute-Thesis
root=/REDACTED/FAU_PROJECT_ROOT/datasets/MVUB
python=/REDACTED/FAU_ACCOUNT_ROOT/software/private/conda/envs/unicorn-me-py38/bin/python
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
