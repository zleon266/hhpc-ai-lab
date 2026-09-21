#!/bin/bash
#SBATCH --job-name=hhpc-prof-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=PROFILING/logs/baseline-%j.out
#SBATCH --error=PROFILING/logs/baseline-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8

source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

TRACE_DIR="PROFILING/traces/baseline-${SLURM_JOB_ID}"

echo "===== BASELINE PROFILING ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "TRACE_DIR=$TRACE_DIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "===== PROFILING START ====="
date --iso-8601=seconds

python -u PROFILING/profile_baseline.py \
  --model google-bert/bert-base-uncased \
  --batch-size 12 \
  --steps 15 \
  --trace-dir "$TRACE_DIR"

echo "===== PROFILING END ====="
date --iso-8601=seconds
