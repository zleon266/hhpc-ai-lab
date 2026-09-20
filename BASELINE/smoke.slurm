#!/bin/bash
#SBATCH --job-name=hhpc-d1-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=BASELINE/logs/smoke-%j.out
#SBATCH --error=BASELINE/logs/smoke-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8

source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "===== GPU INFO ====="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "===== PYTORCH TEST ====="
python -u BASELINE/smoke_test.py
