#!/bin/bash
#SBATCH --job-name=hhpc-d1-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=BASELINE/logs/run-%j.out
#SBATCH --error=BASELINE/logs/run-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8
source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

echo "===== GPU ====="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "===== TRAINING START ====="
date --iso-8601=seconds

python -u BASELINE/run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name rajpurkar/squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --logging_steps 50 \
  --save_strategy no \
  --report_to none \
  --seed 42 \
  --output_dir "BASELINE/outputs/run-${SLURM_JOB_ID}"

echo "===== TRAINING END ====="
date --iso-8601=seconds
