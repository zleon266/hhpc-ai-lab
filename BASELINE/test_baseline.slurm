#!/bin/bash
#SBATCH --job-name=hhpc-d1-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=BASELINE/logs/test-%j.out
#SBATCH --error=BASELINE/logs/test-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8
source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

python -u BASELINE/run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name rajpurkar/squad \
  --do_train \
  --do_eval \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --logging_steps 20 \
  --save_strategy no \
  --report_to none \
  --seed 42 \
  --output_dir BASELINE/outputs/test
