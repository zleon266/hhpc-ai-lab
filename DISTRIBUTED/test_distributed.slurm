#!/bin/bash
#SBATCH --job-name=hhpc-d2-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=DISTRIBUTED/logs/test-ddp-%j.out
#SBATCH --error=DISTRIBUTED/logs/test-ddp-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8

source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "===== DISTRIBUTED TEST ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODES:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

echo "===== TRAINING START ====="
date --iso-8601=seconds

srun bash -c '
    NODE_RANK=$SLURM_NODEID

    torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      DISTRIBUTED/run_qa.py \
        --model_name_or_path google-bert/bert-base-uncased \
        --dataset_name rajpurkar/squad \
        --do_train \
        --do_eval \
        --max_train_samples 1000 \
        --max_eval_samples 200 \
        --per_device_train_batch_size 3 \
        --per_device_eval_batch_size 3 \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --logging_steps 20 \
        --save_strategy no \
        --report_to none \
        --seed 42 \
        --output_dir "DISTRIBUTED/outputs/test-${SLURM_JOB_ID}"
'

echo "===== TRAINING END ===="
date --iso-8601=seconds
