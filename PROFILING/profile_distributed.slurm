#!/bin/bash
#SBATCH --job-name=hhpc-prof-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=PROFILING/logs/distributed-%j.out
#SBATCH --error=PROFILING/logs/distributed-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8

source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

TRACE_DIR="PROFILING/traces/distributed-${SLURM_JOB_ID}"

echo "===== DISTRIBUTED PROFILING ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "TRACE_DIR=$TRACE_DIR"

echo "NODES:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

echo "===== GPU ALLOCATION ====="
srun nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "===== PROFILING START ====="
date --iso-8601=seconds

srun bash -c '
    NODE_RANK=$SLURM_NODEID

    torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      PROFILING/profile_distributed.py \
        --model google-bert/bert-base-uncased \
        --batch-size 3 \
        --steps 15 \
        --trace-dir "'"$TRACE_DIR"'"
'

echo "===== PROFILING END ====="
date --iso-8601=seconds
