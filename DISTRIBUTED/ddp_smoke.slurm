#!/bin/bash
#SBATCH --job-name=hhpc-d2-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=DISTRIBUTED/logs/ddp-smoke-%j.out
#SBATCH --error=DISTRIBUTED/logs/ddp-smoke-%j.err

set -euo pipefail

module load cesga/2022
module load python/3.10.8
source "$STORE/mypython/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$STORE/hf-cache"
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODES:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

srun bash -c '
    NODE_RANK=$SLURM_NODEID

    torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      DISTRIBUTED/ddp_smoke.py
'
