#!/usr/bin/env bash
# Generate the Deliverable 1 BASELINE README from the project root.

set -euo pipefail

mkdir -p BASELINE

cat <<'EOF' > BASELINE/README.md
# HPC Tools AI Lab - Deliverable 1: BASELINE

## Objective

Fine-tune `google-bert/bert-base-uncased` on the SQuAD question-answering dataset using one NVIDIA A100 GPU.

## Implementation

The implementation is based on the official Hugging Face Transformers PyTorch Question Answering example.

Main files:

- `run_qa.py`: main training and evaluation script.
- `trainer_qa.py`: question-answering trainer support.
- `utils_qa.py`: question-answering post-processing utilities.
- `run_baseline.slurm`: Slurm script for the official baseline run.
- `smoke_test.py`: simple CUDA/A100 verification script.

The exact upstream Transformers commit is recorded in `UPSTREAM_COMMIT.txt`.

## Environment

- Python: 3.10.8
- PyTorch: 2.11.0+cu128
- Transformers: 5.17.0
- Datasets: 5.0.1
- Accelerate: 1.15.0
- Evaluate: 0.4.6
- GPU: NVIDIA A100-PCIE-40GB

The complete Python environment is recorded in `requirements.lock.txt`.

## Training configuration

- Model: `google-bert/bert-base-uncased`
- Dataset: `rajpurkar/squad`
- Number of GPUs: 1
- Epochs: 2
- Train batch size: 12
- Eval batch size: 12
- Learning rate: 3e-5
- Maximum sequence length: 384
- Document stride: 128
- Seed: 42
- Mixed precision: disabled

## Execution

Submit the baseline job with:

```bash
sbatch BASELINE/run_baseline.slurm
```

## Results

Official Slurm job ID: `9878508`

- GPU: NVIDIA A100-PCIE-40GB
- Training runtime: 44 min 46.58 s
- Training samples per second: 65.877
- Training steps per second: 5.49
- Final training loss: 0.9788
- Exact Match: 80.8988
- F1: 88.2655
- Evaluation runtime: 51.98 s

The original Slurm logs are stored in `BASELINE/logs/`.

## Notes

- This baseline used one GPU only and did not use distributed training.
- Mixed-precision training was disabled for this run.
- The reported metrics are from the official Slurm job listed above.
EOF

echo "Created BASELINE/README.md"

