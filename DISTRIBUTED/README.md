# Distributed Question Answering with PyTorch DDP

This directory contains the distributed-training part of the HPC Tools AI Lab.
The goal is to train the same question-answering workload used in D1 on multiple
GPUs and nodes, validate that distributed communication works correctly, and
compare the D2 execution with the single-GPU D1 baseline.

## Implementation

The D2 workflow uses PyTorch Distributed Data Parallel (DDP):

- `torchrun` starts one training process per GPU.
- NCCL is the communication backend for GPU collective operations.
- Slurm allocates two nodes with two GPUs per node and launches the distributed
  job.

The Python environment and the training setup are the same as D1, except for
the DDP launch and the per-device batch size needed to preserve the global batch
size.

## Resources

| Resource | D2 configuration |
| --- | --- |
| Nodes | 2 |
| GPUs per node | 2 NVIDIA A100 |
| Total GPUs | 4 NVIDIA A100 |
| CPUs per node | 64 |
| Total CPUs | 128 |
| Distributed processes | 4 ranks (one rank per GPU) |

## Validation runs

### NCCL/DDP smoke test

Slurm job `9883533` validated the distributed launch and collective
communication:

- 4 ranks across two nodes, with two GPUs per node;
- `all_reduce_sum = 10.0`;
- result: **passed**.

### Small distributed training test

Slurm job `9885006` performed a short end-to-end distributed training run:

| Setting or metric | Value |
| --- | --- |
| Training examples | 1,000 |
| Evaluation examples | 200 |
| Epochs | 1 |
| Per-device batch size | 3 |
| Global batch size | 12 |
| Training runtime | 9.325 s |
| Exact match (EM) | 20.5 |
| F1 | 23.6411 |

## Formal D2 run

The formal distributed run was submitted as Slurm job `9898964`.

### Allocation and training configuration

- Nodes: `a100-18`, `a100-19`
- GPUs: 4 x `NVIDIA A100-PCIE-40GB`
- Distributed training: `True`
- `ddp_find_unused_parameters=False`
- Per-device training batch size: 3
- Global batch size: 12
- Learning rate: `3e-5`
- Epochs: 2
- `max_seq_length`: 384
- `doc_stride`: 128
- Seed: 42
- Mixed precision: disabled

### Results

| Metric | D2 result |
| --- | --- |
| `train_runtime` | 21m 10.47s (approximately 1270.47 s) |
| `train_loss` | approximately 0.9830 |
| `train_samples_per_second` | approximately 139.305 |
| `train_steps_per_second` | approximately 11.61 |
| `eval_exact_match` | approximately 81.0974 |
| `eval_f1` | approximately 88.2824 |
| `eval_runtime` | approximately 16.40 s |

## D1 versus D2

The D1 reference run used one A100 GPU with batch size 12. The global batch
size was intentionally held at 12 in D2, so that the comparison does not change
the effective batch size.

| Metric | D1 | D2 |
| --- | --- | --- |
| GPUs | 1 A100 | 4 A100 |
| Global batch size | 12 | 12 |
| Training runtime | 44m 46.58s (approximately 2686.58 s) | 21m 10.47s (approximately 1270.47 s) |
| EM | approximately 80.8988 | approximately 81.0974 |
| F1 | approximately 88.2655 | approximately 88.2824 |

The observed training speedup is:

```text
speedup = D1 runtime / D2 runtime
        = 2686.58 / 1270.47
        ≈ 2.11x
```

The D2 run approximately halves training time while preserving comparable EM and
F1. The speedup is not expected to reach the ideal 4x because DDP introduces
NCCL communication and gradient synchronization, including across nodes. In
addition, each GPU processes a small local batch (3), which reduces per-GPU
compute efficiency, and data loading plus other serial overheads do not scale
with the number of GPUs.

## Directory contents

| File or directory | Purpose |
| --- | --- |
| `run_qa.py` | Question-answering training and evaluation entry point. |
| `trainer_qa.py` | QA training logic used by the workflow. |
| `utils_qa.py` | QA data and preprocessing utilities. |
| `ddp_smoke.py` | Minimal distributed communication validation program. |
| `ddp_smoke.slurm` | Slurm submission script for the DDP smoke test. |
| `test_distributed.slurm` | Slurm script for the small distributed end-to-end test. |
| `run_distributed.slurm` | Slurm script for the formal D2 run. |
| `requirements.lock.txt` | Locked Python dependencies for the environment. |
| `logs/` | Slurm standard-output and standard-error logs for submitted jobs. |

## Running the workflow

From the repository root, activate the same Python environment used for D1,
then submit the validation and formal jobs in order:

```bash
sbatch DISTRIBUTED/ddp_smoke.slurm
sbatch DISTRIBUTED/test_distributed.slurm
sbatch DISTRIBUTED/run_distributed.slurm
```

Check a submitted job with:

```bash
squeue -j <job_id>
scontrol show job <job_id>
```

The output and error logs are written under `DISTRIBUTED/logs/`. For the formal
run, inspect:

```bash
cat DISTRIBUTED/logs/run-9898964.out
cat DISTRIBUTED/logs/run-9898964.err
```

## Conclusion

The D2 implementation successfully ran PyTorch DDP over four A100 GPUs on two
nodes. The communication smoke test passed, the small distributed test completed
end-to-end, and the formal run achieved approximately 2.11x training speedup
over D1 while maintaining essentially the same QA evaluation quality.

