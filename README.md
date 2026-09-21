# HPC Tools AI Lab

This repository contains the deliverables for the HPC Tools AI Lab.

## Deliverables

- `BASELINE/`
  - Deliverable 1
  - Single NVIDIA A100 baseline
  - Git tag: `BASELINE`

- `DISTRIBUTED/`
  - Deliverable 2
  - PyTorch DistributedDataParallel (DDP)
  - 2 nodes × 2 NVIDIA A100 GPUs per node
  - 4 GPUs in total
  - Git tag: `DISTRIBUTED`

## Optional profiling analysis

- `PROFILING/`
  - Additional PyTorch Profiler analysis
  - Compares a controlled single-GPU workload with a 4-GPU DDP workload
  - Examines CUDA computation, NCCL AllReduce communication, and GPU memory usage
  - This is an optional profiling microbenchmark and is not a direct profile of the complete D1/D2 training jobs

## Repository structure

```text
.
├── BASELINE/
├── DISTRIBUTED/
├── PROFILING/
└── README.md
```
