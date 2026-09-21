# Optional Profiling Analysis

This directory contains an additional profiling experiment for the HPC Tools AI Lab.

The goal is to analyze why the distributed D2 experiment achieved an end-to-end speedup of approximately 2.11x instead of the theoretical 4x when scaling from one A100 GPU to four A100 GPUs.

## Experimental setup

The profiling experiments use the same general BERT question-answering workload as D1 and D2.

### Baseline profiling

- Job ID: `9905566`
- Nodes: 1
- GPUs: 1 x NVIDIA A100-PCIE-40GB
- Per-GPU batch size: 12
- Global batch size: 12
- Sequence length: 384
- Profiling schedule:
  - wait: 2 steps
  - warmup: 3 steps
  - active: 10 steps
- Total steps: 15
- Slurm elapsed time: approximately 55 seconds

### Distributed profiling

- Job ID: `9906244`
- Nodes: 2
- GPUs: 2 x A100 per node
- Total GPUs: 4
- Per-GPU batch size: 3
- Global batch size: 12
- Sequence length: 384
- PyTorch DistributedDataParallel (DDP)
- NCCL backend
- Profiling schedule:
  - wait: 2 steps
  - warmup: 3 steps
  - active: 10 steps
- Total steps: 15
- Slurm elapsed time: approximately 51 seconds

The same global batch size of 12 is preserved in both profiling experiments.

## Baseline results

For the single-GPU profiling run:

- `aten::mm`: approximately 936.5 ms of self CUDA time
- `aten::addmm`: approximately 453.5 ms
- attention backward: approximately 102.7 ms
- AdamW optimizer step: approximately 69.3 ms
- peak CUDA memory allocated: approximately 4.093 GiB
- total self CUDA time reported by the profiler: approximately 1.722 s

The main GPU cost is therefore dominated by matrix multiplication and attention-related computation.

## Distributed results

The four-GPU DDP run completed successfully across two nodes and four ranks.

Important profiling results from rank 0 include:

- `nccl:all_reduce`: approximately 603.3 ms
- NCCL AllReduce self CUDA percentage: approximately 46.94%
- `aten::mm`: approximately 313.5 ms
- `aten::addmm`: approximately 124.5 ms
- DDP forward: approximately 169.6 ms
- AdamW optimizer step: approximately 69.8 ms
- peak CUDA memory allocated per GPU: approximately 2.474 GiB
- total self CUDA time reported by the profiler: approximately 1.285 s

The profiler traces were successfully generated for all four ranks.

## Comparison

| Metric | 1 GPU | 4 GPU DDP |
|---|---:|---:|
| GPUs | 1 | 4 |
| Nodes | 1 | 2 |
| Per-GPU batch size | 12 | 3 |
| Global batch size | 12 | 12 |
| `aten::mm` self CUDA time | ~936.5 ms | ~313.5 ms |
| `aten::addmm` self CUDA time | ~453.5 ms | ~124.5 ms |
| Peak CUDA memory per GPU | ~4.093 GiB | ~2.474 GiB |
| NCCL AllReduce | N/A | ~603.3 ms |

The distributed configuration substantially reduces the computation performed by each individual GPU.

However, DDP must synchronize gradients between the four processes after backward computation.

In this experiment, NCCL AllReduce accounts for approximately 603 ms and almost 47% of the reported self CUDA time on rank 0.

This communication and synchronization overhead explains an important part of the gap between ideal linear scaling and the measured end-to-end D2 speedup.

## Relation to D1 and D2

The formal D1 runtime was approximately:

- 2686.58 seconds

The formal D2 runtime was approximately:

- 1270.47 seconds

Therefore:

```text
speedup = 2686.58 / 1270.47 ≈ 2.11x

Although four GPUs provide significantly more compute capacity, the workload does not achieve a 4x speedup because distributed training introduces:

- gradient synchronization
- NCCL collective communication
- inter-node communication
- synchronization barriers
- reduced per-GPU batch size
- other serial and data-loading overheads

The profiler provides direct evidence of the NCCL AllReduce component of this overhead.

## Memory behavior

The peak allocated GPU memory decreased from approximately:

- 4.093 GiB on the single GPU

to:

- 2.474 GiB per GPU in the distributed run

because the global batch is partitioned across four GPUs.

This demonstrates another practical benefit of distributed data parallel training: reduced per-device memory pressure.

## Trace files

The raw PyTorch profiler trace files are intentionally not committed to Git because of their size.

Approximate local trace sizes:

- baseline trace: 47 MB
- distributed traces: 223 MB

The directory `PROFILING/traces/` is excluded through `.gitignore`.

## Files

- `profile_baseline.py`: single-GPU profiling program
- `profile_baseline.slurm`: single-A100 Slurm job
- `profile_distributed.py`: four-GPU DDP profiling program
- `profile_distributed.slurm`: two-node distributed Slurm job
- `logs/`: profiling execution logs
- `traces/`: local raw profiler traces, ignored by Git
```
