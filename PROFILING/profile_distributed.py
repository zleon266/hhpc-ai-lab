import argparse
import os
import socket

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--trace-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(
        f"rank={rank} world_size={world_size} "
        f"host={socket.gethostname()} "
        f"local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset(
        "rajpurkar/squad",
        split="train[:512]",
    )

    def tokenize(batch):
        encoded = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=384,
            padding="max_length",
        )

        n = len(encoded["input_ids"])
        encoded["start_positions"] = [0] * n
        encoded["end_positions"] = [0] * n

        return encoded

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset.set_format("torch")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    rank_trace_dir = os.path.join(args.trace_dir, f"rank-{rank}")
    os.makedirs(rank_trace_dir, exist_ok=True)

    prof_schedule = schedule(
        wait=2,
        warmup=3,
        active=10,
        repeat=1,
    )

    sampler.set_epoch(0)
    data_iter = iter(loader)

    torch.cuda.reset_peak_memory_stats()

    dist.barrier()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(rank_trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        for step in range(args.steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            print(
                f"rank={rank} "
                f"step={step + 1}/{args.steps} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

            prof.step()

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    if rank == 0:
        print("\n===== DISTRIBUTED PROFILER SUMMARY =====", flush=True)

        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=40,
            ),
            flush=True,
        )

    print(
        f"rank={rank} peak_cuda_memory={peak_memory:.3f} GiB",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print("DISTRIBUTED PROFILING PASSED", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
