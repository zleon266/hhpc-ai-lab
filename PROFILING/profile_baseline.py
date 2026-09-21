import argparse
import os

import torch
from datasets import load_dataset
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--trace-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is unavailable"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Profile steps:", args.steps, flush=True)

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

        # Profiling focuses on execution cost rather than QA quality.
        # Dummy answer positions preserve the QA training computation graph.
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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    os.makedirs(args.trace_dir, exist_ok=True)

    prof_schedule = schedule(
        wait=2,
        warmup=3,
        active=10,
        repeat=1,
    )

    data_iter = iter(loader)

    torch.cuda.reset_peak_memory_stats()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(args.trace_dir),
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
                f"step={step + 1}/{args.steps} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

            prof.step()

    print("\n===== PROFILER SUMMARY =====", flush=True)

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        ),
        flush=True,
    )

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(
        f"\nPeak CUDA memory allocated: {peak_memory:.3f} GiB",
        flush=True,
    )

    print("BASELINE PROFILING PASSED", flush=True)


if __name__ == "__main__":
    main()
