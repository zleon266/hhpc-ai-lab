import os
import socket

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(local_rank)

    print(
        f"rank={rank} "
        f"world_size={world_size} "
        f"host={hostname} "
        f"local_rank={local_rank} "
        f"gpu={gpu_name}",
        flush=True,
    )

    x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(
        f"rank={rank} all_reduce_sum={x.item()}",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print("DDP SMOKE TEST PASSED", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()



