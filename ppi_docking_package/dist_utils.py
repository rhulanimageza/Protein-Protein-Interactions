import os
import datetime
import torch
import torch.distributed as dist
from logging_gpu import log


def setup(rank: int, world_size: int, timeout_sec: int = 3600, retries: int = 5) -> None:
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '54321')

    for attempt in range(1, retries + 1):
        try:
            log(rank, f"DDP init attempt {attempt}/{retries} (timeout={timeout_sec}s)")
            dist.init_process_group(
                "gloo",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=timeout_sec),
            )
            torch.cuda.set_device(rank)
            torch.backends.cudnn.benchmark = True
            log(rank, f"DDP initialized on cuda:{rank}")
            return
        except RuntimeError as e:
            log(rank, f"DDP init failed: {str(e)}", level="ERROR")
            if attempt < retries:
                dist.destroy_process_group() if dist.is_initialized() else None
                torch.cuda.empty_cache()
            import time
            time.sleep(10)

    raise RuntimeError(f"Failed to initialize DDP after {retries} attempts")


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
