import datetime
import torch
from typing import Literal

LogLevel = Literal["INFO", "WARNING", "ERROR"]


def log(rank: int, msg: str, level: LogLevel = "INFO") -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [Rank {rank}] {level}: {msg}", flush=True)


def print_gpu_status(rank: int) -> None:
    if not torch.cuda.is_available():
        log(rank, "No CUDA detected")
        return
    try:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(rank, f"GPU Mem: alloc {alloc:.2f} GB | res {reserved:.2f} GB | peak {peak:.2f} GB")
    except Exception as e:
        log(rank, f"GPU status error: {e}", level="WARNING")
