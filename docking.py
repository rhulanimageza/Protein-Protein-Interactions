import time
import torch
import torch.distributed as dist
from typing import Tuple, List, Optional
from config import DockingConfig
from geometry import get_com, apply_transformation
from scoring import score_complex
from interface import analyze_interface, InterfaceMetrics
from logging_gpu import log, print_gpu_status

def dock_chain_B_to_A(chain_A: torch.Tensor,
                            chain_B: torch.Tensor,
                            interface_A: Optional[torch.Tensor] = None,
                            interface_B: Optional[torch.Tensor] = None,
                            config: DockingConfig = DockingConfig(),
                            hydro_A: Optional[torch.Tensor] = None,
                            hydro_B: Optional[torch.Tensor] = None,
                            batch_size: int = 8,
                            verbose: bool = True
                            ) -> Tuple[torch.Tensor, List[InterfaceMetrics]]:
    """
    Batched docking of chain_B to chain_A using multiple restarts in parallel.
    Uses original functions only, fully GPU-aware.
    Returns:
        - b_docked_batch: Tensor [batch_size, n_atoms_B, 3]
        - metrics_batch: List of InterfaceMetrics per batch
    """
    rank = dist.get_rank()
    device = chain_A.device
    log(rank, f"Starting batched docking: batch_size={batch_size}")
    print_gpu_status(rank)
    start_time = time.time()

    n_atoms_B = chain_B.shape[0]

    # Center chains
    com_A = get_com(chain_A)
    ca_c = chain_A - com_A
    cb_c = chain_B - get_com(chain_B)

    # Compute normal vector
    normal = torch.tensor([0., 0., 1.], device=device)
    if interface_A is not None and len(interface_A):
        com_ia = get_com(ca_c[interface_A])
        com_ib = get_com(cb_c[interface_B]) if interface_B is not None else get_com(cb_c)
        normal = com_ia - com_ib
        normal /= torch.linalg.norm(normal) + 1e-9

    # Prepare batch of initial translations and rotations
    t_init = normal.unsqueeze(0).repeat(batch_size, 1) * config.initial_separation
    t_init += torch.randn(batch_size, 3, device=device) * 2.0
    r_init = torch.randn(batch_size, 3, device=device) * 0.4

    best_B = torch.zeros(batch_size, n_atoms_B, 3, device=device)
    best_metrics: List[InterfaceMetrics] = [None] * batch_size
    best_scores = torch.full((batch_size,), float('inf'), device=device)

    # ===================== LOOP OVER BATCH =====================
    for i in range(batch_size):
        # Run the normal dock_chain_B_to_A function for each restart
        B_docked, metrics = dock_chain_B_to_A(
            chain_A, cb_c,
            interface_A, interface_B,
            config, hydro_A, hydro_B,
            verbose=False
        )
        if B_docked is not None:
            best_B[i] = B_docked
            best_metrics[i] = metrics
            score = metrics.score if hasattr(metrics, "score") else 0.0
            best_scores[i] = score

        if verbose:
            log(rank, f" Batch {i+1}/{batch_size} docked, score={best_scores[i]:.2f}")
            print_gpu_status(rank)

    total_duration = time.time() - start_time
    log(rank, f"Batched docking complete ({batch_size} restarts) in {total_duration:.2f}s")
    print_gpu_status(rank)

    return best_B, best_metrics