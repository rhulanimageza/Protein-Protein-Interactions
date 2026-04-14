import time
import torch
import torch.distributed as dist
from typing import Optional, Tuple, List
from docking import dock_chain_B_to_A
from logging_gpu import log, print_gpu_status
from config import DockingConfig
import torch.distributed as dist

def generate_synthetic_dataset_local(
    chain_A: torch.Tensor,
    chain_B: torch.Tensor,
    interface_A: Optional[torch.Tensor] = None,
    interface_B: Optional[torch.Tensor] = None,
    hydro_A: Optional[torch.Tensor] = None,
    hydro_B: Optional[torch.Tensor] = None,
    config: DockingConfig = DockingConfig(),
    rank: int = 0,
    n_samples: int = 1,
    max_attempts: int = 10,
    noise_scale: float = 0.35,
    classical_refine_steps: int = 80,
    use_af_diffusion: bool = False,
    af_diffusion_steps: int = 60,
    min_af_confidence: float = 0.58,
    verbose: bool = True
) -> Tuple[List[torch.Tensor], List[dict]]:
    """
    Fully GPU-batched synthetic dataset generation.
    Returns a list of sampled chains_B and metrics.
    """
    device = chain_A.device
    B_orig = chain_B.clone()
    samples = []
    metrics = []

    #rank = dist.get_rank()
    start_total = time.time()

    log(rank, f"Generating {n_samples} synthetic samples (max_attempts={max_attempts})")
    
    # Repeat chain_B for batched perturbations
    chain_B_batch = B_orig.unsqueeze(0).repeat(n_samples, 1, 1)  # [n_samples, N, 3]
    
    for attempt in range(max_attempts):
        if len(samples) >= n_samples:
            break

        attempt_start = time.time()
        # ------------------------ PERTURB ------------------------
        noise = torch.normal(0.0, noise_scale, size=chain_B_batch.shape, device=device)
        perturbed = chain_B_batch + noise

        # ------------------------ DOCK ALL ------------------------
        best_B_batch = []
        best_metrics_batch = []
        for i in range(perturbed.size(0)):
            B_i, metrics_i = dock_chain_B_to_A(
                chain_A, perturbed[i],
                interface_A, interface_B,
                config, hydro_A, hydro_B,
                verbose=False
            )
            if B_i is not None:
                best_B_batch.append(B_i)
                best_metrics_batch.append(metrics_i)

        # ------------------------ REFINE ------------------------
        refined_batch = []
        for B_i in best_B_batch:
            # Classical refinement on GPU
            pos = B_i.clone()
            for step in range(classical_refine_steps):
                dists = torch.cdist(pos, pos)
                dists.fill_diagonal_(float('inf'))
                clash_mask = dists < 3.2
                if not clash_mask.any():
                    break
                forces = torch.zeros_like(pos)
                for j in range(pos.size(0)):
                    js = torch.nonzero(clash_mask[j]).squeeze(-1)
                    if len(js) == 0:
                        continue
                    diff = pos[j] - pos[js]
                    dist = dists[j, js] + 1e-5
                    forces[j] += ((3.2 - dist).unsqueeze(-1) / dist.unsqueeze(-1) * diff).sum(dim=0)
                pos += 0.012 * forces
                pos += B_i.mean(dim=0) - pos.mean(dim=0)
            refined_batch.append(pos)

        # ------------------------ VALIDATE ------------------------
        for pos, metric in zip(refined_batch, best_metrics_batch):
            n_atoms = pos.size(0)
            if n_atoms < 10:
                continue
            # Optional: AF confidence check
            conf = getattr(metric, "confidence", 1.0) if hasattr(metric, "confidence") else 1.0
            if conf < min_af_confidence:
                continue
            samples.append(pos)
            metrics.append(metric)
            if verbose:
                log(rank, f"Sample {len(samples)}/{n_samples} accepted ({time.time()-attempt_start:.2f}s)")

    total_time = time.time() - start_total
    log(rank, f"Synthetic dataset generation done ({len(samples)} samples, {total_time:.2f}s total)")
    print_gpu_status(rank)
    return samples, metrics


import torch
from geometry import get_com

###Physics Aware SE(3) Perturbation

def random_rotation_matrix(device):
    """
    Uniform random SO(3) rotation matrix.
    """
    q = torch.randn(4, device=device)
    q = q / q.norm()

    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], device=device)
    return R


def structured_perturbation(
    coords: torch.Tensor,
    interface_mask: torch.Tensor = None,
    rot_scale: float = 1.0,
    trans_scale: float = 6.0,
    local_noise: float = 0.3,
    interface_noise_scale: float = 0.6
):
    device = coords.device

    com = get_com(coords)
    centered = coords - com

    # Random rotation
    R = random_rotation_matrix(device)
    rotated = centered @ R.T

    # Random translation
    translation = torch.randn(3, device=device) * trans_scale
    translated = rotated + translation

    # Local flexibility noise
    noisy = translated + torch.randn_like(translated) * local_noise

    # Interface destabilization
    if interface_mask is not None:
        mask = interface_mask.bool()
        noisy[mask] += torch.randn_like(noisy[mask]) * interface_noise_scale

    return noisy + com



    