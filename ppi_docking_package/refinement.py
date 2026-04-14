import time
import torch
import torch.distributed as dist
import numpy as np
from typing import Tuple

from logging_gpu import log, print_gpu_status

# -------------------------
# 1. GPU-perturbation
# -------------------------
def perturb_chain(coords: torch.Tensor, scale: float = 0.35) -> torch.Tensor:
    return coords + torch.randn_like(coords) * scale

# -------------------------
# 2. Classical refinement (vectorized)
# -------------------------
def refine_chain(coords: torch.Tensor, n_steps: int = 80, step_size: float = 0.012,
                 clash_dist: float = 3.2) -> torch.Tensor:
    pos = coords.clone()
    device = pos.device
    rank = dist.get_rank()
    log(rank, f"Starting classical refinement ({n_steps} steps)")
    start = time.time()

    for step in range(n_steps):
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float('inf'))

        clash_mask = d < clash_dist  # [N,N]
        if not clash_mask.any():
            log(rank, f" Refinement converged at step {step}")
            break

        # Vectorized forces
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N,N,3]
        mag = torch.clamp((clash_dist - d) / (d + 1e-8), min=0.0).unsqueeze(-1)  # [N,N,1]
        forces = (mag * diff * clash_mask.unsqueeze(-1)).sum(dim=1)  # [N,3]

        pos += step_size * forces
        pos += coords.mean(dim=0) - pos.mean(dim=0)

    log(rank, f"Classical refinement done ({time.time() - start:.2f}s)")
    return pos

# -------------------------
# 3. AF-Diffusion refinement (vectorized)
# -------------------------
def af_diffusion_refine(coords: torch.Tensor,
                        n_steps: int = 60,
                        noise_start: float = 0.65,
                        noise_end: float = 0.045,
                        bond_length: float = 3.81,
                        interface_cutoff: float = 8.2,
                        interface_weight: float = 2.8,
                        step_size: float = 0.020) -> torch.Tensor:
    rank = dist.get_rank()
    log(rank, "Starting AF-Diffusion refinement")
    start_time = time.time()
    print_gpu_status(rank)

    refined = coords.clone()
    device = refined.device
    sigmas = torch.linspace(noise_start, noise_end, n_steps, device=device)

    for step, sigma in enumerate(sigmas):
        step_start = time.time()
        noisy = refined + torch.randn_like(refined) * sigma

        # Vectorized pairwise distances
        dists = torch.cdist(noisy, noisy)
        dists.fill_diagonal_(float('inf'))

        diff = noisy.unsqueeze(1) - noisy.unsqueeze(0)  # [N,N,3]
        delta = dists - bond_length  # [N,N]
        force = -(delta.unsqueeze(-1) * diff) / (dists.unsqueeze(-1) + 1e-6)
        force[dists > interface_cutoff] = 0.0  # only within cutoff
        forces = force.sum(dim=1)  # [N,3]

        # boost interface atoms
        interface_atoms = (dists < interface_cutoff).any(dim=1)
        forces[interface_atoms] *= interface_weight

        refined = noisy + step_size * forces
        refined -= refined.mean(dim=0) - coords.mean(dim=0)

        if step % 10 == 0 or step == n_steps - 1:
            log(rank, f" Diffusion step {step+1}/{n_steps} finished (sigma={sigma:.3f}, {time.time()-step_start:.2f}s)")
            print_gpu_status(rank)

    log(rank, f"AF-Diffusion complete ({time.time() - start_time:.2f}s total)")
    print_gpu_status(rank)
    return refined

# -------------------------
# 4. Structure validation (vectorized)
# -------------------------
def is_valid_structure(coords: torch.Tensor,
                       min_dist: float = 2.0,
                       clash_dist: float = 3.0,
                       max_clash_frac: float = 0.04,
                       min_contacts_per_atom: float = 0.6) -> Tuple[bool, str]:
    n = coords.size(0)
    if n < 10:
        return False, "too few atoms"

    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float('inf'))

    if d.min() < min_dist:
        return False, f"atoms too close ({d.min().item():.2f} Å)"

    n_clash = (d < clash_dist).sum().item()
    if n_clash / (n * (n-1)) > max_clash_frac:
        return False, f"clash fraction too high ({n_clash/(n*(n-1)):.3%})"

    n_contact = ((d >= clash_dist) & (d < 8.0)).sum().item()
    if n_contact / n < min_contacts_per_atom:
        return False, f"too few contacts per atom ({n_contact/n:.2f})"

    return True, "valid"

# -------------------------
# 5. AF-confidence estimator (vectorized)
# -------------------------
def estimate_af_confidence(coords: torch.Tensor) -> float:
    dists = torch.cdist(coords, coords)
    dists.fill_diagonal_(float('inf'))
    clash_frac = (dists < 2.0).float().mean().item()
    compact_frac = (dists < 8.0).float().mean().item()
    smoothness = dists[dists < 10.0].std().item() if (dists < 10.0).any() else 10.0
    conf = 1.0 - 6.0 * clash_frac + 0.025 * compact_frac - 0.015 * smoothness
    return float(np.clip(conf, 0.0, 1.0))