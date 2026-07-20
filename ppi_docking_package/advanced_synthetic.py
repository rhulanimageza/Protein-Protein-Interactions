"""
Advanced Physics-Aware Synthetic Generator
==========================================

Research-grade synthetic dataset engine
for small-data PPI learning.

Design goals:
- SE(3)-aware perturbation
- Multi-restart docking
- Physics refinement
- Diffusion optional
- Diversity filtering
- Hard negative generation
- Curriculum tagging
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

from docking import dock_chain_B_to_A
from refinement import refine_chain, af_diffusion_refine, is_valid_structure
from geometry import get_com
from config import DockingConfig


# ============================================================
# 1. SO(3) Random Rotation
# ============================================================

def random_rotation_matrix(device):
    q = torch.randn(4, device=device)
    q = q / q.norm()

    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], device=device)
    return R


# ============================================================
# 2. Structured SE(3) Perturbation
# ============================================================

def structured_perturbation(
    coords: torch.Tensor,
    interface_mask: Optional[torch.Tensor] = None,
    trans_scale: float = 6.0,
    local_noise: float = 0.3,
    interface_noise_scale: float = 0.6
) -> torch.Tensor:

    device = coords.device
    com = get_com(coords)
    centered = coords - com

    R = random_rotation_matrix(device)
    rotated = centered @ R.T

    translation = torch.randn(3, device=device) * trans_scale
    translated = rotated + translation

    noisy = translated + torch.randn_like(translated) * local_noise

    if interface_mask is not None:
        mask = interface_mask.bool()
        noisy[mask] += torch.randn_like(noisy[mask]) * interface_noise_scale

    return noisy + com


# ============================================================
# 3. Multi-Restart Docking
# ============================================================

def multi_restart_dock(
    chain_A: torch.Tensor,
    chain_B: torch.Tensor,
    config: DockingConfig,
    interface_A=None,
    interface_B=None,
    hydro_A=None,
    hydro_B=None
):

    best_pose = None
    best_metrics = None
    best_score = -1e9

    for _ in range(config.n_restarts):

        pose, metrics = dock_chain_B_to_A(
            chain_A,
            chain_B,
            interface_A,
            interface_B,
            config,
            hydro_A,
            hydro_B,
            verbose=False
        )

        if pose is None:
            continue

        score = getattr(metrics, "score", 0.0)

        if score > best_score:
            best_score = score
            best_pose = pose
            best_metrics = metrics

    return best_pose, best_metrics


# ============================================================
# 4. Diversity Filtering
# ============================================================

def rmsd(a: torch.Tensor, b: torch.Tensor):
    return torch.sqrt(((a - b) ** 2).sum(dim=1).mean())


def is_diverse(new_pose: torch.Tensor,
               existing: List[torch.Tensor],
               threshold: float = 2.0):

    for pose in existing:
        if rmsd(new_pose, pose) < threshold:
            return False
    return True


# ============================================================
# 5. Hard Negative Generation
# ============================================================

def generate_hard_negative(coords: torch.Tensor,
                           strength: float = 4.0):
    offset = torch.randn(3, device=coords.device) * strength
    return coords + offset


# ============================================================
# 6. Curriculum Tagging
# ============================================================

def assign_difficulty(score: float, confidence: float):

    if score > 0.8 and confidence > 0.8:
        return "easy"
    elif score > 0.5:
        return "medium"
    return "hard"


# ============================================================
# 7. Main Advanced Generator
# ============================================================

def generate_physics_aware_synthetic_dataset(
    chain_A: torch.Tensor,
    chain_B: torch.Tensor,
    config: DockingConfig,
    interface_A=None,
    interface_B=None,
    hydro_A=None,
    hydro_B=None,
    n_samples: int = 5,
    max_attempts: int = 15,
    use_diffusion: bool = False
) -> Tuple[List[torch.Tensor], List[Dict]]:

    samples = []
    metadata = []

    for attempt in range(max_attempts):

        if len(samples) >= n_samples:
            break

        # 1. Perturb
        perturbed = structured_perturbation(chain_B)

        # 2. Dock
        best_pose, best_metrics = multi_restart_dock(
            chain_A,
            perturbed,
            config,
            interface_A,
            interface_B,
            hydro_A,
            hydro_B
        )

        if best_pose is None:
            continue

        # 3. Refine
        refined = refine_chain(best_pose)

        if use_diffusion:
            refined = af_diffusion_refine(refined)

        # 4. Validate
        valid, _ = is_valid_structure(refined)
        if not valid:
            continue

        # 5. Diversity
        if not is_diverse(refined, samples):
            continue

        score = getattr(best_metrics, "score", 0.0)
        confidence = getattr(best_metrics, "confidence", 1.0)

        difficulty = assign_difficulty(score, confidence)

        samples.append(refined)
        metadata.append({
            "score": score,
            "confidence": confidence,
            "difficulty": difficulty,
            "label": 1
        })

        # 6. Add hard negative
        negative = generate_hard_negative(refined)

        samples.append(negative)
        metadata.append({
            "score": 0.0,
            "confidence": 0.0,
            "difficulty": "hard",
            "label": 0
        })

    return samples, metadata