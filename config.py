from dataclasses import dataclass
from typing import Optional

@dataclass
class DockingConfig:
    n_restarts: int = 1
    maxiter: int = 1000
    initial_separation: float = 30.0

    weight_clash_vdw: float = 1.0
    weight_attraction: float = 0.18
    weight_contacts: float = 0.12
    weight_desolv_hydro: float = 0.9
    weight_com_restraint: float = 0.04

    clash_threshold: float = 3.2
    contact_min: float = 3.8
    contact_max: float = 8.0
    com_restraint_dist: float = 25.0


@dataclass
class GenerationConfig:
    n_samples: int = 1
    max_attempts: int = 3
    noise_scale: float = 0.35
    classical_refine_steps: int = 80
    use_af_diffusion: bool = False          # often memory heavy → default off
    af_diffusion_steps: int = 60
    min_af_confidence: float = 0.58
