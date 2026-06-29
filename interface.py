from dataclasses import dataclass
import torch
from typing import Optional
from config import DockingConfig
from scoring import score_complex, estimate_hydrophobic_contacts

@dataclass
class InterfaceMetrics:
    n_contacts: int
    min_dist: float
    mean_dist: float
    std_dist: float
    n_hbond_compatible: int
    n_hydrophobic_contacts: int
    score: float = 0.0

    def is_valid(self, min_contacts: int = 22, max_mean_dist: float = 7.4) -> bool:
        return (self.n_contacts >= min_contacts and
                self.min_dist >= 2.3 and
                self.mean_dist <= max_mean_dist)

def analyze_interface(coords_A: torch.Tensor, coords_B: torch.Tensor,
                      interface_A: Optional[torch.Tensor] = None,
                      interface_B: Optional[torch.Tensor] = None,
                      config: DockingConfig = DockingConfig(),
                      hydro_A: Optional[torch.Tensor] = None,
                      hydro_B: Optional[torch.Tensor] = None) -> InterfaceMetrics:
    a = coords_A[interface_A] if interface_A is not None else coords_A
    b = coords_B[interface_B] if interface_B is not None else coords_B
    dists = torch.cdist(a, b)
    return InterfaceMetrics(
        n_contacts=((dists >= config.contact_min) & (dists <= config.contact_max)).sum().item(),
        min_dist=dists.min().item(),
        mean_dist=dists.mean().item(),
        std_dist=dists.std().item(),
        n_hbond_compatible=((dists >= 2.6) & (dists <= 3.4)).sum().item(),
        n_hydrophobic_contacts=estimate_hydrophobic_contacts(coords_A, coords_B, hydro_A, hydro_B) if hydro_A is not None else 0,
        score=score_complex(coords_A, coords_B, interface_A, interface_B, config, hydro_A, hydro_B).item()
    )
