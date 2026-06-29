import torch
from config import DockingConfig
from geometry import get_com


def lennard_jones_potential(r: torch.Tensor, epsilon: float = 0.55, sigma: float = 4.0) -> torch.Tensor:
    r = torch.maximum(r, torch.tensor(0.1, device=r.device))
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6**2 - sr6)


def estimate_hydrophobic_contacts(
    coords_A: torch.Tensor,
    coords_B: torch.Tensor,
    mask_A: torch.Tensor,
    mask_B: torch.Tensor,
    cutoff: float = 5.0
) -> int:
    if not (mask_A.any() and mask_B.any()):
        return 0
    dists = torch.cdist(coords_A[mask_A], coords_B[mask_B])
    return (dists < cutoff).sum().item()


def score_complex(
    coords_A: torch.Tensor,
    coords_B: torch.Tensor,
    interface_A = None,
    interface_B = None,
    config: DockingConfig = DockingConfig(),
    hydro_A = None,
    hydro_B = None
) -> torch.Tensor:
    a = coords_A[interface_A] if interface_A is not None else coords_A
    b = coords_B[interface_B] if interface_B is not None else coords_B

    dists = torch.cdist(a, b)

    # Clash
    clash_mask = dists < config.clash_threshold
    clash_penalty = torch.sum((config.clash_threshold - dists[clash_mask]) ** 2)

    # LJ attraction
    lj = lennard_jones_potential(dists)
    attraction = lj[lj < 0].sum()

    # Good contacts
    n_good = ((dists >= config.contact_min) & (dists <= config.contact_max)).sum()

    # Hydrophobic
    hydro_bonus = torch.tensor(0.0, device=dists.device)
    if hydro_A is not None and hydro_B is not None:
        n_h = estimate_hydrophobic_contacts(coords_A, coords_B, hydro_A, hydro_B)
        hydro_bonus = -config.weight_desolv_hydro * n_h

    # COM restraint fallback
    com_penalty = torch.tensor(0.0, device=dists.device)
    if interface_A is None or len(interface_A) == 0:
        d_com = torch.linalg.norm(get_com(coords_A) - get_com(coords_B))
        delta = d_com - config.com_restraint_dist
        com_penalty = config.weight_com_restraint * delta**2

    total = (
        config.weight_clash_vdw * clash_penalty +
        config.weight_attraction * attraction +
        -config.weight_contacts * n_good +
        hydro_bonus +
        com_penalty
    )
    return total
