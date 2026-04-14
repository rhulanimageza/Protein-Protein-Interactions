import torch
from typing import List, Optional, Tuple
from interface import InterfaceMetrics
from logging_gpu import log, print_gpu_status

# Placeholder: replace with your trained diffusion model import
from learned_diffusion_model import DiffDockModel

# -----------------------------
# Learned diffusion docking
# -----------------------------
def perform_docking_diffusion(chain_A: torch.Tensor,
                              chain_B: torch.Tensor,
                              hydro_A: Optional[torch.Tensor] = None,
                              hydro_B: Optional[torch.Tensor] = None,
                              n_samples: int = 8,
                              device: torch.device = torch.device("cuda:0")
                              ) -> Tuple[List[torch.Tensor], List[InterfaceMetrics]]:

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    log(rank, f"Starting diffusion docking for batch size={n_samples}")
    print_gpu_status(rank)

    # Load pretrained diffusion model
    model = DiffDockModel().to(device)
    model.eval()

    # Preallocate outputs
    docked_list = []
    metrics_list = []

    with torch.no_grad():
        for i in range(n_samples):
            # Predict docked pose
            B_docked = model.sample(chain_A, chain_B, hydro_A=hydro_A, hydro_B=hydro_B)
            docked_list.append(B_docked)

            # Compute interface metrics (optional)
            metrics = InterfaceMetrics.from_chains(chain_A, B_docked)
            metrics_list.append(metrics)

            log(rank, f" Sample {i+1}/{n_samples} generated")
            print_gpu_status(rank)

    return docked_list, metrics_list
