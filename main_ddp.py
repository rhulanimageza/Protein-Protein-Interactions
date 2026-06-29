"""
PPI Docking Package - Physics-aware DDP Multi-GPU
=================================================
Dr. James Jr. Mashiyane
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import torch.distributed as dist
# ====================== MODULE IMPORTS ======================
from dist_utils import setup, cleanup
from logging_gpu import log
from synthetic import generate_synthetic_dataset_local
from advanced_synthetic import generate_physics_aware_synthetic_dataset
from config import DockingConfig
from loadingPPI_data import load_pdb_complex_fast
from docking_diffusion import perform_docking_diffusion


# ============================================================
# DDP SAFE BROADCAST
# ============================================================

def broadcast_ndarray(rank: int, src_rank: int, arr: Optional[np.ndarray], dtype, device):
    """DDP-safe broadcast for any ndarray, including empty arrays."""
    # Step 1: Determine shape
    if rank == src_rank and arr is not None:
        shape = np.array(arr.shape, dtype=np.int64)
        size = np.array([arr.size], dtype=np.int64)
    else:
        shape = np.zeros(2, dtype=np.int64)
        size = np.array([0], dtype=np.int64)

    # Step 2: Broadcast shape and size
    shape_t = torch.from_numpy(shape).to(device)
    size_t = torch.from_numpy(size).to(device)
    dist.broadcast(shape_t, src=src_rank)
    dist.broadcast(size_t, src=src_rank)

    shape = shape_t.cpu().numpy()
    size_val = int(size_t.item())

    # Step 3: Prepare array on non-src ranks
    if rank != src_rank:
        if size_val == 0:
            # Return empty array of correct dtype
            return np.zeros((0,), dtype=dtype)
        arr = np.zeros(shape, dtype=dtype)

    # Step 4: Broadcast actual data if size > 0
    if size_val > 0:
        t = torch.from_numpy(arr.astype(dtype)).to(device)
        dist.broadcast(t, src=src_rank)
        arr = t.cpu().numpy()

    return arr


# ============================================================
# LOAD COMPLEX
# ============================================================

def load_complex(protein_file: str, ligand_file: str, rank: int):
    struct_A, struct_B, idx_A_np, idx_B_np, hydro_A_np, hydro_B_np = load_pdb_complex_fast(
        protein_file,
        ligand_file,
        chain_A_id="A",
        use_ca_only=True,
        contact_threshold=8.0,
        max_atoms=1500,
        rank=rank,
    )

    if struct_A is None or struct_B is None:
        return None

    return (
        struct_A.coords.astype(np.float32),
        struct_B.coords.astype(np.float32),
        hydro_A_np,
        hydro_B_np,
    )


# ============================================================
# MAIN
# ============================================================

def main(rank: int, world_size: int, args):

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    setup(rank, world_size)

    config = DockingConfig()

    # -------------------------------
    # Load single complex or list
    # -------------------------------
    if getattr(args, "protein", None) and getattr(args, "ligand", None):
        if not (os.path.exists(args.protein) and os.path.exists(args.ligand)):
            if rank == 0:
                log(rank, f"Protein or ligand file does not exist.", level="ERROR")
            dist.barrier()
            cleanup()
            return
        complex_list = [("single", args.protein, args.ligand)]
    else:
        complex_list = get_complex_list(args.refined_dir, args.mol2_dir)
        if args.pdbs is not None:
            complex_list = [item for item in complex_list if item[0] in args.pdbs]

    if len(complex_list) == 0:
        if rank == 0:
            log(rank, "No complexes found.", level="ERROR")
        dist.barrier()
        cleanup()
        return

    # -------------------------------
    # Loop over complexes
    # -------------------------------
    for pdb_id, protein_path, ligand_path in tqdm(complex_list, desc=f"Rank {rank}", position=rank):
        try:
            chain_A_np, chain_B_np, hydro_A_np, hydro_B_np = load_complex(protein_path, ligand_path, rank)
        except Exception as e:
            if rank == 0:
                log(rank, f"Loading failed for {pdb_id}: {e}", level="ERROR")
            chain_A_np = np.zeros((0,3), dtype=np.float32)
            chain_B_np = np.zeros((0,3), dtype=np.float32)
            hydro_A_np = np.zeros((0,), dtype=np.float32)
            hydro_B_np = np.zeros((0,), dtype=np.float32)

        # -------------------------------
        # DDP-safe broadcast
        # -------------------------------
        chain_A_np = broadcast_ndarray(rank, 0, chain_A_np, np.float32, device)
        chain_B_np = broadcast_ndarray(rank, 0, chain_B_np, np.float32, device)
        hydro_A_np = broadcast_ndarray(rank, 0, hydro_A_np, np.float32, device)
        hydro_B_np = broadcast_ndarray(rank, 0, hydro_B_np, np.float32, device)

        # If broadcast yielded empty arrays → skip
        if chain_A_np.size == 0 or chain_B_np.size == 0:
            if rank == 0:
                log(rank, f"No valid structure for {pdb_id}. Skipping.", level="WARNING")
            continue

        # -------------------------------
        # Convert to torch tensors
        # -------------------------------
        chain_A = torch.tensor(chain_A_np, dtype=torch.float32, device=device)
        chain_B = torch.tensor(chain_B_np, dtype=torch.float32, device=device)
        hydro_A = torch.tensor(hydro_A_np, dtype=torch.float32, device=device) if hydro_A_np.size else None
        hydro_B = torch.tensor(hydro_B_np, dtype=torch.float32, device=device) if hydro_B_np.size else None

        # -------------------------------
        # Generator switch
        # -------------------------------
        if args.generator == "baseline":
            samples_local, metrics_local = generate_synthetic_dataset_local(
                chain_A,
                chain_B,
                hydro_A=hydro_A,
                hydro_B=hydro_B,
                config=config,
                rank=rank,
                n_samples=args.n_samples,
                max_attempts=args.max_attempts,
                verbose=True
            )
        elif args.generator == "advanced":
            samples_local, metrics_local = generate_physics_aware_synthetic_dataset(
                chain_A,
                chain_B,
                config=config,
                hydro_A=hydro_A,
                hydro_B=hydro_B,
                n_samples=args.n_samples,
                max_attempts=args.max_attempts,
            )
        elif args.generator == "diffusion":
        # --- NEW LEARNED DIFFUSION GENERATOR ---

            samples_local, metrics_local = perform_docking_diffusion(
                chain_A,
                chain_B,
                hydro_A=hydro_A,
                hydro_B=hydro_B,
                n_samples=args.n_samples,
                device=device
            )
        else:
            raise ValueError("Unknown generator type")

        # -------------------------------
        # Save (Rank 0 Only)
        # -------------------------------
        if rank == 0 and samples_local:
            out_dir = Path(args.output_dir) / pdb_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(samples_local):
                np.save(out_dir / f"sample_{i}.npy", sample.cpu().numpy())
            print(f"[Rank 0] Saved {len(samples_local)} samples for {pdb_id}")

    # -------------------------------
    # DDP barrier & cleanup
    # -------------------------------
    dist.barrier()
    cleanup()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Physics-aware PPI Docking (DDP)")

    # =========================
    # Dataset control
    # =========================
    parser.add_argument("--refined-dir", type=str,default="/dev1/genomeGPT/FABind/PDB_Data/refined-set")
    parser.add_argument("--mol2-dir", type=str,default="/dev1/genomeGPT/FABind/PDB_Data/mol2")
    parser.add_argument("--pdbs", nargs="+", default=None,help="Specific PDB IDs to process")
    parser.add_argument("--protein", type=str, default=None,help="Single protein PDB file")
    parser.add_argument("--ligand", type=str, default=None,help="Single ligand MOL2 file")

    # =========================
    # Output
    # =========================
    parser.add_argument("--output-dir", type=str,default="/dev1/genomeGPT/LelapaAI_PPI_/ppi_docking_package/docking_outputs")

    # =========================
    # Generator Control
    # =========================
    parser.add_argument("--generator", type=str,default="baseline",choices=["baseline", "advanced", "diffusion"])

    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=10)

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log(rank, f"Launching DDP world_size={world_size}, rank={rank}")
    main(rank, world_size, args)

    
