import time
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from dataclasses import dataclass

# Simple log
def log(rank, msg, level="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [Rank {rank}] {level}: {msg}", flush=True)

@dataclass
class ProteinStructure:
    coords: np.ndarray
    residue_names: list
    chain_id: str
    residue_ids: list
    hydrophobic_mask: np.ndarray = None

# ================================================================
# Interface-aware atom limiting
# ================================================================
def limit_atoms(
    coords: np.ndarray,
    interface_idx: np.ndarray,
    max_atoms: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n_atoms = len(coords)
    if n_atoms <= max_atoms:
        return coords, interface_idx
    interface_idx = np.unique(interface_idx)
    n_interface = len(interface_idx)
    if n_interface >= max_atoms:
        selected = rng.choice(interface_idx, size=max_atoms, replace=False)
    else:
        remaining = np.setdiff1d(np.arange(n_atoms), interface_idx)
        n_needed = max_atoms - n_interface
        sampled = rng.choice(remaining, size=n_needed, replace=False)
        selected = np.concatenate([interface_idx, sampled])
    selected = np.sort(selected)
    new_coords = coords[selected]
    old_to_new = {old: new for new, old in enumerate(selected)}
    new_interface = np.array(
        [old_to_new[i] for i in interface_idx if i in old_to_new], dtype=int
    )
    return new_coords, new_interface

# ================================================================
# FAST PDB + ligand loader
# ================================================================
def load_pdb_complex_fast(
    protein_file: str,
    ligand_file: Optional[str] = None,
    chain_A_id: str = 'A',
    use_ca_only: bool = True,
    contact_threshold: float = 8.0,
    max_atoms: int = 1500,
    seed: int = 42,
    rank: int = 0
) -> Tuple[
    Optional[ProteinStructure],
    Optional[ProteinStructure],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray]
]:
    if rank != 0:
        return None, None, None, None, None, None

    start_time = time.time()
    rng = np.random.default_rng(seed)
    log(rank, f"Loading protein {protein_file} + ligand {ligand_file}")

    # Load protein
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", protein_file)
    except Exception as e:
        log(rank, f"Protein parse error: {e}", "ERROR")
        return None, None, None, None, None, None

    def load_protein_chain(chain_id: str):
        coords = []
        resnames = []
        resids = []
        for model in structure:
            if chain_id not in model:
                continue
            chain = model[chain_id]
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resname = res.resname.strip()
                resid = res.id[1]
                atoms = []
                if use_ca_only:
                    if "CA" in res:
                        atoms = [res["CA"]]
                else:
                    atoms = [a for a in res if a.element != "H"]
                for atom in atoms:
                    coords.append(atom.coord)
                    resnames.append(resname)
                    resids.append(resid)
            break  # only first model
        if not coords:
            return None
        coords = np.asarray(coords)
        hydro_mask = np.array([r in {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO"} for r in resnames])
        return ProteinStructure(
            coords=coords,
            residue_names=resnames,
            chain_id=chain_id,
            residue_ids=resids,
            hydrophobic_mask=hydro_mask
        )

    struct_A = load_protein_chain(chain_A_id)

    # Load ligand as single "chain"
    struct_B = None
    if ligand_file:
        try:
            coords = []
            with open(ligand_file, 'r') as f:
                atom_section = False
                for line in f:
                    if line.startswith("@<TRIPOS>ATOM"):
                        atom_section = True
                        continue
                    if line.startswith("@<TRIPOS>"):
                        atom_section = False
                    if atom_section:
                        parts = line.split()
                        if len(parts) < 9:
                            continue
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        coords.append([x, y, z])
            coords = np.asarray(coords)
            struct_B = ProteinStructure(
                coords=coords,
                residue_names=["LIG"]*len(coords),
                chain_id="LIG",
                residue_ids=list(range(len(coords))),
                hydrophobic_mask=np.ones(len(coords), dtype=bool)
            )
        except Exception as e:
            log(rank, f"Ligand parse error: {e}", "ERROR")
            struct_B = None

    if struct_A is None or struct_B is None:
        log(rank, "Chain loading failed", "ERROR")
        return None, None, None, None, None, None

    log(rank, f"Raw atoms: A {len(struct_A.coords)} / B {len(struct_B.coords)}")

    # Interface detection
    dists = cdist(struct_A.coords, struct_B.coords)
    interface_A = np.where(np.any(dists < contact_threshold, axis=1))[0]
    interface_B = np.where(np.any(dists < contact_threshold, axis=0))[0]
    log(rank, f"Interface atoms: A {len(interface_A)} / B {len(interface_B)}")

    # Limit atoms
    struct_A.coords, interface_A = limit_atoms(struct_A.coords, interface_A, max_atoms, rng)
    struct_B.coords, interface_B = limit_atoms(struct_B.coords, interface_B, max_atoms, rng)

    log(rank, f"Limited atoms: A {len(struct_A.coords)} / B {len(struct_B.coords)}")
    log(rank, f"Load done in {time.time()-start_time:.2f}s")

    return (
        struct_A, struct_B,
        interface_A.astype(np.int64),
        interface_B.astype(np.int64),
        struct_A.hydrophobic_mask,
        struct_B.hydrophobic_mask
    )