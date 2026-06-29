import subprocess
from pathlib import Path
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import torch
from transformers import AutoModel, AutoTokenizer , pipeline

# -----------------------------
# CONFIG
# -----------------------------

pdb_dir = Path("diffdock/examples")
ligand_dir = Path("diffdock/examples")
fasta_dir = Path("diffdock/examples/fasta")
ankh_output_dir = Path("diffdock/data/ankh_output")
diffdock_model_dir = Path("diffdock/workdir/paper_score_model")
diffdock_conf_model_dir = Path("diffdock/workdir/paper_confidence_model")
diffdock_out_dir = Path("diffdock/test_output")
ankh_model_name = "ElnaggarLab/ankh2-ext2"

fasta_dir.mkdir(parents=True, exist_ok=True)
ankh_output_dir.mkdir(parents=True, exist_ok=True)
diffdock_out_dir.mkdir(parents=True, exist_ok=True)

extract_script = Path("diffdock/ankh/scripts/extract.py").resolve()
esm_root = Path("diffdock/ankh").resolve()

print("===================================")
print("DiffDock Automatic Docking Pipeline")
print("===================================")


# -----------------------------
# FIX 1: Per-chain FASTA generation
# -----------------------------

def pdb_to_fasta(pdb_file: Path, fasta_file: Path):
    """
    Write one FASTA entry PER CHAIN, using the chain letter as the sequence ID.
    DiffDock's ESM loader looks up embeddings by chain ID (e.g. 'A', 'B'),
    so the FASTA headers must match those chain letters exactly.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, str(pdb_file))

    records = []

    for model in structure:
        for chain in model:
            seq = ""
            for residue in chain:
                if PDB.is_aa(residue, standard=True):  # skip non-standard/HETATM
                    try:
                        # BioPython >= 1.78 compatibility
                        one_letter = PDB.Polypeptide.protein_letters_3to1.get(
                            residue.get_resname(), 'X'
                        )
                    except AttributeError:
                        # Fallback for older BioPython
                        try:
                            one_letter = PDB.Polypeptide.three_to_one(residue.get_resname())
                        except Exception:
                            one_letter = 'X'
                    seq += one_letter

            if seq:
                record = SeqRecord(
                    Seq(seq),
                    id=chain.id,          # <-- chain letter, e.g. 'A', 'B'
                    description=""
                )
                records.append(record)
                print(f"  [INFO] Chain {chain.id}: {len(seq)} residues")

        break  # only process first MODEL

    if records:
        SeqIO.write(records, fasta_file, "fasta")
        print(f"[INFO] FASTA generated: {fasta_file} ({len(records)} chain(s))")
        return True

    print(f"[WARNING] No amino acids found in {pdb_file}")
    return False


# -----------------------------
# MAIN PIPELINE
# -----------------------------

pdb_files = list(pdb_dir.glob("*.pdb"))
print(f"[INFO] Found {len(pdb_files)} PDB files")

for pdb_file in pdb_files:

    base_name = pdb_file.stem
    base_name = base_name.replace("_protein_processed", "")
    base_name = base_name.replace("_protein", "")

    ligand_file = ligand_dir / f"{base_name}_ligand.sdf"

    if not ligand_file.exists():
        print(f"[WARNING] Ligand missing for {pdb_file.name}, skipping.")
        continue

    print(f"\nProcessing complex: {pdb_file.stem}")

    # -----------------------------
    # FASTA generation
    # -----------------------------

    fasta_file = fasta_dir / f"{pdb_file.stem}.fasta"

    # FIX 2: Always regenerate FASTA to avoid stale single-chain cache
    success = pdb_to_fasta(pdb_file, fasta_file)
    if not success:
        continue

    # -----------------------------
    # Ankh embeddings
    # -----------------------------

    ankh_output_path = ankh_output_dir / pdb_file.stem
    ankh_output_path.mkdir(parents=True, exist_ok=True)

    # FIX 3: Always regenerate embeddings after fixing FASTA
    # (stale single-chain .pt files cause the chain mismatch error)
    import shutil
    if ankh_output_path.exists():
        shutil.rmtree(ankh_output_path)
    ankh_output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        ankh_model_name
    )

    anhk_model = AutoModel.from_pretrained(
        ankh_model_name
    )

    input_fasta_file = tokenizer(
        str(fasta_file.resolve()),
        return_tensors = "pt"
    )
    print("[INFO] Generating Ankh embeddings...")
    with torch.no_grad():
        outputs =anhk_model(**input_fasta_file)
    
    embedding = outputs.last_hidden_state.mean(dim=1)

    torch.save(embedding, ankh_output_path + f"/{pdb_file.stem}.pt" )

    # -----------------------------
    # DiffDock inference
    # -----------------------------

    cmd_diffdock = [
        "python",
        "diffdock/inference.py",
        "--protein_path", str(pdb_file),
        "--ligand", str(ligand_file),
        "--model_dir", str(diffdock_model_dir),
        "--confidence_model_dir", str(diffdock_conf_model_dir),
        "--out_dir", str(diffdock_out_dir),
        # FIX 4: Explicitly pass the embeddings path so DiffDock finds them
        "--anhk_embeddings_path", str(ankh_output_path.resolve()),
        "--samples_per_complex", "5"
    ]

    print("[INFO] Running DiffDock inference...")
    print(" ".join(cmd_diffdock))
    subprocess.run(cmd_diffdock, check=True)

    print(f"[SUCCESS] Docking complete → {diffdock_out_dir}/{pdb_file.stem}")

    







    





