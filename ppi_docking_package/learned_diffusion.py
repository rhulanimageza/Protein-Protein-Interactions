import torch
from pathlib import Path

class DiffDockModel:
    """
    Clean wrapper for loading DiffDock score and confidence models.
    Designed for integration into DDP docking pipeline.
    """

    def __init__(
        self,
        score_ckpt_path: str,
        confidence_ckpt_path: str = None,
        device: torch.device = torch.device("cuda")
    ):
        self.device = device

        # Load score model
        self.score_model = self._load_model(score_ckpt_path)

        # Load confidence model (optional but recommended)
        self.confidence_model = None
        if confidence_ckpt_path is not None:
            self.confidence_model = self._load_model(confidence_ckpt_path)

        self.score_model.eval()
        if self.confidence_model:
            self.confidence_model.eval()

    def _load_model(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # DiffDock checkpoints usually store full model state dict
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint

        # ⚠️ IMPORTANT:
        # You must import the exact DiffDock model architecture
        # from their codebase.
        from models.score_model import ScoreModel  # adjust path if needed

        model = ScoreModel()
        model.load_state_dict(model_state)
        model.to(self.device)

        return model

    @torch.no_grad()
    def dock(self, protein_graph, ligand_graph):
        """
        Runs diffusion inference for docking.
        protein_graph / ligand_graph must match DiffDock expected input format.
        """

        # Forward diffusion sampling loop
        # (you will adapt this from DiffDock inference.py)
        output = self.score_model(protein_graph, ligand_graph)

        if self.confidence_model:
            confidence = self.confidence_model(protein_graph, ligand_graph)
            return output, confidence

        return output, None