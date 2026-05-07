"""
ThermostabilityOracle
─────────────────────
Predicts protein thermostability (ΔΔG) from sequence.

Architecture:
    soft one-hot (B, 20, L)
    → ESM2SoftBackbone (frozen)
    → mean-pool over L positions
    → MLP regression head
    → (B,) ΔΔG predictions

Training data (choose one):
    ① ProteinGym stability assays  [recommended]
       URL:  https://marks.hms.harvard.edu/proteingym/
       File: DMS_ProteinGym_substitutions.zip
       Use the assays with phenotype tag "Thermostability" or "Stability".
       Columns: mutated_sequence, DMS_score
       ~2.7M variants across 217 assays (use stability-tagged subset)

    ② ProTherm  [classic, smaller]
       URL:  https://web.iitm.ac.in/bioinfo2/prothermdb/
       File: ProThermDB.xlsx  (Nikam et al., NAR 2021)
       Columns: WILD_TYPE, MUTATION, DDG
       ~30k single-point mutations with experimental ΔΔG values

Usage:
    # Training
    python train_thermostability.py --data path/to/stability_assays.csv

    # Inference / optimization
    oracle = ThermostabilityOracle.load("checkpoints/thermo.pt", backbone)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from .base import BaseOracle, OracleConfig
from .esm2_backbone import ESM2SoftBackbone


# ── Regression head ───────────────────────────────────────────────────────────

class _ThermostabilityHead(nn.Module):
    """
    MLP that maps a pooled ESM-2 representation to a ΔΔG scalar.

    Architecture: LayerNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear(1)
    Input:  (B, d_model) mean-pooled ESM-2 representation
    Output: (B,)         predicted ΔΔG
    """

    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: (B, d_model)
        Returns:
            (B,)
        """
        return self.net(h_pooled).squeeze(-1)


# ── Oracle ────────────────────────────────────────────────────────────────────

class ThermostabilityOracle(BaseOracle):
    """
    Thermostability oracle for use with Fast SeqProp / GRACE.

    Higher score = more thermostable (we internally negate ΔΔG so that
    "maximize score" = "maximize stability").

    Accepts soft one-hot input (B, 20, L) for end-to-end gradient flow.
    """

    def __init__(
        self,
        backbone: ESM2SoftBackbone,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(OracleConfig(name="thermostability", maximize=True))
        self.backbone = backbone
        self.head = _ThermostabilityHead(
            d_model=backbone.d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 20, L) soft one-hot sequences

        Returns:
            scores: (B,) — higher is more thermostable
        """
        h = self.backbone.soft_forward(x)            # (B, L+2, d_model)
        h_pooled = h[:, 1:-1, :].mean(dim=1)         # (B, d_model)  — exclude BOS/EOS
        ddg = self.head(h_pooled)                     # (B,)
        # Convention: higher ΔΔG from wildtype = destabilizing.
        # We negate so "higher score = more stable".
        return -ddg

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state_dict": self.head.state_dict(),
            "model_name": self.backbone.model_name,
            "hidden_dim": self.head.net[1].out_features,
        }, path)
        print(f"[ThermostabilityOracle] Saved to {path}")

    @classmethod
    def load(cls, path: str, backbone: "ESM2SoftBackbone",
             oracle_name: str = "thermostability") -> "ThermostabilityOracle":
        ckpt = torch.load(path, map_location="cpu")
        oracle = cls(backbone, hidden_dim=ckpt.get("hidden_dim", 256))
        oracle.head.load_state_dict(ckpt["head_state_dict"])
        # Override the oracle name so GRACE treats it as a distinct signal
        oracle.config.name = oracle_name
        oracle.name        = oracle_name
        print(f"[{oracle_name}] Loaded from {path}")
        return oracle





