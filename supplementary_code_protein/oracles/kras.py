"""
KRASOracle
──────────
Wraps the PropertyMLP heads trained by CW_RAS_full_profile/train.py.

Checkpoint format (from train.py):
    {
        "model_state_dict": ...,          # PropertyMLP weights (640→512→256→128→1)
        "normalisation":    {"mu": float, "std": float},
        "property":         str,          # e.g. "RAF"
        "val_spearman":     float,
        "test_spearman":    float,
        "esm_model":        str,          # "facebook/esm2_t30_150M_UR50D"
    }

The head was trained on fitness values normalised as z = (y - mu) / std.
We un-normalise in forward() so oracle outputs are in raw fitness units
(WT fitness ≈ 0 by convention in the CW-RAS dataset).
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base import BaseOracle, OracleConfig
from .esm2_backbone import ESM2SoftBackbone


# ── Regression head (mirrors PropertyMLP in train.py) ────────────────────────

class _PropertyMLP(nn.Module):
    """640→512→256→128→1 with LayerNorm + GELU + Dropout."""

    def __init__(self, input_dim: int = 640, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),       nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),       nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Oracle ────────────────────────────────────────────────────────────────────

class KRASOracle(BaseOracle):
    """
    KRAS property oracle for use with GRACE / optimize_final.py.

    Accepts soft one-hot input (B, 20, L) for end-to-end gradient flow.
    Outputs raw fitness units (un-normalised from the z-scored training target).
    """

    ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

    def __init__(
        self,
        backbone: ESM2SoftBackbone,
        head:     _PropertyMLP,
        mu:       float,
        std:      float,
        name:     str,
    ):
        super().__init__(OracleConfig(name=name, maximize=True))
        self.backbone = backbone
        self.head     = head
        self.register_buffer("mu",  torch.tensor(mu,  dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 20, L) soft one-hot sequences
        Returns:
            fitness: (B,) in raw fitness units (WT ≈ 0)
        """
        h        = self.backbone.soft_forward(x)       # (B, L+2, d_model)
        h_pooled = h[:, 1:-1, :].mean(dim=1)           # (B, d_model) — exclude BOS/EOS
        z        = self.head(h_pooled)                  # (B,) normalised
        return z * self.std + self.mu                   # un-normalise → raw fitness

    @classmethod
    def load(
        cls,
        path:        str,
        backbone:    ESM2SoftBackbone,
        oracle_name: str = None,
    ) -> "KRASOracle":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        head = _PropertyMLP(input_dim=backbone.d_model)
        head.load_state_dict(ckpt["model_state_dict"])
        mu   = float(ckpt["normalisation"]["mu"])
        std  = float(ckpt["normalisation"]["std"])
        name = oracle_name or ckpt.get("property", Path(path).parent.name)
        oracle = cls(backbone, head, mu, std, name)
        print(f"[KRASOracle:{name}] Loaded from {path}  "
              f"(val ρ={ckpt.get('val_spearman', float('nan')):.4f})")
        return oracle
