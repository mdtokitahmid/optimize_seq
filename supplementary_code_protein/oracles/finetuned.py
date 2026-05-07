"""
FineTunedESMOracle
──────────────────
Wraps fine-tuned ESM-2 checkpoints where the backbone weights are stored
inside the checkpoint itself (unlike KRASOracle / ThermostabilityOracle which
use a frozen external backbone).

Checkpoint format:
    {
        "model":   state_dict,   # keys prefixed with "esm." and "head."
        "metrics": {...},
        "args":    {...},        # optional; may contain esm_model, pooling
        ...
    }

Normalization is loaded from a companion file:
    <same directory as checkpoint>/normalization.npy  →  [mean, std]

Usage via optimize_final.py:
    --ckpts binding:path/to/gb1.pt:facebook/esm2_t30_150M_UR50D:mean \\
            stability:path/to/stab.pt:facebook/esm2_t12_35M_UR50D:cls
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseOracle, OracleConfig
from .esm2_backbone import ESM2SoftBackbone

logger = logging.getLogger(__name__)


class _RegressionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FineTunedESMOracle(BaseOracle):
    """
    Oracle backed by a fine-tuned ESM-2 model (backbone + head in one checkpoint).

    Accepts soft one-hot input (B, 20, L) for end-to-end gradient flow.
    """

    def __init__(
        self,
        backbone:    ESM2SoftBackbone,
        head:        _RegressionHead,
        pooling:     str,
        output_mean: float,
        output_std:  float,
        name:        str,
    ):
        super().__init__(OracleConfig(name=name, maximize=True))
        self.backbone    = backbone
        self.head        = head
        self.pooling     = pooling
        self.register_buffer("output_mean", torch.tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_std",  torch.tensor(output_std,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.soft_forward(x)          # (B, L+2, d_model)
        if self.pooling == "mean":
            pooled = h[:, 1:-1, :].mean(dim=1)    # exclude BOS/EOS tokens
        elif self.pooling == "cls":
            pooled = h[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}'")
        pred = self.head(pooled)
        return pred * self.output_std + self.output_mean

    @classmethod
    def load(
        cls,
        path:        str,
        oracle_name: str,
        model_name:  str                  = "facebook/esm2_t30_150M_UR50D",
        pooling:     str                  = "mean",
        norm_path:   Optional[str]        = None,
    ) -> "FineTunedESMOracle":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Read model_name / pooling from checkpoint args if embedded
        ckpt_args = ckpt.get("args") or {}
        model_name = ckpt_args.get("esm_model", model_name)
        pooling    = ckpt_args.get("pooling",   pooling)

        backbone = ESM2SoftBackbone(model_name=model_name, freeze=True)
        head     = _RegressionHead(d_model=backbone.d_model)

        state     = ckpt["model"]
        esm_state = {k[len("esm."):]: v for k, v in state.items() if k.startswith("esm.")}
        head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        if head_state and not any(k.startswith("net.") for k in head_state):
            head_state = {f"net.{k}": v for k, v in head_state.items()}

        backbone.esm.load_state_dict(esm_state, strict=False)
        head.load_state_dict(head_state, strict=True)
        backbone.aa_embed_matrix = backbone._build_aa_embed_matrix().to(
            backbone.aa_embed_matrix.device
        )

        # Freeze all parameters
        for p in list(backbone.parameters()) + list(head.parameters()):
            p.requires_grad_(False)

        # Load normalization
        if norm_path is None:
            candidate = Path(path).parent / "normalization.npy"
            norm_path = str(candidate) if candidate.exists() else None

        output_mean, output_std = 0.0, 1.0
        if norm_path is not None:
            arr = np.load(norm_path)
            output_mean, output_std = float(arr[0]), float(arr[1])
            logger.info("[%s] Loaded normalization from %s (mean=%.4f, std=%.4f)",
                        oracle_name, norm_path, output_mean, output_std)
        else:
            logger.warning("[%s] No normalization.npy found — outputs are raw logits", oracle_name)

        logger.info("[FineTunedESMOracle:%s] Loaded from %s (model=%s, pooling=%s)",
                    oracle_name, path, model_name, pooling)
        return cls(backbone, head, pooling, output_mean, output_std, oracle_name)
