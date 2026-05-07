"""
BaseOracle — the interface every oracle must implement.

To add a new oracle:
1. Subclass BaseOracle
2. Implement forward(x) → (B,) tensor
3. Pass it to optimize() as target_oracle or in constraint_oracles

That's it. GRACE handles the rest automatically.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OracleConfig:
    """
    Metadata for an oracle.

    Attributes:
        name:     human-readable label (used in logs and plots)
        maximize: True  → we want to maximize this property (target oracle)
                  False → not used directly; GRACE freezes constraint oracles
                          regardless of this flag. Kept for bookkeeping.
    """
    name: str
    maximize: bool = True


class BaseOracle(ABC, nn.Module):
    """
    Abstract oracle base class.

    Every oracle takes a batch of soft one-hot protein sequences and returns
    a scalar prediction per sequence.

    Input convention:
        x : (B, 20, L) float tensor
            B = batch size (number of ST samples, typically K=8)
           20 = amino acid alphabet size
            L = sequence length
        Each column x[b, :, i] is a (soft) probability vector over 20 AAs.
        For discrete (hard) sequences, exactly one entry per column is 1.0.

    Output:
        (B,) float tensor — one scalar prediction per sequence in the batch.

    The SeqProp / GRACE optimizer uses these scalars to compute gradients
    back through the soft sequence distribution.
    """

    def __init__(self, config: OracleConfig):
        super().__init__()
        self.config = config
        self.name   = config.name
        self.maximize = config.maximize

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 20, L) soft one-hot protein sequences

        Returns:
            scores: (B,) scalar predictions
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, maximize={self.maximize})"
