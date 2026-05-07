"""
ProteinSeqProp
──────────────
Fast SeqProp for protein sequences (Linder & Seelig, 2021 — adapted for proteins).

Maintains a *distribution over sequences* via learned logits:

    ell   ∈ ℝ^{20 × L}   raw logits  (the main "what sequence?" parameters)
    gamma ∈ ℝ             global scale (prevents saturation)
    beta  ∈ ℝ^{20}        per-AA bias

Distribution:
    z    = gamma * InstanceNorm(ell) + beta      (normalized logits, stays bounded)
    P    = softmax(z, dim=0)                      (20, L) probability matrix

Sampling:
    Y_hard ~ Categorical(P)                       discrete, in-distribution for oracle
    Y_ST   = Y_hard + (P - sg(P))                straight-through: forward=hard, grad=soft

Why InstanceNorm?
    Without it, logits grow large during gradient ascent → softmax saturates →
    Jacobian ≈ 0 → gradients die → optimization stalls. InstanceNorm keeps logits
    bounded so the Jacobian stays nonzero throughout optimization.

Why straight-through (ST)?
    Oracles (e.g., ESM-2) are trained on discrete sequences. Feeding soft P
    is out-of-distribution. The ST trick: forward pass sees hard one-hot (in-distribution),
    backward pass sees gradients through soft P (differentiable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from utils.encoding import AA_ALPHABET, AA_TO_IDX, IDX_TO_AA, NUM_AAS


class ProteinSeqProp(nn.Module):
    """
    Differentiable protein sequence distribution for gradient-based optimization.

    Args:
        L:               sequence length
        init_sequence:   starting protein sequence (str). If None → uniform distribution.
        init_logit_scale: multiplier on initial one-hot logits.
                          4.0 → P(correct AA) ≈ 0.97 initially (near-discrete but not saturated).
    """

    def __init__(
        self,
        L: int,
        init_sequence: str = None,
        init_logit_scale: float = 4.0,
        init_gamma: float = 1.0,
    ):
        super().__init__()
        self.L = L

        # ── Parameters ────────────────────────────────────────────────────────
        init_logits = self._make_init_logits(L, init_sequence, init_logit_scale)
        self.logits = nn.Parameter(init_logits)                        # (20, L)
        # gamma controls how peaked the initial distribution is.
        # NOTE: init_logit_scale has no effect when InstanceNorm is used —
        # InstanceNorm normalizes away the absolute logit scale, making the
        # normalized value at WT positions = sqrt(L/n_aa - 1) regardless of scale.
        # gamma (applied AFTER InstanceNorm) is the real sharpness knob.
        # init_gamma=1.0 → expected hamming ≈ L/4 from WT (too spread for single-mutant oracles)
        # init_gamma=3.0 → expected hamming ≈ 3-5 from WT (near-WT, oracle-calibrated)
        self.gamma  = nn.Parameter(torch.tensor([init_gamma]))         # scalar
        self.beta   = nn.Parameter(torch.zeros(NUM_AAS))              # (20,)

        # InstanceNorm over the L dimension (normalizes each AA channel across positions)
        # affine=False because we apply our own learned gamma/beta above
        self._inst_norm = nn.InstanceNorm1d(NUM_AAS, affine=False, eps=1e-5)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_init_logits(
        L: int,
        sequence: str,
        scale: float,
    ) -> torch.Tensor:
        """
        Initialize logit matrix.
        - If sequence given: set logit[correct_aa, pos] = scale, others = 0.
        - Add small noise to break symmetry.
        """
        logits = torch.zeros(NUM_AAS, L)

        if sequence is not None:
            if len(sequence) != L:
                raise ValueError(
                    f"init_sequence length {len(sequence)} ≠ L={L}"
                )
            for i, aa in enumerate(sequence):
                if aa not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid '{aa}' at position {i}")
                logits[AA_TO_IDX[aa], i] = scale

        logits += 0.01 * torch.randn_like(logits)
        return logits

    # ── Public API ────────────────────────────────────────────────────────────

    def normalized_logits(self) -> torch.Tensor:
        """
        Apply InstanceNorm + affine rescaling to raw logits.
        Keeps logit magnitudes bounded → Jacobian of softmax stays nonzero.

        Returns:
            z: (20, L) bounded logits
        """
        # InstanceNorm1d expects (B, C, L); add/remove batch dimension
        z = self._inst_norm(self.logits.unsqueeze(0)).squeeze(0)   # (20, L)
        z = self.gamma * z + self.beta.unsqueeze(1)                 # (20, L)
        return z

    def probabilities(self) -> torch.Tensor:
        """
        Current amino acid probability distribution.

        Returns:
            P: (20, L)  — each column sums to 1, entries in [0, 1]
        """
        return F.softmax(self.normalized_logits(), dim=0)  # (20, L)

    def st_sample(self, K: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Draw K straight-through samples from the current distribution.

        Each returned sample Y_ST satisfies:
            forward pass:  Y_ST  ≡  Y_hard  (valid discrete one-hot sequence)
            backward pass: ∂Y_ST/∂P = I      (gradient flows through soft P)

        This is the core trick: oracles see valid sequences; gradients flow anyway.

        Args:
            K: number of Monte Carlo samples

        Returns:
            samples: list of K tensors, each (20, L)
            P:       (20, L) current distribution (shared across all samples)
        """
        P = self.probabilities()   # (20, L)
        samples = []

        for _ in range(K):
            # Sample one amino acid per position from the categorical distribution
            # P.T: (L, 20) — multinomial needs (n_distributions, n_categories)
            c = torch.multinomial(P.T, num_samples=1).squeeze(1)   # (L,)

            # Convert to one-hot: (L, 20) → (20, L)
            Y_hard = F.one_hot(c, num_classes=NUM_AAS).T.float()   # (20, L)

            # Straight-through estimator:
            #   Forward:  Y_hard + (P - P) = Y_hard           ← discrete ✓
            #   Backward: ∂/∂P [Y_hard + P - sg(P)] = I       ← differentiable ✓
            Y_st = Y_hard + (P - P.detach())                       # (20, L)
            samples.append(Y_st)

        return samples, P

    def decode(self, method: str = "argmax") -> str:
        """
        Convert the current distribution to a sequence string.

        Args:
            method: "argmax" → take most probable AA per position (deterministic)
                    "sample" → sample one AA per position from P (stochastic)

        Returns:
            sequence string of length L
        """
        P = self.probabilities()  # (20, L)

        if method == "argmax":
            indices = P.argmax(dim=0)
        elif method == "sample":
            indices = torch.multinomial(P.T, 1).squeeze(1)
        else:
            raise ValueError(f"decode method must be 'argmax' or 'sample', got '{method}'")

        return ''.join(IDX_TO_AA[i.item()] for i in indices)

    def entropy(self) -> torch.Tensor:
        """
        Mean positional entropy of the distribution.

        H = 0     → fully committed to one sequence (low uncertainty)
        H = log20 → fully uniform (maximum uncertainty)

        Useful for monitoring convergence: entropy should decrease over training.
        """
        P = self.probabilities()                       # (20, L)
        H = -(P * (P + 1e-9).log()).sum(dim=0)        # (L,)  per-position entropy
        return H.mean()

    def mutation_probabilities(self, original_sequence: str) -> torch.Tensor:
        """
        Per-position probability of being mutated away from the original sequence.

        P_mut[i] = 1 - P[original_aa_at_i, i]

        Args:
            original_sequence: string of length L

        Returns:
            (L,) tensor of mutation probabilities ∈ [0, 1]
        """
        P = self.probabilities()  # (20, L)
        mut_probs = torch.ones(self.L, device=P.device)
        for i, aa in enumerate(original_sequence):
            if aa in AA_TO_IDX:
                mut_probs[i] = 1.0 - P[AA_TO_IDX[aa], i]
        return mut_probs
