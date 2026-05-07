"""
Amino acid encoding / decoding utilities.

All oracles and the SeqProp model use the same AA ordering:
    index 0 = A, 1 = C, ..., 19 = Y   (alphabetical over 20 canonical AAs)

One-hot convention:  tensor shape (20, L)
    dim 0 = amino acid identity (20 values)
    dim 1 = sequence position   (L positions)
    Each column is a probability vector (or hard one-hot) over the 20 AAs.
"""

import torch

# ── Alphabet ─────────────────────────────────────────────────────────────────

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"       # 20 canonical AAs, alphabetical
AA_TO_IDX   = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA   = {i: aa for i, aa in enumerate(AA_ALPHABET)}

NUM_AAS = 20


def sequence_to_onehot(seq: str) -> torch.Tensor:
    """
    Convert a protein sequence string to a one-hot tensor.

    Args:
        seq: string of canonical amino acids (length L)

    Returns:
        x: (20, L) float32 tensor — exactly one 1.0 per column
    """
    L = len(seq)
    x = torch.zeros(NUM_AAS, L, dtype=torch.float32)
    for i, aa in enumerate(seq):
        if aa not in AA_TO_IDX:
            raise ValueError(
                f"Unknown amino acid '{aa}' at position {i}. "
                f"Supported: {AA_ALPHABET}"
            )
        x[AA_TO_IDX[aa], i] = 1.0
    return x


def onehot_to_sequence(x: torch.Tensor) -> str:
    """
    Convert a one-hot (or soft) tensor to a sequence string by argmax.

    Args:
        x: (20, L) tensor

    Returns:
        seq: string of length L (argmax at each position)
    """
    assert x.shape[0] == NUM_AAS, f"Expected 20 channels, got {x.shape[0]}"
    indices = x.argmax(dim=0)   # (L,)
    return ''.join(IDX_TO_AA[i.item()] for i in indices)


def hamming_distance(seq_a: str, seq_b: str) -> int:
    """Number of positions where two sequences differ."""
    assert len(seq_a) == len(seq_b), "Sequences must have the same length."
    return sum(a != b for a, b in zip(seq_a, seq_b))


def expected_hamming(P: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
    """
    Differentiable expected Hamming distance from original sequence.

    E[Hamming] = Σ_i (1 - P[:, i] · X_orig[:, i])
               = L - Σ_i P[:, i]^T X_orig[:, i]

    Args:
        P:      (20, L) current probability distribution
        X_orig: (20, L) one-hot original sequence

    Returns:
        scalar tensor — expected number of mutations
    """
    return (1.0 - (P * X_orig).sum(dim=0)).sum()
