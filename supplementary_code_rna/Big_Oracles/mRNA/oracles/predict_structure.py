"""
predict_structure.py — Differentiable Structure Oracle (CNN Surrogate)
=======================================================================
Predicts ViennaRNA secondary structure properties for 5' UTR sequences
using a trained CNN surrogate — fully differentiable in PyTorch.

Train the surrogate first (one-time):
    python ../train_structure_surrogate.py --n_seqs 500000 --out models/structure_surrogate.pt

Predicted outputs (B, 4):
    col 0: mfe            — minimum free energy (kcal/mol, negative = more structured)
    col 1: unpaired_frac  — fraction of unpaired positions globally [0, 1]
    col 2: five_prime_frac— fraction of unpaired positions in 5' half [0, 1]
    col 3: n_stems        — number of independent stem groups

Usage — CLI:
    python predict_structure.py ACGAUGCAUAGCAGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGACC
    python predict_structure.py SEQ1 SEQ2 --model models/structure_surrogate.pt

Usage — Python (non-differentiable, returns dicts):
    from predict_structure import load_structure_model, structure_scores
    model  = load_structure_model()
    scores = structure_scores(model, ["ACGAUGCAU..."])
    # [{'mfe': -8.1, 'unpaired_frac': 0.72, 'five_prime_frac': 0.80,
    #   'n_stems': 2.1, 'struct_score': 0.68}, ...]

Usage — Python (differentiable, backprop through):
    from predict_structure import load_structure_model
    from predict_mrl import one_hot_encode
    import torch, numpy as np

    model = load_structure_model()
    x = torch.tensor(
            np.stack([one_hot_encode(s) for s in seqs]),
            requires_grad=True)         # (B, 50, 4)
    out = model(x)                      # (B, 4) raw normalized outputs
    # out[:, 1] = unpaired_frac (normalized) → maximize for open structure
    loss = -out[:, 1].mean()
    loss.backward()
    print(x.grad.norm())                # nonzero → backprop works
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn

_HERE        = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PT  = os.path.join(_HERE, "models", "structure_surrogate.pt")


# ── Model (must match train_structure_surrogate.py exactly) ───────────────────

class StructureSurrogate(nn.Module):
    """
    CNN surrogate predicting ViennaRNA structure properties.
    Architecture mirrors Optimus 5-Prime for consistency.
    Input:  (B, L, 4)  one-hot RNA sequence
    Output: (B, 4)     normalized [mfe, unpaired_frac, five_prime_frac, n_stems]
                       — call denormalize() to get real-valued outputs
    """

    def __init__(self, seq_len: int = 50, n_filters: int = 120, k: int = 8,
                 hidden: int = 40, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, k, padding='same')
        self.conv2 = nn.Conv1d(n_filters, n_filters, k, padding='same')
        self.conv3 = nn.Conv1d(n_filters, n_filters, k, padding='same')
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(seq_len * n_filters, hidden)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4) → (B, 4, L) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        B = h.size(0)
        h = h.permute(0, 2, 1).contiguous().view(B, -1)
        h = self.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_structure_model(pt_path: str = _DEFAULT_PT,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"
                         ) -> tuple:
    """
    Load surrogate from checkpoint.
    Returns (model, norm_stats) where norm_stats = (mfe_mean, mfe_std, stems_max).
    Pass both to structure_scores() or denormalize() for real-valued outputs.
    """
    ckpt = torch.load(pt_path, map_location=device)
    cfg  = ckpt.get("config", {})
    model = StructureSurrogate(
        seq_len   = ckpt.get("seq_len", 50),
        n_filters = cfg.get("n_filters", 120),
        k         = cfg.get("k", 8),
        hidden    = cfg.get("hidden", 40),
        dropout   = cfg.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    norm_stats = (ckpt["mfe_mean"], ckpt["mfe_std"], ckpt["stems_max"])
    return model, norm_stats


def denormalize(raw: np.ndarray, mfe_mean: float, mfe_std: float,
                stems_max: float) -> dict:
    """
    Convert raw model output (normalized) to real-valued structure properties.
    raw: (4,) or (B, 4) numpy array
    """
    squeeze = raw.ndim == 1
    if squeeze:
        raw = raw[None]
    out = {
        "mfe":             raw[:, 0] * mfe_std + mfe_mean,
        "unpaired_frac":   np.clip(raw[:, 1], 0, 1),
        "five_prime_frac": np.clip(raw[:, 2], 0, 1),
        "n_stems":         np.clip(raw[:, 3] * stems_max, 0, None),
    }
    if squeeze:
        return {k: float(v[0]) for k, v in out.items()}
    return out


def _struct_score(mfe: float, unpaired_frac: float, five_prime_frac: float,
                  n_stems: float) -> float:
    """Composite score matching reward_calculator.py score_global_structure()."""
    mfe_target, mfe_sigma = -8.0, 6.0
    s_mfe        = float(np.exp(-0.5 * ((mfe - mfe_target) / mfe_sigma) ** 2))
    s_unpaired   = float(min(unpaired_frac / 0.80, 1.0))
    s_stems      = float(np.exp(-0.7 * max(0, n_stems - 1)))
    s_five_prime = float(min(five_prime_frac / 0.80, 1.0))
    return float((s_mfe * s_unpaired * s_stems * s_five_prime) ** 0.25)


def structure_scores(model_and_stats: tuple, sequences: list,
                     seq_len: int = 50) -> list:
    """
    Non-differentiable inference. Returns list of dicts with real-valued outputs.
    For differentiable use, call model(x) directly with requires_grad=True input.
    """
    from predict_mrl import one_hot_encode

    model, (mfe_mean, mfe_std, stems_max) = model_and_stats
    device  = next(model.parameters()).device
    encoded = np.stack([one_hot_encode(s, seq_len) for s in sequences])
    x       = torch.from_numpy(encoded).to(device)

    with torch.no_grad():
        raw = model(x).cpu().numpy()  # (B, 4) normalized

    results = []
    d = denormalize(raw, mfe_mean, mfe_std, stems_max)
    for i in range(len(sequences)):
        mfe   = float(d["mfe"][i])
        unp   = float(d["unpaired_frac"][i])
        fp    = float(d["five_prime_frac"][i])
        ns    = float(d["n_stems"][i])
        score = _struct_score(mfe, unp, fp, ns)
        results.append({
            "mfe":             mfe,
            "unpaired_frac":   unp,
            "five_prime_frac": fp,
            "n_stems":         ns,
            "struct_score":    score,
        })
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict structure scores for 5' UTR sequences")
    parser.add_argument("sequences", nargs="+", help="RNA/DNA sequences")
    parser.add_argument("--model",  default=_DEFAULT_PT, help="Path to structure_surrogate.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_and_stats = load_structure_model(args.model, args.device)
    scores = structure_scores(model_and_stats, args.sequences)

    print(f"\n{'Sequence':<30} {'MFE':>7} {'unp':>6} {'5p':>6} {'stems':>6} {'score':>7}")
    print("-" * 66)
    for seq, s in zip(args.sequences, scores):
        disp = seq[:27] + "..." if len(seq) > 30 else seq
        print(f"{disp:<30} {s['mfe']:>7.2f} {s['unpaired_frac']:>6.3f}"
              f" {s['five_prime_frac']:>6.3f} {s['n_stems']:>6.1f} {s['struct_score']:>7.4f}")


if __name__ == "__main__":
    main()
