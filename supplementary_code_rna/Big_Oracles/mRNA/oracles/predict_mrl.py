"""
predict_mrl.py — Differentiable MRL Oracle (Optimus 5-Prime)
=============================================================
Predicts Mean Ribosome Load (translation efficiency) for 5' UTR sequences.
Self-contained: no imports from the parent 5_prime_optimization directory.

Model: Sample et al. 2019 CNN trained on 280k human 5'UTRs (GEO: GSE114002).
Architecture: 3×Conv1D(120, k=8) → Flatten → Dense(40) → Dense(1)
Input: (B, 50, 4) one-hot  |  Output: MRL in [1, 10] range

Usage — CLI:
    python predict_mrl.py ACGAUGCAUAGCAGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGACC
    python predict_mrl.py SEQ1 SEQ2 SEQ3 --model models/main_MRL_model.hdf5

Usage — Python (non-differentiable, returns numpy):
    from predict_mrl import load_mrl_model, mrl_scores
    model = load_mrl_model()
    scores = mrl_scores(model, ["ACGAUGCAU..."])  # list[float]

Usage — Python (differentiable, backprop through):
    from predict_mrl import load_mrl_model, one_hot_encode
    import torch, numpy as np

    model = load_mrl_model()            # weights loaded, eval mode
    x = torch.tensor(
            np.stack([one_hot_encode(s) for s in seqs]),
            requires_grad=True)         # (B, 50, 4)
    out = model(x)                      # (B,) scaled MRL — grads flow
    loss = -out.mean()
    loss.backward()
    print(x.grad.norm())                # nonzero → backprop works
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_H5 = os.path.join(_HERE, "models", "main_MRL_model.hdf5")

_SCALER_MEAN = 4.6509
_SCALER_STD  = 1.0540

_NT = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'u': 3}


# ── Encoding ───────────────────────────────────────────────────────────────────

def one_hot_encode(seq: str, seq_len: int = 50) -> np.ndarray:
    """(seq_len, 4) float32 array. Left-zero-padded, right-truncated."""
    seq    = seq.lower()[:seq_len]
    oh     = np.zeros((seq_len, 4), dtype=np.float32)
    offset = seq_len - len(seq)
    for i, nt in enumerate(seq):
        idx = _NT.get(nt, -1)
        if idx >= 0:
            oh[offset + i, idx] = 1.0
    return oh


# ── Weight loading ─────────────────────────────────────────────────────────────

def _read_h5_weights(h5_path: str) -> dict:
    layers = {}
    with h5py.File(h5_path, "r") as f:
        root = f.get("model_weights") or f

        def visit(path, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            layer = path.split("/")[0]
            wname = path.split("/")[-1].split(":")[0]
            layers.setdefault(layer, {})[wname] = obj[()]

        root.visititems(visit)
    return layers


def _sort_key(name: str) -> tuple:
    import re
    m    = re.search(r"_(\d+)$", name)
    base = re.sub(r"_\d+$", "", name)
    idx  = int(m.group(1)) if m else 0
    pri  = 0 if "conv" in base else (1 if "dense" in base else 2)
    return (pri, idx)


# ── Model ──────────────────────────────────────────────────────────────────────

class Optimus5Prime(nn.Module):
    """
    PyTorch port of the Optimus 5-Prime Keras CNN.
    Weights loaded from main_MRL_model.hdf5.
    Forward pass is fully differentiable.
    Input:  (B, L, 4)  one-hot
    Output: (B,)       scaled MRL (apply inverse_scale for real values)
    """

    def __init__(self, lw: dict):
        super().__init__()
        self.layers_list = nn.ModuleList()
        self._ops        = []
        self._build(lw)

    def _build(self, lw: dict):
        n_dense_seen = 0
        for name in sorted(lw.keys(), key=_sort_key):
            w = lw[name]
            if "kernel" not in w:
                continue
            kernel = w["kernel"]
            bias   = w.get("bias")

            if kernel.ndim == 3:                        # Conv1D layer
                ks, in_ch, out_ch = kernel.shape
                conv = nn.Conv1d(in_ch, out_ch, ks, padding=0)
                conv.weight.data = torch.from_numpy(kernel.transpose(2, 1, 0))
                if bias is not None:
                    conv.bias.data = torch.from_numpy(bias)
                self.layers_list.append(conv)
                self._ops.append(("conv", conv, ks))

            elif kernel.ndim == 2:                      # Dense layer
                in_f, out_f = kernel.shape
                linear = nn.Linear(in_f, out_f)
                linear.weight.data = torch.from_numpy(kernel.T)
                if bias is not None:
                    linear.bias.data = torch.from_numpy(bias)
                self.layers_list.append(linear)
                self._ops.append(("dense", linear, n_dense_seen))
                n_dense_seen += 1

        self._n_dense = n_dense_seen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4) → (B, 4, L) for Conv1d
        out = x.permute(0, 2, 1)

        for op in self._ops:
            if op[0] == "conv":
                _, layer, ks = op
                # Keras 'same' padding for even kernel (ks=8): 3 left, 4 right
                out = F.pad(out, ((ks - 1) // 2, ks // 2))
                out = torch.relu(layer(out))
            else:
                _, layer, idx = op
                if idx == 0:
                    # Match Keras channels_last flatten order
                    out = out.permute(0, 2, 1).contiguous().flatten(1)
                out = layer(out)
                if idx < self._n_dense - 1:
                    out = torch.relu(out)

        return out.squeeze(-1)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_mrl_model(h5_path: str = _DEFAULT_H5,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"
                   ) -> Optimus5Prime:
    """Load Optimus 5-Prime from .hdf5 weights. Returns model in eval mode."""
    lw    = _read_h5_weights(h5_path)
    model = Optimus5Prime(lw).to(device)
    model.eval()
    return model


def mrl_scores(model: Optimus5Prime, sequences: list,
               seq_len: int = 50) -> list:
    """
    Non-differentiable inference. Returns list of real MRL floats in [1, 10].
    For differentiable use, call model(x) directly with requires_grad=True input.
    """
    device  = next(model.parameters()).device
    encoded = np.stack([one_hot_encode(s, seq_len) for s in sequences])
    x       = torch.from_numpy(encoded).to(device)
    with torch.no_grad():
        scaled = model(x).cpu().numpy()
    return list(scaled * _SCALER_STD + _SCALER_MEAN)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict MRL for 5' UTR sequences")
    parser.add_argument("sequences", nargs="+", help="RNA/DNA sequences")
    parser.add_argument("--model",  default=_DEFAULT_H5, help="Path to .hdf5 weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model  = load_mrl_model(args.model, args.device)
    scores = mrl_scores(model, args.sequences)

    print(f"\n{'Sequence':<55} {'MRL':>6}")
    print("-" * 63)
    for seq, mrl in zip(args.sequences, scores):
        disp = seq[:52] + "..." if len(seq) > 55 else seq
        print(f"{disp:<55} {mrl:>6.3f}")


if __name__ == "__main__":
    main()
