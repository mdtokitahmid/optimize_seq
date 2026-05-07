"""
eval_dna_test.py — Run test evaluation on a pre-trained DNA CNN model.

Usage:
    python eval_dna_test.py --task sknsh
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Reuse everything from the training script
from train_cnn_dna import DNADataset, DNARegressor, evaluate, collate_fn, OUT_ROOT, FP16

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        required=True)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out_dir = OUT_ROOT / args.task
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task: {args.task}  Device: {device}")

    norm      = np.load(out_dir / "normalization.npy")
    mean, std = float(norm[0]), float(norm[1])

    test_df   = pd.read_csv(out_dir / "test.csv")
    print(f"Test samples: {len(test_df):,}")

    test_ds     = DNADataset(test_df, mean, std)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    ckpt  = torch.load(out_dir / "best_model.pt",
                       map_location=device, weights_only=False)
    model = DNARegressor().to(device)
    model.load_state_dict(ckpt["model"])
    val_spearman = ckpt.get("metrics", {}).get("spearman", float("nan"))
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val spearman={val_spearman:.4f})")

    metrics = evaluate(model, test_loader, device, mean, std, FP16)
    pd.DataFrame([metrics]).to_csv(out_dir / "test_results.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Test Results — {args.task}")
    print(f"{'='*50}")
    print(f"  Spearman ρ : {metrics['spearman']:.4f}")
    print(f"  Pearson  r : {metrics['pearson']:.4f}")
    print(f"  RMSE       : {metrics['rmse']:.4f}")
    print(f"{'='*50}")
    print(f"Saved to {out_dir / 'test_results.csv'}")

if __name__ == "__main__":
    main()
