"""
Train simple CNN regressors for the three DNA assays:
  - hepG2
  - k562
  - sknsh

Architecture:
  Input: 4 x L one-hot DNA
  Conv1: 4   -> 128, kernel=15, same
  BN + GELU + MaxPool1d(2) + Dropout(0.1)
  Conv2: 128 -> 192, kernel=11, same
  BN + GELU + MaxPool1d(2) + Dropout(0.1)
  Conv3: 192 -> 256, kernel=7, same
  BN + GELU + Dropout(0.1)
  GlobalMaxPool1d
  Linear 256 -> 128
  GELU + Dropout(0.2)
  Linear 128 -> 1

Example:
  python train_cnn_dna.py --task all
  python train_cnn_dna.py --task hepG2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = BUNDLE_ROOT / "data" / "Big_Oracles" / "DNA"
DATA_DIR = DATA_ROOT / "data"
OUT_ROOT = DATA_ROOT / "cnn_dna_models"
TASKS = ["hepG2", "k562", "sknsh"]
DNA_ALPHABET = "ACGT"
DNA_TO_IDX = {b: i for i, b in enumerate(DNA_ALPHABET)}

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
FP16 = True
SEED = 42
VAL_FRAC = 0.10
TEST_FRAC = 0.10

loss_fn = nn.MSELoss()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all", *TASKS])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_dna(seq: str) -> torch.Tensor:
    x = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, base in enumerate(seq):
        idx = DNA_TO_IDX.get(base)
        if idx is not None:
            x[idx, i] = 1.0
    return x


def is_valid_dna(seq: str) -> bool:
    return bool(seq) and all(base in DNA_TO_IDX for base in seq)


def deduplicate_and_split(df: pd.DataFrame, seed: int):
    df = (
        df.groupby("sequence", as_index=False)
        .agg(score=("score", "mean"))
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_test = max(1, int(len(df) * TEST_FRAC))
    n_val = max(1, int(len(df) * VAL_FRAC))

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


class DNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, mean: float, std: float):
        self.seqs = df["sequence"].tolist()
        self.labels = torch.tensor(((df["score"].values - mean) / std), dtype=torch.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": encode_dna(self.seqs[idx]),
            "label": self.labels[idx],
        }


def collate_fn(batch):
    max_len = max(item["x"].size(1) for item in batch)
    x_pad = torch.zeros(len(batch), 4, max_len, dtype=torch.float32)
    labels = torch.stack([item["label"] for item in batch])
    for i, item in enumerate(batch):
        length = item["x"].size(1)
        x_pad[i, :, :length] = item["x"]
    return {"x": x_pad, "label": labels}


class DNARegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=15, padding="same"),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(128, 192, kernel_size=11, padding="same"),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(192, 256, kernel_size=7, padding="same"),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.max(dim=-1).values
        return self.head(h).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device, mean, std, fp16):
    model.eval()
    preds_norm, trues_norm = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        with autocast(enabled=fp16):
            pred = model(x)
        preds_norm.append(pred.cpu())
        trues_norm.append(y.cpu())

    preds_norm = torch.cat(preds_norm).numpy()
    trues_norm = torch.cat(trues_norm).numpy()
    preds = preds_norm * std + mean
    trues = trues_norm * std + mean

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    r, _ = pearsonr(preds, trues)
    rho, _ = spearmanr(preds, trues)
    return {"rmse": rmse, "pearson": float(r), "spearman": float(rho)}


def load_task_frame(task: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{task}.csv"
    df = pd.read_csv(csv_path)
    df["sequence"] = df["sequence"].astype(str).str.upper()
    df = df[df["sequence"].map(is_valid_dna)].copy()
    return df.reset_index(drop=True)


def train_one_task(task: str, args):
    out_dir = OUT_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_task_frame(task)
    train_df, valid_df, test_df = deduplicate_and_split(df, args.seed)

    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    mean_score = float(train_df["score"].mean())
    std_score = float(train_df["score"].std())
    if std_score == 0:
        std_score = 1.0
    np.save(out_dir / "normalization.npy", np.array([mean_score, std_score]))

    print(f"\nTask: {task}")
    print(f"Train: {len(train_df):,}  Val: {len(valid_df):,}  Test: {len(test_df):,}")
    print(f"score mean={mean_score:.4f}  std={std_score:.4f}")

    train_ds = DNADataset(train_df, mean_score, std_score)
    valid_ds = DNADataset(valid_df, mean_score, std_score)
    test_ds = DNADataset(test_df, mean_score, std_score)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNARegressor().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=FP16)

    best_val_spearman = -np.inf
    log_rows = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"{task} epochs"):
        model.train()
        for batch in tqdm(train_loader, desc=f"{task} epoch {epoch}"):
            optimizer.zero_grad()
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=FP16):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        val_metrics = evaluate(model, valid_loader, device, mean_score, std_score, FP16)
        print(f"{task} epoch {epoch}: val spearman={val_metrics['spearman']:.4f}")
        log_rows.append({"epoch": epoch, **val_metrics})
        pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)

        if val_metrics["spearman"] > best_val_spearman:
            best_val_spearman = val_metrics["spearman"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "task": task,
                },
                out_dir / "best_model.pt",
            )
            print(f"Saved best model for {task} at epoch {epoch} (val spearman={best_val_spearman:.4f})")

    print(f"\nLoading best checkpoint for {task} test evaluation...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, mean_score, std_score, FP16)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_results.csv", index=False)
    print(
        f"{task} test: spearman={test_metrics['spearman']:.4f} "
        f"pearson={test_metrics['pearson']:.4f} rmse={test_metrics['rmse']:.4f}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]
    for task in tasks:
        train_one_task(task, args)


if __name__ == "__main__":
    main()
