from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = BUNDLE_ROOT / "data" / "Big_Oracles" / "DNA" / "data"
ASSAYS = ["k562", "hepG2", "sknsh"]
TASKS = {
    "k562_vs_others": ("k562", ["hepG2", "sknsh"]),
    "hepG2_vs_others": ("hepG2", ["k562", "sknsh"]),
    "sknsh_vs_others": ("sknsh", ["hepG2", "k562"]),
}


def load_merged_table() -> pd.DataFrame:
    dfs = []
    for assay in ASSAYS:
        df = pd.read_csv(DATA_DIR / f"{assay}.csv").rename(columns={"score": assay})
        dfs.append(df)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="sequence", how="inner")
    for assay in ASSAYS:
        merged[f"{assay}_rank"] = merged[assay].rank(pct=True, method="average")
    return merged


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def select_for_task(
    df: pd.DataFrame,
    task_name: str,
    target: str,
    constraints: list[str],
    n_starts: int,
    target_max_rank: float,
    constraint_min_rank: float,
    constraint_max_rank: float,
    candidate_pool: int,
    diversity_weight: float,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out[f"{target}_rank"] <= target_max_rank].copy()
    for c in constraints:
        out = out[
            (out[f"{c}_rank"] >= constraint_min_rank)
            & (out[f"{c}_rank"] <= constraint_max_rank)
        ].copy()

    if len(out) < n_starts:
        raise ValueError(
            f"Not enough candidates for {task_name}: found {len(out)} after filtering."
        )

    out["selection_score"] = out[f"{target}_rank"]
    for c in constraints:
        out["selection_score"] += 0.5 * (out[f"{c}_rank"] - 0.45).abs()

    pool = out.nsmallest(min(candidate_pool, len(out)), "selection_score").reset_index(drop=True)
    chosen_rows = []
    chosen_indices = []

    first_idx = int(pool["selection_score"].idxmin())
    chosen_rows.append(pool.loc[first_idx])
    chosen_indices.append(first_idx)

    while len(chosen_rows) < n_starts:
        best_idx = None
        best_value = None
        for idx, row in pool.iterrows():
            if idx in chosen_indices:
                continue
            min_h = min(hamming(row["sequence"], chosen["sequence"]) for chosen in chosen_rows)
            diversity_term = diversity_weight * (min_h / max(1, len(row["sequence"])))
            value = diversity_term - float(row["selection_score"])
            if best_value is None or value > best_value:
                best_value = value
                best_idx = idx
        chosen_rows.append(pool.loc[best_idx])
        chosen_indices.append(best_idx)

    selected = pd.DataFrame(chosen_rows).reset_index(drop=True)
    selected.insert(0, "task", task_name)
    selected.insert(1, "start_id", [f"{i+1:02d}" for i in range(len(selected))])
    selected.insert(2, "target", target)
    selected.insert(3, "constraints", [" ".join(constraints)] * len(selected))
    return selected


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA_DIR / "dna_multistart_starts.csv"))
    p.add_argument("--n_starts", type=int, default=5)
    p.add_argument("--target_max_rank", type=float, default=0.35)
    p.add_argument("--constraint_min_rank", type=float, default=0.20)
    p.add_argument("--constraint_max_rank", type=float, default=0.60)
    p.add_argument("--candidate_pool", type=int, default=1000)
    p.add_argument("--diversity_weight", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    merged = load_merged_table()

    all_selected = []
    for task_name, (target, constraints) in TASKS.items():
        chosen = select_for_task(
            merged,
            task_name=task_name,
            target=target,
            constraints=constraints,
            n_starts=args.n_starts,
            target_max_rank=args.target_max_rank,
            constraint_min_rank=args.constraint_min_rank,
            constraint_max_rank=args.constraint_max_rank,
            candidate_pool=args.candidate_pool,
            diversity_weight=args.diversity_weight,
        )
        all_selected.append(
            chosen[
                [
                    "task",
                    "start_id",
                    "target",
                    "constraints",
                    "sequence",
                    "k562",
                    "hepG2",
                    "sknsh",
                    "k562_rank",
                    "hepG2_rank",
                    "sknsh_rank",
                    "selection_score",
                ]
            ]
        )

    out_df = pd.concat(all_selected, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} selected starts to {out_path}")
    print(out_df[["task", "start_id", "k562", "hepG2", "sknsh", "selection_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
