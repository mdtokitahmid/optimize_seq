from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ["k562_vs_others", "hepG2_vs_others", "sknsh_vs_others"]
MODES = ["unconstrained", "grace_only", "lagrangian_only", "grace_lagrangian", "simulated_annealing"]


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def best_feasible_from_json(data: dict, eps_abs: float, smooth_w: int):
    target = data["target_oracle"]
    constraints = data.get("constraint_oracles", [])
    scores = data["real_scores"]
    c0 = data["real_c_initial"]

    target_scores = smooth(scores[target], smooth_w)
    best_step = None
    best_score = None

    for i in range(len(target_scores)):
        feasible = all(abs(scores[c][i] - c0[c]) <= eps_abs for c in constraints)
        if feasible and (best_score is None or target_scores[i] > best_score):
            best_step = i
            best_score = float(target_scores[i])

    return best_step, best_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(Path(__file__).resolve().parent / "data" / "results" / "dna"))
    p.add_argument("--eps_abs", type=float, default=0.1)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--out_csv", default=str(Path(__file__).resolve().parent / "data" / "results" / "dna" / "multistart_summary.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)

    rows = []
    for task in TASKS:
        task_dir = root / task
        start_dirs = sorted([p for p in task_dir.glob("start_*") if p.is_dir()])
        for start_dir in start_dirs:
            start_id = start_dir.name
            for mode in MODES:
                result_path = start_dir / f"{mode}.json"
                if not result_path.exists():
                    continue
                with open(result_path) as f:
                    data = json.load(f)
                best_step, best_score = best_feasible_from_json(data, args.eps_abs, args.smooth)
                rows.append(
                    {
                        "task": task,
                        "start_id": start_id,
                        "mode": mode,
                        "best_step": best_step,
                        "best_score": best_score,
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        print(f"No multistart DNA results found under {root}")
        return

    summary = (
        df.groupby(["task", "mode"], as_index=False)
        .agg(
            n_starts=("best_score", lambda x: int(x.notna().sum())),
            mean_best_score=("best_score", "mean"),
            std_best_score=("best_score", "std"),
            mean_best_step=("best_step", "mean"),
        )
        .sort_values(["task", "mean_best_score"], ascending=[True, False])
    )

    print("\n" + "=" * 84)
    print(f"DNA MULTISTART SUMMARY  (real decoded scores, absolute ±{args.eps_abs:.3f})")
    print("=" * 84)
    for task in TASKS:
        task_df = summary[summary["task"] == task]
        if task_df.empty:
            continue
        print(f"\nTask: {task}")
        print(task_df.to_string(index=False))
    print("\n" + "=" * 84)
    print(f"Per-start summary saved to {out_csv}")


if __name__ == "__main__":
    main()
