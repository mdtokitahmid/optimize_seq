"""
table_feasible_by_task_ablation.py

Print a table of ablation metrics per task:
- Max normalized feasible improvement
- Feasible percentage over the first 500 steps
- AvgViol: average constraint violation over infeasible steps only

Feasibility and improvement follow the same logic as the paper figure scripts:
use real_scores when available, inspect up to the first 500 steps, and keep
only steps that both satisfy the constraints and improve over the initial score
when computing feasible-improvement statistics.
"""

import json
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
DNA_BASE = BASE / "data" / "results" / "dna"
STARTS = [f"start_{i:02d}" for i in range(1, 6)]

METHODS = ["GRACE", "GRACE only", "Lagrangian only"]

FNAME = {
    "GRACE": "grace_lagrangian",
    "GRACE only": "grace_only",
    "Lagrangian only": "lagrangian_only",
}

TASK_CFG = [
    ("k562", DNA_BASE / "k562_vs_others", STARTS, FNAME),
    ("HepG2", DNA_BASE / "hepG2_vs_others", STARTS, FNAME),
    ("SK-N-SH", DNA_BASE / "sknsh_vs_others", STARTS, FNAME),
]


def resolve_files(task_dir: Path, starts, stem: str) -> list[Path]:
    if starts is None:
        return [task_dir / f"{stem}.json"]
    return [task_dir / start / f"{stem}.json" for start in starts]


def collect_file_stats(path: Path):
    if not path.exists():
        return None

    with open(path) as f:
        d = json.load(f)

    target = d["target_oracle"]
    constraints = d.get("constraint_oracles", [])
    c0 = d.get("real_c_initial") or d["c_initial"]
    t0 = c0[target]
    ctype = d.get("constraint_type", "absolute")
    eps = d.get("constraint_eps", 0.1)
    scores = d.get("real_scores", d["scores"])

    feasible_scores = []
    feasible_steps = 0
    violations = []
    best = None
    for i in range(min(500, len(scores[target]))):
        viol = 0.0
        for c in constraints:
            drift = abs(scores[c][i] - c0[c])
            viol += (
                max(0.0, drift - eps)
                if ctype == "absolute"
                else max(0.0, drift / (abs(c0[c]) + 1e-4) - eps)
            )
        if viol == 0.0:
            feasible_steps += 1
        else:
            violations.append(float(viol))

        if viol == 0.0 and scores[target][i] > t0:
            val = float(scores[target][i])
            feasible_scores.append(val)
            if best is None or val > best:
                best = val

    total_steps = min(500, len(scores[target]))
    return {
        "t0": float(t0),
        "best_feasible_score": best,
        "feasible_scores": feasible_scores,
        "total_steps": total_steps,
        "feasible_steps": feasible_steps,
        "violations": violations,
    }


def main():
    header = ["Task"]
    for method in METHODS:
        header.extend([
            f"{method} MaxNormFeasImp",
            f"{method} Feasible%",
            f"{method} AvgViol",
        ])
    print("\t".join(header))

    for label, task_dir, starts, fmap in TASK_CFG:
        method_stats = {}
        task_best = None
        for method in METHODS:
            stats = [collect_file_stats(p) for p in resolve_files(task_dir, starts, fmap[method])]
            stats = [s for s in stats if s is not None]
            method_stats[method] = stats
            for s in stats:
                for score in s["feasible_scores"]:
                    if task_best is None or score > task_best:
                        task_best = score

        row = [label]
        for method in METHODS:
            stats = method_stats[method]
            best_vals = [s["best_feasible_score"] for s in stats if s["best_feasible_score"] is not None]
            total_steps = sum(s["total_steps"] for s in stats)
            feasible_steps = sum(s["feasible_steps"] for s in stats)
            violations = [v for s in stats for v in s["violations"]]

            if best_vals and task_best is not None:
                best_score = max(best_vals)
                t0_for_best = next(
                    s["t0"] for s in stats if s["best_feasible_score"] == best_score
                )
                denom = task_best - t0_for_best
                max_norm = 1.0 if denom <= 1e-12 else min(1.0, (best_score - t0_for_best) / denom)
                row.append(f"{max_norm:.6f}")
            else:
                row.append("n/a")

            if total_steps > 0:
                row.append(f"{100.0 * feasible_steps / total_steps:.2f}")
            else:
                row.append("n/a")

            if violations:
                row.append(f"{sum(violations) / len(violations):.6f}")
            elif total_steps > 0:
                row.append("0.000000")
            else:
                row.append("n/a")

        print("\t".join(row))


if __name__ == "__main__":
    main()
