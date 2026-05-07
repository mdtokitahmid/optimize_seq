"""
Plot real-score trajectories for a single DNA run directory.

For each method, this script plots the real target score over optimization steps.
Markers encode whether a point improved over the initial target and whether it
remains feasible with respect to the real constraints:

- solid circle  : improved and feasible
- hollow circle : improved and constraint-violating
- faint circle  : not improved

Usage:
  python paper_figures/plot_dna_single_run_trajectory.py \
      --run-dir data/results/dna/sknsh_vs_others/start_03
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


def darken_color(color: str, amount: float = 0.65) -> tuple[float, float, float]:
    rgb = to_rgb(color)
    return tuple(max(0.0, c * amount) for c in rgb)


METHOD_ORDER = [
    "grace_lagrangian",
    "botorch_baseline",
    "directed_evolution",
    "simulated_annealing",
    "unconstrained",
]

METHOD_LABELS = {
    "grace_lagrangian": "GRACE",
    "botorch_baseline": "Constrained BO",
    "directed_evolution": "Grad. Guided Evol.",
    "simulated_annealing": "AdaLead",
    "unconstrained": "FastSeqProp",
}

METHOD_COLORS = {
    "grace_lagrangian": "#4A90C2",
    "botorch_baseline": "#57AE63",
    "directed_evolution": "#8646A8",
    "simulated_annealing": "#E07A5F",
    "unconstrained": "#9E9E9E",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot real-score DNA trajectories for a single run.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "results" / "dna" / "sknsh_vs_others" / "start_03",
        help="Directory containing per-method JSON result files.",
    )
    p.add_argument(
        "--step-stride",
        type=int,
        default=10,
        help="Plot every Nth step from the saved trajectories.",
    )
    p.add_argument(
        "--out-stem",
        type=str,
        default=None,
        help="Optional output filename stem. Defaults to <task>_<start>_real_trajectory.",
    )
    return p.parse_args()


def is_within_tolerance(val: float, c0: float, eps: float, constraint_type: str) -> bool:
    abs_drift = abs(val - c0)
    if constraint_type == "percent":
        return abs_drift / (abs(c0) + 1e-4) <= eps
    return abs_drift <= eps


def load_method(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def classify_points(data: dict, stride: int):
    target = data["target_oracle"]
    constraints = data.get("constraint_oracles", [])
    real_scores = data["real_scores"]
    initial_target = real_scores[target][0]
    real_c_initial = data["real_c_initial"]
    eps = data["constraint_eps"]
    constraint_type = data["constraint_type"]
    eval_n = min(500, len(real_scores[target]))

    best_feasible_step = None
    best_feasible_score = None
    for step in range(eval_n):
        target_val = real_scores[target][step]
        feasible = all(
            is_within_tolerance(real_scores[c][step], real_c_initial[c], eps, constraint_type)
            for c in constraints
        )
        improved = target_val > initial_target
        if feasible and improved and (best_feasible_score is None or target_val > best_feasible_score):
            best_feasible_step = step
            best_feasible_score = target_val

    steps = list(range(0, len(real_scores[target]), stride))
    if steps[-1] != len(real_scores[target]) - 1:
        steps.append(len(real_scores[target]) - 1)

    x = []
    y = []
    improved_feasible = []
    improved_violating = []
    not_improved = []

    for step in steps:
        target_val = real_scores[target][step]
        feasible = all(
            is_within_tolerance(real_scores[c][step], real_c_initial[c], eps, constraint_type)
            for c in constraints
        )
        improved = target_val > initial_target

        x.append(step)
        y.append(target_val)
        if improved and feasible:
            improved_feasible.append(True)
            improved_violating.append(False)
            not_improved.append(False)
        elif improved and not feasible:
            improved_feasible.append(False)
            improved_violating.append(True)
            not_improved.append(False)
        else:
            improved_feasible.append(False)
            improved_violating.append(False)
            not_improved.append(True)

    return {
        "target": target,
        "initial_target": initial_target,
        "best_feasible_step": best_feasible_step,
        "best_feasible_score": best_feasible_score,
        "x": x,
        "y": y,
        "improved_feasible": improved_feasible,
        "improved_violating": improved_violating,
        "not_improved": not_improved,
    }


def masked(values, mask):
    return [v for v, keep in zip(values, mask) if keep]


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = Path(__file__).resolve().parent

    methods = [m for m in METHOD_ORDER if (run_dir / f"{m}.json").exists()]
    if not methods:
        raise FileNotFoundError(f"No known method JSON files found in {run_dir}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 40,
        "legend.title_fontsize": 35,
    })

    fig, ax = plt.subplots(figsize=(20, 8))

    target_name = None
    initial_target = None
    best_scores = {}
    method_step_maxes = {}

    for method in methods:
        data = load_method(run_dir / f"{method}.json")
        classified = classify_points(data, args.step_stride)
        target_name = target_name or classified["target"]
        initial_target = initial_target if initial_target is not None else classified["initial_target"]

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        x = classified["x"]
        y = classified["y"]
        method_step_maxes[method] = x[-1]

        ax.plot(x, y, color=color, lw=3, alpha=0.8, zorder=1)

        ax.scatter(
            masked(x, classified["not_improved"]),
            masked(y, classified["not_improved"]),
            s=36,
            facecolors=color,
            edgecolors=color,
            linewidths=0.6,
            alpha=0.18,
            zorder=2,
        )
        ax.scatter(
            masked(x, classified["improved_feasible"]),
            masked(y, classified["improved_feasible"]),
            s=200,
            facecolors=color,
            edgecolors=darken_color(color),
            linewidths=2.5,
            alpha=1,
            zorder=3,
            label=label,
        )
        ax.scatter(
            masked(x, classified["improved_violating"]),
            masked(y, classified["improved_violating"]),
            s=200,
            facecolors="white",
            edgecolors=color,
            linewidths=2.5,
            alpha=1,
            zorder=4,
        )

        best_scores[method] = classified["best_feasible_score"]
        if classified["best_feasible_score"] is not None:
            ax.scatter(
                [classified["best_feasible_step"]],
                [classified["best_feasible_score"]],
                s=1000,
                marker="*",
                facecolors=color,
                edgecolors=darken_color(color),
                linewidths=3.0,
                alpha=1.0,
                zorder=5,
            )
            print(
                f"For {method}, best feasible real-trajectory step: "
                f"{classified['best_feasible_step']}, value: {classified['best_feasible_score']}"
            )

    ax.axhline(initial_target, color="black", lw=1.0, ls="--", alpha=0.55)

    ax.set_xlabel("Optimization step", fontsize=40)
    ax.set_ylabel("SK-N-SH Expression (DNA)", fontsize=40, labelpad=15)
    ax.set_xlim(-5, 505)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="major", length=10, width=2, labelsize=35)

    max_step = max(method_step_maxes.values(), default=505)
    if max_step > 505:
        ylim = ax.get_ylim()
        y_line = ylim[1] - 0.04 * (ylim[1] - ylim[0])
        ax.plot([500, 505], [y_line, y_line], color="grey", ls="--", lw=1.2, zorder=6)
        label_text = f"data continues to {max_step} steps"
        if method_step_maxes.get("directed_evolution", 0) == max_step:
            label_text = f"Dir. Evol. continues to {max_step} steps"
        ax.text(
            505,
            y_line,
            label_text,
            ha="right",
            va="center",
            color="grey",
            fontsize=18,
            backgroundcolor="white",
            alpha=0.9,
        )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[m],
            marker="o",
            markersize=15,
            lw=2.0,
            label=(
                f"{METHOD_LABELS[m]}: {best_scores[m]:.3f}"
                if best_scores.get(m) is not None
                else f"{METHOD_LABELS[m]}: n/a"
            ),
        )
        for m in methods
    ]
    state_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", markersize=15, lw=0, label="Improved + feasible"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markersize=15, lw=1.2, label="Improved + constraints violated"),
        Line2D([0], [0], color="black", lw=2, alpha=0.18, label="Not improved"),
    ]
    fig.legend(
        handles=method_handles,
        title="",
        loc="upper left",
        bbox_to_anchor=(0.1, 1.3),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=False,
        fontsize=38,
        title_fontsize=24,
    )
    fig.legend(
        handles=state_handles,
        title="",
        loc="upper right",
        bbox_to_anchor=(0.9, 1.3),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=False,
        fontsize=38,
        title_fontsize=24,
    )

    fig.subplots_adjust(top=0.80)

    out_stem = args.out_stem
    if out_stem is None:
        out_stem = f"dna_{run_dir.parent.name}_{run_dir.name}_real_trajectory"

    for ext in ("png", "pdf", "svg"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
