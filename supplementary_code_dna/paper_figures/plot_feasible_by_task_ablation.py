"""
plot_feasible_by_task_ablation.py

Strip-scatter of normalized feasible improvement across the 3 DNA tasks for the
GRACE ablation methods.
Each dot = one feasible-improving step. Color = method.
Normalization: (score - t0) / (task_best - t0)
  task_best = max raw score seen by ANY ablation method on that task,
              computed from the same score array as the points → always <= 1.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             13,
    "axes.labelsize":        14,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       12,
    "legend.title_fontsize": 12,
    "figure.dpi":            300,
    "pdf.fonttype":          42,
})

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DNA_BASE = BASE / "data" / "results" / "dna"
STARTS = [f"start_{i:02d}" for i in range(1, 6)]

METHODS = ["GRACE", "Projection Only", "Lagrangian only"]

FNAME = {
    "GRACE": "grace_lagrangian",
    "Projection Only": "grace_only",
    "Lagrangian only": "lagrangian_only",
}

COLORS = {
    "GRACE": "#3498db",
    "Projection Only": "#27ae60",
    "Lagrangian only": "#f39c12",
}

# 3 DNA tasks: (label, task_dir, starts, fname_map)
task_cfg = [
    ("k562\n(DNA)",       DNA_BASE / "k562_vs_others",    STARTS, FNAME),
    ("HepG2\n(DNA)",      DNA_BASE / "hepG2_vs_others",   STARTS, FNAME),
    ("SK-N-SH\n(DNA)",    DNA_BASE / "sknsh_vs_others",   STARTS, FNAME),
]


def get_scores(path):
    """Return (scores_array, t0, constraint info) using real_scores if available."""
    d = json.load(open(path))
    target = d["target_oracle"]
    constraints = d.get("constraint_oracles", [])
    c0 = d.get("real_c_initial") or d["c_initial"]
    t0 = c0[target]
    ctype = d.get("constraint_type", "absolute")
    eps = d.get("constraint_eps", 0.1)
    sc = d.get("real_scores", d["scores"])
    return sc, target, constraints, c0, t0, ctype, eps


def collect_raw_points(task_dir, starts, fname_map, method):
    """Return list of (raw_score, t0) for every feasible-improving step."""
    stem = fname_map[method]
    files = (
        [task_dir / (stem + ".json")]
        if starts is None
        else [task_dir / s / (stem + ".json") for s in starts]
    )

    points = []
    for p in files:
        if not p.exists():
            continue
        sc, target, constraints, c0, t0, ctype, eps = get_scores(p)
        use_n = min(500, len(sc[target]))
        for i in range(use_n):
            viol = 0.0
            for c in constraints:
                drift = abs(sc[c][i] - c0[c])
                viol += (
                    max(0.0, drift - eps)
                    if ctype == "absolute"
                    else max(0.0, drift / (abs(c0[c]) + 1e-4) - eps)
                )
            if viol == 0.0 and sc[target][i] > t0:
                points.append((float(sc[target][i]), float(t0)))
    return points


def task_best_score(task_dir, starts, fname_map):
    """Max raw score any ablation method achieves on this task."""
    best = None
    for m in METHODS:
        for score, _ in collect_raw_points(task_dir, starts, fname_map, m):
            if best is None or score > best:
                best = score
    return best


def collect_normalized(task_dir, starts, fname_map, method):
    """Normalized improvements for one method on one task. Always in [0, 1]."""
    best = task_best_score(task_dir, starts, fname_map)
    if best is None:
        return np.array([])
    points = collect_raw_points(task_dir, starts, fname_map, method)
    out = []
    for score, t0 in points:
        denom = best - t0
        if denom > 1e-12:
            out.append(min(1.0, (score - t0) / denom))
    return np.array(out)


all_data = {}
for ti, (_, tdir, starts, fmap) in enumerate(task_cfg):
    for mi, m in enumerate(METHODS):
        all_data[(ti, mi)] = collect_normalized(tdir, starts, fmap, m)


group_gap = 1.25
method_w = 0.13
offsets = np.linspace(-(len(METHODS) - 1) / 2, (len(METHODS) - 1) / 2, len(METHODS)) * method_w
jitter_std = 0.018

rng = np.random.default_rng(0)

ALL_IDX = list(range(len(task_cfg)))


def draw_strip(ax, task_indices):
    x_local = np.arange(len(task_indices)) * group_gap
    for ti_local, ti_global in enumerate(task_indices):
        for mi, m in enumerate(METHODS):
            vals = all_data[(ti_global, mi)]
            xc = x_local[ti_local] + offsets[mi]
            if len(vals) == 0:
                ax.scatter(
                    [xc], [0.0],
                    marker="x", s=60,
                    color=COLORS[m], alpha=1.0,
                    linewidths=1.8, zorder=6,
                )
                ax.text(
                    xc, -0.015, "n/a",
                    ha="center", va="top", fontsize=6,
                    color=COLORS[m], style="italic",
                )
                continue
            jit = rng.normal(0, jitter_std, size=len(vals))
            ax.scatter(
                xc + jit, vals,
                color=COLORS[m],
                alpha=0.30,
                s=5,
                linewidths=0,
                zorder=3,
            )
            bp = ax.boxplot(
                vals,
                positions=[xc],
                widths=method_w * 0.85,
                patch_artist=True,
                showfliers=False,
                whiskerprops=dict(color="black", linewidth=1.0, linestyle="-"),
                capprops=dict(color="black", linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.8),
                boxprops=dict(color="black", linewidth=1.0),
                zorder=4,
            )
            bp["boxes"][0].set_facecolor(COLORS[m])
            bp["boxes"][0].set_alpha(0.18)
            ax.scatter(
                [xc], [vals.max()],
                marker="*", s=320,
                color=COLORS[m],
                edgecolors="white", linewidths=0.8,
                alpha=1.0,
                zorder=7,
            )
    return x_local


def style_ax(ax, task_indices, x_local):
    labels = [task_cfg[i][0] for i in task_indices]
    ax.set_xticks(x_local)
    ax.set_xticklabels(labels)
    ax.set_xlim(x_local[0] - group_gap * 0.6, x_local[-1] + group_gap * 0.6)
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_ylabel("Normalized Feasible Improvement")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.8)


handles = [
    mpatches.Patch(facecolor=COLORS[m], label=m, alpha=0.9, edgecolor="grey", linewidth=0.4)
    for m in METHODS
]

fig, ax = plt.subplots(figsize=(8.0, 5.2))
fig.subplots_adjust(top=0.78, bottom=0.22)
x_local = draw_strip(ax, ALL_IDX)
style_ax(ax, ALL_IDX, x_local)

ax.legend(
    handles=handles,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    bbox_transform=ax.transAxes,
    framealpha=0.95,
    edgecolor="#cccccc",
    frameon=True,
    columnspacing=1.2,
    handlelength=1.2,
)

fig.tight_layout(rect=[0, 0, 1, 0.78])
for ext in ("svg", "png"):
    out = OUT_DIR / f"feasible_by_task_ablation_dna.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
plt.close(fig)
