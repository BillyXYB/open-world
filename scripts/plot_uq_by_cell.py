"""Plot UQ metrics grouped by 2x2 uncertainty cell.

Reads either uq_per_chunk.csv (from --chunk_jsonl mode) or
uq_per_episode.csv (from --replay_summary mode) and produces:
  - violin plots for key UQ metrics
  - a 2x2 bar chart (mean ± std) matching the cell grid

Usage:
    cd open-world
    uv run python scripts/plot_uq_by_cell.py \
        --csv results/uq_2x2_v0/uq_per_chunk.csv \
        --output_dir results/uq_2x2_v0/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELLS = [
    "high_var/large_data",
    "high_var/small_data",
    "low_var/large_data",
    "low_var/small_data",
]
CELL_LABELS = [
    "hi-var\nlarge-data",
    "hi-var\nsmall-data",
    "lo-var\nlarge-data",
    "lo-var\nsmall-data",
]
CELL_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

# Metric groups: (group_title, [metric_cols])
METRIC_GROUPS = [
    ("Aleatoric uncertainty", [
        "mean_aleatoric_var",
        "t0.9_mean_aleatoric_var",
        "t0.5_mean_aleatoric_var",
        "t0.1_mean_aleatoric_var",
    ]),
    ("Epistemic uncertainty (LTV)", [
        "mean_epi_ltv",
        "t0.9_mean_epi_ltv",
        "t0.5_mean_epi_ltv",
        "t0.1_mean_epi_ltv",
    ]),
    ("Epistemic uncertainty (var / KL)", [
        "mean_epi_var",
        "mean_pdf_diff",
        "mean_kl",
    ]),
    ("Reconstruction quality", [
        "latent_mse",
        "psnr_db",
    ]),
]


def violin_grid(df: pd.DataFrame, metrics: list[str], title: str,
                output_path: Path) -> None:
    """One subplot per metric, violin plots grouped by cell."""
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return
    ncols = min(len(metrics), 4)
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, metric in zip(axes, metrics):
        data_by_cell = [df[df["uncertainty_cell"] == c][metric].dropna().values
                        for c in CELLS]
        data_by_cell = [d for d in data_by_cell]  # keep empty arrays

        parts = ax.violinplot(
            [d if len(d) > 0 else [np.nan] for d in data_by_cell],
            positions=range(len(CELLS)),
            showmeans=True, showmedians=False, showextrema=True,
        )
        for pc, color in zip(parts["bodies"], CELL_COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for part in ("cmeans", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color("black")
                parts[part].set_linewidth(1.2)

        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels(CELL_LABELS, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def bar_2x2(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """2x2 bar chart (mean ± std) matching the cell grid layout."""
    if metric not in df.columns:
        return
    # Layout: row = variance_level, col = data_level
    grid_cells = [
        ["high_var/large_data", "high_var/small_data"],
        ["low_var/large_data",  "low_var/small_data"],
    ]
    row_labels = ["High variance", "Low variance"]
    col_labels = ["Large data", "Small data"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=True)
    for r, (row, rl) in enumerate(zip(grid_cells, row_labels)):
        for c, (cell, cl) in enumerate(zip(row, col_labels)):
            ax = axes[r][c]
            vals = df[df["uncertainty_cell"] == cell][metric].dropna().values
            mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std  = float(np.std(vals))  if len(vals) > 0 else 0.0
            color = CELL_COLORS[CELLS.index(cell)]
            ax.bar([0], [mean], yerr=[std], color=color, alpha=0.75,
                   capsize=6, width=0.5, error_kw={"linewidth": 1.5})
            ax.set_xticks([])
            ax.set_title(f"{rl}\n{cl}", fontsize=9)
            ax.text(0, mean + std * 0.05, f"{mean:.4f}", ha="center",
                    va="bottom", fontsize=8)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            if c == 0:
                ax.set_ylabel(metric, fontsize=8)

    fig.suptitle(f"2×2 grid: {metric}  (mean ± std)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="uq_per_chunk.csv or uq_per_episode.csv from aggregate_uq_by_cell.py")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write plot PNGs")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] loaded {len(df)} rows, {df['uncertainty_cell'].value_counts().to_dict()}")

    # Violin grids per metric group
    for group_title, metrics in METRIC_GROUPS:
        slug = group_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        violin_grid(df, metrics, group_title, output_dir / f"violin_{slug}.png")

    # 2x2 bar charts for primary metrics
    for metric in ["mean_aleatoric_var", "mean_epi_ltv", "mean_epi_var",
                   "mean_kl", "t0.9_mean_aleatoric_var", "t0.9_mean_epi_ltv"]:
        if metric in df.columns:
            bar_2x2(df, metric, output_dir / f"bar2x2_{metric}.png")

    print(f"[plot] done — plots written to {output_dir}/")


if __name__ == "__main__":
    main()
