"""Compare |mean_epi_var_signed| (and its t-target siblings) across multiple
UQ checkpoints ("methods"), using each checkpoint's uq_per_chunk.csv from
aggregate_uq_by_cell.py.

mean_epi_var_signed = mean(exp(epi_logvar) - exp(logvar)) is already averaged
over all pixels/ODE-steps in a chunk -- positive and negative per-pixel
disagreement can cancel inside that mean, hiding chunks where the model
disagrees with itself a lot but in both directions. abs(mean_epi_var_signed)
removes that cancellation at the chunk level (it's a post-hoc transform of
the already-logged per-chunk scalar, not a re-derivation from raw per-pixel
tensors -- those aren't saved).

Usage:
    cd open-world
    uv run scripts/plot_abs_epi_var_signed_comparison.py \
        --csv v1=checkpoints/wm_droid/droid_flow_matching_uq_false_future_v1/replay_manual_check/uq_by_quadrant/uq_per_chunk.csv \
        --csv v2=checkpoints/wm_droid/droid_flow_matching_uq_false_future_v2/replay_manual_check/uq_by_quadrant/uq_per_chunk.csv \
        --csv v3=checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3/replay_manual_check/uq_by_quadrant/uq_per_chunk.csv \
        --output_dir results/manual_check_abs_epi_var_comparison
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
    "low_aleatoric/low_epistemic",
    "low_aleatoric/high_epistemic",
    "high_aleatoric/low_epistemic",
    "high_aleatoric/high_epistemic",
]
CELL_LABELS = [c.replace("/", "\n") for c in CELLS]

# Dataviz skill reference palette, categorical slots 1-4 (blue/orange/aqua/
# yellow) -- clears the *adjacent*-pair CVD/normal-vision floor used for
# grouped bars/violins (stricter all-pairs floor only needed for
# scatter/choropleth, where it's just the first 3 slots). Fixed order, not
# cycled. Shared by both the per-method and per-cell charts below -- they're
# never shown in the same legend, so slot reuse across the two categorical
# dimensions (method vs. cell) is fine. Extend here (not by cycling) if a 5th
# method/cell is ever added.
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CELL_COLORS_4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load(csv_specs: list[str]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for spec in csv_specs:
        label, path = spec.split("=", 1)
        df = pd.read_csv(path)
        df["abs_mean_epi_var_signed"] = df["mean_epi_var_signed"].abs()
        for t in ("t0.1", "t0.5", "t0.9"):
            col = f"{t}_mean_epi_var_signed"
            if col in df.columns:
                df[f"{t}_abs_mean_epi_var_signed"] = df[col].abs()
        dfs[label] = df
    return dfs


def violin_by_method(dfs: dict[str, pd.DataFrame], metric: str, title: str,
                      output_path: Path) -> None:
    """One violin per method, pooled across all cells/episodes/chunks."""
    methods = list(dfs.keys())
    data = [dfs[m][metric].dropna().values for m in methods]

    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 4.5))
    parts = ax.violinplot(data, positions=range(len(methods)),
                           showmeans=True, showmedians=False, showextrema=True)
    for pc, color in zip(parts["bodies"], METHOD_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor(color)
    for part in ("cmeans", "cbars", "cmins", "cmaxes"):
        if part in parts:
            parts[part].set_color("#52514e")
            parts[part].set_linewidth(1.2)

    for i, (m, d) in enumerate(zip(methods, data)):
        if len(d):
            ax.text(i, float(np.mean(d)), f"{np.mean(d):.4f}", ha="center",
                    va="bottom", fontsize=9, color="#0b0b0b")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def violin_by_method_and_cell(dfs: dict[str, pd.DataFrame], metric: str, title: str,
                               output_path: Path) -> None:
    """2x2 grid (one subplot per uncertainty cell), each with one violin per method."""
    methods = list(dfs.keys())
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
    axes = axes.flatten()

    for ax, cell, cell_label in zip(axes, CELLS, CELL_LABELS):
        data = [dfs[m][dfs[m]["uncertainty_cell"] == cell][metric].dropna().values
                for m in methods]
        data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
        parts = ax.violinplot(data, positions=range(len(methods)),
                               showmeans=True, showmedians=False, showextrema=True)
        for pc, color in zip(parts["bodies"], METHOD_COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor(color)
        for part in ("cmeans", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color("#52514e")
                parts[part].set_linewidth(1.0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(cell_label, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def violin_single_method_by_quadrant(df: pd.DataFrame, metric: str, method_label: str,
                                      title: str, output_path: Path) -> None:
    """One figure for a single method: one violin per of the 4 uncertainty cells."""
    data = [df[df["uncertainty_cell"] == c][metric].dropna().values for c in CELLS]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(data, positions=range(len(CELLS)),
                           showmeans=True, showmedians=False, showextrema=True)
    for pc, color in zip(parts["bodies"], CELL_COLORS_4):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor(color)
    for part in ("cmeans", "cbars", "cmins", "cmaxes"):
        if part in parts:
            parts[part].set_color("#52514e")
            parts[part].set_linewidth(1.2)

    for i, d in enumerate(data):
        if len(d) and not np.all(np.isnan(d)):
            ax.text(i, float(np.nanmean(d)), f"{np.nanmean(d):.4f}", ha="center",
                    va="bottom", fontsize=9, color="#0b0b0b")

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELL_LABELS, fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"{title} -- {method_label}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def bar_by_method(dfs: dict[str, pd.DataFrame], metric: str, title: str,
                   output_path: Path) -> None:
    """One bar per method (mean +/- std), pooled across all cells/episodes/chunks."""
    methods = list(dfs.keys())
    means = [float(dfs[m][metric].dropna().mean()) for m in methods]
    stds = [float(dfs[m][metric].dropna().std()) for m in methods]

    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 4.5))
    ax.bar(range(len(methods)), means, yerr=stds, color=METHOD_COLORS, alpha=0.8,
           capsize=6, width=0.5, error_kw={"linewidth": 1.5})
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.002, f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"{title}  (mean ± std)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def bar_single_method_by_quadrant(df: pd.DataFrame, metric: str, method_label: str,
                                   title: str, output_path: Path) -> None:
    """One figure for a single method: one bar (mean +/- std) per uncertainty cell."""
    means, stds = [], []
    for c in CELLS:
        vals = df[df["uncertainty_cell"] == c][metric].dropna().values
        means.append(float(np.mean(vals)) if len(vals) else 0.0)
        stds.append(float(np.std(vals)) if len(vals) else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(CELLS)), means, yerr=stds, color=CELL_COLORS_4, alpha=0.8,
           capsize=6, width=0.5, error_kw={"linewidth": 1.5})
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.002, f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELL_LABELS, fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"{title} -- {method_label}  (mean ± std)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", action="append", required=True,
                    help="label=path/to/uq_per_chunk.csv; repeat once per method.")
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()

    dfs = load(a.csv)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(dfs) > len(METHOD_COLORS):
        raise ValueError(
            f"{len(dfs)} methods but only {len(METHOD_COLORS)} colors in METHOD_COLORS "
            "-- add another validated dataviz palette slot rather than letting a "
            "method silently fall back to matplotlib's default color.")

    for m, df in dfs.items():
        print(f"[plot] {m}: {len(df)} chunks, "
              f"mean |epi_var_signed|={df['abs_mean_epi_var_signed'].mean():.4f}")

    violin_by_method(
        dfs, "abs_mean_epi_var_signed",
        "|mean_epi_var_signed| by method (pooled across cells)",
        output_dir / "violin_abs_epi_var_signed_overall.png")

    violin_by_method_and_cell(
        dfs, "abs_mean_epi_var_signed",
        "|mean_epi_var_signed| by method, split by uncertainty cell",
        output_dir / "violin_abs_epi_var_signed_by_cell.png")

    for m, df in dfs.items():
        violin_single_method_by_quadrant(
            df, "abs_mean_epi_var_signed", m,
            "|mean_epi_var_signed| by uncertainty cell",
            output_dir / f"violin_abs_epi_var_signed_quadrants_{m}.png")
        bar_single_method_by_quadrant(
            df, "abs_mean_epi_var_signed", m,
            "|mean_epi_var_signed| by uncertainty cell",
            output_dir / f"bar_abs_epi_var_signed_quadrants_{m}.png")

    bar_by_method(
        dfs, "abs_mean_epi_var_signed",
        "|mean_epi_var_signed| by method (pooled across cells)",
        output_dir / "bar_abs_epi_var_signed_overall.png")

    for t in ("t0.1", "t0.5", "t0.9"):
        col = f"{t}_abs_mean_epi_var_signed"
        if all(col in df.columns for df in dfs.values()):
            violin_by_method(
                dfs, col, f"|{t}_mean_epi_var_signed| by method (pooled)",
                output_dir / f"violin_abs_epi_var_signed_{t}_overall.png")

    print(f"[plot] done -> {output_dir}/")


if __name__ == "__main__":
    main()
