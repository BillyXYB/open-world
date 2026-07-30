"""Aggregate UQ replay metrics by 2x2 uncertainty cell.

Two input modes (use one or both in the same call):

  --replay_summary  replay_summary.json written at the end of replay_libero_wm_traj.py.
                    Joined with val annotations to get cell labels.
                    Writes: uq_per_episode.csv, uq_by_cell.json

  --chunk_jsonl     chunk_metrics.jsonl streamed live during replay (available mid-run).
                    Already contains uncertainty_cell — no annotation join needed.
                    Writes: uq_per_chunk.csv, uq_by_cell_chunks.json

Usage:
    cd open-world
    # From finished replay (episode-level):
    uv run python scripts/aggregate_uq_by_cell.py \\
        --replay_summary .../replay_summary.json \\
        --data_root data/libero_uq_2x2 \\
        --output_dir results/uq_2x2_v0

    # From live/partial JSONL (chunk-level, mid-run):
    uv run python scripts/aggregate_uq_by_cell.py \\
        --chunk_jsonl .../chunk_metrics.jsonl \\
        --output_dir results/uq_2x2_v0

Example seaborn plot:
    import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
    df = pd.read_csv("results/uq_2x2_v0/uq_per_chunk.csv")
    sns.violinplot(data=df, x="uncertainty_cell", y="mean_aleatoric_var")
    plt.tight_layout(); plt.savefig("uq_violin.png")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fields from the annotation JSON that we want to join in
ANN_FIELDS = [
    "uncertainty_cell", "variance_level", "data_level",
    "mu", "sigma_cm", "is_success",
]

CELLS = [
    "high_var/large_data",
    "high_var/small_data",
    "low_var/large_data",
    "low_var/small_data",
]


def cvar95(vals: np.ndarray) -> float:
    """Mean of the top-5% (worst-case) values."""
    if len(vals) == 0:
        return float("nan")
    threshold = np.percentile(vals, 95)
    tail = vals[vals >= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)


KEY_METRICS = [
    "mean_aleatoric_var", "mean_epi_ltv", "mean_kl",
    "t0.9_mean_aleatoric_var", "t0.9_mean_epi_ltv",
    "latent_mse", "psnr_db",
]


def aggregate_by_cell(df: pd.DataFrame, numeric_cols: list[str],
                      count_key: str = "n_rows") -> dict[str, dict]:
    """Group df by uncertainty_cell and compute mean/std/CVaR95 per numeric column."""
    cell_agg: dict[str, dict] = {}
    for cell in CELLS:
        group = df[df["uncertainty_cell"] == cell]
        if group.empty:
            print(f"[aggregate] WARNING: no rows for cell '{cell}'")
            cell_agg[cell] = {count_key: 0}
            continue
        agg: dict[str, object] = {count_key: len(group)}
        for col in numeric_cols:
            vals = group[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            agg[col] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "cvar95": cvar95(vals),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            }
        cell_agg[cell] = agg
    return cell_agg


def print_summary_table(cell_agg: dict, numeric_cols: list[str], count_key: str = "n_rows") -> None:
    # KEY_METRICS first, then any remaining numeric cols not in the list
    available = [m for m in KEY_METRICS if m in numeric_cols]
    available += [m for m in numeric_cols if m not in available]
    if not available:
        return
    header = f"{'cell':35s}" + "".join(f"  {m[:18]:>18s}" for m in available)
    print()
    print(header)
    print("-" * len(header))
    for cell in CELLS:
        agg = cell_agg.get(cell, {})
        n = agg.get(count_key, 0)
        row = f"{cell:35s}"
        for m in available:
            if m in agg and isinstance(agg[m], dict):
                row += f"  {agg[m]['mean']:>18.4f}"
            else:
                row += f"  {'—':>18s}"
        print(f"{row}   (n={n})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay_summary", default=None,
                   help="Path to replay_summary.json from replay_libero_wm_traj.py")
    p.add_argument("--chunk_jsonl", default=None,
                   help="Path to chunk_metrics.jsonl (streamed live; usable mid-run)")
    p.add_argument("--data_root", default="data/libero_uq_2x2",
                   help="Root of UQ 2x2 dataset (to find val annotations; used with --replay_summary)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write output files")
    a = p.parse_args()

    if not a.replay_summary and not a.chunk_jsonl:
        p.error("At least one of --replay_summary or --chunk_jsonl is required.")

    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Mode A: chunk_metrics.jsonl  (per-chunk, available mid-run)
    # ------------------------------------------------------------------ #
    if a.chunk_jsonl:
        jsonl_path = Path(a.chunk_jsonl)
        print(f"[aggregate] loading {jsonl_path}")
        df_chunks = pd.read_json(jsonl_path, lines=True)
        print(f"[aggregate] {len(df_chunks)} chunk records  "
              f"({df_chunks['episode'].nunique()} episodes so far)")

        non_numeric_chunk = {"suite", "episode", "uncertainty_cell",
                             "chunk_idx", "frame_now", "chunk_frames"}
        numeric_cols_chunk = [c for c in df_chunks.columns
                              if c not in non_numeric_chunk
                              and pd.api.types.is_numeric_dtype(df_chunks[c])]

        csv_path = output_dir / "uq_per_chunk.csv"
        df_chunks.to_csv(csv_path, index=False)
        print(f"[aggregate] wrote {len(df_chunks)} rows → {csv_path}")

        cell_agg_chunks = aggregate_by_cell(df_chunks, numeric_cols_chunk, count_key="n_chunks")

        json_path = output_dir / "uq_by_cell_chunks.json"
        json_path.write_text(json.dumps(cell_agg_chunks, indent=2))
        print(f"[aggregate] wrote {json_path}")

        print_summary_table(cell_agg_chunks, numeric_cols_chunk, count_key="n_chunks")

    # ------------------------------------------------------------------ #
    # Mode B: replay_summary.json  (per-episode, written at end of replay)
    # ------------------------------------------------------------------ #
    if a.replay_summary:
        replay_path = Path(a.replay_summary)
        data_root = Path(a.data_root)

        print(f"\n[aggregate] loading {replay_path}")
        summary = json.loads(replay_path.read_text())
        print(f"[aggregate] {len(summary)} episode records")

        # Join with val annotations if uncertainty_cell not already present
        missing = []
        for rec in summary:
            if rec.get("uncertainty_cell") is not None:
                # Already populated by replay script (manifest mode)
                for field in ANN_FIELDS:
                    if field not in rec or rec[field] is None:
                        suite = rec.get("suite", "")
                        ep_id = rec.get("episode", "")
                        ann_path = data_root / suite / "annotation" / "val" / f"{ep_id}.json"
                        if not ann_path.exists():
                            ann_path = data_root / suite / "annotation" / "train" / f"{ep_id}.json"
                        if ann_path.exists():
                            ann = json.loads(ann_path.read_text())
                            rec[field] = ann.get(field)
                continue
            suite = rec.get("suite", "")
            ep_id = rec.get("episode", "")
            ann_path = data_root / suite / "annotation" / "val" / f"{ep_id}.json"
            if not ann_path.exists():
                ann_path = data_root / suite / "annotation" / "train" / f"{ep_id}.json"
            if ann_path.exists():
                ann = json.loads(ann_path.read_text())
                for field in ANN_FIELDS:
                    rec[field] = ann.get(field)
            else:
                missing.append(f"{suite}/{ep_id}")
                for field in ANN_FIELDS:
                    rec[field] = None

        if missing:
            print(f"[aggregate] WARNING: {len(missing)} episodes have no annotation file:")
            for m in missing[:5]:
                print(f"    {m}")

        df = pd.DataFrame(summary)
        if "episode" in df.columns and "episode_id" not in df.columns:
            df = df.rename(columns={"episode": "episode_id"})

        non_numeric = {"suite", "episode_id", "text",
                       "uncertainty_cell", "variance_level", "data_level"}
        numeric_cols = [c for c in df.columns
                        if c not in non_numeric
                        and pd.api.types.is_numeric_dtype(df[c])]

        csv_path = output_dir / "uq_per_episode.csv"
        df.to_csv(csv_path, index=False)
        print(f"[aggregate] wrote {len(df)} rows → {csv_path}")

        cell_agg = aggregate_by_cell(df, numeric_cols, count_key="n_episodes")

        # Add success stats
        for cell in CELLS:
            group = df[df["uncertainty_cell"] == cell]
            if not group.empty and "is_success" in group.columns:
                n = len(group)
                n_success = int(group["is_success"].sum())
                cell_agg[cell]["n_success"] = n_success
                cell_agg[cell]["success_rate"] = round(n_success / n, 3)

        json_path = output_dir / "uq_by_cell.json"
        json_path.write_text(json.dumps(cell_agg, indent=2))
        print(f"[aggregate] wrote {json_path}")

        print_summary_table(cell_agg, numeric_cols, count_key="n_episodes")


if __name__ == "__main__":
    main()
