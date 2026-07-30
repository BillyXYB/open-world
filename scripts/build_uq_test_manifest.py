"""Build a cell-grouped test manifest from the UQ 2x2 val split.

Scans all annotation/val/*.json files across the 4 LIBERO suites in the
UQ data root, groups episode records by uncertainty_cell, and writes
test_manifest_by_cell.json.  This manifest is the index used for
post-training UQ metric evaluation.

Also generates data/libero_uq_2x2/stat.json (action normalization stats)
from training annotations, which is required before training.

Usage:
    cd open-world
    uv run python scripts/build_uq_test_manifest.py \\
        --data_root data/libero_uq_2x2 \\
        --output data/libero_uq_2x2/test_manifest_by_cell.json

    # Also generate stat.json (do this before training):
    uv run python scripts/build_uq_test_manifest.py \\
        --data_root data/libero_uq_2x2 --gen_stat
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CELLS = [
    "high_var/large_data",
    "high_var/small_data",
    "low_var/large_data",
    "low_var/small_data",
]


def gen_stat(data_root: Path) -> None:
    """Compute 1st/99th percentile of (cartesian_position, gripper_position)
    across all training annotations and write data_root/stat.json.

    LiberoLatentDataset._load_stat() finds this file as the fallback
    <meta_root>/stat.json path (second candidate), so a single shared file
    at the UQ root covers all 4 suites.
    """
    all_states: list[np.ndarray] = []
    ann_files = sorted(data_root.glob("*/annotation/train/*.json"))
    if not ann_files:
        raise FileNotFoundError(
            f"No training annotations found under {data_root}/*/annotation/train/"
        )
    for ann_path in ann_files:
        ann = json.loads(ann_path.read_text())
        cart = np.array(ann["observation.state.cartesian_position"], dtype=np.float32)  # (T, 6)
        grip = np.array(ann["observation.state.gripper_position"], dtype=np.float32)    # (T,) or (T,1)
        if grip.ndim == 1:
            grip = grip[:, None]
        state = np.concatenate([cart, grip], axis=-1)  # (T, 7)
        all_states.append(state)

    stacked = np.concatenate(all_states, axis=0)  # (N, 7)
    stat = {
        "state_01": np.percentile(stacked, 1, axis=0).tolist(),
        "state_99": np.percentile(stacked, 99, axis=0).tolist(),
    }
    out = data_root / "stat.json"
    out.write_text(json.dumps(stat, indent=2))
    print(f"Written {out}  ({len(all_states)} episodes, {stacked.shape[0]} frames)")
    print("  state_01:", [f"{v:.4f}" for v in stat["state_01"]])
    print("  state_99:", [f"{v:.4f}" for v in stat["state_99"]])


def build_test_manifest(data_root: Path, output: Path) -> dict:
    """Scan val annotations, group by uncertainty_cell, write manifest."""
    cells: dict[str, list[dict]] = defaultdict(list)
    missing_cell: list[str] = []

    ann_files = sorted(data_root.glob("*/annotation/val/*.json"))
    if not ann_files:
        raise FileNotFoundError(
            f"No val annotations found under {data_root}/*/annotation/val/"
        )

    for ann_path in ann_files:
        ann = json.loads(ann_path.read_text())
        cell = ann.get("uncertainty_cell")
        suite = ann_path.parts[-4]  # .../libero_goal/annotation/val/000097.json
        episode_id = ann_path.stem

        if not cell:
            missing_cell.append(str(ann_path))
            continue

        # relative path from data_root so the manifest is portable
        rel_ann = str(ann_path.relative_to(data_root))

        cells[cell].append({
            "episode_id": episode_id,
            "suite": suite,
            "ann_path": rel_ann,
            "task_name": ann.get("task_name", ann.get("language_instruction", "")),
            "task_id": ann.get("task_id"),
            "mu": ann.get("mu"),
            "sigma_cm": ann.get("sigma_cm"),
            "variance_level": ann.get("variance_level"),
            "data_level": ann.get("data_level"),
            "is_success": ann.get("is_success"),
            "episode_steps": ann.get("episode_steps"),
        })

    if missing_cell:
        print(f"WARNING: {len(missing_cell)} annotations missing 'uncertainty_cell' field")
        for p in missing_cell[:5]:
            print(f"  {p}")

    # Summary per cell
    summary: dict[str, dict] = {}
    for cell in CELLS:
        eps = cells.get(cell, [])
        tasks = sorted({e["task_name"] for e in eps})
        suites = sorted({e["suite"] for e in eps})
        n_success = sum(1 for e in eps if e.get("is_success"))
        summary[cell] = {
            "n_episodes": len(eps),
            "n_success": n_success,
            "success_rate": round(n_success / len(eps), 3) if eps else 0.0,
            "n_tasks": len(tasks),
            "suites": suites,
            "tasks": tasks,
        }

    manifest = {"cells": dict(cells), "summary": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(f"\nWritten {output}")
    print("\nCell summary:")
    for cell, s in summary.items():
        print(f"  {cell:30s}  n={s['n_episodes']:4d}  success={s['success_rate']:.1%}"
              f"  tasks={s['n_tasks']}  suites={s['suites']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/libero_uq_2x2",
                        help="Root of UQ 2x2 dataset")
    parser.add_argument("--output", default=None,
                        help="Output manifest path (default: <data_root>/test_manifest_by_cell.json)")
    parser.add_argument("--gen_stat", action="store_true",
                        help="Also generate stat.json for action normalization")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    output = Path(args.output) if args.output else data_root / "test_manifest_by_cell.json"

    if args.gen_stat:
        print("=== Generating stat.json ===")
        gen_stat(data_root)
        print()

    print("=== Building test manifest ===")
    build_test_manifest(data_root, output)


if __name__ == "__main__":
    main()
