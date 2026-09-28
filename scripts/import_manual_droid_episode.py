"""
Import one or more hand-collected DROID episode bundles (produced locally by
droid/scripts/convert/export_episode_for_wm_uq.py) into the `droid_ctrl_world`-style
on-disk schema (annotation JSON + latent .pt + raw mp4) that `scripts/replay_libero_wm_traj.py`
and `openworld.training.world_model.dataset.LiberoLatentDataset` consume.

Meant to run on the GPU machine that holds the SVD VAE weights and the training data root
(the same environment `jobs/replay_droid_wm_traj_epi_false_future.sh` runs in).

Each bundle is a directory `<bundle_dir>/<episode_id>/` containing:
    bundle.npz   -- cam_rgb_left/right/wrist (T,H,W,3) uint8, cart (T,6), grip (T,), joint_position (T,7)
    meta.json    -- {"language": str, "fps": int, "down_sample": int, ...}

Usage:
    uv run scripts/import_manual_droid_episode.py \
        --bundle_dir /path/to/wm_uq_export \
        --data_root /scratch/gpfs/AM43/yy4041/data \
        --suite droid_manual_uq_check \
        --split val \
        --svd_model_path external/svd_weights   # same path used elsewhere for LatentEncoder
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_libero_for_wm import LatentEncoder  # noqa: E402

from openworld.utils.droid_export import next_episode_id, write_droid_episode  # noqa: E402


def find_bundle_dirs(bundle_dir: Path) -> list:
    """A bundle_dir is either a single episode dir (has bundle.npz directly) or a parent
    dir containing one or more episode subdirs."""
    if (bundle_dir / "bundle.npz").exists():
        return [bundle_dir]
    return sorted(p for p in bundle_dir.iterdir() if p.is_dir() and (p / "bundle.npz").exists())


def ensure_stat_json(data_root: Path, suite: str, reference_suite: str = "droid_ctrl_world") -> None:
    stat_path = data_root / "dataset_meta_info" / suite / "stat.json"
    if stat_path.exists():
        return
    ref_path = data_root / "dataset_meta_info" / reference_suite / "stat.json"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"No stat.json for suite '{suite}' and no reference stat.json at {ref_path} to copy from. "
            f"Either place a stat.json at {stat_path} yourself, or make sure '{reference_suite}' exists."
        )
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref_path, stat_path)
    print(f"[*] Bootstrapped {stat_path} from {ref_path} (reusing '{reference_suite}' normalization stats).")


def import_bundle(bundle_path: Path, *, data_root: Path, suite: str, split: str, encoder) -> None:
    data = np.load(bundle_path / "bundle.npz")
    meta = json.loads((bundle_path / "meta.json").read_text())

    cam_rgb = [data["cam_rgb_left"], data["cam_rgb_right"], data["cam_rgb_wrist"]]
    cart = data["cart"]
    grip = data["grip"]
    joint_position = data["joint_position"]

    episode_id = f"{next_episode_id(data_root, suite):06d}"
    write_droid_episode(
        suite=suite,
        split=split,
        episode_id=episode_id,
        output_root=data_root,
        encoder=encoder,
        cam_rgb=cam_rgb,
        cart=cart,
        grip=grip,
        joint_position=joint_position,
        language=meta["language"],
        fps=meta["fps"],
        down_sample=meta["down_sample"],
        write_raw=True,
        extra_annotation={
            "source": "hand_collected_manual_uq_check",
            "source_bundle": str(bundle_path),
            "original_episode_id": meta.get("episode_id"),
        },
    )
    print(f"[*] Imported '{bundle_path.name}' -> {suite}/{split}/{episode_id} (task='{meta['language']}')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import hand-collected DROID episode bundle(s) for WM/UQ replay.")
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--suite", default="droid_manual_uq_check")
    parser.add_argument("--split", default="val")
    parser.add_argument("--svd_model_path", default=None, help="Path to SVD weights for LatentEncoder; "
                         "omit to skip latent encoding (raw videos are still written).")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    ensure_stat_json(args.data_root, args.suite)

    encoder = None
    if args.svd_model_path is not None:
        encoder = LatentEncoder(svd_path=args.svd_model_path, target_h=args.height, target_w=args.width)

    bundle_dirs = find_bundle_dirs(args.bundle_dir)
    if not bundle_dirs:
        raise FileNotFoundError(f"No bundle.npz found under {args.bundle_dir}")

    for bundle_path in bundle_dirs:
        import_bundle(bundle_path, data_root=args.data_root, suite=args.suite, split=args.split, encoder=encoder)


if __name__ == "__main__":
    main()
