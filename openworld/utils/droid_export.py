"""Export collected DROID hardware trajectories into the on-disk schema the
real ``droid_ctrl_world`` training set already uses (verified against a live
annotation JSON at
``/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world/annotation/train/*.json``),
so newly collected trajectories are directly loadable by
``openworld.training.world_model.dataset.LiberoLatentDataset`` alongside the
existing training data -- no dataset-loader changes needed.

This is DROID's 3-camera analog of
``scripts/preprocess_libero_for_wm.py::write_episode``, which hardcodes
LIBERO's 2-camera (agentview/wrist) schema and therefore can't be reused
as-is here.

On-disk layout written per episode:

    <output_root>/<suite>/annotation/<split>/<episode_id>.json
    <output_root>/<suite>/latent_videos/<split>/<episode_id>/<cam_idx>.pt
    <output_root>/<suite>/raw_videos/<split>/<episode_id>/<cam_idx>.mp4   (if write_raw)

The dataset loader only ever reads ``texts``, ``latent_videos`` and
``observation.state.{cartesian_position,gripper_position}`` from the
annotation JSON (see ``LiberoLatentDataset.__getitem__``), and
``write_sample_list`` (reused as-is from ``preprocess_libero_for_wm.py``)
only reads ``observation.state.cartesian_position``'s length -- so this
writer only needs to populate those fields plus whatever free-form
``extra_annotation`` the caller wants recorded for provenance.

CAUTION for future training runs that mix this suite with the original
``droid_ctrl_world``: episodes written here have camera frames and state
sampled at the SAME round-trip rate (``down_sample=1`` is correct), whereas
``droid_ctrl_world`` itself needs ``down_sample=3`` (its state/action arrays
are natively 3x denser than its pre-encoded latents). ``LiberoLatentDataset``
applies one ``args.down_sample`` across every dataset named in
``dataset_names``, so training on both suites in the same run with a single
down_sample value will misalign whichever one it wasn't tuned for. Train on
this suite alone, or with other ``down_sample=1`` data, until the
dataset loader supports a per-dataset down_sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from preprocess_libero_for_wm import LatentEncoder


def _write_mp4(*args, **kwargs) -> None:
    """Lazy import of preprocess_libero_for_wm._write_mp4 -- ``scripts/`` is
    not an importable package (no __init__.py), so it's only reachable via
    sys.path, same as scripts/run_data_collection_active_uq.py does for its
    own imports from that directory."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from preprocess_libero_for_wm import _write_mp4 as _impl

    return _impl(*args, **kwargs)


def write_droid_episode(
    *,
    suite: str,
    split: str,
    episode_id: str,
    output_root: Path,
    encoder: Optional["LatentEncoder"],
    cam_rgb: list[np.ndarray],   # len == num_cams, each (T_raw, H, W, 3) uint8, native rate
    cart: np.ndarray,            # (T_raw, 6) FK-derived xyz+euler, native rate
    grip: np.ndarray,            # (T_raw,) native rate
    joint_position: np.ndarray,  # (T_raw, 7) native rate
    language: str,
    fps: int,
    down_sample: int,
    write_raw: bool = True,
    extra_annotation: Optional[dict[str, Any]] = None,
) -> None:
    """Write one DROID hardware episode in the droid_ctrl_world layout.

    ``down_sample`` is recorded in the annotation so
    ``LiberoLatentDataset``/``load_full_episode`` (both of which read
    ``down_sample`` from the training config, not the JSON) pair the correct
    native-rate state row with each latent-rate video frame -- match
    whatever ``down_sample`` the collection config's world model used.
    """
    suite_root = Path(output_root) / suite
    ann_dir = suite_root / "annotation" / split
    ann_dir.mkdir(parents=True, exist_ok=True)

    num_cams = len(cam_rgb)
    latent_videos_meta: list[dict] = []
    if encoder is not None:
        lat_dir = suite_root / "latent_videos" / split / episode_id
        lat_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx in range(num_cams):
            latent = encoder.encode(cam_rgb[cam_idx])
            rel_path = f"latent_videos/{split}/{episode_id}/{cam_idx}.pt"
            torch.save(latent, suite_root / rel_path)
            latent_videos_meta.append({"latent_video_path": rel_path})

    raw_videos_meta: list[dict] = []
    if write_raw:
        raw_dir = suite_root / "raw_videos" / split / episode_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx in range(num_cams):
            rel_path = f"raw_videos/{split}/{episode_id}/{cam_idx}.mp4"
            _write_mp4(suite_root / rel_path, cam_rgb[cam_idx], fps)
            raw_videos_meta.append({"video_path": rel_path})

    annotation = {
        "texts": [language],
        "language_instruction": language,
        "task_suite": suite,
        "episode_id": episode_id,
        "fps": fps,
        "down_sample": down_sample,
        "video_length": int(cam_rgb[0].shape[0]) if encoder is not None else None,
        "state_length": int(len(cart)),
        "raw_length": int(len(cart)),
        "observation.state.cartesian_position": np.asarray(cart, dtype=np.float32).tolist(),
        "observation.state.gripper_position": np.asarray(grip, dtype=np.float32).tolist(),
        "observation.state.joint_position": np.asarray(joint_position, dtype=np.float32).tolist(),
        "latent_videos": latent_videos_meta,
        "raw_videos": raw_videos_meta,
        "source": "droid_hardware_active_uq",
    }
    if extra_annotation:
        annotation.update(extra_annotation)
    with open(ann_dir / f"{episode_id}.json", "w") as f:
        json.dump(annotation, f)


def next_episode_id(output_root: Path, suite: str) -> int:
    """Smallest unused integer episode id under <output_root>/<suite>/annotation/*/."""
    suite_root = Path(output_root) / suite
    used = set()
    for split_dir in (suite_root / "annotation").glob("*"):
        for p in split_dir.glob("*.json"):
            if p.stem.isdigit():
                used.add(int(p.stem))
    n = 0
    while n in used:
        n += 1
    return n
