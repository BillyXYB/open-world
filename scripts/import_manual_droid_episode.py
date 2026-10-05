"""
Import one or more hand-collected DROID episode bundles (produced locally by
droid/scripts/convert/export_episode_for_wm_uq.py) into the `droid_ctrl_world`-style
on-disk schema (annotation JSON + latent .pt + raw mp4) that `scripts/replay_libero_wm_traj.py`
and `openworld.training.world_model.dataset.LiberoLatentDataset` consume.

Meant to run on the GPU machine that holds the SVD VAE weights and the training data root
(the same environment `jobs/replay_droid_wm_traj_epi_false_future.sh` runs in).

Each bundle is a directory `<bundle_dir>/<episode_id>/`. Two layouts are accepted:

    format 2 (current)            format 1 (legacy)
      31177322.mp4  left            bundle.npz  -- cam_rgb_left/right/wrist (T,H,W,3) uint8,
      38872458.mp4  right                          cart (T,6), grip (T,), joint_position (T,7)
      10501775.mp4  wrist           meta.json
      state.npz     cart/grip/joint_position
      meta.json     bundle_format=2, num_steps, cameras[], ...

Format 2 ships the MP4s the format-1 arrays were decoded from, which is bit-identical at
~1/50th the size (~36 MB vs 740 MB - 2 GB per episode). `meta["bundle_format"]` selects the
branch; bundles with no such key are format 1.

Meant to run on the GPU machine that holds the SVD VAE weights and the training data root
(the same environment `jobs/replay_droid_wm_traj_epi_false_future.sh` runs in).

Usage:
    uv run scripts/import_manual_droid_episode.py \
        --bundle_dir /scratch/gpfs/AM43/yx2653/data/wm_uq_export \
        --data_root /scratch/gpfs/AM43/yx2653/data \
        --stat_reference_root /scratch/gpfs/AM43/yy4041/data \
        --suite droid_manual_uq_check \
        --split val \
        --svd_model_path external/stable-video-diffusion-img2vid
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_libero_for_wm import LatentEncoder  # noqa: E402

from openworld.utils.droid_export import next_episode_id, write_droid_episode  # noqa: E402


def _decode_mp4(path: Path) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB, decoded by frame iteration.

    Tiered because the three decoders have different availability here: `imageio[ffmpeg]` and
    `decord` are core deps of this repo, while `opencv-python` only arrives with the
    `policy-openpi` extra. preprocess_libero_for_wm._write_mp4 already exercises the imageio
    FFMPEG writer in this env, so its reader is the safest default.

    NEVER seek by timestamp: these MP4s carry a 60 fps container header (the ZED camera
    config) while the true content rate is 15 Hz, so any time-based access is 4x wrong.
    Frame iteration is immune to the bad header.
    """
    errors = []
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path), format="FFMPEG")
        try:
            frames = [np.asarray(f) for f in reader]
        finally:
            reader.close()
        if frames:
            return np.stack(frames, axis=0)
        errors.append("imageio: decoded 0 frames")
    except Exception as e:  # noqa: BLE001 - fall through to the next decoder
        errors.append(f"imageio: {e}")

    try:
        import decord

        vr = decord.VideoReader(str(path))
        return vr.get_batch(range(len(vr))).asnumpy()
    except Exception as e:  # noqa: BLE001
        errors.append(f"decord: {e}")

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(frame_bgr[..., ::-1])  # BGR -> RGB
        cap.release()
        if frames:
            return np.stack(frames, axis=0)
        errors.append("cv2: decoded 0 frames")
    except Exception as e:  # noqa: BLE001
        errors.append(f"cv2: {e}")

    raise RuntimeError(f"Could not decode {path}; tried three decoders:\n  " + "\n  ".join(errors))


def _align_to_length(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or pad-by-repeating-last-frame to exactly `target_len` timesteps (handles
    off-by-one frame counts between the H5 state log and the MP4 encode).

    Verbatim copy of what used to live in
    droid/scripts/convert/export_episode_for_wm_uq.py -- alignment moved to wherever the
    frames get decoded, which is here now. Not a no-op in practice: a 148-step episode was
    observed with a 143-frame wrist video.
    """
    t = frames.shape[0]
    if t == target_len:
        return frames
    if t > target_len:
        return frames[:target_len]
    pad = np.repeat(frames[-1:], target_len - t, axis=0)
    return np.concatenate([frames, pad], axis=0)


class _LazyFrames:
    """One camera's frames, decoded on first element access.

    ``.shape`` is answered from meta without touching the file, which matters because
    ``write_droid_episode`` reads ``cam_rgb[0].shape[0]`` *after* its latent loop has moved
    on to camera 2 -- a plain memoized array would re-decode camera 0 for a single integer.
    ``LatentEncoder.encode`` likewise reads ``.shape[0]`` before iterating, so the decode is
    triggered by the first real frame access and happens exactly once.
    """

    def __init__(self, path: Path, num_steps: int, declared_frames: int):
        self._path = path
        self._num_steps = num_steps
        self._declared = declared_frames
        self._frames = None
        self._hw = None

    @property
    def shape(self) -> tuple:
        """(T, H, W, 3) with T from meta. H/W come from a single-frame probe when the full
        array isn't already decoded, so asking for the shape never costs 5.5 GB."""
        if self._frames is not None:
            return self._frames.shape
        if self._hw is None:
            self._hw = self._probe_hw()
        return (self._num_steps, *self._hw)

    def _probe_hw(self) -> tuple:
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(self._path), format="FFMPEG")
            try:
                return tuple(np.asarray(reader.get_next_data()).shape)
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - fall back to a full decode rather than guessing
            return tuple(self._materialize().shape[1:])

    def _materialize(self) -> np.ndarray:
        if self._frames is None:
            frames = _decode_mp4(self._path)
            if abs(frames.shape[0] - self._num_steps) > 2:
                print(f"[!] {self._path.name}: decoded {frames.shape[0]} frames, exporter "
                      f"recorded {self._declared}, state has {self._num_steps} steps "
                      f"-- trimming/padding to {self._num_steps}.")
            self._frames = _align_to_length(frames, self._num_steps)
        return self._frames

    def release(self) -> None:
        self._frames = None

    def __len__(self) -> int:
        return self._num_steps

    def __getitem__(self, idx):
        return self._materialize()[idx]

    def __array__(self, dtype=None):
        arr = self._materialize()
        return arr.astype(dtype) if dtype is not None else arr


class LazyCams:
    """Sequence of per-camera frames that holds at most one decoded camera at a time.

    A 2003-step episode is 5.5 GB per camera at 720p; materializing all three plus the
    transient np.stack copy is ~22 GB against the job's --mem=64G. Handing out _LazyFrames
    and releasing the previous one keeps peak RSS at roughly one camera.
    """

    def __init__(self, bundle_path: Path, meta: dict):
        num_steps = int(meta["num_steps"])
        self._views = [
            _LazyFrames(bundle_path / cam["file"], num_steps, cam.get("mp4_num_frames") or 0)
            for cam in meta["cameras"]
        ]
        self._last = None

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> _LazyFrames:
        view = self._views[idx]
        if self._last is not None and self._last is not view:
            self._last.release()
        self._last = view
        return view


def find_bundle_dirs(bundle_dir: Path) -> list:
    """A bundle_dir is either a single episode dir (holds a bundle marker directly) or a
    parent dir containing one or more episode subdirs."""
    def is_bundle(d: Path) -> bool:
        return (d / "bundle.npz").exists() or (d / "state.npz").exists()

    if is_bundle(bundle_dir):
        return [bundle_dir]
    return sorted(p for p in bundle_dir.iterdir() if p.is_dir() and is_bundle(p))


def ensure_stat_json(data_root: Path, suite: str, reference_suite: str = "droid_ctrl_world",
                     reference_root: Path = None) -> None:
    """Bootstrap <data_root>/dataset_meta_info/<suite>/stat.json from a reference suite.

    ``reference_root`` may differ from ``data_root``: we write into yx2653's scratch but
    droid_ctrl_world's stats live in yy4041's, which is read-only to us.
    """
    reference_root = Path(reference_root) if reference_root is not None else Path(data_root)
    stat_path = Path(data_root) / "dataset_meta_info" / suite / "stat.json"
    if stat_path.exists():
        return
    ref_path = reference_root / "dataset_meta_info" / reference_suite / "stat.json"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"No stat.json for suite '{suite}' at {stat_path}, and no reference to copy from "
            f"at {ref_path}.\n"
            f"  write root     : {data_root}\n"
            f"  reference root : {reference_root}   (set with --stat_reference_root)\n"
            f"Either place a stat.json at {stat_path} yourself, or point --stat_reference_root "
            f"at a tree that has '{reference_suite}'."
        )
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    # copyfile + explicit chmod, NOT copy2: copy2 preserves mode bits, so a 0444 file in a
    # read-only reference tree would land here unwritable and could never be replaced.
    shutil.copyfile(ref_path, stat_path)
    os.chmod(stat_path, 0o644)
    print(f"[*] Bootstrapped {stat_path} from {ref_path} (reusing '{reference_suite}' normalization stats).")


def find_imported_source_bundles(data_root: Path, suite: str, split: str) -> tuple:
    """(by_relpath, by_bundle_path) -> episode_id, for every already-imported episode in
    <data_root>/<suite>/annotation/<split>/*.json, so re-running this script against the
    same bundle_dir skips what it already imported instead of duplicating it under a new
    episode id (next_episode_id() has no content-awareness of its own).

    source_relpath ("success/2026-10-01/Thu_...") is the primary key because it is invariant
    under moving the bundle dir or the scratch root. source_bundle, an absolute path, is kept
    only as a fallback for annotations written before source_relpath existed -- keying on it
    alone means a changed BUNDLE_DIR makes every episode look new, silently double-counting
    every chunk in the per-quadrant aggregation.
    """
    ann_dir = data_root / suite / "annotation" / split
    by_relpath: dict = {}
    by_bundle: dict = {}
    for p in sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []:
        try:
            ann = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if ann.get("source_relpath"):
            by_relpath[ann["source_relpath"]] = p.stem
        if ann.get("source_bundle"):
            by_bundle[ann["source_bundle"]] = p.stem
    return by_relpath, by_bundle


def load_bundle(bundle_path: Path, meta: dict) -> tuple:
    """-> (cam_rgb, cart, grip, joint_position). cam_rgb is a list for format 1, a LazyCams
    for format 2; both index the same way write_droid_episode expects."""
    fmt = int(meta.get("bundle_format", 1))
    if fmt == 1:
        data = np.load(bundle_path / "bundle.npz")
        cam_rgb = [data["cam_rgb_left"], data["cam_rgb_right"], data["cam_rgb_wrist"]]
        return cam_rgb, data["cart"], data["grip"], data["joint_position"]
    if fmt == 2:
        state = np.load(bundle_path / meta.get("state_file", "state.npz"))
        return (LazyCams(bundle_path, meta), state["cart"], state["grip"],
                state["joint_position"])
    raise ValueError(
        f"{bundle_path}: unsupported bundle_format={fmt}. This importer understands 1 and 2; "
        f"upgrade scripts/import_manual_droid_episode.py to match the exporter that wrote it."
    )


def import_bundle(bundle_path: Path, *, data_root: Path, suite: str, split: str, encoder,
                  write_raw: bool = False) -> None:
    meta = json.loads((bundle_path / "meta.json").read_text())
    cam_rgb, cart, grip, joint_position = load_bundle(bundle_path, meta)

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
        # Pinned by the exporter to the 15 Hz control rate, NOT the MP4 header's bogus 60.
        # replay_libero_wm_traj.py turns this into stride = round(fps / target_hz) to reach
        # the world model's 5 Hz, so a wrong value here silently changes the eval rate.
        fps=meta["fps"],
        down_sample=meta["down_sample"],
        write_raw=write_raw,
        extra_annotation={
            "source": "hand_collected_manual_uq_check",
            "source_bundle": str(bundle_path),
            # The stable dedup key -- see find_imported_source_bundles().
            "source_relpath": meta.get("source_relpath"),
            "bundle_format": int(meta.get("bundle_format", 1)),
            "original_episode_id": meta.get("episode_id"),
            "num_steps": meta.get("num_steps"),
            "exported_at": meta.get("exported_at"),
            # Human-labeled UQ ground truth carried through from the GELLO collection
            # prompt (see droid/scripts/convert/export_episode_for_wm_uq.py), when present.
            # "uncertainty_cell" reuses the generic field name replay_libero_wm_traj.py /
            # aggregate_uq_by_cell.py / plot_uq_by_cell.py already group metrics by.
            "uncertainty_cell": meta.get("quadrant"),
            "quadrant": meta.get("quadrant"),
            "aleatoric_level": meta.get("aleatoric_level"),
            "epistemic_level": meta.get("epistemic_level"),
            # aggregate_uq_by_cell.py groups on uncertainty_cell alone, so successes and
            # failures in the same quadrant land in one mean. These let you post-filter
            # chunk_metrics.jsonl by outcome afterwards.
            "outcome": meta.get("outcome"),
            "success": meta.get("success"),
            "failure": meta.get("failure"),
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
    parser.add_argument("--stat_reference_root", type=Path, default=None,
                        help="Root to copy the reference suite's stat.json FROM when this suite "
                             "has none yet. Defaults to --data_root; point it at yy4041's tree "
                             "when writing into a scratch dir that has no droid_ctrl_world.")
    parser.add_argument("--stat_reference_suite", default="droid_ctrl_world")
    parser.add_argument("--write_raw", action="store_true",
                        help="Also write raw_videos/*.mp4. Off by default: replay_libero_wm_traj.py "
                             "reads only latent_video_path, so this is a full-resolution decode and "
                             "re-encode nothing downstream consumes.")
    parser.add_argument("--wipe_suite", action="store_true",
                        help="Move an existing <data_root>/<suite> aside to <suite>.bak.<timestamp> "
                             "before importing (never deletes). Use when re-exporting a bundle for "
                             "the same source_relpath, which dedup would otherwise skip.")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    if args.wipe_suite:
        suite_dir = args.data_root / args.suite
        if suite_dir.exists():
            backup = suite_dir.with_name(f"{args.suite}.bak.{time.strftime('%Y%m%d_%H%M%S')}")
            shutil.move(str(suite_dir), str(backup))
            print(f"[*] Moved existing suite aside: {suite_dir} -> {backup}")

    ensure_stat_json(args.data_root, args.suite,
                     reference_suite=args.stat_reference_suite,
                     reference_root=args.stat_reference_root)

    encoder = None
    if args.svd_model_path is not None:
        encoder = LatentEncoder(svd_path=args.svd_model_path, target_h=args.height, target_w=args.width)

    bundle_dirs = find_bundle_dirs(args.bundle_dir)
    if not bundle_dirs:
        raise FileNotFoundError(
            f"No bundle found under {args.bundle_dir} (looked for state.npz or bundle.npz)"
        )

    by_relpath, by_bundle = find_imported_source_bundles(args.data_root, args.suite, args.split)
    n_imported = n_skipped = 0
    for bundle_path in bundle_dirs:
        meta_path = bundle_path / "meta.json"
        relpath = None
        if meta_path.exists():
            try:
                relpath = json.loads(meta_path.read_text()).get("source_relpath")
            except (json.JSONDecodeError, OSError):
                relpath = None

        existing_id = (by_relpath.get(relpath) if relpath else None) or by_bundle.get(str(bundle_path))
        if existing_id is not None:
            print(f"[*] Skipping '{bundle_path.name}' -- already imported as "
                  f"{args.suite}/{args.split}/{existing_id}")
            n_skipped += 1
            continue

        import_bundle(bundle_path, data_root=args.data_root, suite=args.suite,
                      split=args.split, encoder=encoder, write_raw=args.write_raw)
        n_imported += 1
        if relpath:
            by_relpath[relpath] = "just-imported"

    print(f"[*] {n_imported} imported, {n_skipped} already present "
          f"-> {args.data_root}/{args.suite}/annotation/{args.split}")


if __name__ == "__main__":
    main()
