"""LIBERO world-model dataset loader.

Mirrors ``Fast-Control-World/dataset/dataset_droid_exp33.py`` for LIBERO.

Key differences from DROID:
* Two cameras (agentview + wrist) stacked vertically into latent shape
  ``(4, num_cams * 24, 40)`` -- that's ``(4, 48, 40)`` by default vs
  DROID's ``(4, 72, 40)``.
* Action conditioning is the absolute end-effector pose (xyz + axis-angle)
  plus the absolute gripper command, normalized to [-1, 1] using percentile
  stats from ``dataset_meta_info/<suite>/stat.json`` (or, as a fallback,
  ``dataset_meta_info/libero/stat.json``).
* On-disk latents and states both live at the WM prediction rate (5 Hz),
  pre-strided by the preprocessor / collector. ``down_sample`` is therefore
  1 by default (indexing latents and states 1:1).

The on-disk format is what ``scripts/preprocess_libero_for_wm.py`` writes:

    <dataset_root_path>/<suite>/annotation/<split>/<episode_id>.json
    <dataset_root_path>/<suite>/latent_videos/<cam>/<episode_id>.pt
    <dataset_root_path>/<suite>/{train,val}_sample.json
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_stat(meta_root: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (state_p01, state_p99) for normalization. Accepts either a
    per-suite file (``<meta_root>/<suite>/stat.json``) or a single shared
    file at ``<meta_root>/stat.json`` (or the legacy pooled
    ``<meta_root>/libero/stat.json``)."""
    candidates = [
        os.path.join(meta_root, dataset_name, "stat.json"),
        os.path.join(meta_root, "stat.json"),
        os.path.join(meta_root, "libero", "stat.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                stat = json.load(f)
            p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]
            p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
            return p01, p99
    raise FileNotFoundError(f"No stat.json found under {candidates}")


class LiberoLatentDataset(Dataset):
    """Loads pre-encoded LIBERO VAE latents + per-frame EEF/gripper actions.

    The interface (``__getitem__`` returns ``{'latent','action','text'}``)
    is identical to ``Fast-Control-World/dataset/dataset_droid_exp33.py``
    so the trainer can be a near-verbatim port.
    """

    def __init__(self, args, mode: str = "train"):
        super().__init__()
        self.args = args
        self.mode = mode

        self.dataset_path_all: list[list[str]] = []
        self.samples_all: list[list[dict[str, Any]]] = []
        self.samples_len: list[int] = []
        self.norm_all: list[tuple[np.ndarray, np.ndarray]] = []

        dataset_root = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        meta_root = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = list(args.prob)
        if len(self.prob) != len(dataset_names):
            raise ValueError(
                f"len(prob)={len(self.prob)} != len(dataset_names)={len(dataset_names)}"
            )

        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            sample_path = os.path.join(dataset_root, dataset_cfg, f"{mode}_sample.json")
            with open(sample_path) as f:
                samples = json.load(f)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
            self.dataset_path_all.append(
                [os.path.join(dataset_root, dataset_name) for _ in samples]
            )
            self.norm_all.append(_load_stat(meta_root, dataset_name))
            print(f"[LiberoLatentDataset {mode}] {dataset_name}: {len(samples)} samples")

        if not self.samples_all:
            raise RuntimeError("No LIBERO datasets found.")
        self.max_id = max(self.samples_len)

        # Index-aligned with samples_all/dataset_path_all/norm_all -- lets
        # __getitem__ recover which sub-dataset a sample came from (needed to
        # key into the bootstrap-future cache, which is laid out per
        # dataset_name; see p_bootstrap_future's docstring in config.py).
        self.dataset_names_all: list[str] = dataset_names
        bootstrap_cache_root = getattr(args, "bootstrap_cache_root", None)
        self.bootstrap_cache_root = Path(bootstrap_cache_root) if bootstrap_cache_root else None

    def __len__(self) -> int:
        return self.max_id

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_latent_video(video_path: str, frame_ids: list[int]) -> torch.Tensor:
        with open(video_path, "rb") as f:
            video_tensor = torch.load(f)
        video_tensor.requires_grad = False
        max_frames = video_tensor.shape[0]
        clamped = [min(int(i), max_frames - 1) for i in frame_ids]
        return video_tensor[clamped]

    @staticmethod
    def normalize_bound(
        data: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        return np.clip(2 * (data - lo) / (hi - lo + eps) - 1, -1, 1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _build_frame_ids(self, frame_now: int, frame_len: int,
                         apply_future_in_history: bool = False) -> tuple[list[int], np.ndarray, int]:
        """Same temporal layout as the DROID loader:
        ``num_history`` frames going back, then ``num_frames`` future frames.
        Random skip with occasional zeroing for history augmentation.

        ``apply_future_in_history`` is pre-computed by the caller as part of
        a joint mode draw (mutually exclusive with single_history).

        Also returns the drawn ``skip`` (the future-frame stride) -- the
        caller needs it to key the p_bootstrap_future cache lookup (see
        __getitem__'s use_bootstrap_future branch) with the SAME skip this
        sample's own true future uses, so the cached peek's implied
        timestamps (frame_now + i*skip) match what this sample's true future
        actually represents at its own cadence, instead of an arbitrary
        fixed stride (see generate_bootstrap_futures_droid.py, which now
        generates BOTH skip=1 and skip=2 cache variants per anchor for
        exactly this reason)."""
        skip = random.randint(1, 2)
        skip_his = int(skip * 4)
        if random.random() < 0.15:
            skip_his = 0

        if apply_future_in_history:
            future_shift = random.randint(1, self.args.num_frames - 1) * skip
            frame_now = frame_now + future_shift
            # Existing np.clip(..., 0, frame_len) below handles out-of-bounds.

        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.clip(np.asarray(rgb_id), 0, frame_len).tolist()
        rgb_id = [int(x) for x in rgb_id]
        state_id = np.asarray(rgb_id) * self.args.down_sample
        return rgb_id, state_id, skip

    def _sample_other_episode_future(
        self, samples: list[dict[str, Any]], dataset_path: list[str],
        sample: dict[str, Any], index: int,
        total_h: int, per_cam_h: int, latent_w: int,
    ) -> torch.Tensor:
        """Load a ``num_frames-1``-length future window from a randomly-chosen
        DIFFERENT episode in the same sub-dataset -- the p_false_future
        distractor, also used as the fallback when a shifted-future window
        doesn't fit or a bootstrap-future cache lookup misses. Factored out
        of __getitem__ so all three call sites share one implementation."""
        j = index
        for _ in range(5):  # avoid accidentally picking the same episode
            j = random.randrange(len(samples))
            distractor = samples[j]
            if distractor["episode_id"] != sample["episode_id"]:
                break
        distractor = samples[j]
        distractor_dir = dataset_path[j]
        distractor_ann = os.path.join(
            distractor_dir, self.args.annotation_name, self.mode, f"{distractor['episode_id']}.json"
        )
        with open(distractor_ann) as f:
            distractor_label = json.load(f)
        distractor_frame_now = int(distractor["frame_ids"][0])
        distractor_rgb_id = [distractor_frame_now + i for i in range(1, self.args.num_frames)]
        distractor_cam_specs = distractor_label.get("latent_videos", [])
        false_future_latent = torch.zeros(
            (self.args.num_frames - 1, 4, total_h, latent_w), dtype=torch.float32
        )
        for cam_idx in range(self.args.num_cams):
            video_path = os.path.join(distractor_dir, distractor_cam_specs[cam_idx]["latent_video_path"])
            cam_latent = self._load_latent_video(video_path, distractor_rgb_id)
            false_future_latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = cam_latent
        return false_future_latent

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Pick a sub-dataset weighted by prob.
        dataset_id = int(np.random.choice(len(self.samples_all), p=self.prob))
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        frame_ids = sample["frame_ids"]
        ann_file = os.path.join(
            dataset_dir, self.args.annotation_name, self.mode, f"{sample['episode_id']}.json"
        )
        with open(ann_file) as f:
            label = json.load(f)

        # Frame indices live in WM-rate units. Both latents and state arrays
        # on disk are pre-strided to the same WM rate by the
        # preprocessor / collector, so with down_sample == 1 this is the
        # identity. down_sample > 1 is retained only for legacy data where
        # states were stored at a higher native rate than latents.
        joint_len = len(label["observation.state.cartesian_position"]) - 1
        frame_len = int(np.floor(joint_len / self.args.down_sample))
        frame_now = int(frame_ids[0])

        # Joint mode draw: future_in_history and single_history are mutually exclusive.
        p_fih = getattr(self.args, 'p_future_in_history', 0.0)
        p_sh = getattr(self.args, 'p_single_history', 0.0)
        r = random.random()
        _do_fih = r < p_fih
        _do_sh = (not _do_fih) and r < p_fih + p_sh

        rgb_id, state_id, sample_skip = self._build_frame_ids(frame_now, frame_len,
                                                               apply_future_in_history=_do_fih)

        # Stack two camera latents vertically along H.
        per_cam_h = self.args.height // 8  # 24 at height=192
        total_h = self.args.num_cams * per_cam_h
        latent_w = self.args.width // 8
        latent = torch.zeros(
            (self.args.num_frames + self.args.num_history, 4, total_h, latent_w),
            dtype=torch.float32,
        )
        cam_specs = label.get("latent_videos", [])
        if len(cam_specs) < self.args.num_cams:
            raise ValueError(
                f"{ann_file} declares {len(cam_specs)} cameras but config asks for {self.args.num_cams}"
            )
        for cam_idx in range(self.args.num_cams):
            video_path = os.path.join(dataset_dir, cam_specs[cam_idx]["latent_video_path"])
            cam_latent = self._load_latent_video(video_path, rgb_id)
            latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = cam_latent

        # False/shifted/bootstrap-future augmentation: conditional on the
        # history-future-overlap branch firing (config.py's
        # p_history_future_overlap), replace the peeked future frames with
        # one of: (a) a temporally-SHIFTED window from the SAME episode
        # (p_shifted_future), (b) a mismatched future from a DIFFERENT
        # episode entirely (p_false_future), or (c) a SELF-SAMPLED synthetic
        # future generated by the model's own sampler, conditioned on THIS
        # episode's real history+actions (p_bootstrap_future -- see
        # scripts/generate_bootstrap_futures_droid.py and config.py's
        # docstring). Spliced into the history-future-overlap context in
        # CrtlWorld.forward() in place of the true peeked future -- the
        # diffusion target stays the real continuation of THIS episode, so
        # the model must learn to judge whether the peeked content is
        # actually plausible instead of just trusting "peek == answer". See
        # config.py's p_false_future/p_shifted_future/p_bootstrap_future
        # docstrings.
        p_shift = getattr(self.args, 'p_shifted_future', 0.0)
        p_ff = getattr(self.args, 'p_false_future', 0.0)
        p_bs = getattr(self.args, 'p_bootstrap_future', 0.0)
        r = random.random()
        use_shifted_future = p_shift > 0.0 and r < p_shift
        use_other_future = (not use_shifted_future) and p_ff > 0.0 and r < p_shift + p_ff
        use_bootstrap_future = (
            not use_shifted_future and not use_other_future
            and p_bs > 0.0 and r < p_shift + p_ff + p_bs
        )
        false_future_latent = torch.zeros(
            (self.args.num_frames - 1, 4, total_h, latent_w), dtype=torch.float32
        )

        if use_shifted_future:
            # Window of num_frames-1 frames starting at least min_gap (and at
            # most max_gap) frames beyond the true peeked window
            # (frame_now+1..frame_now+num_frames-1) -- same episode, wrong
            # point in time.
            min_gap = int(getattr(self.args, 'shifted_future_min_gap', 5))
            max_gap = int(getattr(self.args, 'shifted_future_max_gap', 30))
            gap = random.randint(min_gap, max(min_gap, max_gap))
            shift_start = frame_now + (self.args.num_frames - 1) + gap
            shifted_rgb_id = [shift_start + i for i in range(self.args.num_frames - 1)]
            if shifted_rgb_id[-1] > frame_len:
                # Episode too short for a valid shifted window -- fall back
                # to the other-episode distractor path so this sample still
                # gets a false peek (preserving the configured total false
                # rate) instead of silently reverting to the true future.
                use_shifted_future = False
                use_other_future = True
            else:
                for cam_idx in range(self.args.num_cams):
                    video_path = os.path.join(dataset_dir, cam_specs[cam_idx]["latent_video_path"])
                    cam_latent = self._load_latent_video(video_path, shifted_rgb_id)
                    false_future_latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = cam_latent

        if use_bootstrap_future:
            dataset_name = self.dataset_names_all[dataset_id]
            # Keyed by THIS sample's own drawn `sample_skip` (not a fixed
            # stride) so the cached peek's implied timestamps
            # (frame_now + i*sample_skip) exactly match what this sample's
            # true future represents at its own cadence --
            # generate_bootstrap_futures_droid.py generates BOTH skip=1 and
            # skip=2 cache variants per covered anchor for exactly this
            # reason (see its module docstring).
            cache_path = (
                self.bootstrap_cache_root / dataset_name / str(sample["episode_id"])
                / f"{frame_now}_skip{sample_skip}.pt"
            )
            if cache_path.exists():
                false_future_latent = torch.load(cache_path, map_location="cpu").float()
            else:
                # Partial-coverage cache (DROID's train split is far too
                # large for full coverage, unlike push_cube's bootstrap
                # cache -- see generate_bootstrap_futures_droid.py) -- fall
                # back to the other-episode distractor so this sample still
                # gets a false peek at the configured total rate, instead of
                # silently reverting to the true future. Mirrors the
                # shifted-window-too-short fallback above. This dilutes the
                # *realized* p_bootstrap_future rate below its configured
                # value by roughly the cache miss rate -- worth measuring
                # (e.g. a periodic hit/miss log) rather than assuming.
                use_bootstrap_future = False
                use_other_future = True

        if use_other_future:
            false_future_latent = self._sample_other_episode_future(
                samples, dataset_path, sample, index, total_h, per_cam_h, latent_w
            )

        use_false_future = use_shifted_future or use_other_future or use_bootstrap_future

        # Action conditioning: cartesian + gripper.
        cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.cartesian_position"]) - 1)
        ]
        grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.gripper_position"]) - 1)
        ]
        if grip.ndim == 1:
            grip = grip[:, None]
        action = np.concatenate([cart, grip], axis=-1)
        action = self.normalize_bound(action, state_p01, state_p99)

        if _do_sh:
            num_history = self.args.num_history
            # Repeat most-recent history frame for all older history slots
            latent[:num_history - 1] = latent[num_history - 1 : num_history]
            action[:num_history - 1] = action[num_history - 1 : num_history]

        return {
            "text": label["texts"][0] if label.get("texts") else label.get("language_instruction", ""),
            "latent": latent.float(),
            "action": torch.tensor(action, dtype=torch.float32),
            "false_future_latent": false_future_latent,
            "use_false_future": torch.tensor(use_false_future),
        }


__all__ = ["LiberoLatentDataset"]
