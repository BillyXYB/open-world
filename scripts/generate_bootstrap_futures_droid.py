"""Generate bootstrap (self-sampled) future latents for DROID's
``p_bootstrap_future`` distractor (see config.py's ``p_bootstrap_future``/
``bootstrap_cache_root`` docstrings and dataset.py's ``use_bootstrap_future``
branch).

Ports ``wm_uq``'s bootstrap idea
(``wm_uq/wm_uq/scripts/generate_bootstrap_futures.py``, run against a frozen
"Phase A" checkpoint) to DROID -- reusing
``scripts/replay_libero_wm_traj.py``'s existing model-loading/frame-id
helpers (``load_crtl_world``, ``build_frame_ids``) rather than re-deriving
them, since this repo already has all the standalone-inference plumbing a
push_cube-style generation script would otherwise need to write from
scratch. The frozen sampler here is the existing
``droid_flow_matching_uq_false_future_v3`` checkpoint itself -- no separate
Phase-A training run is needed (see
configs/training/droid_wm_uq_bootstrap_v1.py's module docstring).

For each covered ``(episode_id, frame_now)`` anchor, and for EACH possible
``skip`` a training sample might later draw for its own true future
(``dataset.py``: ``skip = random.randint(1, 2)``):
  1. Build the SAME history/current/action conditioning a true (non-overlap)
     training sample would use at that anchor AND that skip -- matching
     ``dataset.py``'s ``_build_frame_ids`` exactly (history stride
     ``skip_his = skip*4``, or 0 w.p. 0.15) via ``build_frame_ids``.
  2. Run the frozen checkpoint's flow-matching sampler with NO peek
     (``overlap_k=0, overlap_active=False``) -- a single, plain forward
     sample conditioned on real history+actions, starting from pure noise.
     This is what makes the result a harder distractor than
     p_false_future/p_shifted_future: it looks like a plausible continuation
     of THIS trajectory (real history, real actions), but is only a
     stochastic sample -- not guaranteed to match the true outcome.
  3. Cache ``pred_latents[1:]`` (the ``num_frames - 1`` future frames,
     matching ``false_future_latent``'s shape in dataset.py) to
     ``<output_root>/<dataset_name>/<episode_id>/<frame_now>_skip<skip>.pt``
     (half precision, matching push_cube's cache).

Why BOTH skip variants, not just one: at train time, ``dataset.py``'s
``use_bootstrap_future`` branch loads whichever skip=1/skip=2 cache file
matches the SPECIFIC training sample's own randomly-drawn ``skip`` (see its
docstring) -- because the peek's implied timestamps (``frame_now +
i*skip``) must match what that sample's true future actually represents at
its own cadence. Caching only one skip variant would mean a random ~50% of
samples that draw the OTHER skip silently miss the cache and fall back to
the (easier) other-episode distractor -- generating both variants per
anchor guarantees a timestep-exact match whenever that anchor was covered
at all, at the cost of ~2x the generation compute per anchor. ``skip_his``
(history stride) is NOT part of the cache key: it only affects how good/
representative the CONDITIONING was during generation, not what timestamps
the OUTPUT represents (the real GT history is always loaded fresh at train
time regardless of what the generation script used) -- so it's drawn
per-variant matching the training distribution, but doesn't need its own
key.

DROID's ``train_sample.json`` has ~8.7M rows across ~94k episodes -- far too
large for full coverage (unlike push_cube's bootstrap cache, which covers
every group). Use ``--every_nth_episode``/``--anchors_per_episode``/
``--shard_id``/``--num_shards`` (a plain SLURM array job, see
``jobs/generate_bootstrap_futures_droid_array.sh`` -- no submitit dependency
in this repo, unlike wm_uq/) to generate a bounded subset; dataset.py
gracefully falls back to the other-episode distractor for any (anchor, skip)
combination not found in the cache. The cache is resumable (skip-if-exists,
checked per skip variant independently), so scaling up coverage later is
additive, not wasted work.

Usage (smoke test, no SLURM):
    uv run scripts/generate_bootstrap_futures_droid.py \\
        --checkpoint checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3/checkpoint-220000.pt \\
        --data_root /scratch/gpfs/AM43/yy4041/data \\
        --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \\
        --output_root outputs/bootstrap_futures_cache_droid \\
        --split val --max_anchors 200 --n_steps 25

Sharded (SLURM array) usage: add ``--shard_id $SLURM_ARRAY_TASK_ID
--num_shards $SLURM_ARRAY_TASK_COUNT`` and drop ``--max_anchors`` (or keep it
as a per-shard cap).

Breadth-first coverage: ``--every_nth_episode 1 --anchors_per_episode K``
covers ALL selected episodes with up to K base anchors each (evenly spaced
across each episode's timeline), instead of full-density coverage of a
random episode subset -- see ``--anchors_per_episode``'s help. Each base
anchor still expands into up to 2 (skip=1, skip=2) generation calls per the
above.

Batch-size note (found via --sweep_batch_sizes on an H200, torch 2.10.0+cu128):
letting PyTorch auto-select the SDPA backend crashes with "CUDA error:
invalid configuration argument" at batch_size=16+ -- the dispatcher picks
FLASH_ATTENTION or CUDNN_ATTENTION for some attention call in the UNet
(cuDNN SDPA is documented to not support a key/value sequence length of 1,
which the action cross-attention hits) and that combination hits an invalid
kernel-launch config instead of falling back gracefully, regardless of batch
size in principle -- it happened not to trigger at batch_size<=8 in testing,
but nothing guarantees that holds for other shapes/hardware. Forcing MATH
OOMs (naive O(N^2) memory). Forcing EFFICIENT_ATTENTION explicitly works
cleanly at every batch size tested and is what `_run_batch` below always
uses -- never rely on the auto-dispatch default for this model. 32 was the
most efficient batch size measured (throughput plateaus around 16-32;
compute-bound by the sequential n_steps Euler loop, not batch-parallelism;
128 OOMs at ~140GB).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from replay_libero_wm_traj import build_frame_ids, load_crtl_world  # noqa: E402

from openworld.training.world_model.config import LiberoWMArgs  # noqa: E402
from openworld.training.world_model.dataset import _load_stat, LiberoLatentDataset  # noqa: E402

# The only two `skip` values dataset.py's _build_frame_ids ever draws
# (random.randint(1, 2)) for a sample's own true future -- see module
# docstring's "Why BOTH skip variants" section.
SKIP_VALUES = (1, 2)
# Matches dataset.py's _build_frame_ids exactly: skip_his = skip*4, then
# overridden to 0 with this probability (simulates "only the immediately
# preceding frame is available").
PROB_SKIP_HIS_ZERO = 0.15


def _group_by_episode(samples: list[dict]) -> dict[str, list[int]]:
    """episode_id -> list of frame_now anchors (sample['frame_ids'][0]),
    deduped. Groups once so each episode's latent videos are loaded a single
    time regardless of how many anchors it contributes -- required at
    DROID's scale (push_cube's script samples one ``ds[idx]`` at a time,
    which would be far too slow here)."""
    by_ep: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    for s in samples:
        ep = str(s["episode_id"])
        frame_now = int(s["frame_ids"][0])
        if frame_now not in seen[ep]:
            seen[ep].add(frame_now)
            by_ep[ep].append(frame_now)
    return by_ep


def _select_episodes(episode_ids: list[str], every_nth: int, shard_id: int, num_shards: int) -> list[str]:
    """Deterministic coverage + shard selection. Coverage first (every_nth
    over the full sorted episode list), THEN shard the selected subset --
    so shards stay balanced regardless of coverage fraction."""
    episode_ids = sorted(episode_ids)
    covered = episode_ids[::every_nth] if every_nth > 1 else episode_ids
    return covered[shard_id::num_shards] if num_shards > 1 else covered


def _draw_skip_his(skip: int) -> int:
    """Matches dataset.py's _build_frame_ids exactly (see PROB_SKIP_HIS_ZERO)."""
    skip_his = skip * 4
    if random.random() < PROB_SKIP_HIS_ZERO:
        skip_his = 0
    return skip_his


def _build_batch_conditioning(
    batch: list[tuple[str, int, int, int]], dataset_dir: str, args_ns, a, ann_cache: dict,
    state_p01: np.ndarray, state_p99: np.ndarray, per_cam_h: int, latent_w: int, total_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build (history, current, action, text_list) CPU tensors for one batch of
    (episode_id, frame_now, skip, skip_his) anchors -- shared by the main
    generation loop and the --sweep_batch_sizes benchmark below."""
    history_list, current_list, action_list, text_list = [], [], [], []
    for ep, frame_now, skip, skip_his in batch:
        if ep not in ann_cache:
            ann_file = os.path.join(dataset_dir, args_ns.annotation_name, a.split, f"{ep}.json")
            with open(ann_file) as f:
                ann_cache[ep] = json.load(f)
        label = ann_cache[ep]

        rgb_id = build_frame_ids(frame_now, a.num_history, a.num_frames, skip, skip_his)
        frame_len = int(np.floor((len(label["observation.state.cartesian_position"]) - 1) / a.down_sample))
        rgb_id_clamped = [int(np.clip(r, 0, frame_len)) for r in rgb_id]
        state_id = [r * a.down_sample for r in rgb_id_clamped]

        cam_specs = label["latent_videos"]
        latent = torch.zeros((a.num_history + 1, 4, total_h, latent_w), dtype=torch.float32)
        for cam_idx in range(a.num_cams):
            video_path = os.path.join(dataset_dir, cam_specs[cam_idx]["latent_video_path"])
            cam_latent = LiberoLatentDataset._load_latent_video(video_path, rgb_id_clamped[: a.num_history + 1])
            latent[:, :, cam_idx * per_cam_h: (cam_idx + 1) * per_cam_h] = cam_latent

        cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.cartesian_position"]) - 1)
        ]
        grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.gripper_position"]) - 1)
        ]
        if grip.ndim == 1:
            grip = grip[:, None]
        action = np.concatenate([cart, grip], axis=-1)
        action = LiberoLatentDataset.normalize_bound(action, state_p01, state_p99)

        history_list.append(latent[: a.num_history])
        current_list.append(latent[a.num_history])
        action_list.append(torch.tensor(action, dtype=torch.float32))
        text_list.append(label["texts"][0] if label.get("texts") else label.get("language_instruction", ""))

    history = torch.stack(history_list, 0)   # (B, num_history, 4, total_h, latent_w)
    current = torch.stack(current_list, 0)   # (B, 4, total_h, latent_w)
    action = torch.stack(action_list, 0)     # (B, num_history+1, action_dim)
    return history, current, action, text_list


def _run_batch(
    model, pipeline, pipeline_cls, args_ns, a, device,
    batch: list[tuple[str, int, int, int]],
    history: torch.Tensor, current: torch.Tensor, action: torch.Tensor, text_list: list[str],
) -> torch.Tensor:
    """Run one no-peek forward sample for a batch of anchors already built into
    tensors. Returns pred_latents (B, num_frames, 4, total_h, latent_w), float32, CPU."""
    history = history.to(device)
    current = current.to(device)
    action = action.to(device)
    # EFFICIENT_ATTENTION forced explicitly -- see module docstring's
    # "Batch-size note": the SDPA auto-dispatcher crashes at larger batch
    # sizes by picking FLASH_ATTENTION/CUDNN_ATTENTION for an attention call
    # those backends don't actually support here (key/value seq len 1).
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16), \
            sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        action_latent = model.action_encoder(
            action, text_list, model.tokenizer, model.text_encoder, args_ns.frame_level_cond)
        gen = torch.Generator(device=device).manual_seed(
            abs(hash((batch[0][0], batch[0][1], batch[0][2]))) % (2 ** 31)
        )
        _, pred_latents = pipeline_cls.__call__(
            pipeline, image=current, text=action_latent,
            width=a.width, height=int(a.num_cams * a.height),
            num_frames=a.num_frames, history=history,
            num_inference_steps=a.n_steps, decode_chunk_size=args_ns.decode_chunk_size,
            max_guidance_scale=args_ns.guidance_scale, fps=args_ns.fps,
            motion_bucket_id=args_ns.motion_bucket_id, mask=None,
            output_type="latent", return_dict=False,
            frame_level_cond=args_ns.frame_level_cond, his_cond_zero=False,
            flow_map_type=args_ns.flow_map_type, flow_map_loss_type=args_ns.flow_map_loss_type,
            return_uncertainty=False, generator=gen,
            overlap_k=0, overlap_active=False,  # no peek -- plain sample from real history+actions
        )
    return pred_latents.float().cpu()


def _run_batch_size_sweep(
    model, pipeline, pipeline_cls, args_ns, a, device,
    anchors: list[tuple[str, int, int, int]], dataset_dir: str, ann_cache: dict,
    state_p01: np.ndarray, state_p99: np.ndarray, per_cam_h: int, latent_w: int, total_h: int,
    out_root: Path,
) -> None:
    """Benchmark --sweep_batch_sizes on the SAME first --sweep_anchors anchors for each
    size (apples-to-apples), reports anchors/sec + s/anchor + peak GPU memory per size.
    Outputs are cached to a separate `_sweep/bs<B>/` namespace (not the real cache) so a
    sweep run isn't wasted work, but also can't collide with/pollute the real cache."""
    batch_sizes = [int(b) for b in a.sweep_batch_sizes.split(",") if b.strip()]
    pool = anchors[: a.sweep_anchors]
    if len(pool) < a.sweep_anchors:
        print(f"[sweep] WARNING: only {len(pool)} anchors available "
              f"(requested --sweep_anchors {a.sweep_anchors})")
    print(f"[sweep] {len(pool)} anchors x batch sizes {batch_sizes}")

    results = []
    for bs in batch_sizes:
        n_batches = len(pool) // bs
        if n_batches == 0:
            print(f"[sweep] skipping batch_size={bs}: fewer than {bs} sweep anchors available")
            continue
        used = pool[: n_batches * bs]
        sweep_out_root = out_root.parent / "_sweep" / f"bs{bs}"

        # One warmup batch (untimed): cuDNN/cuBLAS autotuning and any lazy
        # CUDA-kernel compilation for a batch size seen for the first time
        # would otherwise pollute the measured throughput, especially for
        # larger batch sizes.
        torch.cuda.reset_peak_memory_stats(device)
        warm_batch = used[:bs]
        history, current, action, text_list = _build_batch_conditioning(
            warm_batch, dataset_dir, args_ns, a, ann_cache, state_p01, state_p99, per_cam_h, latent_w, total_h)
        _run_batch(model, pipeline, pipeline_cls, args_ns, a, device, warm_batch, history, current, action, text_list)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        n_done_bs = 0
        for batch_start in range(0, len(used), bs):
            batch = used[batch_start: batch_start + bs]
            history, current, action, text_list = _build_batch_conditioning(
                batch, dataset_dir, args_ns, a, ann_cache, state_p01, state_p99, per_cam_h, latent_w, total_h)
            pred_latents = _run_batch(
                model, pipeline, pipeline_cls, args_ns, a, device, batch, history, current, action, text_list)
            for i, (ep, frame_now, skip, _skip_his) in enumerate(batch):
                out_dir = sweep_out_root / ep
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(pred_latents[i, 1:].half(), out_dir / f"{frame_now}_skip{skip}.pt")
            n_done_bs += len(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        rate = n_done_bs / elapsed if elapsed > 0 else 0.0
        results.append((bs, n_done_bs, elapsed, rate, peak_mem_gb))
        print(f"[sweep] batch_size={bs}: {n_done_bs} anchors in {elapsed:.2f}s "
              f"({rate:.3f} anchors/s, {(1.0 / rate if rate > 0 else float('inf')):.3f} s/anchor, "
              f"peak_mem={peak_mem_gb:.1f}GB)")

    print("\n[sweep] ==== SUMMARY ====")
    print(f"{'batch_size':>10} {'anchors/s':>10} {'s/anchor':>10} {'peak_mem_GB':>12}")
    for bs, _n, _elapsed, rate, mem in results:
        s_per_anchor = 1.0 / rate if rate > 0 else float("inf")
        print(f"{bs:>10} {rate:>10.3f} {s_per_anchor:>10.3f} {mem:>12.1f}")
    if results:
        best = max(results, key=lambda r: r[3])
        print(f"\n[sweep] most efficient (highest anchors/s): batch_size={best[0]} "
              f"({best[3]:.3f} anchors/s, {(1.0 / best[3]):.3f} s/anchor, peak_mem={best[4]:.1f}GB)")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Frozen sampler checkpoint, e.g. the v3 false-future checkpoint.")
    p.add_argument("--data_root", default="/scratch/gpfs/AM43/yy4041/data")
    p.add_argument("--stat_root", default="/scratch/gpfs/AM43/yy4041/data/dataset_meta_info")
    p.add_argument("--dataset_name", default="droid_ctrl_world")
    p.add_argument("--output_root", default="outputs/bootstrap_futures_cache_droid")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--every_nth_episode", type=int, default=1,
                   help="Coverage control: generate for every Nth episode (sorted by episode_id). "
                        "1 = full coverage. Combine with --anchors_per_episode to prioritize "
                        "BREADTH (every episode gets at least some coverage) over DEPTH (full "
                        "density in a random episode subset) -- see --anchors_per_episode.")
    p.add_argument("--anchors_per_episode", type=int, default=0,
                   help="Cap BASE anchors (distinct frame_now values) PER EPISODE (0 = no cap -- "
                        "every valid frame_now anchor in each selected episode, the original "
                        "behavior). With --every_nth_episode 1 (the default -- all episodes "
                        "selected) and this set to a small number, EVERY episode gets touched "
                        "instead of a random subset getting full-density coverage while the rest "
                        "get none -- e.g. --anchors_per_episode 2 covers all ~94k DROID train "
                        "episodes with up to 2 base anchors each (each expanding into up to 2 "
                        "skip variants -- see module docstring -- so ~360k generation calls "
                        "total) rather than full-density coverage of a ~2%% episode subset. Base "
                        "anchors are chosen EVENLY SPACED across each episode's valid frame_now "
                        "range (not just the first K), so coverage spans the episode's timeline "
                        "rather than clustering near its start.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_anchors", type=int, default=0,
                   help="Cap total (anchor, skip) generation calls this run (0 = no cap; each "
                        "base anchor contributes up to len(SKIP_VALUES)=2 calls). Useful for a "
                        "smoke test.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Anchors per forward call (may span multiple episodes AND skip variants "
                        "-- GPU efficiency, unlike push_cube's one-at-a-time script). 32 is the "
                        "most efficient size measured on an H200 (--sweep_batch_sizes "
                        "4,8,16,32,64,128): throughput plateaus around 16-32 (compute-bound by "
                        "the sequential n_steps Euler loop, not batch-parallelism-bound) and 128 "
                        "OOMs at ~140GB. ALWAYS uses the EFFICIENT_ATTENTION SDPA backend "
                        "explicitly (see module docstring's 'Batch-size note') -- letting PyTorch "
                        "auto-select crashes at larger batch sizes.")
    p.add_argument("--n_steps", type=int, default=25,
                   help="Euler/flow-matching inference steps. Open tuning knob -- sweep on the smoke "
                        "test (visually inspect decoded sharpness) before committing to a full run: "
                        "too few risks a blurry/easy distractor, too many multiplies GPU-hours.")
    p.add_argument("--svd_model_path", default="external/stable-video-diffusion-img2vid")
    p.add_argument("--clip_model_path", default="external/clip-vit-base-patch32")
    p.add_argument("--num_cams", type=int, default=3)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--down_sample", type=int, default=3)
    p.add_argument("--num_history", type=int, default=6)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--sweep_batch_sizes", default=None,
                   help="Comma-separated batch sizes to benchmark (e.g. '8,16,32,64'). When set, "
                        "loads the model ONCE, then times the SAME first --sweep_anchors anchors "
                        "for each batch size in turn (apples-to-apples throughput comparison), "
                        "prints an anchors/sec + s/anchor table, and skips the normal generation "
                        "loop. Outputs are still cached (to <output_root>/_sweep/bs<B>/... , a "
                        "separate namespace from the real cache) so a sweep run isn't wasted work.")
    p.add_argument("--sweep_anchors", type=int, default=64,
                   help="Anchors per batch size in --sweep_batch_sizes mode. Rounded down to a "
                        "whole number of batches for each size.")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, pipeline_cls, args, _use_uq = load_crtl_world(
        a.checkpoint,
        svd_model_path=a.svd_model_path, clip_model_path=a.clip_model_path,
        data_root=a.data_root, stat_root=a.stat_root, suites=[a.dataset_name], device=device,
        predict_uncertainty=False,  # sampling only -- the UQ head's output is unused here
        tag="bootstrap_future_sampler",
        num_cams=a.num_cams, height=a.height, width=a.width, down_sample=a.down_sample,
    )
    pipeline = model.pipeline

    dataset_dir = os.path.join(a.data_root, a.dataset_name)
    sample_path = os.path.join(a.data_root, "dataset_meta_info", a.dataset_name, f"{a.split}_sample.json")
    with open(sample_path) as f:
        samples = json.load(f)
    state_p01, state_p99 = _load_stat(a.stat_root, a.dataset_name)

    by_episode = _group_by_episode(samples)
    selected_episodes = _select_episodes(list(by_episode.keys()), a.every_nth_episode, a.shard_id, a.num_shards)
    print(f"[gen_bootstrap] {a.split}: {len(by_episode)} episodes total, "
          f"{len(selected_episodes)} selected (every_nth={a.every_nth_episode}, "
          f"shard {a.shard_id}/{a.num_shards})")

    out_root = Path(a.output_root) / a.dataset_name
    per_cam_h, latent_w = a.height // 8, a.width // 8
    total_h = a.num_cams * per_cam_h

    # Select BASE (episode_id, frame_now) anchors, skipping ones too close to
    # episode start for a full history window under the WORST-CASE skip_his
    # (skip=2 -> skip_his=8) so neither skip variant below gets a clamped/
    # degenerate history. Per-episode cap (--anchors_per_episode) applies
    # AFTER this filter (so a short episode isn't shortchanged relative to
    # its actual valid range).
    max_skip_his = max(SKIP_VALUES) * 4
    min_first_anchor = a.num_history * max_skip_his
    base_anchors: list[tuple[str, int]] = []
    n_capped_episodes = 0
    for ep in selected_episodes:
        valid_frames = sorted(f for f in by_episode[ep] if f >= min_first_anchor)
        if a.anchors_per_episode > 0 and len(valid_frames) > a.anchors_per_episode:
            n_capped_episodes += 1
            # Evenly-spaced subsample across the episode's valid timeline
            # (not just the first K) -- see --anchors_per_episode's help.
            idxs = sorted({int(round(i)) for i in
                           np.linspace(0, len(valid_frames) - 1, a.anchors_per_episode)})
            valid_frames = [valid_frames[i] for i in idxs]
        for frame_now in valid_frames:
            base_anchors.append((ep, frame_now))
    print(f"[gen_bootstrap] {len(base_anchors)} base anchors"
          + (f" ({n_capped_episodes}/{len(selected_episodes)} episodes capped to "
             f"<= {a.anchors_per_episode} each)" if a.anchors_per_episode > 0 else ""))

    # Expand each base anchor into up to len(SKIP_VALUES) generation calls --
    # one per possible skip a training sample might later draw for its own
    # true future -- skipping (episode, frame_now, skip) combinations
    # already cached (resumable, see module docstring). skip_his is drawn
    # fresh per call, matching dataset.py's exact distribution.
    anchors: list[tuple[str, int, int, int]] = []
    for ep, frame_now in base_anchors:
        for skip in SKIP_VALUES:
            cache_path = out_root / ep / f"{frame_now}_skip{skip}.pt"
            if cache_path.exists():
                continue
            anchors.append((ep, frame_now, skip, _draw_skip_his(skip)))
    if a.max_anchors > 0:
        anchors = anchors[: a.max_anchors]
    print(f"[gen_bootstrap] {len(anchors)} (anchor, skip) generation calls "
          f"to run (after skip-if-exists, up to {len(SKIP_VALUES)} skip variants each)")

    ann_cache: dict[str, dict] = {}

    if a.sweep_batch_sizes:
        _run_batch_size_sweep(
            model, pipeline, pipeline_cls, args, a, device,
            anchors, dataset_dir, ann_cache, state_p01, state_p99, per_cam_h, latent_w, total_h,
            out_root,
        )
        return

    n_done = 0
    t_start = time.time()

    for batch_start in range(0, len(anchors), a.batch_size):
        batch = anchors[batch_start: batch_start + a.batch_size]
        history, current, action, text_list = _build_batch_conditioning(
            batch, dataset_dir, args, a, ann_cache, state_p01, state_p99, per_cam_h, latent_w, total_h)
        pred_latents = _run_batch(
            model, pipeline, pipeline_cls, args, a, device, batch, history, current, action, text_list)

        for i, (ep, frame_now, skip, _skip_his) in enumerate(batch):
            out_dir = out_root / ep
            out_dir.mkdir(parents=True, exist_ok=True)
            future_latent = pred_latents[i, 1:].half()  # (num_frames-1, 4, total_h, latent_w)
            torch.save(future_latent, out_dir / f"{frame_now}_skip{skip}.pt")
            n_done += 1

        if (batch_start // a.batch_size) % 10 == 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            print(f"[gen_bootstrap] {n_done}/{len(anchors)} anchors "
                  f"({rate:.2f}/s, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t_start
    print(f"[gen_bootstrap] done: {n_done} anchors in {elapsed:.1f}s "
          f"({n_done / elapsed if elapsed > 0 else 0.0:.2f}/s)")


if __name__ == "__main__":
    main()
