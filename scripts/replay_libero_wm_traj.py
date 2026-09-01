"""Auto-regressive trajectory replay for the LIBERO world model.

Replays recorded LIBERO trajectories through a trained ``CrtlWorld`` checkpoint
and writes ground-truth vs world-model-predicted videos side by side as two
separate sets.

The checkpoint is trained at 5 Hz (``data/wm_training/libero_processed_5hz``)
while collected rollouts are stored at 20 Hz, so each trajectory is strided
down to 5 Hz before replay (``--target_hz``). Actions are normalized with the
*training* stats (``--stat_root``), not the eval data.

Rollout (per chunk, at 5 Hz, mirroring LiberoLatentDataset._build_frame_ids
with deterministic skip=1, skip_his=4):

    frame_now -> history = rolled[frame_now - 6*4 .. -4]   (mostly past predictions)
                 current = rolled[frame_now]
                 action  = state[frame_now - 24 .. +4] (normalized)
                 pred[5] = pipeline(image=current, history=history, text=action_latent)
    overwrite rolled[frame_now .. frame_now+4] with pred so the next chunk
    conditions on its own predictions (closed loop).

Output:
    <output_dir>/gt/<suite>_<episode>.mp4     # decoded ground-truth latents
    <output_dir>/pred/<suite>_<episode>.mp4   # decoded predictions (same frames)
    <output_dir>/replay_summary.json          # per-episode latent/pixel MSE, PSNR

Usage:
    uv run scripts/replay_libero_wm_traj.py \\
        --checkpoint checkpoints/wm/libero/checkpoint-30000.pt \\
        --data_root  data/libero_collected \\
        --stat_root  dataset_meta_info \\
        --output_dir checkpoints/wm/libero/replay
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch

from openworld.training.world_model.config import LiberoWMArgs
from openworld.training.world_model.dataset import _load_stat
from openworld.world_models.ctrl_world import CrtlWorld, CtrlWorldDiffusionPipeline


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def list_suites(data_root: str) -> list[str]:
    return sorted(p.name for p in Path(data_root).iterdir() if (p / "annotation").is_dir())


def list_episode_ids(data_root: str, suite: str, split: str) -> list[str]:
    ann_dir = Path(data_root) / suite / "annotation" / split
    return sorted(p.stem for p in ann_dir.glob("*.json"))


def load_full_episode(
    data_root: str, suite: str, episode_id: str, args: LiberoWMArgs, split: str,
    down_sample: int = 1, native_fps_default: int = 20,
) -> tuple[torch.Tensor, np.ndarray, str, int]:
    """Return (latent[T,4,total_h,latent_w] fp32, action[T,7] fp32, text, native_fps).

    ``down_sample`` corrects for datasets (e.g. DROID's droid_ctrl_world)
    where the action/state arrays are stored at a higher native rate than
    the pre-encoded latents: state index ``i * down_sample`` is paired with
    latent frame ``i``. Defaults to 1 (LIBERO's collected data has state and
    latents at the same rate, so this is a no-op there).
    """
    suite_dir = os.path.join(data_root, suite)
    with open(os.path.join(suite_dir, args.annotation_name, split, f"{episode_id}.json")) as f:
        label = json.load(f)

    cam_specs = label["latent_videos"]
    if len(cam_specs) < args.num_cams:
        raise ValueError(f"{episode_id}: {len(cam_specs)} cameras < num_cams={args.num_cams}")

    cam_tensors = []
    for cam_idx in range(args.num_cams):
        with open(os.path.join(suite_dir, cam_specs[cam_idx]["latent_video_path"]), "rb") as fh:
            cam_tensors.append(torch.load(fh, map_location="cpu").float())

    # Stack the two camera latents vertically along H, exactly as the dataset does.
    per_cam_h, latent_w = args.height // 8, args.width // 8
    T = min(int(t.shape[0]) for t in cam_tensors)
    latent = torch.zeros((T, 4, args.num_cams * per_cam_h, latent_w), dtype=torch.float32)
    for cam_idx, t in enumerate(cam_tensors):
        latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = t[:T]

    cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)
    grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]
    full_state = np.concatenate([cart, grip], axis=-1)  # (S, 7), at native state rate
    state_idx = np.clip(np.arange(T) * down_sample, 0, len(full_state) - 1)
    action_native = full_state[state_idx]  # (T, 7), aligned 1:1 with latent frames

    text = label["texts"][0] if label.get("texts") else label.get("language_instruction", "")
    return latent, action_native, text, int(label.get("fps", native_fps_default))


def normalize_actions(action: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    return np.clip(2 * (action - p01) / (p99 - p01 + 1e-8) - 1, -1, 1)


def load_crtl_world(
    checkpoint: str,
    *,
    svd_model_path: str,
    clip_model_path: str,
    data_root: str,
    stat_root: str,
    suites: list[str],
    device: torch.device,
    predict_uncertainty: bool = True,
    uq_vis_t_targets: tuple[float, ...] = (0.9, 0.5, 0.1),
    tag: str = "libero_replay",
    num_cams: int = 2,
    height: int = 320,
    width: int = 320,
    down_sample: int = 1,
) -> tuple[CrtlWorld, type[CtrlWorldDiffusionPipeline], LiberoWMArgs, bool]:
    """Build ``LiberoWMArgs``, construct ``CrtlWorld``, and load a checkpoint.

    Shared by ``replay_libero_wm_traj.py``'s CLI and any other script that
    needs to score candidates with this world model (see
    ``scripts/run_data_collection_active_uq.py``).

    ``num_cams``/``height``/``width``/``down_sample`` default to LIBERO's
    values; pass DROID's (3, 192, 320, 3) to replay a DROID checkpoint.

    Returns (model, pipeline_cls, args, use_uq).
    """
    args = LiberoWMArgs(
        svd_model_path=svd_model_path, clip_model_path=clip_model_path,
        dataset_root_path=data_root, dataset_meta_info_path=stat_root,
        dataset_names="+".join(suites), dataset_cfgs="+".join(suites),
        prob=tuple([1.0 / len(suites)] * len(suites)),
        num_cams=num_cams, height=height, width=width,
        num_frames=5, num_history=6, action_dim=7, down_sample=down_sample,
        flow_map_type="flow_matching", distance_conditioning=False, tag=tag,
        predict_uncertainty=predict_uncertainty,
        uq_vis_t_targets=tuple(uq_vis_t_targets),
    )

    print(f"[wm] loading checkpoint {checkpoint}")
    model = CrtlWorld(args)
    missing, unexpected = model.load_state_dict(
        torch.load(checkpoint, map_location="cpu"), strict=False)
    if missing:
        print(f"[wm] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[wm] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.to(device).eval()
    use_uq = args.predict_uncertainty and getattr(model.unet, "predict_uncertainty", False)
    if args.predict_uncertainty and not use_uq:
        print("[wm] WARNING: --predict_uncertainty set but checkpoint has no UQ head; disabling.")
    return model, CtrlWorldDiffusionPipeline, args, use_uq


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def build_frame_ids(frame_now: int, num_history: int, num_frames: int,
                     skip: int, skip_his: int) -> list[int]:
    """History going back by skip_his, then num_frames future at stride skip."""
    rgb_id = [int(frame_now - i * skip_his) for i in range(num_history, 0, -1)]
    rgb_id.append(int(frame_now))
    rgb_id += [int(frame_now + i * skip) for i in range(1, num_frames)]
    return rgb_id


@torch.no_grad()
def replay_episode(
    model: CrtlWorld, pipeline, pipeline_cls, args: LiberoWMArgs,
    latent_gt: torch.Tensor,        # (T, 4, total_h, latent_w) fp32, at WM (5 Hz) rate
    action_norm: np.ndarray,        # (T, 7) normalized, at WM rate, 1:1 with latents
    text: str, device: torch.device,
    num_inference_steps: int, skip: int, max_chunks: int | None,
    use_uq: bool = False,
    uq_epi_mode: str = "none",      # "none" | "zero_history" | "flat_history" | "single_history" | "iterative" | "future_overlap"
    epi_overlap_k: int = 0,         # uq_epi_mode="future_overlap" only: 0 = use num_frames-1 (max overlap)
    overlap_zero_action: bool = False,  # uq_epi_mode="future_overlap" only: zero the overlap slot's action
                                         # conditioning instead of the real candidate action -- must match
                                         # the checkpoint's zero_overlap_action training setting.
    on_chunk=None,                  # Callable[[int,int,list,list,list,list,list,float|None,float|None],None] | None
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[torch.Tensor],
           torch.Tensor | None, list[torch.Tensor]]:
    """Closed-loop rollout.

    Returns:
        gt_stack, pred_stack, predicted_indices, logvar_by_step (pass 1),
        epi_pred_stack (pass 2, or None if uq_epi_mode=="none"),
        epi_logvar_by_step (pass 2, or [] if unavailable).
    """
    T = int(latent_gt.shape[0])
    num_history, num_frames = args.num_history, args.num_frames
    skip_his = skip * 4  # matches dataset._build_frame_ids (history is 4x coarser)

    # rolled[t] = latent we believe represents frame t; seed with GT, overwrite with preds.
    rolled = [latent_gt[t].clone() for t in range(T)]
    # rolled2 is used only by the iterative mode (conditions pass 2 on its own predictions).
    if uq_epi_mode == "iterative":
        rolled2 = [latent_gt[t].clone() for t in range(T)]
    else:
        rolled2 = None

    first_anchor = num_history * skip_his  # so history indices stay >= 0
    step = (num_frames - 1) * skip         # chain: last pred frame -> next anchor
    if first_anchor + (num_frames - 1) * skip >= T:
        raise RuntimeError(f"episode too short: T={T}")

    predicted_indices: list[int] = []
    # frame_logvar mirrors `rolled`: frame_index -> list[logvar_per_ode_step]
    # each entry is a list of length n_ode_steps, each element (1, total_h, latent_w)
    frame_logvar: dict[int, list[torch.Tensor]] = {}
    # frame_vel: same structure but velocity predictions (4, total_h, latent_w)
    frame_vel: dict[int, list[torch.Tensor]] = {}
    # epi accumulators (pass 2)
    epi_frame_pred: dict[int, torch.Tensor] = {}
    epi_frame_logvar: dict[int, list[torch.Tensor]] = {}
    epi_frame_vel: dict[int, list[torch.Tensor]] = {}

    chunk_idx = 0
    while True:
        frame_now = first_anchor + chunk_idx * step
        if frame_now + (num_frames - 1) * skip >= T:
            break
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

        rgb_id = build_frame_ids(frame_now, num_history, num_frames, skip, skip_his)
        state_id = [min(max(r, 0), T - 1) for r in rgb_id]  # action at same frame as latent

        history = torch.stack([rolled[rgb_id[i]] for i in range(num_history)], 0).unsqueeze(0).to(device)
        current = rolled[rgb_id[num_history]].unsqueeze(0).to(device)
        action = torch.tensor(action_norm[state_id], dtype=torch.float32).unsqueeze(0).to(device)

        # Shared seed for pass 1 / pass 2: a fresh torch.Generator per call
        # (not one shared object reused across both __call__s, which would
        # advance its state between calls and defeat the point) so the two
        # passes draw identical initial diffusion noise -- isolating the
        # measured pass1/pass2 divergence to the conditioning difference
        # instead of ordinary sampling variance. See pipeline_flow_map_ctrl_world.py's
        # generator-handling fix.
        noise_seed = torch.Generator().seed()  # random seed, no side effect on global RNG
        gen1 = torch.Generator(device=device).manual_seed(noise_seed)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            action_latent = model.action_encoder(
                action, [text], model.tokenizer, model.text_encoder, args.frame_level_cond)
            result = pipeline_cls.__call__(
                pipeline, image=current, text=action_latent,
                width=args.width, height=int(args.num_cams * args.height),
                num_frames=num_frames, history=history,
                num_inference_steps=num_inference_steps, decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale, fps=args.fps,
                motion_bucket_id=args.motion_bucket_id, mask=None,
                output_type="latent", return_dict=False,
                frame_level_cond=args.frame_level_cond, his_cond_zero=args.his_cond_zero,
                flow_map_type=args.flow_map_type, flow_map_loss_type=args.flow_map_loss_type,
                return_uncertainty=use_uq, generator=gen1,
            )
        if use_uq:
            _, pred_latents, logvar_steps, vel_steps = result
        else:
            _, pred_latents = result
            logvar_steps = []
            vel_steps = []
        pred = pred_latents[0].float().cpu()  # (num_frames, 4, total_h, latent_w)

        # ---- Pass 2 (epistemic) ----
        overlap_k_used: int | None = None
        overlap_latent_mse: float | None = None
        full_latent_mse: float | None = None
        if uq_epi_mode != "none":
            action_latent2 = action_latent  # default: unchanged (matches all modes except future_overlap)
            if uq_epi_mode == "zero_history":
                history2, current2, his_cond_zero2 = history, current, True
            elif uq_epi_mode == "flat_history":
                # repeat current frame for all history slots
                history2 = current.unsqueeze(1).expand(-1, num_history, -1, -1, -1).contiguous()
                current2, his_cond_zero2 = current, False
            elif uq_epi_mode == "single_history":
                # repeat most-recent history frame for all slots (mirrors training augmentation)
                last_hist = history[:, -1:, :, :, :]  # (1, 1, 4, total_h, latent_w)
                history2 = last_hist.expand(-1, num_history, -1, -1, -1).contiguous()
                current2, his_cond_zero2 = current, False
            elif uq_epi_mode == "future_overlap":
                # Splice pass-1's OWN predicted future frames into history, keep current
                # UNCHANGED, and re-predict the SAME target range (mirrors training's
                # p_history_future_overlap augmentation in CrtlWorld.forward()).
                overlap_k_used = epi_overlap_k if epi_overlap_k > 0 else (num_frames - 1)
                overlap_k_used = max(1, min(overlap_k_used, num_frames - 1))
                overlap_frames = pred[1:1 + overlap_k_used].unsqueeze(0).to(device)  # pred = pass-1 output
                history2 = torch.cat([history, overlap_frames], dim=1)
                current2, his_cond_zero2 = current, False  # MUST reuse pass-1's `current` verbatim -- target range/current frame unchanged; do not recompute
                if overlap_zero_action:
                    overlap_action_t = torch.zeros(1, overlap_k_used, action_norm.shape[-1],
                                                    dtype=torch.float32, device=device)
                else:
                    overlap_action_vals = action_norm[[state_id[num_history + i] for i in range(1, 1 + overlap_k_used)]]
                    overlap_action_t = torch.tensor(overlap_action_vals, dtype=torch.float32).unsqueeze(0).to(device)
                action2_for_encoder = torch.cat([action, overlap_action_t], dim=1)
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    action_latent2 = model.action_encoder(
                        action2_for_encoder, [text], model.tokenizer, model.text_encoder, args.frame_level_cond)
            else:  # iterative
                history2 = torch.stack([rolled2[rgb_id[i]] for i in range(num_history)], 0).unsqueeze(0).to(device)
                current2 = rolled2[rgb_id[num_history]].unsqueeze(0).to(device)
                his_cond_zero2 = False

            gen2 = torch.Generator(device=device).manual_seed(noise_seed)  # same seed as pass 1
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                result2 = pipeline_cls.__call__(
                    pipeline, image=current2, text=action_latent2,
                    width=args.width, height=int(args.num_cams * args.height),
                    num_frames=num_frames, history=history2,
                    num_inference_steps=num_inference_steps, decode_chunk_size=args.decode_chunk_size,
                    max_guidance_scale=args.guidance_scale, fps=args.fps,
                    motion_bucket_id=args.motion_bucket_id, mask=None,
                    output_type="latent", return_dict=False,
                    frame_level_cond=args.frame_level_cond, his_cond_zero=his_cond_zero2,
                    flow_map_type=args.flow_map_type, flow_map_loss_type=args.flow_map_loss_type,
                    return_uncertainty=use_uq, generator=gen2,
                )
            if use_uq:
                _, pred_latents2, logvar_steps2, vel_steps2 = result2
            else:
                _, pred_latents2 = result2
                logvar_steps2 = []
                vel_steps2 = []
            pred2 = pred_latents2[0].float().cpu()  # (num_frames, 4, total_h, latent_w)

            if overlap_k_used is not None:
                # Checkpoint-agnostic divergence metric -- works even without a UQ head.
                overlap_latent_mse = float(((pred[1:1 + overlap_k_used] - pred2[1:1 + overlap_k_used]) ** 2).mean())
                full_latent_mse = float(((pred - pred2) ** 2).mean())

        for k in range(num_frames):
            t_native = rgb_id[num_history + k]
            if t_native < T:
                rolled[t_native] = pred[k]
                predicted_indices.append(t_native)
                if logvar_steps:
                    frame_logvar[t_native] = [
                        logvar_steps[s][0, k].cpu()  # (1, total_h, latent_w)
                        for s in range(len(logvar_steps))
                    ]
                if vel_steps:
                    frame_vel[t_native] = [
                        vel_steps[s][0, k].cpu()  # (4, total_h, latent_w)
                        for s in range(len(vel_steps))
                    ]
                if uq_epi_mode != "none":
                    epi_frame_pred[t_native] = pred2[k]
                    if logvar_steps2:
                        epi_frame_logvar[t_native] = [
                            logvar_steps2[s][0, k].cpu()
                            for s in range(len(logvar_steps2))
                        ]
                    if vel_steps2:
                        epi_frame_vel[t_native] = [
                            vel_steps2[s][0, k].cpu()
                            for s in range(len(vel_steps2))
                        ]
                    if uq_epi_mode == "iterative" and rolled2 is not None:
                        rolled2[t_native] = pred[k].clone()  # use Pass-1 predictions

        # Per-chunk callback: stream UQ metrics for this chunk's predicted frames.
        # Fires even without a UQ head (logvar_steps empty) when overlap_latent_mse is
        # available, so uq_epi_mode="future_overlap" is smoke-testable on any checkpoint.
        if on_chunk is not None:
            chunk_frame_indices = [
                rgb_id[num_history + k] for k in range(num_frames)
                if rgb_id[num_history + k] < T
            ]
            if chunk_frame_indices and (logvar_steps or overlap_latent_mse is not None):
                n_ode = len(logvar_steps)
                chunk_lv = [
                    torch.stack([frame_logvar[t][s] for t in chunk_frame_indices], 0)
                    for s in range(n_ode)
                ]
                chunk_vel = [
                    torch.stack([frame_vel[t][s] for t in chunk_frame_indices], 0)
                    for s in range(n_ode)
                ] if vel_steps else []
                chunk_epi_lv = [
                    torch.stack([epi_frame_logvar[t][s] for t in chunk_frame_indices], 0)
                    for s in range(n_ode)
                ] if epi_frame_logvar else []
                chunk_epi_vel = [
                    torch.stack([epi_frame_vel[t][s] for t in chunk_frame_indices], 0)
                    for s in range(n_ode)
                ] if epi_frame_vel else []
                on_chunk(chunk_idx, frame_now, chunk_frame_indices,
                         chunk_lv, chunk_vel, chunk_epi_lv, chunk_epi_vel,
                         overlap_latent_mse, full_latent_mse)
        chunk_idx += 1

    if chunk_idx == 0:
        raise RuntimeError("no chunks produced")
    predicted_indices = sorted(set(predicted_indices))
    pred_stack = torch.stack([rolled[t] for t in predicted_indices], 0)
    gt_stack = torch.stack([latent_gt[t] for t in predicted_indices], 0)

    # Assemble per-ODE-step logvar stacks aligned with predicted_indices
    # logvar_by_step[s]: (T_pred, 1, total_h, latent_w)
    logvar_by_step: list[torch.Tensor] = []
    if frame_logvar:
        n_ode = len(next(iter(frame_logvar.values())))
        for s in range(n_ode):
            logvar_by_step.append(
                torch.stack([frame_logvar[t][s] for t in predicted_indices], 0)
            )

    # vel_by_step[s]: (T_pred, 4, total_h, latent_w)
    vel_by_step: list[torch.Tensor] = []
    if frame_vel:
        n_ode = len(next(iter(frame_vel.values())))
        for s in range(n_ode):
            vel_by_step.append(
                torch.stack([frame_vel[t][s] for t in predicted_indices], 0)
            )

    # Assemble epistemic stacks
    epi_pred_stack: torch.Tensor | None = None
    epi_logvar_by_step: list[torch.Tensor] = []
    epi_vel_by_step: list[torch.Tensor] = []
    if uq_epi_mode != "none" and epi_frame_pred:
        epi_pred_stack = torch.stack([epi_frame_pred[t] for t in predicted_indices], 0)
        if epi_frame_logvar:
            n_ode2 = len(next(iter(epi_frame_logvar.values())))
            for s in range(n_ode2):
                epi_logvar_by_step.append(
                    torch.stack([epi_frame_logvar[t][s] for t in predicted_indices], 0)
                )
        if epi_frame_vel:
            n_ode2 = len(next(iter(epi_frame_vel.values())))
            for s in range(n_ode2):
                epi_vel_by_step.append(
                    torch.stack([epi_frame_vel[t][s] for t in predicted_indices], 0)
                )

    return (gt_stack, pred_stack, predicted_indices,
            logvar_by_step, vel_by_step,
            epi_pred_stack, epi_logvar_by_step, epi_vel_by_step)


# ---------------------------------------------------------------------------
# Decode / IO
# ---------------------------------------------------------------------------


def decode_per_cam(latents: torch.Tensor, pipeline, args: LiberoWMArgs) -> np.ndarray:
    """(T, 4, num_cams*per_cam_h, latent_w) -> (T, num_cams*H, W, 3) uint8.

    Decode each camera view separately (avoids cross-view conv bleed), stack vertically."""
    device, dtype = pipeline.unet.device, pipeline.unet.dtype
    per_cam = einops.rearrange(latents, "t c (m h) w -> m t c h w", m=args.num_cams)
    decoded = []
    for cam_idx in range(args.num_cams):
        flat = per_cam[cam_idx].to(device=device, dtype=dtype)
        chunks = []
        for i in range(0, flat.shape[0], args.decode_chunk_size):
            chunk = flat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            chunks.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        out = torch.cat(chunks, 0)  # (T, 3, H, W)
        out = ((out / 2.0 + 0.5).clamp(0, 1) * 255).float().cpu().numpy().astype(np.uint8)
        decoded.append(out.transpose(0, 2, 3, 1))  # (T, H, W, 3)
    return np.concatenate(decoded, axis=1)


def _render_uq_video(
    logvar_stack: torch.Tensor,  # (T, 1, total_h, latent_w) for one ODE step
    args: LiberoWMArgs,
) -> np.ndarray:
    """Turbo heatmap video with globally consistent colour scale -> (T, num_cams*H, W, 3) uint8."""
    import torch.nn.functional as F_nn
    import matplotlib.cm as cm

    # Upsample entire stack in one call
    var_all = logvar_stack.exp().float()  # (T, 1, total_h, latent_w)
    var_up = F_nn.interpolate(
        var_all, size=(args.num_cams * args.height, args.width),
        mode="bilinear", align_corners=False,
    )  # (T, 1, num_cams*H, W)

    # Global vmin/vmax -> no frame-to-frame flicker
    var_np = var_up[:, 0].cpu().numpy()  # (T, num_cams*H, W)
    vmin, vmax = float(var_np.min()), float(var_np.max())

    # Split cameras for stacking
    per_cam = var_np.reshape(var_np.shape[0], args.num_cams, args.height, args.width)
    frames = []
    for t in range(per_cam.shape[0]):
        cam_frames = []
        for cam_idx in range(args.num_cams):
            v_np = per_cam[t, cam_idx]  # (H, W)
            norm = (v_np - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v_np)
            cam_frames.append((cm.turbo(norm)[..., :3] * 255).astype(np.uint8))
        frames.append(np.concatenate(cam_frames, axis=0))  # (num_cams*H, W, 3)
    return np.stack(frames, 0)  # (T, num_cams*H, W, 3)


def _turbo_heatmap(
    signal: torch.Tensor,   # (T, 1, total_h, latent_w) — already non-negative or signed
    args: LiberoWMArgs,
) -> np.ndarray:
    """Shared helper: upsample -> global normalize -> turbo -> (T, num_cams*H, W, 3) uint8."""
    import torch.nn.functional as F_nn
    import matplotlib.cm as cm

    up = F_nn.interpolate(
        signal.float(), size=(args.num_cams * args.height, args.width),
        mode="bilinear", align_corners=False,
    )[:, 0].cpu().numpy()  # (T, num_cams*H, W)
    vmin, vmax = float(up.min()), float(up.max())
    norm = (up - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(up)
    per_cam = norm.reshape(norm.shape[0], args.num_cams, args.height, args.width)
    frames = [
        np.concatenate(
            [(cm.turbo(per_cam[t, c])[..., :3] * 255).astype(np.uint8)
             for c in range(args.num_cams)], axis=0
        )
        for t in range(per_cam.shape[0])
    ]
    return np.stack(frames, 0)


def _diverging_heatmap(
    signal: torch.Tensor,   # (T, 1, total_h, latent_w) — SIGNED, e.g. exp(lv2)-exp(lv1) unclamped
    args: LiberoWMArgs,
    cmap_name: str = "coolwarm",
) -> np.ndarray:
    """Like _turbo_heatmap but symmetric around 0 with a diverging colormap, so sign is
    visible (not just magnitude). Normalizes by the whole stack's max |value| (not
    independent min/max) so 0 always maps to the colormap's neutral center -- a plain
    min-max normalize would shift the zero point off-center whenever the signal is
    skewed to one side, which turbo-style sequential heatmaps don't need to care about
    but a diverging one does."""
    import matplotlib
    import torch.nn.functional as F_nn

    up = F_nn.interpolate(
        signal.float(), size=(args.num_cams * args.height, args.width),
        mode="bilinear", align_corners=False,
    )[:, 0].cpu().numpy()  # (T, num_cams*H, W)
    vmax = float(np.abs(up).max())
    norm = (up / vmax + 1.0) / 2.0 if vmax > 0 else np.full_like(up, 0.5)  # [-vmax,vmax] -> [0,1], 0 -> 0.5
    norm = np.clip(norm, 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    per_cam = norm.reshape(norm.shape[0], args.num_cams, args.height, args.width)
    frames = [
        np.concatenate(
            [(cmap(per_cam[t, c])[..., :3] * 255).astype(np.uint8)
             for c in range(args.num_cams)], axis=0
        )
        for t in range(per_cam.shape[0])
    ]
    return np.stack(frames, 0)


def _render_vel_ltv_video(
    vel1: torch.Tensor,   # (T, 4, total_h, latent_w) — v_pred from pass 1
    vel2: torch.Tensor,   # (T, 4, total_h, latent_w) — v_pred from pass 2
    args: LiberoWMArgs,
) -> np.ndarray:
    """LTV epistemic: mean_channel((vel1 - vel2)^2) → turbo heatmap."""
    epi = ((vel1 - vel2) ** 2).mean(dim=1, keepdim=True)   # (T, 1, total_h, latent_w)
    return _turbo_heatmap(epi, args)


def _render_pdf_diff_video(
    lv1: torch.Tensor,   # (T, 1, total_h, latent_w)
    lv2: torch.Tensor,   # (T, 1, total_h, latent_w)
    args: LiberoWMArgs,
) -> np.ndarray:
    """PDF diff: 0.5 * |logvar1 - logvar2| → turbo heatmap."""
    diff = 0.5 * (lv1 - lv2).abs()
    return _turbo_heatmap(diff, args)


def _render_kl_video(
    vel1: torch.Tensor,   # (T, 4, total_h, latent_w)
    lv1:  torch.Tensor,   # (T, 1, total_h, latent_w)
    vel2: torch.Tensor,   # (T, 4, total_h, latent_w)
    lv2:  torch.Tensor,   # (T, 1, total_h, latent_w)
    args: LiberoWMArgs,
) -> np.ndarray:
    """KL(N(vel1,exp(lv1)) || N(vel2,exp(lv2))) per pixel → turbo heatmap."""
    sq_diff = ((vel1 - vel2) ** 2).sum(dim=1, keepdim=True)   # (T, 1, ...)
    kl = 0.5 * (
        4.0 * (lv2 - lv1)
        + 4.0 * lv1.exp() / lv2.exp()
        + sq_diff / lv2.exp()
        - 4.0
    ).clamp(min=0)
    return _turbo_heatmap(kl, args)


def _render_epi_var_video(
    logvar_stack1: torch.Tensor,  # (T, 1, total_h, latent_w)  — pass 1
    logvar_stack2: torch.Tensor,  # (T, 1, total_h, latent_w)  — pass 2
    args: LiberoWMArgs,
) -> np.ndarray:
    """Epistemic var: (exp(lv2) - exp(lv1)).clamp(0) → turbo heatmap.

    One-directional by design: built to check "did the model get MORE uncertain
    given worse conditioning" (zero_history/flat_history/single_history/iterative,
    where pass 2 conditions on strictly less/worse info than pass 1). For
    future_overlap, pass 2 conditions on strictly MORE info, so the "expected"
    direction is reversed -- clamping here would show ~0 even when there's a large,
    real (just oppositely-signed) effect. Use _render_epi_var_signed_video instead
    to see both directions at once."""
    epi_var = (logvar_stack2.exp() - logvar_stack1.exp()).clamp(min=0)
    return _turbo_heatmap(epi_var, args)


def _render_epi_var_signed_video(
    logvar_stack1: torch.Tensor,  # (T, 1, total_h, latent_w)  — pass 1
    logvar_stack2: torch.Tensor,  # (T, 1, total_h, latent_w)  — pass 2
    args: LiberoWMArgs,
) -> np.ndarray:
    """Signed epistemic var: exp(lv2) - exp(lv1), UNCLAMPED → diverging heatmap.
    Same sign convention as _render_epi_var_video (positive = pass 2 MORE uncertain
    than pass 1) but keeps the negative side visible instead of clamping it away.
    Warm/positive = pass 2 more uncertain (the anomalous direction for future_overlap,
    since pass 2 has strictly more context -- localized warm regions here would be
    genuine content-driven epistemic disagreement). Cool/negative = pass 2 more
    confident (the direction a global length-driven confidence collapse would show
    as, uniformly, across the whole frame)."""
    epi_var_signed = logvar_stack2.exp() - logvar_stack1.exp()
    return _diverging_heatmap(epi_var_signed, args)


def compute_uq_metrics(
    logvar_by_step: list[torch.Tensor],       # list[N_s] of (T, 1, H, W)
    vel_by_step: list[torch.Tensor],          # list[N_s] of (T, 4, H, W)
    epi_logvar_by_step: list[torch.Tensor],   # list[N_s] of (T, 1, H, W)  — pass 2
    epi_vel_by_step: list[torch.Tensor],      # list[N_s] of (T, 4, H, W)  — pass 2
) -> dict[str, float]:
    """Aggregate UQ signals across all ODE sampling steps, then compute scalar metrics.

    Step 1: stack per-step tensors → mean over N_s (ODE step axis).
      - logvar: (N_s, T, 1, H, W) → mean → (T, 1, H, W) → squeeze C=1 → (T, H, W)
      - velocity: (N_s, T, 4, H, W) → mean → (T, 4, H, W)  [C kept]
    Step 2: scalar = mean / max over all remaining dims.
    """
    metrics: dict[str, float] = {}

    if logvar_by_step:
        agg_var = torch.stack([lv.exp() for lv in logvar_by_step], 0).mean(0).squeeze(1)  # (T,H,W)
        metrics["mean_aleatoric_var"] = float(agg_var.mean())
        metrics["max_aleatoric_var"]  = float(agg_var.max())

    if vel_by_step and epi_vel_by_step:
        n = min(len(vel_by_step), len(epi_vel_by_step))
        # EpiLTV: (T,4,H,W) per step; keep C, stack N_s → mean → (T,4,H,W)
        agg_ltv = torch.stack(
            [(vel_by_step[s] - epi_vel_by_step[s]) ** 2 for s in range(n)], 0
        ).mean(0)  # (T, 4, H, W)
        metrics["mean_epi_ltv"] = float(agg_ltv.mean())
        metrics["max_epi_ltv"]  = float(agg_ltv.max())

    if logvar_by_step and epi_logvar_by_step:
        n = min(len(logvar_by_step), len(epi_logvar_by_step))
        # EpiVar: (T,1,H,W) → squeeze → (T,H,W) per step; stack N_s → mean → (T,H,W)
        agg_epi_var = torch.stack(
            [(epi_logvar_by_step[s].exp() - logvar_by_step[s].exp()).clamp(0).squeeze(1)
             for s in range(n)], 0
        ).mean(0)
        metrics["mean_epi_var"] = float(agg_epi_var.mean())
        # EpiVarSigned: same quantity, UNCLAMPED -- keeps the "pass 2 more confident"
        # direction visible instead of discarding it. See _render_epi_var_signed_video.
        agg_epi_var_signed = torch.stack(
            [(epi_logvar_by_step[s].exp() - logvar_by_step[s].exp()).squeeze(1)
             for s in range(n)], 0
        ).mean(0)
        metrics["mean_epi_var_signed"] = float(agg_epi_var_signed.mean())
        # PDFDiff: 0.5*|lv1-lv2|
        agg_pdf = torch.stack(
            [0.5 * (logvar_by_step[s] - epi_logvar_by_step[s]).abs().squeeze(1)
             for s in range(n)], 0
        ).mean(0)
        metrics["mean_pdf_diff"] = float(agg_pdf.mean())

    if vel_by_step and epi_vel_by_step and logvar_by_step and epi_logvar_by_step:
        n = min(len(vel_by_step), len(epi_vel_by_step),
                len(logvar_by_step), len(epi_logvar_by_step))
        kl_steps = []
        for s in range(n):
            lv1, lv2 = logvar_by_step[s], epi_logvar_by_step[s]
            sq = ((vel_by_step[s] - epi_vel_by_step[s]) ** 2).sum(dim=1, keepdim=True)  # (T,1,H,W)
            kl = 0.5 * (
                4 * (lv2 - lv1) + 4 * lv1.exp() / lv2.exp() + sq / lv2.exp() - 4
            ).clamp(0).squeeze(1)  # (T, H, W)
            kl_steps.append(kl)
        metrics["mean_kl"] = float(torch.stack(kl_steps, 0).mean(0).mean())

    return metrics


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # h264 requires even dimensions; crop 1px if needed
    H, W = frames.shape[1], frames.shape[2]
    if H % 2 != 0 or W % 2 != 0:
        frames = frames[:, : H - (H % 2), : W - (W % 2)]
    mediapy.write_video(str(path), frames, fps=fps, codec="h264",
                        ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/wm/libero/checkpoint-30000.pt")
    p.add_argument("--data_root", default="data/libero_collected")
    p.add_argument("--stat_root", default="dataset_meta_info",
                   help="Where stat.json lives (the TRAINING stats; not the eval data).")
    p.add_argument("--output_dir", default=None, help="Default: <checkpoint dir>/replay.")
    p.add_argument("--suites", default=None, help="Comma-separated; default: all under data_root.")
    p.add_argument("--episode_id", default=None, help="Single episode; default: all in suite.")
    p.add_argument("--num_episodes", type=int, default=0, help="Cap episodes per suite (0=all).")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--target_hz", type=int, default=5, help="WM rate; data is strided down to this.")
    p.add_argument("--skip", type=int, default=1, help="Frame stride at WM rate (training used 1 or 2).")
    p.add_argument("--max_chunks", type=int, default=0, help="Cap autoregressive chunks (0=to episode end).")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--video_fps", type=int, default=5, help="5 = real time at the 5 Hz WM rate.")
    p.add_argument("--svd_model_path", default="external/stable-video-diffusion-img2vid")
    p.add_argument("--clip_model_path", default="external/clip-vit-base-patch32")
    p.add_argument("--predict_uncertainty", action="store_true",
                   help="Enable UQ visualization (requires a UQ-trained checkpoint).")
    p.add_argument("--uq_vis_t_targets", type=float, nargs="+", default=[0.9, 0.5, 0.1],
                   help="Target t-values for UQ visualization columns (closest ODE step is used).")
    p.add_argument("--uq_epi_mode", default="none",
                   choices=["none", "zero_history", "flat_history", "single_history",
                            "iterative", "future_overlap"],
                   help="Epistemic UQ mode: 'zero_history'=his_cond_zero pass, "
                        "'flat_history'=repeat current frame as history, "
                        "'single_history'=repeat most-recent history frame for all slots, "
                        "'iterative'=self-consistency (pass 2 uses pass-1 predictions as history), "
                        "'future_overlap'=splice pass-1's own predicted future frames into history "
                        "and re-predict the SAME target range (requires a checkpoint trained with "
                        "p_history_future_overlap>0 for meaningful results; see --epi_overlap_k).")
    p.add_argument("--epi_overlap_k", type=int, default=0,
                   help="Overlap width k for uq_epi_mode=future_overlap; 0 (default) = "
                        "num_frames-1 (max overlap, uses all of pass-1's predicted future "
                        "as pass-2 context). Must be in [1, num_frames-1].")
    p.add_argument("--overlap_zero_action", action="store_true",
                   help="For uq_epi_mode=future_overlap: zero the action conditioning at the "
                        "overlap slot instead of using the real candidate action. Must match "
                        "the checkpoint's zero_overlap_action training setting -- set this for "
                        "checkpoints trained with zero_overlap_action=True (e.g. "
                        "droid_flow_matching_uq_false_future_v1), leave unset for "
                        "droid_flow_matching_uq_future_overlap_v1 and earlier.")
    p.add_argument("--manifest", default=None,
                   help="Path to test_manifest_by_cell.json for cell-balanced episode selection.")
    p.add_argument("--max_episodes_per_cell", type=int, default=0,
                   help="Cap episodes per cell from manifest (0 = all).")
    p.add_argument("--num_cams", type=int, default=2,
                   help="Camera views stacked into the latent (LIBERO=2, DROID=3).")
    p.add_argument("--height", type=int, default=320,
                   help="Per-camera pixel height (LIBERO=320, DROID=192).")
    p.add_argument("--width", type=int, default=320,
                   help="Per-camera pixel width (LIBERO=320, DROID=320).")
    p.add_argument("--down_sample", type=int, default=1,
                   help="Native-state-rate / latent-rate ratio (LIBERO=1, DROID=3; "
                        "see load_full_episode docstring).")
    p.add_argument("--native_fps_default", type=int, default=20,
                   help="Fallback native fps when the annotation JSON has no 'fps' key "
                        "(LIBERO collected data always has one; DROID's droid_ctrl_world "
                        "never does — pass e.g. 5 there so the restride below is a no-op).")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suites = ([s.strip() for s in a.suites.split(",") if s.strip()]
              if a.suites else list_suites(a.data_root))
    out_root = Path(a.output_dir) if a.output_dir else Path(a.checkpoint).resolve().parent / "replay"

    model, pipeline_cls, args, use_uq = load_crtl_world(
        a.checkpoint,
        svd_model_path=a.svd_model_path, clip_model_path=a.clip_model_path,
        data_root=a.data_root, stat_root=a.stat_root, suites=suites, device=device,
        predict_uncertainty=a.predict_uncertainty, uq_vis_t_targets=a.uq_vis_t_targets,
        tag="libero_replay",
        num_cams=a.num_cams, height=a.height, width=a.width, down_sample=a.down_sample,
    )
    pipeline = model.pipeline

    max_chunks = a.max_chunks if a.max_chunks > 0 else None

    # Pre-load action normalization stats for all suites
    suite_stats: dict[str, tuple] = {
        suite: _load_stat(a.stat_root, suite) for suite in suites
    }

    # Build flat episode list as (suite, ep_id, cell|None), optionally from manifest
    if a.manifest:
        from collections import deque
        _CELLS = ["high_var/large_data", "high_var/small_data",
                  "low_var/large_data",  "low_var/small_data"]
        _manifest = json.loads(Path(a.manifest).read_text())
        _queues: dict = {}
        for _cell in _CELLS:
            _eps = _manifest["cells"].get(_cell, [])
            if a.max_episodes_per_cell > 0:
                _eps = _eps[:a.max_episodes_per_cell]
            _queues[_cell] = deque(_eps)
        episode_list: list[tuple] = []
        while any(_queues.values()):
            for _cell in _CELLS:
                if _queues[_cell]:
                    _ep = _queues[_cell].popleft()
                    episode_list.append((_ep["suite"], _ep["episode_id"], _cell))
        print(f"[replay] manifest: {len(episode_list)} episodes across {len(_CELLS)} cells")
    else:
        episode_list = []
        for suite in suites:
            ep_ids = ([a.episode_id] if a.episode_id
                      else list_episode_ids(a.data_root, suite, a.split))
            if a.num_episodes > 0:
                ep_ids = ep_ids[:a.num_episodes]
            for ep_id in ep_ids:
                episode_list.append((suite, ep_id, None))

    # Open streaming JSONL for per-chunk metrics (append mode so resuming works)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "chunk_metrics.jsonl"
    _jsonl_fh = open(jsonl_path, "a")

    def _make_on_chunk(suite: str, ep_id: str, cell):
        """Return per-chunk callback; streams one JSONL record per autoregressive chunk."""
        def _on_chunk(chunk_idx: int, frame_now: int, chunk_frame_indices: list,
                      chunk_lv, chunk_vel, chunk_epi_lv, chunk_epi_vel,
                      overlap_latent_mse: float | None = None,
                      full_latent_mse: float | None = None) -> None:
            agg = compute_uq_metrics(chunk_lv, chunk_vel, chunk_epi_lv, chunk_epi_vel)
            if overlap_latent_mse is not None:
                agg["epi_overlap_latent_mse"] = overlap_latent_mse
            if full_latent_mse is not None:
                agg["epi_full_latent_mse"] = full_latent_mse
            uq_t: dict = {}
            if chunk_lv and args.uq_vis_t_targets:
                n_ode = len(chunk_lv)
                _t_arr = np.linspace(0.0, 1.0, n_ode + 1)[:-1]
                _idxs = sorted({int(np.argmin(np.abs(_t_arr - t)))
                                 for t in args.uq_vis_t_targets})
                for si in _idxs:
                    t_tag = f"t{float(_t_arr[si]):.1f}"
                    t_met = compute_uq_metrics(
                        [chunk_lv[si]],
                        [chunk_vel[si]]     if chunk_vel     else [],
                        [chunk_epi_lv[si]]  if chunk_epi_lv  else [],
                        [chunk_epi_vel[si]] if chunk_epi_vel else [],
                    )
                    for k, v in t_met.items():
                        uq_t[f"{t_tag}_{k}"] = v
            record = {
                "suite": suite, "episode": ep_id, "uncertainty_cell": cell,
                "chunk_idx": chunk_idx, "frame_now": frame_now,
                **agg, **uq_t,
            }
            _jsonl_fh.write(json.dumps(record) + "\n")
            _jsonl_fh.flush()
        return _on_chunk

    summary = []
    for suite, ep_id, cell in episode_list:
        p01, p99 = suite_stats[suite]
        latent_gt, action_native, text, native_fps = load_full_episode(
            a.data_root, suite, ep_id, args, a.split,
            down_sample=a.down_sample, native_fps_default=a.native_fps_default)

        # Stride 20 Hz -> target_hz (5 Hz) so spacing matches how the WM was trained.
        stride = max(1, round(native_fps / a.target_hz))
        latent5 = latent_gt[::stride]
        action5 = normalize_actions(action_native[::stride], p01, p99)
        print(f"[replay] {suite}/{ep_id}: {latent_gt.shape[0]}@{native_fps}Hz "
              f"-> {latent5.shape[0]}@{a.target_hz}Hz  {text!r}")

        try:
            (gt_stack, pred_stack, idxs,
             logvar_by_step, vel_by_step,
             epi_pred_stack, epi_logvar_by_step, epi_vel_by_step) = replay_episode(
                model, pipeline, pipeline_cls, args, latent5, action5, text, device,
                a.num_inference_steps, a.skip, max_chunks,
                use_uq=use_uq, uq_epi_mode=a.uq_epi_mode, epi_overlap_k=a.epi_overlap_k,
                overlap_zero_action=a.overlap_zero_action,
                on_chunk=_make_on_chunk(suite, ep_id, cell))
        except RuntimeError as e:
            print(f"[replay]   skipped: {e}")
            continue

        gt_vid = decode_per_cam(gt_stack, pipeline, args)
        pred_vid = decode_per_cam(pred_stack, pipeline, args)
        # Pass-2's own decoded video (e.g. future_overlap's same-range re-prediction).
        # One extra VAE decode when has_epi -- same cost class as pred_vid's decode above.
        epi_vid = decode_per_cam(epi_pred_stack, pipeline, args) if epi_pred_stack is not None else None

        # ODE step indices closest to each t-target
        uq_indices: list[int] = []
        ref_steps = logvar_by_step or epi_logvar_by_step
        if use_uq and ref_steps:
            n_ode = len(ref_steps)
            step_t_values = np.linspace(0.0, 1.0, n_ode + 1)[:-1]
            uq_indices = sorted({
                int(np.argmin(np.abs(step_t_values - t)))
                for t in args.uq_vis_t_targets
            })

        has_epi = a.uq_epi_mode != "none" and epi_pred_stack is not None
        has_vel = bool(vel_by_step and epi_vel_by_step)
        # alea2 (pass-2's own raw exp(logvar) heatmap) only needs epi_logvar_by_step --
        # independent of has_vel, unlike the four diff renderers below which also need velocity.
        has_epi_alea = has_epi and use_uq and bool(epi_logvar_by_step)
        use_compare = (use_uq and logvar_by_step) or has_epi

        # Aggregate UQ tensors across all ODE steps for the extra summary row/metrics.
        agg_logvar     = torch.stack(logvar_by_step,     0).mean(0) if logvar_by_step     else None
        agg_vel        = torch.stack(vel_by_step,        0).mean(0) if vel_by_step        else None
        agg_epi_logvar = torch.stack(epi_logvar_by_step, 0).mean(0) if epi_logvar_by_step else None
        agg_epi_vel    = torch.stack(epi_vel_by_step,    0).mean(0) if epi_vel_by_step    else None

        if use_compare:
            # Each t-target is one horizontal row; rows are stacked vertically.
            # Row layout: GT | Pred | Pred2(*) | alea | alea2(**) | epi_var_signed(***)
            #             [| ltv | epi_var | pdf_diff | kl]
            #   (*)   Pred2 = pass-2's own decoded video; present whenever has_epi (no UQ head required).
            #   (**)  alea2 = pass-2's own raw exp(epi_logvar) heatmap; present when has_epi_alea
            #         (requires use_uq + epi_logvar_by_step; independent of has_vel, unlike the
            #         four diff columns below which additionally need velocity).
            #   (***) epi_var_signed = exp(lv2)-exp(lv1), UNCLAMPED, diverging colormap (warm=pass-2
            #         MORE uncertain, cool=pass-2 MORE confident) -- see _render_epi_var_signed_video.
            #         Same gate as alea2; the "epi_var" column below is the original one-directional
            #         (clamped) version, kept for the degradation-style modes it was designed for.
            rows = []
            for step_idx in uq_indices:
                row_cols = [gt_vid, pred_vid]
                if epi_vid is not None:
                    row_cols.append(epi_vid)
                row_cols.append(_render_uq_video(logvar_by_step[step_idx], args))
                if has_epi_alea:
                    row_cols.append(_render_uq_video(epi_logvar_by_step[step_idx], args))
                    row_cols.append(_render_epi_var_signed_video(
                        logvar_by_step[step_idx], epi_logvar_by_step[step_idx], args))
                if has_epi and use_uq and has_vel and epi_logvar_by_step:
                    row_cols += [
                        _render_vel_ltv_video(
                            vel_by_step[step_idx], epi_vel_by_step[step_idx], args),
                        _render_epi_var_video(
                            logvar_by_step[step_idx], epi_logvar_by_step[step_idx], args),
                        _render_pdf_diff_video(
                            logvar_by_step[step_idx], epi_logvar_by_step[step_idx], args),
                        _render_kl_video(
                            vel_by_step[step_idx], logvar_by_step[step_idx],
                            epi_vel_by_step[step_idx], epi_logvar_by_step[step_idx], args),
                    ]
                rows.append(np.concatenate(row_cols, axis=2))

            # Extra row: aggregated (mean across all ODE steps)
            if agg_logvar is not None:
                agg_cols = [gt_vid, pred_vid]
                if epi_vid is not None:
                    agg_cols.append(epi_vid)
                agg_cols.append(_render_uq_video(agg_logvar, args))
                if has_epi_alea and agg_epi_logvar is not None:
                    agg_cols.append(_render_uq_video(agg_epi_logvar, args))
                    agg_cols.append(_render_epi_var_signed_video(agg_logvar, agg_epi_logvar, args))
                if (has_epi and use_uq and agg_vel is not None
                        and agg_epi_logvar is not None and agg_epi_vel is not None):
                    agg_cols += [
                        _render_vel_ltv_video(agg_vel, agg_epi_vel, args),
                        _render_epi_var_video(agg_logvar, agg_epi_logvar, args),
                        _render_pdf_diff_video(agg_logvar, agg_epi_logvar, args),
                        _render_kl_video(agg_vel, agg_logvar, agg_epi_vel, agg_epi_logvar, args),
                    ]
                rows.append(np.concatenate(agg_cols, axis=2))

            if rows:
                T_vid = rows[0].shape[0]
                h_sep = np.full((T_vid, 3, rows[0].shape[2], 3), 200, dtype=np.uint8)
                composite = rows[0]
                for row in rows[1:]:
                    composite = np.concatenate([composite, h_sep, row], axis=1)
                write_video(composite, out_root / "compare" / f"{suite}_{ep_id}.mp4", a.video_fps)
            else:
                # no uq_indices (e.g. uq disabled, epi-only mode) -- still show pass-2's own
                # decoded video here since it doesn't require a UQ head, unlike alea/alea2/diffs.
                fallback_cols = [gt_vid, pred_vid]
                if epi_vid is not None:
                    fallback_cols.append(epi_vid)
                composite = np.concatenate(fallback_cols, axis=2)
                write_video(composite, out_root / "compare" / f"{suite}_{ep_id}.mp4", a.video_fps)
        else:
            write_video(gt_vid, out_root / "gt" / f"{suite}_{ep_id}.mp4", a.video_fps)
            write_video(pred_vid, out_root / "pred" / f"{suite}_{ep_id}.mp4", a.video_fps)

        latent_mse = float(((gt_stack - pred_stack) ** 2).mean())
        pixel_mse = float(np.mean((gt_vid / 255.0 - pred_vid / 255.0) ** 2))
        psnr = float(10 * np.log10(1.0 / max(pixel_mse, 1e-12)))
        uq_metrics = compute_uq_metrics(
            logvar_by_step, vel_by_step, epi_logvar_by_step, epi_vel_by_step)

        # Per-t-target UQ metrics: one compute_uq_metrics call per ODE step
        # index closest to each uq_vis_t_targets value (same indices used for vis).
        uq_t_metrics: dict[str, float] = {}
        if uq_indices and logvar_by_step:
            n_ode = len(logvar_by_step)
            step_t_vals = np.linspace(0.0, 1.0, n_ode + 1)[:-1]
            for step_idx in uq_indices:
                t_val = float(step_t_vals[step_idx])
                t_tag = f"t{t_val:.1f}"
                t_met = compute_uq_metrics(
                    [logvar_by_step[step_idx]],
                    [vel_by_step[step_idx]]         if vel_by_step         else [],
                    [epi_logvar_by_step[step_idx]]  if epi_logvar_by_step  else [],
                    [epi_vel_by_step[step_idx]]     if epi_vel_by_step     else [],
                )
                for k, v in t_met.items():
                    uq_t_metrics[f"{t_tag}_{k}"] = v

        uq_str = "  ".join(f"{k}={v:.4f}" for k, v in uq_metrics.items())
        print(f"[replay]   {len(idxs)} frames  latent-MSE={latent_mse:.4f}  PSNR={psnr:.2f}dB"
              + (f"  {uq_str}" if uq_str else ""))
        summary.append({
            "suite": suite, "episode": ep_id, "uncertainty_cell": cell, "frames": len(idxs),
            "latent_mse": latent_mse, "pixel_mse": pixel_mse, "psnr_db": psnr, "text": text,
            **uq_metrics,      # aggregated (mean over all ODE steps)
            **uq_t_metrics,    # per-t: t0.9_mean_aleatoric_var, t0.5_mean_epi_ltv, ...
        })

    _jsonl_fh.close()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    use_compare_any = use_uq or a.uq_epi_mode != "none"
    vid_dir = "compare" if use_compare_any else "{gt,pred}"
    print(f"[replay] wrote {len(summary)} episodes -> {out_root}/{vid_dir}/  (+ replay_summary.json)")


if __name__ == "__main__":
    main()
