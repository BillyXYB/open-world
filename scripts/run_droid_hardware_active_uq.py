"""Active, epistemic-UQ-guided Pi0.5 rollouts on real DROID hardware.

Server-side counterpart to ``uq_data_collection/examples/droid/main.py``
(the robot side is NOT modified by this script -- it keeps talking to
whatever writes the same comms-dir files it always has). At each
replanning step (every interaction round-trip with the robot) this script:

  1. samples ``num_candidates`` candidate action chunks from the Pi0.5
     policy,
  2. gets each candidate's future joint positions / gripper / FK cartesian
     pose *kinematically* via ``OpenPIActionAdapter`` (policy joint-velocity
     chunk -> Dynamics MLP -> forward kinematics) -- no physical robot
     motion happens during scoring, since real hardware can't be "rewound"
     the way ``scripts/run_data_collection_active_uq.py`` rewinds LIBERO sim
     state,
  3. scores each candidate's EPISTEMIC uncertainty with a UQ-trained world
     model (``CrtlWorld``) via the "iterative" self-consistency mechanism
     (compare a prediction conditioned on real observed history against one
     conditioned on the model's own compounding self-predicted history),
  4. sends back the candidate that maximizes the configured UQ metric.

Collected trajectories are also exported via
``openworld.utils.droid_export.write_droid_episode`` into the same on-disk
schema the real ``droid_ctrl_world`` training set uses, so they are directly
usable as further world-model training data.

Usage:
    uv run python scripts/run_droid_hardware_active_uq.py \\
        --config configs/collection/droid_hardware_active_uq.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess_libero_for_wm import LatentEncoder, write_sample_list  # noqa: E402
from replay_libero_wm_traj import (  # noqa: E402
    build_frame_ids,
    compute_uq_metrics,
    decode_per_cam,
    load_crtl_world,
    normalize_actions,
    write_video,
    _render_epi_var_video,
    _render_kl_video,
    _render_pdf_diff_video,
    _render_uq_video,
    _render_vel_ltv_video,
)

from openworld.policies.openpi_action_adapter import (  # noqa: E402
    AdaptedActionChunk,
    OpenPIActionAdapter,
    get_fk_solution,
)
from openworld.policies.openpi_loader import load_policy_from_checkpoint  # noqa: E402
from openworld.policies.molmoact2_client import MolmoAct2Client  # noqa: E402
from openworld.training.world_model.dataset import _load_stat  # noqa: E402
from openworld.utils.droid_comms import (  # noqa: E402
    FileChannel,
    clear_comms_dir,
    cleanup_trajectory_files,
)
from openworld.utils.droid_comms_socket import serve_trajectories  # noqa: E402
from openworld.utils.droid_export import next_episode_id, write_droid_episode  # noqa: E402
from openworld.utils.io import load_yaml  # noqa: E402

logger = logging.getLogger(__name__)

UQ_METRIC_CHOICES = (
    "mean_aleatoric_var", "max_aleatoric_var",
    "mean_epi_ltv", "max_epi_ltv",
    "mean_epi_var", "mean_pdf_diff", "mean_kl",
)

# Pass-2 (epistemic self-consistency) conditioning strategy. Each mode must
# match how the loaded wm_checkpoint was trained -- see the corresponding
# comment in configs/collection/droid_hardware_active_uq.yaml for the
# checkpoint <-> mode pairing table. Ported from
# replay_libero_wm_traj.py::replay_episode's offline dispatch (lines
# ~269-330 there) into this script's batched-over-candidates, streaming
# (unknown trajectory length) live rollout -- see _build_pass2_inputs.
UQ_EPI_MODE_CHOICES = (
    "none", "zero_history", "flat_history", "single_history", "iterative", "future_overlap",
)
# Only these two metrics are computable with no pass-2 (uq_epi_mode="none").
PASS1_ONLY_METRICS = ("mean_aleatoric_var", "max_aleatoric_var")

# Fixed camera order for the world model's 3-camera vertical latent stack --
# matches droid_ctrl_world's training-data convention. The policy's single
# "exterior" view is a separate, independently configured camera (see
# hardware.external_camera), not necessarily either endpoint of this order.
WM_CAM_KEYS = ("left_image", "right_image", "wrist_image")
WM_TRAIN_SUITE = "droid_ctrl_world"


# ---------------------------------------------------------------------------
# Kinematics / state helpers
# ---------------------------------------------------------------------------


def _fk_pose(joint_position: np.ndarray, gripper_position: float) -> np.ndarray:
    """FK-derived (xyz, euler, gripper) pose -- (7,) float32.

    Matches svd_ac_video_model's
    ``uq_data_collection.py::get_robot_state_from_joint_pos`` exactly (same
    ``get_fk_solution`` + euler convention), so history frames fed to the
    world model use the identical pose representation the training data and
    ``OpenPIActionAdapter`` both use -- NOT the robot's own raw
    ``cartesian_position`` field, whose convention isn't guaranteed to match.
    """
    fk = get_fk_solution(np.asarray(joint_position, dtype=np.float64)[:7])
    xyz = fk[:3, 3]
    euler = _Rotation.from_matrix(fk[:3, :3]).as_euler("xyz")
    return np.concatenate([xyz, euler, [gripper_position]], axis=0).astype(np.float32)


def _encode_current_latent(
    obs: dict, encoder: LatentEncoder, num_cams: int, per_cam_h: int, latent_w: int,
) -> torch.Tensor:
    latent = torch.zeros((4, num_cams * per_cam_h, latent_w), dtype=torch.float32)
    for cam_idx, key in enumerate(WM_CAM_KEYS[:num_cams]):
        img = np.ascontiguousarray(obs[key])
        cam_latent = encoder.encode(img[None])[0]
        latent[:, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = cam_latent
    return latent


# ---------------------------------------------------------------------------
# Policy inference + kinematic (no-rewind) candidate lookahead
# ---------------------------------------------------------------------------


def _policy_infer_raw_chunk(policy, obs: dict, external_camera: str, prompt: str) -> np.ndarray:
    """One stochastic ``policy.infer()`` call -> raw (>=8, 8) joint-velocity
    + gripper chunk. Mirrors uq_data_collection.py::forward_policy's
    ``example`` dict exactly (same openpi observation keys)."""
    from openpi_client import image_tools

    ext_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs[f"{external_camera}_image"], 224, 224))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs["wrist_image"], 224, 224))
    joint_position = np.asarray(obs["joint_position"], dtype=np.float32)[:7]
    gripper_position = np.atleast_1d(
        np.asarray(obs["gripper_position"], dtype=np.float32)).reshape(-1)[:1]
    element = {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    return np.asarray(policy.infer(element)["actions"])


def _generate_candidate_openpi(
    policy, action_adapter: OpenPIActionAdapter, obs: dict,
    external_camera: str, prompt: str, wire_skip: int, wire_len: int,
) -> tuple[AdaptedActionChunk, np.ndarray, list[int]]:
    """Sample one candidate: raw policy chunk -> kinematic (no robot motion)
    lookahead via the Dynamics MLP + forward kinematics -> the fixed 5-step
    slice (stride ``wire_skip``, ``wire_len`` steps) reused for BOTH the
    world model's future-conditioning targets and the robot wire format --
    same stride-2/take-5 convention as
    uq_data_collection.py::integrate_action_vel's ``action_key="joint_pos"``
    path, the one actually in production today.

    Returns (full 15-step AdaptedActionChunk, the (wire_len, 7) xyz+euler+
    gripper future-pose slice used for WM conditioning).
    """
    raw_chunk = _policy_infer_raw_chunk(policy, obs, external_camera, prompt)
    joint_position = np.asarray(obs["joint_position"], dtype=np.float32)[:7]
    gripper_position = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
    adapted = action_adapter.adapt(joint_position, gripper_position, raw_chunk)

    idx = list(range(0, adapted.env_actions.shape[0], wire_skip))[:wire_len]
    future_pose = adapted.env_actions[idx]  # (wire_len, 7) xyz+euler+gripper
    return adapted, future_pose, idx


def _ma2_chunk_to_adapted(
    ma2_chunk: np.ndarray, joint_position: np.ndarray, gripper_position: float,
    wire_skip: int, wire_len: int, gripper_scale: float,
) -> tuple[AdaptedActionChunk, np.ndarray, list[int]]:
    """MolmoAct2 ``(T>=9, 8)`` absolute joint-position + gripper chunk ->
    ``AdaptedActionChunk``.

    Two consumers, deliberately NOT the same rows:
      * ``future_pose`` (WM conditioning) -- the checkpoint expects frame 0 to
        be the CURRENT measured pose, then the model's future at stride
        ``wire_skip``; built from the current-prepended stack's ``env_actions``.
      * the wire action (``_wire_action`` -> ``adapted.joint_positions``) -- the
        robot's real per-tick joint targets, ``ma2[0..]`` DENSE with NO
        prepended no-op tick (unlike openpi, whose adapter genuinely produces
        the current pose as its step 0).

    ``gripper_scale`` is applied here so the WM sees the same gripper value the
    robot executes (``examples/droid/main.py`` does ``pred_action_chunk[:,-1] *= 1.5``
    unconditionally before binarizing).
    """
    ma2_chunk = np.asarray(ma2_chunk, dtype=np.float32)
    joint_position = np.asarray(joint_position, dtype=np.float32).reshape(-1)[:7]

    # dense per-tick model targets, gripper-scaled -- what the robot executes
    ma2 = ma2_chunk[:, :8].astype(np.float32).copy()
    ma2[:, 7] = np.clip(ma2[:, 7] * float(gripper_scale), 0.0, 1.0)

    # current-prepended stack, used ONLY for the WM's future_pose conditioning.
    # cur_row's gripper is scaled the same way (matches the pre-refactor
    # behaviour of scaling the whole stack -- keeps future_pose byte-identical).
    cur_row = np.concatenate(
        [joint_position, [np.clip(float(gripper_position) * float(gripper_scale), 0.0, 1.0)]]
    ).astype(np.float32)
    stack = np.concatenate([cur_row[None, :], ma2], axis=0)  # (T+1, 8)
    idx = list(range(0, stack.shape[0], wire_skip))[:wire_len]
    if len(idx) < wire_len:
        raise ValueError(
            f"MolmoAct2 chunk too short: stacked {stack.shape[0]} rows, "
            f"need stride-{wire_skip} x {wire_len}")
    stack_env = np.stack(
        [_fk_pose(stack[r, :7], float(stack[r, 7])) for r in range(stack.shape[0])], axis=0
    ).astype(np.float32)  # (T+1, 7) xyz+euler+gripper

    adapted = AdaptedActionChunk(
        env_actions=stack_env,                              # (T+1, 7) -- indexed by idx for future_pose
        joint_positions=ma2[:, :7].astype(np.float32),      # (T, 7)   -- dense, NO current-prepend
        gripper_positions=ma2[:, 7:8].astype(np.float32),   # (T, 1)
    )
    future_pose = adapted.env_actions[idx]  # (wire_len, 7); row 0 == _fk_pose(current)
    return adapted, future_pose, idx


def _generate_candidates(
    policy, action_adapter, obs: dict, external_camera: str, prompt: str,
    wire_skip: int, wire_len: int, num_candidates: int, backend: str,
) -> list[tuple[AdaptedActionChunk, np.ndarray, list[int]]]:
    """``num_candidates`` ``(AdaptedActionChunk, future_pose, idx)`` tuples --
    backend-agnostic shape consumed by the rest of the rollout loop.

    openpi: ``num_candidates`` independent stochastic ``policy.infer`` calls.
    molmoact2: ONE ``policy.infer_candidates`` round-trip -> ``(N, T>=9, 8)``
    absolute joint-position chunks; each -> ``_ma2_chunk_to_adapted``.
    """
    if backend == "openpi":
        return [
            _generate_candidate_openpi(policy, action_adapter, obs, external_camera,
                                       prompt, wire_skip, wire_len)
            for _ in range(num_candidates)
        ]
    if backend == "molmoact2":
        chunks = policy.infer_candidates(obs, external_camera, prompt, num_candidates)
        joint_position = np.asarray(obs["joint_position"], dtype=np.float32)[:7]
        gripper_position = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
        gscale = float(getattr(policy, "gripper_scale", 1.0))
        return [
            _ma2_chunk_to_adapted(chunks[i], joint_position, gripper_position,
                                  wire_skip, wire_len, gscale)
            for i in range(num_candidates)
        ]
    raise ValueError(f"unknown policy backend {backend!r}")


def _wire_action(adapted: AdaptedActionChunk, exec_len: int) -> np.ndarray:
    """(exec_len, 8) joint_positions + gripper -- the DENSE, un-strided
    prefix of the chunk (every raw tick, real MolmoAct2/adapted-openpi
    predictions, no interpolation) up to the last raw tick the WM actually
    scored (``exec_len = idx[-1] + 1`` at the call site). This is what
    ``uq_data_collection.py::send_action``'s production ``action_key=
    "joint_pos"`` path sends (future joint positions, NOT cartesian
    velocity, despite the root CLAUDE.md's description) -- previously a
    ``wire_skip``-strided slice of length ``wire_len``; now the robot
    executes the real per-tick trajectory natively at 15Hz instead of
    snapping through strided waypoints at native rate (which commanded
    roughly wire_skip x the intended joint velocity)."""
    joint_pos = adapted.joint_positions[:exec_len]
    grip_pos = adapted.gripper_positions[:exec_len]
    return np.concatenate([joint_pos, grip_pos], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Two-pass epistemic scoring (ported from
# scripts/run_data_collection_active_uq.py's LIBERO version; the sim-rewind
# lookahead is replaced by the kinematic candidates generated above)
# ---------------------------------------------------------------------------


def _pipeline_call(
    wm_model, wm_pipeline_cls, wm_args, action_latent,
    image, history, his_cond_zero, num_frames, num_inference_steps,
    generator=None,
):
    return wm_pipeline_cls.__call__(
        wm_model.pipeline, image=image, text=action_latent,
        width=wm_args.width, height=int(wm_args.num_cams * wm_args.height),
        num_frames=num_frames, history=history,
        num_inference_steps=num_inference_steps,
        decode_chunk_size=wm_args.decode_chunk_size,
        max_guidance_scale=wm_args.guidance_scale, fps=wm_args.fps,
        motion_bucket_id=wm_args.motion_bucket_id, mask=None,
        output_type="latent", return_dict=False,
        frame_level_cond=wm_args.frame_level_cond, his_cond_zero=his_cond_zero,
        flow_map_type=wm_args.flow_map_type, flow_map_loss_type=wm_args.flow_map_loss_type,
        return_uncertainty=True, generator=generator,
    )


def _save_decision_viz(
    viz_path_prefix, i_star: int, pred_latents1, pred_latents2,
    logvar_steps1, vel_steps1, logvar_steps2, vel_steps2,
    pipeline, wm_args, fps: int,
) -> None:
    """Per-decision candidate-comparison videos -- verbatim port of
    scripts/run_data_collection_active_uq.py's ``_save_decision_viz``, minus
    the LIBERO-specific agentview/wrist camera-name split (DROID has 3 cams,
    so this writes one video per WM camera slot instead of two named ones)."""
    num_candidates = pred_latents1.shape[0]
    n_ode = len(logvar_steps1)
    col_order = [i_star] + [i for i in range(num_candidates) if i != i_star]
    per_cam_h = wm_args.height

    def _agg(steps, i):
        return torch.stack([steps[s][i] for s in range(n_ode)], 0).mean(0)

    signal_cols: dict[str, list] = {
        name: [] for name in ("pred1", "pred2", "alea", "ltv", "epivar", "pdfdiff", "kl")
    }
    for i in col_order:
        signal_cols["pred1"].append(decode_per_cam(pred_latents1[i].float().cpu(), pipeline, wm_args))
        signal_cols["pred2"].append(decode_per_cam(pred_latents2[i].float().cpu(), pipeline, wm_args))
        lv1, vel1 = _agg(logvar_steps1, i), _agg(vel_steps1, i)
        lv2, vel2 = _agg(logvar_steps2, i), _agg(vel_steps2, i)
        signal_cols["alea"].append(_render_uq_video(lv1, wm_args))
        signal_cols["ltv"].append(_render_vel_ltv_video(vel1, vel2, wm_args))
        signal_cols["epivar"].append(_render_epi_var_video(lv1, lv2, wm_args))
        signal_cols["pdfdiff"].append(_render_pdf_diff_video(lv1, lv2, wm_args))
        signal_cols["kl"].append(_render_kl_video(vel1, lv1, vel2, lv2, wm_args))

    num_frames = signal_cols["pred1"][0].shape[0]
    for cam_idx in range(wm_args.num_cams):
        cam_slice = slice(cam_idx * per_cam_h, (cam_idx + 1) * per_cam_h)
        signal_rows = [
            np.concatenate([frames[:, cam_slice] for frames in signal_cols[key]], axis=2)
            for key in ("pred1", "pred2", "alea", "ltv", "epivar", "pdfdiff", "kl")
        ]
        h_sep = np.full((num_frames, 3, signal_rows[0].shape[2], 3), 200, dtype=np.uint8)
        composite = signal_rows[0]
        for row in signal_rows[1:]:
            composite = np.concatenate([composite, h_sep, row], axis=1)
        write_video(composite, pathlib.Path(f"{viz_path_prefix}_cam{cam_idx}.mp4"), fps)


def _build_pass2_inputs(
    *, uq_epi_mode: str, epi_overlap_k: int,
    history_latents: torch.Tensor, current_latent: torch.Tensor,
    action_batch: torch.Tensor, action_latent: torch.Tensor,
    pred_latents1: torch.Tensor,
    rolled2: Optional[list], hist_cur_fn,
    num_history: int, num_frames: int,
    texts: list, wm_model, wm_args, device,
    overlap_zero_action: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Build (history2, current2, his_cond_zero2, action_latent2) for one of
    the 5 non-"none" uq_epi_mode strategies. Batched-over-candidates,
    streaming analogue of replay_libero_wm_traj.py::replay_episode's pass-2
    dispatch (lines ~275-304 there); never called for uq_epi_mode=="none".
    All tensors here carry a leading `num_candidates` batch dim (this
    script's convention), vs. replay's batch-of-1.

    NOT called for "iterative" unless `rolled2` is populated by the caller --
    see _rollout_trajectory_hardware's conditional `rolled2` allocation.
    """
    action_latent2 = action_latent  # unchanged for every mode except future_overlap

    if uq_epi_mode == "zero_history":
        # Reuse pass-1's real history/current verbatim; the degradation is
        # the model's own internal "zero out history conditioning" flag.
        return history_latents, current_latent, True, action_latent2

    if uq_epi_mode == "flat_history":
        # Repeat the CURRENT ("now") frame across all history slots -- zero
        # temporal offset anywhere in pass-2's conditioning.
        history2 = current_latent.unsqueeze(1).expand(-1, num_history, -1, -1, -1).contiguous()
        return history2, current_latent, False, action_latent2

    if uq_epi_mode == "single_history":
        # Repeat the LAST (most recent) real history slot -- a single real,
        # slightly-stale reference frame, not the current frame itself.
        # Matches droid_flow_matching_uq_single_hist_v1's training augmentation.
        last_hist = history_latents[:, -1:, :, :, :]
        history2 = last_hist.expand(-1, num_history, -1, -1, -1).contiguous()
        return history2, current_latent, False, action_latent2

    if uq_epi_mode == "future_overlap":
        # Needs pass-1's OUTPUT (pred_latents1) -- caller must run pass-1
        # first and pass its result in here before pass-2 can be built.
        overlap_k_used = epi_overlap_k if epi_overlap_k > 0 else (num_frames - 1)
        overlap_k_used = max(1, min(overlap_k_used, num_frames - 1))
        overlap_frames = pred_latents1[:, 1:1 + overlap_k_used].to(
            device=device, dtype=history_latents.dtype)
        history2 = torch.cat([history_latents, overlap_frames], dim=1)
        # action_batch rows [num_history+1 : num_history+1+overlap_k_used] are
        # exactly this candidate's own overlapped FUTURE action rows --
        # already normalized, no extra slicing/re-normalization needed
        # (unlike replay's offline re-slice from action_norm/state_id).
        if overlap_zero_action:
            overlap_action = torch.zeros(
                action_batch.shape[0], overlap_k_used, action_batch.shape[-1],
                dtype=action_batch.dtype, device=action_batch.device)
        else:
            overlap_action = action_batch[:, num_history + 1: num_history + 1 + overlap_k_used, :]
        action2_for_encoder = torch.cat([action_batch, overlap_action], dim=1)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            action_latent2 = wm_model.action_encoder(
                action2_for_encoder, texts, wm_model.tokenizer, wm_model.text_encoder,
                wm_args.frame_level_cond)
        return history2, current_latent, False, action_latent2

    # "iterative": pass-2 conditions on the model's own self-predicted
    # history (rolled2), seeded/advanced by the caller across round-trips.
    assert rolled2 is not None, "rolled2 must be built by the caller when uq_epi_mode == 'iterative'"
    history2, current2 = hist_cur_fn(rolled2)
    return history2, current2, False, action_latent2


# ---------------------------------------------------------------------------
# Per-trajectory rollout
# ---------------------------------------------------------------------------


def _rollout_trajectory_hardware(
    *, traj_idx: int, backend: str, policy, action_adapter, wm_model, wm_pipeline_cls, wm_args,
    p01, p99, encoder: LatentEncoder, channel, external_camera: str,
    num_candidates: int, uq_metric: str, uq_epi_mode: str, epi_overlap_k: int,
    overlap_zero_action: bool,
    num_inference_steps: int,
    wire_skip: int, wire_len: int, max_steps: int,
    obs_timeout_s: float, poll_interval_s: float,
    viz_enabled: bool, viz_every_n_decisions: int, viz_max_decisions: Optional[int],
    viz_fps: int, viz_dir,
    num_cams: int,
    policy_only: bool = False,
) -> tuple[dict, list[dict], str]:
    """Run one trajectory to completion (robot-signalled done or
    ``max_steps`` round-trips). Returns (buffered-episode payload,
    decision_log, instruction text).

    ``policy_only=True`` bypasses the world model entirely (``wm_model``/
    ``wm_pipeline_cls``/``wm_args``/``p01``/``p99`` may all be ``None``):
    MolmoAct2/openpi candidates are still generated every round trip (so the
    policy + comms/execution pipeline still get exercised end-to-end), but
    the FIRST proposed candidate is sent as-is, no WM scoring/UQ metrics/
    candidate-comparison viz. See the per-round-trip if/else below.
    """
    # num_cams is a plain param now (not wm_args.num_cams) since the
    # save-buffer accumulation loop below needs it regardless of whether the
    # WM (and therefore wm_args) is loaded at all.
    rolled: list[torch.Tensor] = []
    # Only "iterative" mode needs a self-predicted history buffer; other
    # modes derive pass-2's conditioning directly from `rolled`/`current`
    # each step (see _build_pass2_inputs), so skip the bookkeeping for them
    # -- mirrors replay_libero_wm_traj.py's `rolled2 = [...] if mode=="iterative" else None`.
    rolled2: Optional[list[torch.Tensor]] = [] if uq_epi_mode == "iterative" else None
    real_pose_hist: list[np.ndarray] = []
    real_grip_hist: list[float] = []
    joint_hist: list[np.ndarray] = []
    cam_frames: list[list[np.ndarray]] = [[] for _ in range(len(WM_CAM_KEYS))]
    # Save-only buffers, decoupled from the decision-rate buffers above.
    # Decisions (candidate generation + WM scoring) fire once per round trip
    # (now up to wire_skip*(wire_len-1)+1 raw ticks apart, per _wire_action's
    # dense execution), but training data should stay ~wire_skip-tick-spaced
    # regardless -- so these accumulate from EVERY entry of each round
    # trip's received obs_list (a wire_skip-strided window spanning the
    # whole elapsed period, see examples/droid/main.py's send-side slicing),
    # not just obs_list[-1]. See write_droid_episode's payload below.
    save_pose_hist: list[np.ndarray] = []
    save_grip_hist: list[float] = []
    save_joint_hist: list[np.ndarray] = []
    save_cam_frames: list[list[np.ndarray]] = [[] for _ in range(len(WM_CAM_KEYS))]
    instruction = ""
    decision_log: list[dict] = []
    # Self-consistency handoff for `rolled2`: unlike
    # run_data_collection_active_uq.py (which decides once every
    # (num_frames-1) real steps and drains a multi-item queue over the
    # steps in between), THIS loop makes a fresh decision every single
    # round-trip. So each decision only ever needs to seed rolled2's NEXT
    # single entry -- with pass-1's k=1 (one-round-trip-ahead) prediction,
    # since build_frame_ids's k=0 slot is a re-prediction of the CURRENT
    # (already ground-truth) frame, not a future one. Only None for the very
    # first round-trip (t_step==0), before any decision has fired yet --
    # rolled2's first entry then falls back to ground truth, like rolled's.
    next_rolled2_seed: Optional[torch.Tensor] = None

    t_step = 0
    while t_step < max_steps:
        result = channel.poll_observation(
            t_step, timeout_s=obs_timeout_s, poll_interval_s=poll_interval_s)
        if result.get("done"):
            break
        obs_list = result["obs"]
        obs = obs_list[-1]  # most recent of the strided window (see save-buffer loop below)
        instruction = str(obs.get("instruction", instruction))

        # Save-buffer accumulation: every entry in the received window (the
        # robot sends a wire_skip-strided slice spanning the whole elapsed
        # round-trip period, not just the newest tick), so this stays
        # ~wire_skip-tick-spaced and gapless across round trips regardless
        # of how long the round-trip period is. Does NOT feed rolled/
        # real_pose_hist/joint_hist/cam_frames below -- those stay
        # decision-rate (obs_list[-1] only), unchanged.
        for save_obs in obs_list:
            for cam_idx, key in enumerate(WM_CAM_KEYS[:num_cams]):
                save_cam_frames[cam_idx].append(np.ascontiguousarray(save_obs[key]))
            save_joint_position = np.asarray(save_obs["joint_position"], dtype=np.float32)
            save_gripper_position = float(np.asarray(save_obs["gripper_position"]).reshape(-1)[0])
            save_pose = _fk_pose(save_joint_position, save_gripper_position)
            save_pose_hist.append(save_pose[:6])
            save_grip_hist.append(save_pose[6])
            save_joint_hist.append(save_joint_position[:7])

        candidates = _generate_candidates(
            policy, action_adapter, obs, external_camera, instruction,
            wire_skip, wire_len, num_candidates, backend)

        if policy_only:
            # Bypass the world model entirely: take the first proposed
            # candidate as-is. No history/latent bookkeeping, no WM scoring,
            # no UQ metrics, no candidate-comparison viz -- decision_log
            # gets a minimal entry so decision_metrics.jsonl / the done-log
            # line downstream still work (see _process_trajectory's
            # policy_only guards on candidate_metrics-dependent fields).
            i_star = 0
            decision_idx = len(decision_log)
            decision_log.append({
                "decision_idx": decision_idx, "t_step": t_step,
                "policy_only": True, "chosen": i_star,
            })
        else:
            for cam_idx, key in enumerate(WM_CAM_KEYS[:num_cams]):
                cam_frames[cam_idx].append(np.ascontiguousarray(obs[key]))
            joint_position = np.asarray(obs["joint_position"], dtype=np.float32)
            gripper_position = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
            pose = _fk_pose(joint_position, gripper_position)
            real_pose_hist.append(pose[:6])
            real_grip_hist.append(pose[6])
            joint_hist.append(joint_position[:7])

            device = next(wm_model.parameters()).device
            per_cam_h, latent_w = wm_args.height // 8, wm_args.width // 8
            latent = _encode_current_latent(obs, encoder, num_cams, per_cam_h, latent_w)
            rolled.append(latent)
            if rolled2 is not None:
                rolled2.append(next_rolled2_seed if next_rolled2_seed is not None else latent.clone())
                next_rolled2_seed = None

            frame_now = len(rolled) - 1
            # WM-scored decision every step, including early ones where real
            # history doesn't fully exist yet -- hist_ids' max(idx_, 0) clipping
            # below repeats the first real frame for any not-yet-real history
            # slot ("flat history"), matching the same fallback the checkpoint
            # was trained to handle (see dataset.py's `skip_his = 0` augmentation
            # and, for the old server, uq_data_collection.py's explicit
            # broadcast_to-based flat-history seeding).
            num_history, num_frames = wm_args.num_history, wm_args.num_frames
            skip, skip_his = 1, 4  # round-trip units; see module-level rationale in the config comments
            rgb_id = build_frame_ids(frame_now, num_history, num_frames, skip, skip_his)
            hist_ids = [max(idx_, 0) for idx_ in rgb_id[:num_history]]
            hist_rows = np.stack(
                [np.concatenate([real_pose_hist[i], [real_grip_hist[i]]]) for i in hist_ids], axis=0)

            action_tensors = []
            for _adapted, future_pose, _idx in candidates:
                full = np.concatenate([hist_rows, future_pose], axis=0)  # (num_history+num_frames, 7)
                action_tensors.append(normalize_actions(full, p01, p99))
            action_batch = torch.tensor(np.stack(action_tensors, axis=0), dtype=torch.float32, device=device)
            texts = [instruction] * num_candidates

            def _hist_cur(buf):
                hist = torch.stack([buf[i] for i in hist_ids], 0).unsqueeze(0).expand(
                    num_candidates, -1, -1, -1, -1).contiguous().to(device)
                cur = buf[frame_now].unsqueeze(0).expand(
                    num_candidates, -1, -1, -1).contiguous().to(device)
                return hist, cur

            history_latents, current_latent = _hist_cur(rolled)

            # Shared seed for pass 1 / pass 2: a fresh torch.Generator per call
            # (not one shared object reused across both __call__s, which would
            # advance its state between calls and defeat the point) so the two
            # passes draw identical initial diffusion noise -- isolating the
            # measured pass1/pass2 divergence to the conditioning difference
            # instead of ordinary sampling variance. See pipeline_flow_map_ctrl_world.py's
            # generator-handling fix. Mode-agnostic: applies uniformly to all
            # uq_epi_mode variants, since it lives at the call sites, not inside
            # _build_pass2_inputs.
            noise_seed = torch.Generator().seed()  # random seed, no side effect on global RNG
            gen1 = torch.Generator(device=device).manual_seed(noise_seed)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                action_latent = wm_model.action_encoder(
                    action_batch, texts, wm_model.tokenizer, wm_model.text_encoder,
                    wm_args.frame_level_cond)
                _, pred_latents1, logvar_steps1, vel_steps1 = _pipeline_call(
                    wm_model, wm_pipeline_cls, wm_args, action_latent,
                    current_latent, history_latents, wm_args.his_cond_zero,
                    num_frames, num_inference_steps, generator=gen1)

                # Pass-1 always runs first: every mode except "future_overlap"
                # could build pass-2's inputs independently of pass-1's output,
                # but "future_overlap" specifically needs pred_latents1 to build
                # its history2 (splices pass-1's own predicted future frames in)
                # -- so this ordering is applied uniformly (a no-op dependency
                # for the other modes, a real one for future_overlap).
                if uq_epi_mode == "none":
                    pred_latents2 = None
                    logvar_steps2, vel_steps2 = [], []
                else:
                    history2_latents, current2_latent, his_cond_zero2, action_latent2 = _build_pass2_inputs(
                        uq_epi_mode=uq_epi_mode, epi_overlap_k=epi_overlap_k,
                        history_latents=history_latents, current_latent=current_latent,
                        action_batch=action_batch, action_latent=action_latent,
                        pred_latents1=pred_latents1, rolled2=rolled2, hist_cur_fn=_hist_cur,
                        num_history=num_history, num_frames=num_frames,
                        texts=texts, wm_model=wm_model, wm_args=wm_args, device=device,
                        overlap_zero_action=overlap_zero_action,
                    )
                    gen2 = torch.Generator(device=device).manual_seed(noise_seed)  # same seed as pass 1
                    _, pred_latents2, logvar_steps2, vel_steps2 = _pipeline_call(
                        wm_model, wm_pipeline_cls, wm_args, action_latent2,
                        current2_latent, history2_latents, his_cond_zero2,
                        num_frames, num_inference_steps, generator=gen2)

            # Independent step counts: uq_epi_mode=="none" leaves logvar_steps2/
            # vel_steps2 empty, which compute_uq_metrics already handles (it
            # guards each pass-2-dependent metric group), but a single shared
            # `n_ode` would IndexError indexing into the empty pass-2 lists.
            n_ode1, n_ode2 = len(logvar_steps1), len(logvar_steps2)
            candidate_metrics = []
            for i in range(num_candidates):
                lv1_i = [logvar_steps1[s][i] for s in range(n_ode1)]
                vel1_i = [vel_steps1[s][i] for s in range(n_ode1)]
                lv2_i = [logvar_steps2[s][i] for s in range(n_ode2)]
                vel2_i = [vel_steps2[s][i] for s in range(n_ode2)]
                candidate_metrics.append(compute_uq_metrics(lv1_i, vel1_i, lv2_i, vel2_i))

            i_star = max(range(num_candidates), key=lambda i: candidate_metrics[i][uq_metric])
            decision_idx = len(decision_log)
            decision_log.append({
                "decision_idx": decision_idx, "t_step": t_step, "uq_metric": uq_metric,
                "chosen": i_star, "candidate_metrics": candidate_metrics,
            })

            if (viz_enabled and uq_epi_mode != "none"
                    and decision_idx % max(1, viz_every_n_decisions) == 0
                    and (viz_max_decisions is None or decision_idx < viz_max_decisions)):
                viz_dir.mkdir(parents=True, exist_ok=True)
                _save_decision_viz(
                    viz_dir / f"decision_{decision_idx:03d}", i_star,
                    pred_latents1, pred_latents2, logvar_steps1, vel_steps1,
                    logvar_steps2, vel_steps2, wm_model.pipeline, wm_args, viz_fps)

            if rolled2 is not None:  # uq_epi_mode == "iterative" only
                # Seed rolled2's entry for the NEXT round-trip with pass-1's
                # one-step-ahead (k=1) prediction from the chosen candidate, so
                # that decision conditions pass 2 on this decision's
                # true-history prediction rather than ground truth --
                # self-consistency. (k=0 is a re-prediction of THIS already-
                # observed frame, not a future one -- see build_frame_ids.)
                next_rolled2_seed = pred_latents1[i_star, 1].float().cpu()

        adapted, _future_pose, idx = candidates[i_star]
        # exec_len = the last raw tick the WM actually scored (idx[-1]) + 1
        # -- the robot executes the real dense chunk up to exactly as far
        # as this decision was verified for, never further into unverified
        # open-loop territory. (policy_only: idx is still the wire_skip-
        # strided slice _generate_candidates always computes -- WM scoring
        # is what's skipped, not this bookkeeping.)
        exec_len = idx[-1] + 1
        action = _wire_action(adapted, exec_len)

        assert action.shape == (exec_len, 8), f"expected ({exec_len}, 8) wire action, got {action.shape}"
        channel.send_action(t_step, action, instruction)

        t_step += 1

    # Sourced from the dense, wire_skip-spaced save buffers (NOT
    # cam_frames/real_pose_hist/joint_hist, which stay decision-rate and
    # feed only the WM's own history/scoring above) -- so the exported
    # episode's temporal density tracks wire_skip regardless of how long
    # the decision round-trip period (open_loop_horizon) is.
    payload = {
        "cam_rgb": [np.stack(frames, axis=0) if frames else np.zeros((0,), dtype=np.uint8)
                    for frames in save_cam_frames],
        "cart": np.stack(save_pose_hist, axis=0) if save_pose_hist else np.zeros((0, 6), np.float32),
        "grip": np.asarray(save_grip_hist, dtype=np.float32),
        "joint_position": np.stack(save_joint_hist, axis=0) if save_joint_hist else np.zeros((0, 7), np.float32),
    }
    return payload, decision_log, instruction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_root", type=str, default=None,
                    help="Override save.output_root from the config.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_yaml(args.config)

    pol_cfg = cfg.get("policy", {})
    hw_cfg = cfg.get("hardware", {})
    uq_cfg = cfg.get("active_uq", {})
    save_cfg = cfg.get("save", {})

    # "file" (default, unchanged behavior) talks to the robot via the
    # comms-dir file protocol (droid_comms.py); "socket" instead holds one
    # persistent websocket connection per trajectory (droid_comms_socket.py),
    # cutting disk/rsync/polling latency -- see the config comment above for
    # how the robot reaches this over an SSH tunnel. Only "file" mode needs
    # comms_dir at all.
    comms_mode = hw_cfg.get("comms_mode", "file")
    if comms_mode not in ("file", "socket"):
        raise SystemExit(f'hardware.comms_mode must be "file" or "socket", got {comms_mode!r}')

    comms_dir = None
    socket_host = socket_port = None
    if comms_mode == "file":
        # Clear any stale obs/pred/done files left behind by a crashed or
        # interrupted previous run BEFORE loading the (slow, ~2 min) policy
        # and world model -- this server always starts traj_idx at 1, so a
        # leftover droid_observations_instructions_1_0.npz from an earlier
        # session would otherwise be misread as this run's first real
        # observation. Mirrors the old pipeline's launcher-side cleanup (see
        # clear_comms_dir's docstring); unlike that one, this runs
        # regardless of how this script is invoked (manual tmux run or sbatch).
        comms_dir = pathlib.Path(hw_cfg["comms_dir"])
        logger.info("Clearing stale comms files under %s", comms_dir)
        clear_comms_dir(comms_dir)
    else:
        socket_host = hw_cfg.get("socket_host", "127.0.0.1")
        socket_port = int(hw_cfg.get("socket_port", 8765))

    device = torch.device(pol_cfg.get("pytorch_device", "cuda") if torch.cuda.is_available() else "cpu")

    # policy.backend: "openpi" (default -- Pi0.5 loaded in-process + the
    # joint-velocity Dynamics MLP adapter) or "molmoact2" (a separate
    # ~/molmoact2/serve.py process reached over a TCP socket; it returns
    # absolute joint-position candidate chunks so no adapter is needed).
    backend = str(pol_cfg.get("backend", "openpi"))
    if backend not in ("openpi", "molmoact2"):
        raise SystemExit(f'policy.backend must be "openpi" or "molmoact2", got {backend!r}')

    if backend == "openpi":
        logger.info("Loading openpi policy %s from %s", pol_cfg.get("config_name", "pi05_droid"),
                    pol_cfg["checkpoint_path"])
        policy = load_policy_from_checkpoint(
            config_name=pol_cfg.get("config_name", "pi05_droid"),
            checkpoint_path=pol_cfg["checkpoint_path"],
            repo_path=pol_cfg.get("repo_path", "external/openpi"),
            default_prompt=None,
            pytorch_device=pol_cfg.get("pytorch_device", "cuda"),
        )
        action_adapter = OpenPIActionAdapter(
            checkpoint_path=pol_cfg["act_adapter_path"],
            action_num=15, action_dim=7, hidden_size=512,
            gripper_max=float(pol_cfg.get("gripper_max", 0.75)),
            device=pol_cfg.get("pytorch_device", "cuda"),
        )
    else:  # molmoact2
        ma2_cfg = pol_cfg.get("molmoact2", {})
        logger.info("Connecting to MolmoAct2 server at %s:%s",
                    ma2_cfg.get("host", "127.0.0.1"), ma2_cfg.get("port", 9999))
        policy = MolmoAct2Client(
            host=ma2_cfg.get("host", "127.0.0.1"),
            port=int(ma2_cfg.get("port", 9999)),
            norm_tag=ma2_cfg.get("norm_tag", "franka_droid"),
            num_steps=ma2_cfg.get("num_steps", 10),
            n_action_steps=int(ma2_cfg.get("n_action_steps", 15)),
            camera_order=ma2_cfg.get("camera_order",
                                     ["wrist_image", "right_image", "left_image"]),
            seed=int(ma2_cfg.get("seed", 0)),
            gripper_scale=float(ma2_cfg.get("gripper_scale", 1.0 / 1.5)),
            image_size=ma2_cfg.get("image_size"),
            timeout_s=float(ma2_cfg.get("timeout_s", 20.0)),
            debug_dump_dir=ma2_cfg.get("debug_dump_dir"),
            debug_dump_n=int(ma2_cfg.get("debug_dump_n", 0)),
        )
        policy.connect()  # fail fast if serve.py is not running
        logger.info("MolmoAct2 server connected.")
        action_adapter = None

    num_candidates = int(uq_cfg.get("num_candidates", 4))
    uq_metric = uq_cfg.get("uq_metric", "mean_pdf_diff")
    if uq_metric not in UQ_METRIC_CHOICES:
        raise SystemExit(f"active_uq.uq_metric must be one of {UQ_METRIC_CHOICES}, got {uq_metric!r}")
    uq_epi_mode = str(uq_cfg.get("uq_epi_mode", "iterative"))
    if uq_epi_mode not in UQ_EPI_MODE_CHOICES:
        raise SystemExit(f"active_uq.uq_epi_mode must be one of {UQ_EPI_MODE_CHOICES}, got {uq_epi_mode!r}")
    epi_overlap_k = int(uq_cfg.get("epi_overlap_k", 0))
    # Must match the loaded checkpoint's zero_overlap_action training setting
    # (see config.py's docstring): True for droid_flow_matching_uq_false_future_v1
    # and later, False for droid_flow_matching_uq_future_overlap_v1 and earlier.
    overlap_zero_action = bool(uq_cfg.get("overlap_zero_action", False))
    if uq_epi_mode == "none" and uq_metric not in PASS1_ONLY_METRICS:
        raise SystemExit(
            f'active_uq.uq_epi_mode="none" runs no pass-2 scoring, so active_uq.uq_metric '
            f"must be one of {PASS1_ONLY_METRICS}, got {uq_metric!r}")
    num_inference_steps = int(uq_cfg.get("num_inference_steps", 15))
    wire_skip = int(uq_cfg.get("wire_skip", 3))
    wire_len = int(uq_cfg.get("wire_len", 5))
    num_cams = int(uq_cfg.get("num_cams", 3))
    height = int(uq_cfg.get("height", 192))
    width = int(uq_cfg.get("width", 320))

    # policy_only: bypass the world model entirely. MolmoAct2/openpi
    # candidates still get generated every round trip (so the policy +
    # comms/execution pipeline get exercised end-to-end), but the FIRST
    # proposed candidate is sent as-is -- no WM scoring, no UQ metrics, no
    # candidate-comparison viz. Skips loading the (slow, GPU-heavy) WM
    # checkpoint entirely -- useful for a quick, isolated check that the
    # policy + robot execution work correctly before bringing the WM in.
    policy_only = bool(uq_cfg.get("policy_only", False))

    exec_len = wire_skip * (wire_len - 1) + 1
    n_action_steps = int(pol_cfg.get(backend, {}).get("n_action_steps", 0)) if backend == "molmoact2" else None
    if n_action_steps and exec_len > n_action_steps:
        raise SystemExit(
            f"active_uq.wire_skip*(wire_len-1)+1 ({exec_len}) exceeds "
            f"policy.{backend}.n_action_steps ({n_action_steps}) -- the policy chunk isn't "
            "long enough to cover the configured stride/length; lower wire_skip or "
            "raise n_action_steps.")

    if policy_only:
        logger.info(
            "active_uq.policy_only=true -- skipping world model load entirely; the FIRST "
            "proposed candidate from every round trip is sent as-is, no UQ scoring.")
        wm_model = wm_pipeline_cls = wm_args = None
        p01 = p99 = None
        wm_checkpoint = None
    else:
        wm_checkpoint = uq_cfg["wm_checkpoint"]
        stat_root = uq_cfg["stat_root"]

        logger.info("Loading UQ world model from %s", wm_checkpoint)
        wm_model, wm_pipeline_cls, wm_args, wm_use_uq = load_crtl_world(
            wm_checkpoint,
            svd_model_path=save_cfg.get("svd_path", "external/stable-video-diffusion-img2vid"),
            clip_model_path=uq_cfg.get("clip_model_path", "external/clip-vit-base-patch32"),
            data_root=str(save_cfg.get("output_root", "data/droid_hardware_active_uq_collected")),
            stat_root=stat_root,
            suites=[WM_TRAIN_SUITE],
            device=device,
            predict_uncertainty=True,
            tag="droid_hardware_active_uq",
            num_cams=num_cams,
            height=height,
            width=width,
            down_sample=int(uq_cfg.get("down_sample", 3)),
        )
        if not wm_use_uq:
            raise SystemExit(f"checkpoint {wm_checkpoint} has no UQ head; cannot score candidates.")
        if wire_len != wm_args.num_frames:
            raise SystemExit(
                f"active_uq.wire_len ({wire_len}) must equal the checkpoint's num_frames "
                f"({wm_args.num_frames}) -- the {wire_len}-step, wire_skip-strided candidate "
                "slice is the world model's future-conditioning target (the robot's wire "
                "action is a separate, dense, un-strided prefix of the same chunk -- see "
                "_wire_action).")
        if uq_epi_mode == "future_overlap" and not wm_args.frame_level_cond:
            raise SystemExit(
                'active_uq.uq_epi_mode="future_overlap" requires the checkpoint\'s '
                "frame_level_cond=True -- with frame_level_cond=False the action encoder "
                "flattens (T, action_dim) to a fixed-width vector before its first Linear "
                "layer, and appending overlap action rows would silently change that width.")

        p01, p99 = _load_stat(stat_root, WM_TRAIN_SUITE)

    encoder = LatentEncoder(
        save_cfg.get("svd_path", "external/stable-video-diffusion-img2vid"),
        device=save_cfg.get("device", "cuda"),
        target_h=height, target_w=width,
    )

    external_camera = hw_cfg.get("external_camera", "right")
    obs_timeout_s = float(hw_cfg.get("obs_timeout_s", 60))
    poll_interval_s = float(hw_cfg.get("poll_interval_s", 0.2))
    max_steps = int(hw_cfg.get("max_steps", 10_000))
    num_trajectories = hw_cfg.get("num_trajectories")  # null = run forever

    viz_cfg = uq_cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_every_n_decisions = int(viz_cfg.get("every_n_decisions", 1))
    viz_max_decisions = viz_cfg.get("max_decisions_per_episode")
    viz_max_decisions = int(viz_max_decisions) if viz_max_decisions is not None else None
    viz_fps = int(viz_cfg.get("fps", 3))
    if policy_only and viz_enabled:
        logger.info("active_uq.policy_only=true -- ignoring active_uq.viz.enabled "
                    "(no WM scoring to visualize).")
        viz_enabled = False
    if uq_epi_mode == "none" and viz_enabled:
        raise SystemExit(
            'active_uq.viz.enabled=true requires active_uq.uq_epi_mode != "none" -- '
            "_save_decision_viz renders 4 of its 7 rows (ltv/epivar/pdfdiff/kl) from "
            'pass-2 outputs, which do not exist when uq_epi_mode=="none".')

    output_root = pathlib.Path(args.output_root or save_cfg.get("output_root", "data/droid_hardware_active_uq_collected"))
    write_raw = bool(save_cfg.get("write_raw", True))
    encode_latents = bool(save_cfg.get("encode_latents", True))
    raw_fps = int(save_cfg.get("raw_fps", 15))
    export_down_sample = int(save_cfg.get("down_sample", 1))
    num_history_cfg = int(save_cfg.get("num_history", 6))
    num_frames_cfg = int(save_cfg.get("num_frames", 5))
    suite_name = save_cfg.get("suite_name", "droid_ctrl_world_hardware")

    decision_jsonl_path = output_root / suite_name / "decision_metrics.jsonl"
    decision_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    decision_fh = open(decision_jsonl_path, "a")

    train_ids: list[str] = []

    def _process_trajectory(channel, traj_idx: int) -> None:
        """Run one trajectory to completion and export it -- shared between
        both comms modes; only how ``channel`` is obtained differs."""
        viz_dir = output_root / suite_name / "candidate_viz" / f"{traj_idx:06d}"
        logger.info("[traj %d] starting rollout", traj_idx)
        payload, decision_log, instruction = _rollout_trajectory_hardware(
            traj_idx=traj_idx, backend=backend, policy=policy, action_adapter=action_adapter,
            wm_model=wm_model, wm_pipeline_cls=wm_pipeline_cls, wm_args=wm_args,
            p01=p01, p99=p99, encoder=encoder, channel=channel,
            external_camera=external_camera, num_candidates=num_candidates,
            uq_metric=uq_metric, uq_epi_mode=uq_epi_mode, epi_overlap_k=epi_overlap_k,
            overlap_zero_action=overlap_zero_action,
            num_inference_steps=num_inference_steps,
            wire_skip=wire_skip, wire_len=wire_len, max_steps=max_steps,
            obs_timeout_s=obs_timeout_s, poll_interval_s=poll_interval_s,
            viz_enabled=viz_enabled, viz_every_n_decisions=viz_every_n_decisions,
            viz_max_decisions=viz_max_decisions, viz_fps=viz_fps, viz_dir=viz_dir,
            num_cams=num_cams, policy_only=policy_only,
        )

        episode_id = f"{next_episode_id(output_root, suite_name):06d}"
        # decision_log entries carry no "candidate_metrics" in policy_only mode
        # (no WM scoring happened) -- guard both UQ-summary computations.
        chosen_scores = (
            [d["candidate_metrics"][d["chosen"]][uq_metric] for d in decision_log]
            if not policy_only else [])
        write_droid_episode(
            suite=suite_name, split="train", episode_id=episode_id, output_root=output_root,
            encoder=encoder if encode_latents else None,
            cam_rgb=payload["cam_rgb"], cart=payload["cart"], grip=payload["grip"],
            joint_position=payload["joint_position"], language=instruction,
            fps=raw_fps, down_sample=export_down_sample, write_raw=write_raw,
            extra_annotation={
                "policy_backend": backend, "policy_only": policy_only,
                "num_candidates": num_candidates, "uq_metric": uq_metric,
                "uq_epi_mode": uq_epi_mode,
                "wm_checkpoint": str(wm_checkpoint), "num_decisions": len(decision_log),
                "mean_chosen_uq_score": float(np.mean(chosen_scores)) if chosen_scores else None,
                "mean_uq_score_spread": (
                    float(np.mean([
                        max(d["candidate_metrics"][i][uq_metric] for i in range(num_candidates))
                        - min(d["candidate_metrics"][i][uq_metric] for i in range(num_candidates))
                        for d in decision_log
                    ])) if (decision_log and not policy_only) else None),
            },
        )
        for d in decision_log:
            decision_fh.write(json.dumps({
                "traj_idx": traj_idx, "episode_id": episode_id,
                "wm_checkpoint": str(wm_checkpoint), "uq_epi_mode": uq_epi_mode,
                **d,
            }) + "\n")
        decision_fh.flush()
        train_ids.append(episode_id)

        logger.info("[traj %d] done: %d decisions (round-trips), %d saved samples -> episode %s",
                    traj_idx, len(decision_log), payload["cart"].shape[0], episode_id)

    n_done = 0
    if comms_mode == "file":
        traj_idx = 1  # robot increments its counter before its first send; matches uq_data_collection.py
        while num_trajectories is None or n_done < int(num_trajectories):
            _process_trajectory(FileChannel(comms_dir, traj_idx), traj_idx)
            cleanup_trajectory_files(comms_dir, traj_idx)
            traj_idx += 1
            n_done += 1
    else:
        def _handle_and_count(channel, traj_idx: int) -> None:
            nonlocal n_done
            _process_trajectory(channel, traj_idx)
            n_done += 1

        logger.info(
            "Socket comms server listening on %s:%d -- SSH-tunnel this port "
            "from the robot laptop (e.g. ssh -N -L %d:localhost:%d ...)",
            socket_host, socket_port, socket_port, socket_port)
        serve_trajectories(
            socket_host, socket_port, _handle_and_count,
            max_trajectories=int(num_trajectories) if num_trajectories is not None else None,
        )

    decision_fh.close()
    write_sample_list(output_root / suite_name, "train", train_ids,
                       num_history=num_history_cfg, num_frames=num_frames_cfg, down_sample=export_down_sample)
    logger.info("Collected %d trajectories -> %s", n_done, output_root / suite_name)


if __name__ == "__main__":
    main()
