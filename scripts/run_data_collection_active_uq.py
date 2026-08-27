"""Active, epistemic-UQ-guided policy rollouts in LIBERO sim.

Like ``scripts/run_data_collection.py``, but instead of executing whatever
single action chunk the policy proposes, at each replanning step it:

  1. samples ``num_candidates`` candidate action chunks from the policy,
  2. gets each candidate's resulting future EEF pose by physically
     simulating it in the LIBERO sim and then restoring the sim to its
     pre-trial state ("rewind"),
  3. scores each candidate's epistemic uncertainty with a UQ-trained world
     model (``CrtlWorld``) via the online "iterative" self-consistency
     mechanism (compare a prediction conditioned on real history against one
     conditioned on the model's own compounding self-predicted history),
  4. executes only the candidate that maximizes the configured UQ metric.

Collected trajectories are written in the same on-disk format
``run_data_collection.py`` produces, so they remain directly usable as
world-model training data.

Usage:
    uv run python scripts/run_data_collection_active_uq.py \\
        --config configs/collection/libero_pi05_active_uq.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys

import numpy as np
import torch
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess_libero_for_wm import (  # noqa: E402
    LatentEncoder,
    write_episode,
    write_sample_list,
)
from run_data_collection import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    SUITE_MAX_STEPS,
    _ensure_openpi_paths,
    _next_episode_id,
    _quat2axisangle,
    _record_obs,
)
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

from openworld.training.world_model.dataset import _load_stat  # noqa: E402
from openworld.utils.io import load_yaml  # noqa: E402

logger = logging.getLogger(__name__)

UQ_METRIC_CHOICES = (
    "mean_aleatoric_var", "max_aleatoric_var",
    "mean_epi_ltv", "max_epi_ltv",
    "mean_epi_var", "mean_pdf_diff", "mean_kl",
)


# ---------------------------------------------------------------------------
# Sim-state rewind
# ---------------------------------------------------------------------------


def _snapshot_env(env):
    """Physics state + the plain bookkeeping attributes robosuite tracks
    outside of ``sim.get_state()`` (timestep/cur_time/done -- see
    robosuite/environments/base.py; these drive horizon truncation and are
    NOT part of the MuJoCo state, so a discarded lookahead trial would
    otherwise corrupt them).

    LIBERO's ``OffScreenRenderEnv``/``ControlEnv`` is a thin wrapper around
    the actual robosuite env (stored as ``env.env``) -- only a few
    attributes (``sim``, ``robots``, ...) are forwarded as properties, so
    ``timestep``/``cur_time``/``done`` must be read/written on ``env.env``,
    not ``env`` itself (confirmed against
    external/openpi/third_party/libero/.../env_wrapper.py's ``ControlEnv``).
    """
    return env.sim.get_state(), env.env.timestep, env.env.cur_time, env.env.done


def _restore_env(env, snapshot) -> None:
    sim_state, timestep, cur_time, done = snapshot
    env.sim.set_state(sim_state)
    env.sim.forward()
    env.env.timestep = timestep
    env.env.cur_time = cur_time
    env.env.done = done


def _pose_from_obs(obs) -> tuple[np.ndarray, float]:
    """(absolute EEF pose (xyz + axis-angle), (6,)), absolute gripper (scalar))."""
    pose = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
    ).astype(np.float32)
    grip = float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)))
    return pose, grip


def _policy_infer_action(policy, obs, task_description, prompt, resize_size) -> np.ndarray:
    """One stochastic ``policy.infer()`` call -> raw (action_horizon, 8) chunk."""
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(prompt if prompt is not None else task_description),
    }
    return np.asarray(policy.infer(element)["actions"])


def _simulate_lookahead(env, policy, num_steps, task_description, prompt, resize_size):
    """From the env's current (already-restored-elsewhere) state, physically
    roll forward exactly ``num_steps`` real env steps, re-querying the
    policy whenever the current action chunk runs out (pi05_libero's
    action_horizon is 10, shorter than the lookahead this needs).

    Returns (poses[num_steps+1, 6], grips[num_steps+1], first_chunk) where
    poses[0]/grips[0] are the pre-step (current) pose/gripper and
    first_chunk is the raw chunk from the FIRST policy.infer() call (this is
    what gets executed for real if this candidate is selected).
    """
    obs = env.env._get_observations(force_update=True)
    pose0, grip0 = _pose_from_obs(obs)
    poses = [pose0]
    grips = [grip0]

    action_plan: collections.deque = collections.deque()
    first_chunk = None
    for _ in range(num_steps):
        if not action_plan:
            chunk = _policy_infer_action(policy, obs, task_description, prompt, resize_size)
            if first_chunk is None:
                first_chunk = chunk
            action_plan.extend(chunk)
        action = action_plan.popleft()[:7]
        obs, _, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else list(action))
        pose, grip = _pose_from_obs(obs)
        poses.append(pose)
        grips.append(grip)
        if done:
            # Episode would end mid-lookahead; hold the last pose for the
            # remaining (unreachable) offsets -- harmless since the whole
            # trial is discarded/restored right after scoring anyway.
            while len(poses) <= num_steps:
                poses.append(pose)
                grips.append(grip)
            break
    return np.stack(poses, axis=0), np.asarray(grips, dtype=np.float32), first_chunk


# ---------------------------------------------------------------------------
# Candidate-comparison visualization
# ---------------------------------------------------------------------------


def _make_header_row(num_frames: int, col_width: int, num_cols: int, header_h: int = 12) -> np.ndarray:
    """One thin row marking column identity: green under the first ("best")
    column, gray under the rest."""
    row = np.full((num_frames, header_h, col_width * num_cols, 3), 160, dtype=np.uint8)
    row[:, :, :col_width] = np.array([40, 200, 40], dtype=np.uint8)
    return row


def _save_decision_viz(
    viz_path_prefix, i_star: int, pred_latents1, pred_latents2,
    logvar_steps1, vel_steps1, logvar_steps2, vel_steps2,
    pipeline, wm_args, fps: int,
) -> None:
    """Compose two comparison videos for a decision point (one per camera --
    agentview and wrist, matching preprocess_libero_for_wm.py's camera
    order/convention -- rather than one video with both cameras stacked).
    Each video's rows = signal type (pred1/pred2/aleatoric/epi_ltv/epi_var/
    pdf_diff/kl), columns = candidate, chosen candidate first ("best")
    followed by the rest in original index order -- lets you compare one
    signal across every candidate at a glance and check whether the
    selection metric tracks a visible difference.

    Writes ``{viz_path_prefix}_agentview.mp4`` and ``{viz_path_prefix}_wrist.mp4``.

    Reuses replay_libero_wm_traj.py's decode/render functions as-is; needs
    no extra world-model inference since scoring already produced everything
    consumed here (pred_latents/logvar_steps/vel_steps for both passes). The
    render functions stack both cameras vertically into one array
    (T, num_cams*height, width, 3); split that into per-camera slices after
    rendering rather than re-deriving per-camera renderers.
    """
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

    for cam_idx, cam_name in enumerate(("agentview", "wrist")):
        cam_slice = slice(cam_idx * per_cam_h, (cam_idx + 1) * per_cam_h)
        signal_rows = [
            np.concatenate([frames[:, cam_slice] for frames in signal_cols[key]], axis=2)
            for key in ("pred1", "pred2", "alea", "ltv", "epivar", "pdfdiff", "kl")
        ]
        rows = [_make_header_row(num_frames, wm_args.width, num_candidates)] + signal_rows

        h_sep = np.full((num_frames, 3, rows[0].shape[2], 3), 200, dtype=np.uint8)
        composite = rows[0]
        for row in rows[1:]:
            composite = np.concatenate([composite, h_sep, row], axis=1)

        write_video(composite, pathlib.Path(f"{viz_path_prefix}_{cam_name}.mp4"), fps)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _rollout_one_episode_active_uq(
    *, policy, env, wm_model, wm_pipeline_cls, wm_args,
    p01, p99, num_candidates, uq_metric, num_inference_steps,
    init_state, task_description, prompt, max_steps,
    num_steps_wait, replan_steps, resize_size, env_max_reward,
    device, latent_encoder,
    viz_enabled: bool = False, viz_every_n_decisions: int = 1,
    viz_max_decisions: int | None = None, viz_fps: int = 3, viz_dir=None,
):
    """Run a single trajectory with UQ-scored candidate selection.

    Returns (payload-fields, is_success, steps, decision_log), mirroring
    ``run_data_collection._rollout_one_episode`` plus a per-decision log of
    candidate UQ scores for diagnostics.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: collections.deque = collections.deque()

    agent_frames, wrist_frames, cart_list, grip_list = [], [], [], []
    is_success = False
    executed = 0
    decision_log: list[dict] = []

    # For this checkpoint (trained on data/libero_uq_2x2_policy_probe, which
    # is stored at NATIVE 20Hz with down_sample=1 -- confirmed via the
    # dataset's own annotation JSON and LiberoLatentDataset._build_frame_ids
    # -- the WM operates 1:1 with real env steps. "skip"/"skip_his" below are
    # frame-index strides in REAL-STEP units directly, not at some coarser
    # separately-strided "WM rate". This does NOT generalize to checkpoints
    # trained on 5Hz-pre-strided data (e.g. some older libero_collected-based
    # checkpoints) -- confirm a checkpoint's training data's own fps/down_sample
    # before reusing this script against it.
    num_history, num_frames = wm_args.num_history, wm_args.num_frames
    skip = 1
    skip_his = skip * 4  # matches dataset._build_frame_ids / replay_libero_wm_traj.py
    lookahead_steps = (num_frames - 1) * skip  # real steps of pose lookahead needed

    rolled: list[torch.Tensor] = []        # WM latents at real ground truth, one per real step
    rolled2: list[torch.Tensor] = []       # self-predicted after active phase starts
    real_pose_hist: list[np.ndarray] = []  # (6,) absolute pose per real step
    real_grip_hist: list[float] = []
    per_cam_h, latent_w = wm_args.height // 8, wm_args.width // 8

    def _encode_current_latent(obs) -> torch.Tensor:
        agent_img = np.ascontiguousarray(obs["agentview_image"][::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
        agent_lat = latent_encoder.encode(agent_img[None])[0]
        wrist_lat = latent_encoder.encode(wrist_img[None])[0]
        latent = torch.zeros((4, wm_args.num_cams * per_cam_h, latent_w), dtype=torch.float32)
        latent[:, :per_cam_h] = agent_lat
        latent[:, per_cam_h:] = wrist_lat
        return latent

    t = 0
    # Queue of Pass-1 predicted latents (one per committed real step) set on
    # an active decision; drained as those steps are actually executed. Sized
    # to hold exactly (num_frames-1) entries, which is why `replan_steps`
    # must equal (num_frames-1)*skip -- see config comment.
    pending_rolled2_latents: collections.deque = collections.deque()
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list)

        if not action_plan:
            frame_now = len(rolled) - 1
            if frame_now < num_history * skip_his:
                chunk = _policy_infer_action(policy, obs, task_description, prompt, resize_size)
                action_plan.extend(chunk[:replan_steps])
            else:
                snapshot = _snapshot_env(env)
                candidates = []
                for _ in range(num_candidates):
                    _restore_env(env, snapshot)
                    poses, grips, first_chunk = _simulate_lookahead(
                        env, policy, lookahead_steps, task_description, prompt, resize_size)
                    candidates.append((poses, grips, first_chunk))
                _restore_env(env, snapshot)
                obs = env.env._get_observations(force_update=True)

                # rgb_id: [hist_1..hist_num_history, frame_now, +1*skip, ..., +(num_frames-1)*skip]
                # (real-step index space -- identical to replay_libero_wm_traj.py's replay loop,
                # since this checkpoint's WM rate == real env rate; see note above).
                rgb_id = build_frame_ids(frame_now, num_history, num_frames, skip, skip_his)
                hist_ids = [max(idx, 0) for idx in rgb_id[:num_history]]
                future_ids = rgb_id[num_history:]  # length num_frames, future_ids[0] == frame_now

                hist_rows = np.stack(
                    [np.concatenate([real_pose_hist[idx], [real_grip_hist[idx]]]) for idx in hist_ids],
                    axis=0)  # (num_history, 7)

                # Real-step offset from frame_now for each future frame -- these index directly
                # into each candidate's simulated `poses`/`grips` lookahead arrays.
                future_offsets = [idx - frame_now for idx in future_ids]
                action_tensors = []
                for poses, grips, _ in candidates:
                    future_rows = np.stack(
                        [np.concatenate([poses[o], [grips[o]]]) for o in future_offsets], axis=0)
                    full = np.concatenate([hist_rows, future_rows], axis=0)  # (num_history+num_frames, 7)
                    action_tensors.append(normalize_actions(full, p01, p99))
                action_batch = torch.tensor(
                    np.stack(action_tensors, axis=0), dtype=torch.float32, device=device)

                def _hist_cur(buf):
                    hist = torch.stack([buf[idx] for idx in hist_ids], 0).unsqueeze(0).expand(
                        num_candidates, -1, -1, -1, -1).contiguous().to(device)
                    cur = buf[frame_now].unsqueeze(0).expand(
                        num_candidates, -1, -1, -1).contiguous().to(device)
                    return hist, cur

                history_latents, current_latent = _hist_cur(rolled)
                history2_latents, current2_latent = _hist_cur(rolled2)
                texts = [task_description] * num_candidates

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    action_latent = wm_model.action_encoder(
                        action_batch, texts, wm_model.tokenizer, wm_model.text_encoder,
                        wm_args.frame_level_cond)

                    def _pipeline_call(image, history, his_cond_zero):
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
                            flow_map_type=wm_args.flow_map_type,
                            flow_map_loss_type=wm_args.flow_map_loss_type,
                            return_uncertainty=True,
                        )

                    _, pred_latents1, logvar_steps1, vel_steps1 = _pipeline_call(
                        current_latent, history_latents, wm_args.his_cond_zero)
                    _, pred_latents2, logvar_steps2, vel_steps2 = _pipeline_call(
                        current2_latent, history2_latents, False)

                n_ode = len(logvar_steps1)
                candidate_metrics = []
                for i in range(num_candidates):
                    lv1_i = [logvar_steps1[s][i] for s in range(n_ode)]
                    vel1_i = [vel_steps1[s][i] for s in range(n_ode)]
                    lv2_i = [logvar_steps2[s][i] for s in range(n_ode)]
                    vel2_i = [vel_steps2[s][i] for s in range(n_ode)]
                    candidate_metrics.append(compute_uq_metrics(lv1_i, vel1_i, lv2_i, vel2_i))

                decision_idx = len(decision_log)
                i_star = max(range(num_candidates), key=lambda i: candidate_metrics[i][uq_metric])
                decision_log.append({
                    "decision_idx": decision_idx,
                    "real_step": executed,
                    "uq_metric": uq_metric,
                    "chosen": i_star,
                    "candidate_metrics": candidate_metrics,
                })

                if (viz_enabled
                        and decision_idx % max(1, viz_every_n_decisions) == 0
                        and (viz_max_decisions is None or decision_idx < viz_max_decisions)):
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    _save_decision_viz(
                        viz_dir / f"decision_{decision_idx:03d}", i_star,
                        pred_latents1, pred_latents2, logvar_steps1, vel_steps1,
                        logvar_steps2, vel_steps2, wm_model.pipeline, wm_args, viz_fps,
                    )

                _restore_env(env, snapshot)
                obs = env.env._get_observations(force_update=True)
                _, _, winner_chunk = candidates[i_star]
                action_plan.extend(winner_chunk[:replan_steps])
                # Queue Pass-1's predicted frames (k=1..num_frames-1, conditioned on TRUE
                # history) for the chosen candidate -- one per committed real step, drained
                # below to advance the self-consistency buffer as those steps are executed.
                pending_rolled2_latents.clear()
                for k in range(1, num_frames):
                    pending_rolled2_latents.append(pred_latents1[i_star, k].float().cpu())

        action = action_plan.popleft()[:7]
        obs, reward, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else list(action))
        executed += 1

        # This checkpoint's WM operates 1:1 with real env steps, so append a
        # new frame to both buffers after every single step (not just at
        # chunk boundaries).
        latent = _encode_current_latent(obs)
        rolled.append(latent)
        pose, grip = _pose_from_obs(obs)
        real_pose_hist.append(pose)
        real_grip_hist.append(grip)
        if pending_rolled2_latents:
            rolled2.append(pending_rolled2_latents.popleft())
        else:
            rolled2.append(latent.clone())

        if done or reward == env_max_reward:
            is_success = True
            break
        t += 1

    _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list)

    payload = {
        "agent_rgb": np.stack(agent_frames, axis=0),
        "wrist_rgb": np.stack(wrist_frames, axis=0),
        "cart": np.stack(cart_list, axis=0),
        "grip": np.asarray(grip_list, dtype=np.float32),
    }
    return payload, is_success, executed, decision_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_root", type=str, default=None,
                    help="Override save.output_root from the config.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_yaml(args.config)

    pol_cfg = cfg.get("policy", {})
    col_cfg = cfg.get("collection", {})
    env_cfg = cfg.get("env", {})
    save_cfg = cfg.get("save", {})
    uq_cfg = cfg.get("active_uq", {})

    openpi_repo = pol_cfg.get("repo_path", "external/openpi")
    _ensure_openpi_paths(openpi_repo)

    output_root = pathlib.Path(args.output_root or save_cfg["output_root"])
    write_raw = bool(save_cfg.get("write_raw", True))
    down_sample = max(1, int(save_cfg.get("down_sample", 1)))
    raw_fps = int(save_cfg.get("raw_fps", 20))
    out_fps = max(1, raw_fps // down_sample)
    val_fraction = float(save_cfg.get("val_fraction", 0.0))
    num_history_cfg = int(save_cfg.get("num_history", 6))
    num_frames_cfg = int(save_cfg.get("num_frames", 5))

    resolution = int(env_cfg.get("resolution", 256))
    num_steps_wait = int(env_cfg.get("num_steps_wait", 10))
    replan_steps = int(env_cfg.get("replan_steps", 4))
    resize_size = int(env_cfg.get("resize_size", 224))
    seed = int(env_cfg.get("seed", 7))
    env_max_reward = float(env_cfg.get("env_max_reward", 1.0))

    suites = col_cfg.get("task_suites") or col_cfg.get("task_suite")
    if isinstance(suites, str):
        suites = [suites]
    if not suites:
        raise SystemExit("config collection.task_suites must list >=1 suite")
    trajectories_per_task = int(col_cfg.get("trajectories_per_task", 10))
    task_id_filter = col_cfg.get("task_ids")
    prompt_override = col_cfg.get("prompt_override")

    num_candidates = int(uq_cfg.get("num_candidates", 4))
    uq_metric = uq_cfg.get("uq_metric", "mean_epi_ltv")
    if uq_metric not in UQ_METRIC_CHOICES:
        raise SystemExit(f"active_uq.uq_metric must be one of {UQ_METRIC_CHOICES}, got {uq_metric!r}")
    num_inference_steps = int(uq_cfg.get("num_inference_steps", 15))
    wm_checkpoint = uq_cfg["wm_checkpoint"]
    stat_root = uq_cfg["stat_root"]

    viz_cfg = uq_cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_every_n_decisions = int(viz_cfg.get("every_n_decisions", 1))
    viz_max_decisions = viz_cfg.get("max_decisions_per_episode")
    viz_max_decisions = int(viz_max_decisions) if viz_max_decisions is not None else None
    viz_fps = int(viz_cfg.get("fps", 3))

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device(
        pol_cfg.get("pytorch_device", "cuda") if torch.cuda.is_available() else "cpu")

    from openworld.policies.openpi_loader import load_policy_from_checkpoint
    checkpoint_path = pol_cfg["checkpoint_path"]
    logger.info("Loading policy %s from %s",
                pol_cfg.get("config_name", "pi05_libero"), checkpoint_path)
    policy = load_policy_from_checkpoint(
        config_name=pol_cfg.get("config_name", "pi05_libero"),
        checkpoint_path=checkpoint_path,
        repo_path=openpi_repo,
        default_prompt=None,
        pytorch_device=pol_cfg.get("pytorch_device", "cuda"),
    )

    svd_path = save_cfg.get("svd_path")
    if not svd_path:
        raise SystemExit(
            "save.svd_path is required -- also used to build the live rolled/rolled2 "
            "WM latent buffers, not just the final saved trajectory.")
    encoder = LatentEncoder(svd_path, device=save_cfg.get("device", "cuda"))

    logger.info("Loading UQ world model from %s", wm_checkpoint)
    wm_model, wm_pipeline_cls, wm_args, wm_use_uq = load_crtl_world(
        wm_checkpoint,
        svd_model_path=svd_path,
        clip_model_path=uq_cfg.get("clip_model_path", "external/clip-vit-base-patch32"),
        data_root=str(output_root),
        stat_root=stat_root,
        suites=suites,
        device=device,
        predict_uncertainty=True,
        tag="libero_active_uq",
    )
    if not wm_use_uq:
        raise SystemExit(f"checkpoint {wm_checkpoint} has no UQ head; cannot score candidates.")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()

    for suite in suites:
        if suite not in SUITE_MAX_STEPS:
            raise SystemExit(f"unknown task_suite {suite!r}")
        max_steps = SUITE_MAX_STEPS[suite]
        task_suite = bench[suite]()
        task_ids = (
            list(task_id_filter) if task_id_filter is not None
            else list(range(task_suite.n_tasks))
        )

        p01, p99 = _load_stat(stat_root, suite)

        episode_id = _next_episode_id(output_root, suite)
        train_ids: list[str] = []
        val_ids: list[str] = []
        decision_jsonl_path = output_root / suite / "decision_metrics.jsonl"
        decision_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        decision_fh = open(decision_jsonl_path, "a")

        logger.info("[%s] collecting %d traj over %d task(s); first eid=%06d",
                    suite, trajectories_per_task * len(task_ids),
                    len(task_ids), episode_id)

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            bddl_file = task.bddl_file
            if not pathlib.Path(bddl_file).exists():
                from libero.libero import get_libero_path
                bddl_file = str(
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder / task.bddl_file
                )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=resolution,
                camera_widths=resolution,
            )
            env.seed(seed)
            task_description = task.language
            bddl_stem = pathlib.Path(task.bddl_file).stem

            n_ok = 0
            for trial in tqdm.tqdm(
                range(trajectories_per_task),
                desc=f"{suite} task{task_id:02d}",
            ):
                init_state = init_states[trial % len(init_states)]
                eid = f"{episode_id:06d}"
                viz_dir = output_root / suite / "candidate_viz" / eid
                payload, is_success, steps, decision_log = _rollout_one_episode_active_uq(
                    policy=policy, env=env, wm_model=wm_model, wm_pipeline_cls=wm_pipeline_cls,
                    wm_args=wm_args, p01=p01, p99=p99, num_candidates=num_candidates,
                    uq_metric=uq_metric, num_inference_steps=num_inference_steps,
                    init_state=init_state, task_description=task_description, prompt=prompt_override,
                    max_steps=max_steps, num_steps_wait=num_steps_wait, replan_steps=replan_steps,
                    resize_size=resize_size, env_max_reward=env_max_reward, device=device,
                    latent_encoder=encoder,
                    viz_enabled=viz_enabled, viz_every_n_decisions=viz_every_n_decisions,
                    viz_max_decisions=viz_max_decisions, viz_fps=viz_fps, viz_dir=viz_dir,
                )
                n_ok += int(is_success)

                if down_sample > 1:
                    payload = {
                        "agent_rgb": payload["agent_rgb"][::down_sample],
                        "wrist_rgb": payload["wrist_rgb"][::down_sample],
                        "cart": payload["cart"][::down_sample],
                        "grip": payload["grip"][::down_sample],
                    }
                payload["language"] = task_description
                payload["bddl"] = bddl_stem

                split = "val" if rng.random() < val_fraction else "train"

                chosen_scores = [
                    d["candidate_metrics"][d["chosen"]][uq_metric] for d in decision_log
                ]
                write_episode(
                    suite=suite,
                    split=split,
                    episode_id=eid,
                    output_root=output_root,
                    encoder=encoder,
                    payload=payload,
                    fps=out_fps,
                    write_raw=write_raw,
                    extra_annotation={
                        "task_id": int(task_id),
                        "task_name": task_description,
                        "trial": int(trial),
                        "is_success": bool(is_success),
                        "episode_steps": int(steps),
                        "num_steps_wait": int(num_steps_wait),
                        "policy_checkpoint": str(checkpoint_path),
                        "source": "active_uq_policy_rollout",
                        "num_candidates": num_candidates,
                        "uq_metric": uq_metric,
                        "wm_checkpoint": str(wm_checkpoint),
                        "num_decisions": len(decision_log),
                        "mean_chosen_uq_score": (
                            float(np.mean(chosen_scores)) if chosen_scores else None),
                        "mean_uq_score_spread": (
                            float(np.mean([
                                max(d["candidate_metrics"][i][uq_metric] for i in range(num_candidates))
                                - min(d["candidate_metrics"][i][uq_metric] for i in range(num_candidates))
                                for d in decision_log
                            ])) if decision_log else None),
                    },
                )
                for d in decision_log:
                    decision_fh.write(json.dumps({"suite": suite, "episode_id": eid, **d}) + "\n")
                decision_fh.flush()

                (train_ids if split == "train" else val_ids).append(eid)
                episode_id += 1

            env.close()
            logger.info("[%s] task %d (%s): %d/%d success",
                        suite, task_id, task_description, n_ok,
                        trajectories_per_task)

        decision_fh.close()

        suite_root = output_root / suite
        for split in ("train", "val"):
            all_ids = sorted(
                p.stem
                for p in (suite_root / "annotation" / split).glob("*.json")
                if p.stem.isdigit()
            )
            if all_ids:
                write_sample_list(suite_root, split, all_ids,
                                  num_history=num_history_cfg, num_frames=num_frames_cfg,
                                  down_sample=1)
        logger.info("[%s] done this run: %d train + %d val trajectories -> %s",
                    suite, len(train_ids), len(val_ids), suite_root)


if __name__ == "__main__":
    main()
