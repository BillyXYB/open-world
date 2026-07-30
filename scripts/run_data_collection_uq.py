"""Collect pi0.5 policy roll-outs in friction-varied LIBERO environments,
producing a 2x2 UQ benchmark dataset with known uncertainty properties:

  (variance_level: high | low) x (data_level: large | small)

Variance axis: measured automatically from a friction probe (5-point demo-replay
sweep). sigma = std(final object displacement across mu values). Tasks split at
the median sigma: top half -> high, bottom half -> low.
Probe results cached to <output_root>/variance_probe.json (re-runs skip probe).

Data axis: number of distinct friction environments (mu values) per task.

Usage:
  # Probe only (inspect variance classification first):
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0     uv run python scripts/run_data_collection_uq.py       --config configs/collection/libero_pi05_uq_2x2.yaml --probe_only

  # Full collection:
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0     uv run python scripts/run_data_collection_uq.py       --config configs/collection/libero_pi05_uq_2x2.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import sys

import numpy as np
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess_libero_for_wm import (  # noqa: E402
    LatentEncoder,
    write_episode,
    write_sample_list,
)
from collect_libero_branch_sets import (  # noqa: E402
    _build_env,
    _detect_object_body,
    _plate_handles,
    _set_plate_friction,
    _restore_plate_physics,
    _resolve_demo_file,
    _run_counterfactual_branch,
)
from build_uq_dataset import (  # noqa: E402
    _sample_mu,
    _measure_variance,
)

from openworld.utils.io import load_yaml  # noqa: E402

logger = logging.getLogger(__name__)

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a robosuite (x, y, z, w) quaternion to axis-angle."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def _ensure_openpi_paths(openpi_repo: str) -> None:
    repo = pathlib.Path(openpi_repo)
    for p in (
        str(repo),
        str(repo / "src"),
        str(repo / "packages" / "openpi-client" / "src"),
        str(repo / "third_party" / "libero"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    _enable_legacy_torch_load()


def _enable_legacy_torch_load() -> None:
    import torch
    if getattr(torch.load, "_libero_legacy", False):
        return
    _orig_load = torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    _load._libero_legacy = True
    torch.load = _load


def _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list):
    agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]).copy())
    wrist_frames.append(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]).copy()
    )
    cart_list.append(
        np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
        ).astype(np.float32)
    )
    grip_list.append(
        float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)))
    )


def _next_episode_id(output_root: pathlib.Path, suite: str) -> int:
    """First unused integer episode id across both splits for this suite."""
    used = []
    for split in ("train", "val"):
        ann_dir = output_root / suite / "annotation" / split
        if ann_dir.is_dir():
            for p in ann_dir.glob("*.json"):
                if p.stem.isdigit():
                    used.append(int(p.stem))
    return (max(used) + 1) if used else 0


def _rollout_one_episode(
    *, policy, env, init_state, task_description, prompt, max_steps,
    num_steps_wait, replan_steps, resize_size, env_max_reward,
):
    """Run a single trajectory. Returns (payload-fields, is_success, steps)."""
    from openpi_client import image_tools

    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: collections.deque = collections.deque()

    agent_frames, wrist_frames, cart_list, grip_list = [], [], [], []
    is_success = False
    executed = 0

    t = 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list)

        if not action_plan:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, resize_size, resize_size)
            )
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": str(prompt if prompt is not None else task_description),
            }
            chunk = policy.infer(element)["actions"]
            action_plan.extend(chunk[:replan_steps])

        action = action_plan.popleft()[:7]
        obs, reward, done, _ = env.step(
            action.tolist() if hasattr(action, "tolist") else list(action)
        )
        executed += 1
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
    return payload, is_success, executed


def _save_frames_row_video(all_frames: list, probe_dir: pathlib.Path, fps: int = 10) -> None:
    """Save a list of frame arrays as a single side-by-side row video.

    Each element of all_frames is a numpy array of shape (T_i, H, W, 3).
    Shorter sequences are padded with their last frame so all columns are equal length.
    Output: <probe_dir>/probe_row.mp4
    """
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available -- skipping probe video save")
        return

    valid = [f for f in all_frames if hasattr(f, "shape") and f.ndim == 4 and f.shape[0] > 0]
    if not valid:
        return

    probe_dir.mkdir(parents=True, exist_ok=True)
    max_len = max(f.shape[0] for f in valid)
    padded = []
    for f in valid:
        if f.shape[0] < max_len:
            pad = np.repeat(f[-1:], max_len - f.shape[0], axis=0)
            f = np.concatenate([f, pad], axis=0)
        padded.append(f)

    row_frames = np.concatenate(padded, axis=2)
    out_path = probe_dir / "probe_row.mp4"
    try:
        imageio.mimwrite(str(out_path), row_frames, fps=fps)
    except Exception as e:
        logger.warning("Could not write probe row video %s: %s", out_path, e)


def _save_probe_videos_direct(env, object_body, states, actions, mu_values, probe_dir):
    """Save probe videos as a single side-by-side row video (one column per mu value).

    Each mu value occupies one column in the row; frames are aligned in time.
    Shorter trajectories are padded with their last frame so all columns have
    equal length. Output: <probe_dir>/probe_row.mp4
    """
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available -- skipping probe video save")
        return

    probe_dir.mkdir(parents=True, exist_ok=True)
    pbid, geoms, defaults = _plate_handles(env, object_body)

    # Collect frames for each mu value
    all_frames = []  # list of np arrays shape (T_i, H, W, 3)
    for mu in mu_values:
        payload, _, _, _ = _run_counterfactual_branch(
            env, states=states, actions=actions, mu=float(mu), mass_scale=None,
            object_body=object_body, plate_geoms=geoms, pbid_=pbid,
            defaults=defaults, obj_rgba=None)
        frames = payload.get("agent_rgb", [])
        if len(frames):
            all_frames.append(np.stack(frames))  # (T, H, W, 3)

    _restore_plate_physics(env, pbid, geoms, defaults)

    if not all_frames:
        return

    # Pad all sequences to the same length (repeat last frame)
    max_len = max(f.shape[0] for f in all_frames)
    padded = []
    for f in all_frames:
        if f.shape[0] < max_len:
            pad = np.repeat(f[-1:], max_len - f.shape[0], axis=0)
            f = np.concatenate([f, pad], axis=0)
        padded.append(f)

    # Concatenate horizontally: (T, H, W*N, 3)
    row_frames = np.concatenate(padded, axis=2)

    out_path = probe_dir / "probe_row.mp4"
    try:
        imageio.mimwrite(str(out_path), row_frames, fps=10)
    except Exception as e:
        logger.warning("Could not write probe row video %s: %s", out_path, e)


def _resolve_demo_path(suite, task, dataset_path=None):
    """Resolve demo HDF5 path, using dataset_path override if provided."""
    stem = pathlib.Path(task.bddl_file).stem
    if dataset_path:
        return str(pathlib.Path(dataset_path) / suite / f"{stem}_demo.hdf5")
    return _resolve_demo_file(suite, task, None)


def run_variance_probe(cfg, output_root, reprobe=False):
    """Phase A: measure friction sensitivity per task and assign variance_level.

    Results cached to <output_root>/variance_probe.json. On subsequent runs,
    if the cache covers all tasks in the current config it is used directly.
    Pass reprobe=True to force recomputation.

    Returns cfg["objects"] with variance_level filled in for each spec.
    """
    import h5py

    probe_cache_path = output_root / "variance_probe.json"
    probe_cfg = cfg["variance_probe"]
    hv = cfg["hidden_var"]
    obj_specs = cfg["objects"]
    obj_names = {o["name"] for o in obj_specs}
    dataset_path = cfg.get("libero_dataset_path")

    # Check if cache is valid (covers all current tasks)
    if not reprobe and probe_cache_path.exists():
        cached = json.loads(probe_cache_path.read_text())
        if obj_names <= set(cached.keys()):
            logger.info("Loaded variance probe cache from %s", probe_cache_path)
            probe_results = cached
        else:
            missing = obj_names - set(cached.keys())
            logger.info("Cache missing %s -- reprobing", missing)
            probe_results = None
    else:
        probe_results = None

    if probe_results is None:
        probe_mu_values = np.geomspace(
            float(hv["low"]), float(hv["high"]), int(probe_cfg["num_mu"])
        )
        probe_results = {}
        for obj_spec in obj_specs:
            name = obj_spec["name"]
            suite = obj_spec["suite"]
            task_id = obj_spec["task_id"]
            object_body = obj_spec["object_body"]

            logger.info("Probing %s (suite=%s task_id=%d object_body=%s)",
                        name, suite, task_id, object_body)

            env, task = _build_env(suite, task_id, res=96)
            demo_file = _resolve_demo_path(suite, task, dataset_path)
            if not pathlib.Path(demo_file).exists():
                logger.warning("  no demo file (%s) -- skipping %s", demo_file, name)
                env.close()
                continue

            f = h5py.File(demo_file, "r")
            demo_names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
            states = np.asarray(f["data"][demo_names[0]]["states"])
            f.close()
            # actions not needed for displacement measure (kinematic replay uses states)
            actions = np.zeros((len(states), 7))  # placeholder -- kinematic replay ignores them

            # Auto-detect object_body if not specified in config
            if not object_body:
                object_body, det_disp = _detect_object_body(env, states)
                obj_spec["object_body"] = object_body
                logger.info("  auto-detected object_body=%s (demo disp=%.3fm)",
                            object_body, det_disp)

            sigma, range_cm, unstable = _measure_variance(
                env, object_body, [(states, actions)],
                hv, int(probe_cfg["num_mu"]), float(probe_cfg.get("max_disp_cm", 60.0))
            )

            # Save probe videos for visual inspection
            probe_dir = output_root / "probe_videos" / name
            try:
                _save_probe_videos_direct(
                    env, object_body, states, actions, probe_mu_values, probe_dir
                )
                logger.info("  saved probe videos -> %s", probe_dir)
            except Exception as e:
                logger.warning("  probe video save failed: %s", e)

            env.close()

            probe_results[name] = {
                "sigma_cm": float(sigma),
                "range_cm": float(range_cm),
                "unstable": bool(unstable),
                "mu_values": probe_mu_values.tolist(),
                "suite": suite,
                "task_id": task_id,
                "object_body": object_body,  # stored for cache reload
            }
            logger.info("  %s: sigma=%.3f cm  range=%.3f cm  unstable=%s",
                        name, sigma, range_cm, unstable)

        output_root.mkdir(parents=True, exist_ok=True)
        probe_cache_path.write_text(json.dumps(probe_results, indent=2))
        logger.info("Saved variance probe cache -> %s", probe_cache_path)

    # Median split -> assign variance_level; also restore object_body from cache if auto-detected
    sigmas = {n: v["sigma_cm"] for n, v in probe_results.items()
              if n in obj_names}
    median_sigma = float(np.median(list(sigmas.values())))
    logger.info("Variance probe: median sigma = %.3f cm", median_sigma)
    for obj_spec in obj_specs:
        name = obj_spec["name"]
        # Restore auto-detected object_body from cache when config has null
        if not obj_spec.get("object_body") and name in probe_results:
            obj_spec["object_body"] = probe_results[name].get("object_body")
        if name not in sigmas:
            logger.warning("  %s not in probe results -- defaulting to low variance", name)
            obj_spec["variance_level"] = "low"
            obj_spec["sigma_cm"] = 0.0
            continue
        sigma = sigmas[name]
        # Respect explicit override
        if obj_spec.get("variance_level") not in (None, "auto", ""):
            logger.info("  %s: variance_level explicitly set to %s (sigma=%.3f cm)",
                        name, obj_spec["variance_level"], sigma)
        else:
            obj_spec["variance_level"] = "high" if sigma >= median_sigma else "low"
            logger.info("  %s -> %s (sigma=%.3f cm, median=%.3f cm)",
                        name, obj_spec["variance_level"], sigma, median_sigma)
        obj_spec["sigma_cm"] = sigma

    return probe_results


def _probe_rollout_displacement(
    env, policy, init_state, pbid, task_description, cfg_env, probe_steps, max_disp_cm
) -> tuple[float, np.ndarray]:
    """Run policy for probe_steps steps, return (displacement_cm, agent_frames).

    Reads the initial object position after reset, runs the rollout, then reads
    the final position. env still holds the final sim state when _rollout_one_episode
    returns, so we can read body_xpos immediately after.
    """
    # Get initial position (after reset that _rollout_one_episode will do internally)
    # We do a quick pre-reset here to snapshot the start position, then let the
    # rollout do its own reset.
    env.reset()
    env.set_init_state(init_state)
    initial_pos = np.array(env.sim.data.body_xpos[pbid], dtype=np.float64)

    payload, _, _ = _rollout_one_episode(
        policy=policy,
        env=env,
        init_state=init_state,
        task_description=task_description,
        prompt=None,
        max_steps=probe_steps,
        num_steps_wait=int(cfg_env.get("num_steps_wait", 10)),
        replan_steps=int(cfg_env.get("replan_steps", 5)),
        resize_size=int(cfg_env.get("resize_size", 224)),
        env_max_reward=float(cfg_env.get("env_max_reward", 1.0)),
    )
    # env still holds the final sim state at this point
    final_pos = np.array(env.sim.data.body_xpos[pbid], dtype=np.float64)
    disp_cm = float(np.linalg.norm(final_pos - initial_pos) * 100.0)
    disp_cm = min(disp_cm, float(max_disp_cm))
    return disp_cm, payload.get("agent_rgb", np.zeros((0,), dtype=np.uint8))


def run_variance_probe_policy(cfg, output_root, policy, bench, reprobe=False):
    """Phase A (policy variant): measure friction sensitivity via policy rollouts.

    Instead of kinematically replaying a demo, this runs the actual pi0.5 policy
    for probe_steps steps at each probe friction value and measures the resulting
    object displacement. This captures the *effective* variance experienced during
    policy-based data collection, accounting for any compensation the policy makes.

    Results cached to <output_root>/variance_probe_policy.json.
    """
    import h5py

    probe_cache_path = output_root / "variance_probe_policy.json"
    probe_cfg = cfg["variance_probe"]
    hv = cfg["hidden_var"]
    cfg_env = cfg["env"]
    obj_specs = cfg["objects"]
    obj_names = {o["name"] for o in obj_specs}
    dataset_path = cfg.get("libero_dataset_path")

    # Cache check (same pattern as run_variance_probe)
    if not reprobe and probe_cache_path.exists():
        cached = json.loads(probe_cache_path.read_text())
        if obj_names <= set(cached.keys()):
            logger.info("Loaded policy probe cache from %s", probe_cache_path)
            probe_results = cached
        else:
            missing = obj_names - set(cached.keys())
            logger.info("Cache missing %s -- re-probing", missing)
            probe_results = None
    else:
        probe_results = None

    if probe_results is None:
        probe_mu_values = np.geomspace(
            float(hv["low"]), float(hv["high"]), int(probe_cfg["num_mu"])
        )
        probe_steps = int(probe_cfg.get("probe_steps", 80))
        max_disp_cm = float(probe_cfg.get("max_disp_cm", 60.0))
        resolution = int(cfg_env.get("resolution", 256))
        seed = int(cfg_env.get("seed", 7))
        probe_results = {}

        for obj_spec in obj_specs:
            name = obj_spec["name"]
            suite = obj_spec["suite"]
            task_id = obj_spec["task_id"]
            object_body = obj_spec.get("object_body")

            logger.info("Policy-probing %s (suite=%s task_id=%d object_body=%s)",
                        name, suite, task_id, object_body)

            env, task = _build_env(suite, task_id, res=resolution)
            env.seed(seed)

            task_suite = bench[suite]()
            init_state = task_suite.get_task_init_states(task_id)[0]

            # Auto-detect object_body from demo if not specified
            if not object_body:
                demo_file = _resolve_demo_path(suite, task, dataset_path)
                if not pathlib.Path(demo_file).exists():
                    logger.warning("  no demo file (%s) -- cannot auto-detect body, skipping %s",
                                   demo_file, name)
                    env.close()
                    continue
                f = h5py.File(demo_file, "r")
                demo_names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
                states = np.asarray(f["data"][demo_names[0]]["states"])
                f.close()
                object_body, det_disp = _detect_object_body(env, states)
                obj_spec["object_body"] = object_body
                logger.info("  auto-detected object_body=%s (demo disp=%.3fm)",
                            object_body, det_disp)

            # Get plate handles (needs one env reset first)
            env.reset()
            env.set_init_state(init_state)
            pbid, plate_geoms, defaults = _plate_handles(env, object_body)

            displacements, all_frames = [], []
            for mu in probe_mu_values:
                _restore_plate_physics(env, pbid, plate_geoms, defaults)
                _set_plate_friction(env, float(mu), plate_geoms)
                disp_cm, frames = _probe_rollout_displacement(
                    env, policy, init_state, pbid,
                    task.language, cfg_env, probe_steps, max_disp_cm
                )
                displacements.append(disp_cm)
                all_frames.append(frames)
                logger.info("  mu=%.4f  disp=%.2f cm", mu, disp_cm)

            _restore_plate_physics(env, pbid, plate_geoms, defaults)

            # Save side-by-side probe video
            probe_dir = output_root / "probe_videos_policy" / name
            try:
                _save_frames_row_video(all_frames, probe_dir)
                logger.info("  saved policy probe video -> %s", probe_dir)
            except Exception as e:
                logger.warning("  probe video save failed: %s", e)

            env.close()

            disps = np.clip(np.array(displacements, dtype=np.float64), 0.0, max_disp_cm)
            sigma = float(np.std(disps))
            range_cm = float(np.ptp(disps))

            probe_results[name] = {
                "sigma_cm": sigma,
                "range_cm": range_cm,
                "displacements_cm": disps.tolist(),
                "mu_values": probe_mu_values.tolist(),
                "probe_steps": probe_steps,
                "suite": suite,
                "task_id": task_id,
                "object_body": object_body,
            }
            logger.info("  %s: sigma=%.3f cm  range=%.3f cm", name, sigma, range_cm)

        output_root.mkdir(parents=True, exist_ok=True)
        probe_cache_path.write_text(json.dumps(probe_results, indent=2))
        logger.info("Saved policy probe cache -> %s", probe_cache_path)

    # Median split -> assign variance_level (identical to run_variance_probe)
    sigmas = {n: v["sigma_cm"] for n, v in probe_results.items() if n in obj_names}
    median_sigma = float(np.median(list(sigmas.values())))
    logger.info("Policy probe: median sigma = %.3f cm", median_sigma)
    for obj_spec in obj_specs:
        name = obj_spec["name"]
        if not obj_spec.get("object_body") and name in probe_results:
            obj_spec["object_body"] = probe_results[name].get("object_body")
        if name not in sigmas:
            logger.warning("  %s not in probe results -- defaulting to low variance", name)
            obj_spec["variance_level"] = "low"
            obj_spec["sigma_cm"] = 0.0
            continue
        sigma = sigmas[name]
        if obj_spec.get("variance_level") not in (None, "auto", ""):
            logger.info("  %s: variance_level explicitly set to %s (sigma=%.3f cm)",
                        name, obj_spec["variance_level"], sigma)
        else:
            obj_spec["variance_level"] = "high" if sigma >= median_sigma else "low"
            logger.info("  %s -> %s (sigma=%.3f cm, median=%.3f cm)",
                        name, obj_spec["variance_level"], sigma, median_sigma)
        obj_spec["sigma_cm"] = sigma

    return probe_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_root", type=str, default=None,
                    help="Override save.output_root from the config.")
    ap.add_argument("--reprobe", action="store_true",
                    help="Re-run variance probe even if cache exists.")
    ap.add_argument("--probe_only", action="store_true",
                    help="Run probe and print plan, then exit without collecting.")
    ap.add_argument("--policy_probe", action="store_true",
                    help="Probe variance via policy rollouts instead of demo replay. "
                         "Requires the policy to be loaded before the probe runs.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_yaml(args.config)

    pol_cfg = cfg.get("policy", {})
    env_cfg = cfg.get("env", {})
    save_cfg = cfg.get("save", {})
    hv = cfg["hidden_var"]
    probe_cfg = cfg["variance_probe"]
    data_levels = cfg["data_levels"]

    openpi_repo = pol_cfg.get("repo_path", "external/openpi")
    _ensure_openpi_paths(openpi_repo)

    output_root = pathlib.Path(args.output_root or save_cfg["output_root"])

    # Config parameters
    write_raw = bool(save_cfg.get("write_raw", True))
    encode_latents = bool(save_cfg.get("encode_latents", True))
    down_sample = max(1, int(save_cfg.get("down_sample", 1)))
    raw_fps = int(save_cfg.get("raw_fps", 20))
    out_fps = max(1, raw_fps // down_sample)
    val_fraction = float(save_cfg.get("val_fraction", 0.0))
    num_history = int(save_cfg.get("num_history", 6))
    num_frames = int(save_cfg.get("num_frames", 5))

    resolution = int(env_cfg.get("resolution", 256))
    num_steps_wait = int(env_cfg.get("num_steps_wait", 10))
    replan_steps = int(env_cfg.get("replan_steps", 5))
    resize_size = int(env_cfg.get("resize_size", 224))
    seed = int(env_cfg.get("seed", 7))
    env_max_reward = float(env_cfg.get("env_max_reward", 1.0))
    trajectories_per_env = int(data_levels.get("trajectories_per_env", 1))

    rng = np.random.default_rng(seed)
    checkpoint_path = pol_cfg["checkpoint_path"]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()

    # ---- Phase A: variance probe ----------------------------------------- #
    logger.info("=" * 64)
    logger.info("Phase A: variance probe (%s)", "policy-rollout" if args.policy_probe else "demo-replay")

    if args.policy_probe:
        # Policy probe requires the policy to be loaded first
        from openworld.policies.openpi_loader import load_policy_from_checkpoint
        logger.info("Loading policy %s from %s (needed for policy probe)",
                    pol_cfg.get("config_name", "pi05_libero"), checkpoint_path)
        policy = load_policy_from_checkpoint(
            config_name=pol_cfg.get("config_name", "pi05_libero"),
            checkpoint_path=checkpoint_path,
            repo_path=openpi_repo,
            default_prompt=None,
            pytorch_device=pol_cfg.get("pytorch_device", "cuda"),
        )
        probe_results = run_variance_probe_policy(
            cfg, output_root, policy, bench, reprobe=args.reprobe
        )
    else:
        policy = None
        probe_results = run_variance_probe(cfg, output_root, reprobe=args.reprobe)

    obj_specs = cfg["objects"]
    logger.info("=" * 64)
    logger.info("Sampling plan:")
    total_eps = 0
    for o in obj_specs:
        n_envs = int(data_levels[o["data_level"]])
        n_eps = n_envs * trajectories_per_env
        total_eps += n_eps
        logger.info("  %-10s  %s_var / %s_data  n_envs=%d  n_eps=%d  sigma=%.2fcm",
                    o["name"], o.get("variance_level", "?"), o["data_level"],
                    n_envs, n_eps, o.get("sigma_cm", 0.0))
    logger.info("  TOTAL %d episodes", total_eps)

    if args.probe_only:
        logger.info("--probe_only: exiting before collection.")
        return

    # ---- Phase B: load policy and SVD-VAE --------------------------------- #
    if policy is None:
        # Not yet loaded (demo-probe path)
        from openworld.policies.openpi_loader import load_policy_from_checkpoint
        logger.info("Loading policy %s from %s",
                    pol_cfg.get("config_name", "pi05_libero"), checkpoint_path)
        policy = load_policy_from_checkpoint(
            config_name=pol_cfg.get("config_name", "pi05_libero"),
            checkpoint_path=checkpoint_path,
            repo_path=openpi_repo,
            default_prompt=None,
            pytorch_device=pol_cfg.get("pytorch_device", "cuda"),
        )

    encoder = None
    if encode_latents:
        svd_path = save_cfg.get("svd_path")
        if not svd_path:
            raise SystemExit(
                "save.encode_latents is true but save.svd_path is unset.")
        encoder = LatentEncoder(svd_path, device=save_cfg.get("device", "cuda"))

    # ---- Phase C: collection loop ----------------------------------------- #
    logger.info("=" * 64)
    logger.info("Phase C: policy rollout collection")

    # Manifest tracking: per-cell episode IDs
    cell_episodes: dict[str, list[str]] = {
        "high_large": [], "high_small": [],
        "low_large": [], "low_small": [],
    }
    manifest_objects = []

    # Track all IDs per suite for sample-list rebuild
    suite_train_ids: dict[str, list[str]] = {}
    suite_val_ids: dict[str, list[str]] = {}

    for obj_spec in obj_specs:
        suite = obj_spec["suite"]
        task_id = obj_spec["task_id"]
        variance_level = obj_spec["variance_level"]
        data_level = obj_spec["data_level"]
        object_body = obj_spec["object_body"]
        obj_name = obj_spec["name"]

        N_envs = int(data_levels[data_level])
        mu_list = _sample_mu(rng, hv, N_envs)

        cell_key = f"{variance_level}_{data_level}"
        if cell_key not in cell_episodes:
            cell_episodes[cell_key] = []

        task_suite = bench[suite]()
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        bddl_file = task.bddl_file
        if not pathlib.Path(bddl_file).exists():
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

        # Get plate handles for friction control
        pbid, plate_geoms, defaults = _plate_handles(env, object_body)

        max_steps = SUITE_MAX_STEPS.get(suite, 300)
        episode_id = _next_episode_id(output_root, suite)

        if suite not in suite_train_ids:
            suite_train_ids[suite] = []
            suite_val_ids[suite] = []

        obj_episode_ids = []
        logger.info("[%s] %s: %s_var/%s_data, %d envs x %d rollouts",
                    suite, obj_name, variance_level, data_level,
                    N_envs, trajectories_per_env)

        for env_idx, mu in enumerate(
            tqdm.tqdm(mu_list, desc=f"{obj_name}")
        ):
            # Set friction for this environment
            _restore_plate_physics(env, pbid, plate_geoms, defaults)
            _set_plate_friction(env, float(mu), plate_geoms)

            for trial in range(trajectories_per_env):
                init_state = init_states[
                    (env_idx * trajectories_per_env + trial) % len(init_states)
                ]

                payload, is_success, steps = _rollout_one_episode(
                    policy=policy,
                    env=env,
                    init_state=init_state,
                    task_description=task_description,
                    prompt=None,
                    max_steps=max_steps,
                    num_steps_wait=num_steps_wait,
                    replan_steps=replan_steps,
                    resize_size=resize_size,
                    env_max_reward=env_max_reward,
                )

                # Temporal subsample
                if down_sample > 1:
                    payload = {
                        "agent_rgb": payload["agent_rgb"][::down_sample],
                        "wrist_rgb": payload["wrist_rgb"][::down_sample],
                        "cart": payload["cart"][::down_sample],
                        "grip": payload["grip"][::down_sample],
                    }
                payload["language"] = task_description
                payload["bddl"] = pathlib.Path(task.bddl_file).stem

                eid = f"{episode_id:06d}"
                split = "val" if rng.random() < val_fraction else "train"

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
                        "object_name": obj_name,
                        "object_body": object_body,
                        "trial": int(trial),
                        "env_idx": int(env_idx),
                        "mu": float(mu),
                        "variance_level": variance_level,
                        "data_level": data_level,
                        "uncertainty_cell": f"{variance_level}_var/{data_level}_data",
                        "sigma_cm": round(obj_spec.get("sigma_cm", 0.0), 3),
                        "is_success": bool(is_success),
                        "episode_steps": int(steps),
                        "policy_checkpoint": str(checkpoint_path),
                        "source": "policy_rollout_uq",
                    },
                )
                (suite_train_ids[suite] if split == "train" else suite_val_ids[suite]).append(eid)
                cell_episodes[cell_key].append(eid)
                obj_episode_ids.append(eid)
                episode_id += 1

        # Restore physics before closing
        _restore_plate_physics(env, pbid, plate_geoms, defaults)
        env.close()

        manifest_objects.append({
            "name": obj_name,
            "suite": suite,
            "task_id": task_id,
            "task_name": task_description,
            "object_body": object_body,
            "variance_level": variance_level,
            "data_level": data_level,
            "sigma_cm": round(obj_spec.get("sigma_cm", 0.0), 3),
            "mu_values": mu_list.tolist(),
            "n_episodes": len(obj_episode_ids),
            "episode_ids": obj_episode_ids,
        })
        logger.info("[%s] %s done: %d episodes",
                    suite, obj_name, len(obj_episode_ids))

    # ---- Rebuild sample indices ------------------------------------------- #
    for suite, train_ids in suite_train_ids.items():
        suite_root = output_root / suite
        for split in ("train", "val"):
            all_ids = sorted(
                p.stem
                for p in (suite_root / "annotation" / split).glob("*.json")
                if p.stem.isdigit()
            )
            if all_ids:
                write_sample_list(suite_root, split, all_ids,
                                  num_history=num_history, num_frames=num_frames,
                                  down_sample=1)

    # ---- Write manifest ---------------------------------------------------- #
    manifest = {
        "config": str(args.config),
        "hidden_var": hv,
        "variance_probe": probe_results,
        "cells": {k: v for k, v in cell_episodes.items()},
        "objects": manifest_objects,
        "total_episodes": sum(len(v) for v in cell_episodes.values()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "uq_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote manifest -> %s", manifest_path)
    logger.info("Collection complete: %d total episodes -> %s",
                manifest["total_episodes"], output_root)


if __name__ == "__main__":
    main()
