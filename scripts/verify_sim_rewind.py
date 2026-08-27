"""Standalone check: does LIBERO sim-state snapshot/restore actually undo
itself cleanly?

Verifies the "rewind" mechanism ``run_data_collection_active_uq.py`` relies
on: snapshot -> physically step forward -> restore -> re-step the SAME
actions should reproduce identical poses, and the wrapped robosuite env's
timestep/cur_time/done (``env.env.*`` -- LIBERO's ``OffScreenRenderEnv`` is a
thin wrapper, only ``sim``/``robots``/etc. are forwarded as properties) must
be restored too, since they live outside ``sim.get_state()`` (see
robosuite/environments/base.py). No policy or world model is needed --
actions are just small random deltas.

Usage:
    uv run scripts/verify_sim_rewind.py --task_suite libero_spatial --task_id 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_data_collection import (  # noqa: E402
    _enable_legacy_torch_load,
    _ensure_openpi_paths,
)
from run_data_collection_active_uq import (  # noqa: E402
    _pose_from_obs,
    _restore_env,
    _snapshot_env,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi_repo", default="external/openpi")
    ap.add_argument("--task_suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _ensure_openpi_paths(a.openpi_repo)
    _enable_legacy_torch_load()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    task_suite = bench[a.task_suite]()
    task = task_suite.get_task(a.task_id)
    init_states = task_suite.get_task_init_states(a.task_id)
    bddl_file = task.bddl_file
    if not pathlib.Path(bddl_file).exists():
        from libero.libero import get_libero_path
        bddl_file = str(
            pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
    env.seed(a.seed)
    env.reset()
    env.set_init_state(init_states[0])
    for _ in range(10):
        env.step([0.0] * 6 + [-1.0])

    rng = np.random.default_rng(a.seed)
    actions = [
        (rng.uniform(-0.3, 0.3, size=6).tolist() + [-1.0 if i % 2 == 0 else 1.0])
        for i in range(a.num_steps)
    ]

    baseline_timestep, baseline_done = env.env.timestep, env.env.done
    snapshot = _snapshot_env(env)

    def run_actions():
        poses, grips = [], []
        for a_ in actions:
            obs, _, _, _ = env.step(a_)
            pose, grip = _pose_from_obs(obs)
            poses.append(pose)
            grips.append(grip)
        return np.stack(poses), np.asarray(grips), env.env.timestep, env.env.done

    poses_a, grips_a, timestep_a, done_a = run_actions()
    trial_timestep, trial_done = env.env.timestep, env.env.done

    _restore_env(env, snapshot)
    restored_timestep, restored_done = env.env.timestep, env.env.done

    poses_b, grips_b, timestep_b, done_b = run_actions()

    print(f"[verify] baseline timestep={baseline_timestep} done={baseline_done}")
    print(f"[verify] after trial A: timestep={trial_timestep} done={trial_done}")
    print(f"[verify] after restore: timestep={restored_timestep} done={restored_done} "
          f"(should equal baseline: {restored_timestep == baseline_timestep and restored_done == baseline_done})")
    print(f"[verify] after trial B: timestep={timestep_b} done={done_b} "
          f"(should equal trial A's: {timestep_b == timestep_a and done_b == done_a})")

    pose_diff = float(np.abs(poses_a - poses_b).max())
    grip_diff = float(np.abs(grips_a - grips_b).max())
    print(f"[verify] max abs pose diff  A vs B: {pose_diff:.3e}")
    print(f"[verify] max abs grip diff A vs B: {grip_diff:.3e}")

    ok = (
        restored_timestep == baseline_timestep and restored_done == baseline_done
        and timestep_b == timestep_a and done_b == done_a
        and pose_diff < 1e-5 and grip_diff < 1e-5
    )
    print(f"[verify] REWIND {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
