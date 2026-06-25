"""Friction calibration probe for LIBERO same-history / different-future data.

Goal: confirm that sweeping the pushed object's friction produces a *measurable
spread of futures* (different stopping distances) while the *history is shared*
(pre-slide frames are byte-identical across the sweep).

This is the LIBERO analog of mujoco_suite/scripts/collect_sliding_distances.py:
fix the initial state + a fixed open-loop "settle then impart velocity" script,
sweep friction mu, and map mu -> sliding distance.

Method (per branch / per mu):
  1. env.seed(SEED); env.reset()                 -> identical initial state s0
  2. set plate + table sliding friction = mu     -> the only thing that differs
  3. settle K steps with a dummy action          -> plate at rest, frames identical
  4. impart a fixed linear velocity to the plate -> the "push" (open loop)
  5. step T more times, record plate xy each step -> measure where it stops

Then we report mu -> slide distance, and verify that the settle-phase frames are
identical across all mu (the shared-history property).

Run:
  export PATH="$HOME/.local/bin:$PATH"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
    uv run --no-sync python scripts/calib_libero_friction.py
"""

from __future__ import annotations

import os
import numpy as np

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# ---- knobs -----------------------------------------------------------------
SUITE = "libero_goal"
TASK_ID = 5                       # "push the plate to the front of the stove"
OBJECT_BODY = "plate_1_main"      # the pushed object
SEED = 7
RES = 128
SETTLE_STEPS = 8                  # identical-history window (plate at rest)
SLIDE_STEPS = 60                  # observation window for the slide
PUSH_VEL = np.array([0.4, 0.0, 0.0])   # m/s imparted to the plate at release
MU_SWEEP = [0.05, 0.1, 0.2, 0.4, 0.8]  # the hidden variable
STOP_SPEED = 1e-3                 # m/s below which the plate is "stopped"
DUMMY_ACTION = [0.0] * 6 + [-1.0]
OUT_DIR = "outputs/libero_uq_calib"
# ---------------------------------------------------------------------------


def build_env():
    task = get_benchmark(SUITE)().get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), SUITE, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    return env, task


def _plate_handles(env):
    m = env.env.sim.model
    pbid = m.body_name2id(OBJECT_BODY)
    # free-joint linear-velocity dof address
    jid = next(j for j in range(m.njnt) if m.jnt_bodyid[j] == pbid)
    dofadr = m.jnt_dofadr[jid]
    plate_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pbid]
    table_geom = m.geom_name2id("table_collision")
    return pbid, dofadr, plate_geoms, table_geom


def set_friction(env, mu, plate_geoms, table_geom):
    m = env.env.sim.model
    for g in plate_geoms:
        m.geom_friction[g][0] = mu
    m.geom_friction[table_geom][0] = mu
    env.env.sim.forward()


def run_branch(env, mu):
    """Returns (settle_frames, slide_distance, speed_profile, release_xy, stop_xy)."""
    env.seed(SEED)
    obs = env.reset()
    pbid, dofadr, plate_geoms, table_geom = _plate_handles(env)
    set_friction(env, mu, plate_geoms, table_geom)

    d = env.env.sim.data

    settle_frames = [obs["agentview_image"].copy()]
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        settle_frames.append(obs["agentview_image"].copy())

    release_xy = d.body_xpos[pbid][:2].copy()

    # impart the fixed "push" velocity (open loop) -- the same every branch
    d.qvel[dofadr : dofadr + 3] = PUSH_VEL
    env.env.sim.forward()

    speeds = []
    last_xy = release_xy.copy()
    stop_xy = release_xy.copy()
    for t in range(SLIDE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        xy = d.body_xpos[pbid][:2].copy()
        v = d.qvel[dofadr : dofadr + 2]
        sp = float(np.linalg.norm(v))
        speeds.append(sp)
        stop_xy = xy
        if sp < STOP_SPEED and t > 2:
            break
        last_xy = xy

    slide_distance = float(np.linalg.norm(stop_xy - release_xy))
    return settle_frames, slide_distance, speeds, release_xy, stop_xy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env, task = build_env()
    print(f"Task: {task.language}  (suite={SUITE} id={TASK_ID})")
    print(f"Pushed object: {OBJECT_BODY}   push_vel={PUSH_VEL.tolist()} m/s")
    print(f"Friction sweep (sliding coeff): {MU_SWEEP}")
    print("-" * 68)

    results = []
    settle_by_mu = {}
    for mu in MU_SWEEP:
        settle_frames, dist, speeds, rxy, sxy = run_branch(env, mu)
        settle_by_mu[mu] = settle_frames
        peak = max(speeds) if speeds else 0.0
        nstep = len(speeds)
        results.append((mu, dist, peak, nstep))
        print(f"mu={mu:<5}  slide_distance={dist*100:6.2f} cm   "
              f"peak_speed={peak:4.2f} m/s   steps_to_stop={nstep}")

    print("-" * 68)
    # shared-history check: settle frames identical across all mu?
    ref = settle_by_mu[MU_SWEEP[0]]
    max_diff = 0
    for mu in MU_SWEEP[1:]:
        for fa, fb in zip(ref, settle_by_mu[mu]):
            max_diff = max(max_diff, int(np.abs(fa.astype(int) - fb.astype(int)).max()))
    print(f"Shared-history check: max pixel diff over {SETTLE_STEPS+1} settle "
          f"frames across all mu = {max_diff}  "
          f"({'IDENTICAL' if max_diff == 0 else 'DIVERGED'})")

    dists = [r[1] for r in results]
    spread = (max(dists) - min(dists)) * 100
    print(f"Future spread: stopping distance varies by {spread:.2f} cm "
          f"across the mu sweep ({min(dists)*100:.2f}..{max(dists)*100:.2f} cm)")

    # save a montage: settle-end frame + final frame for min and max mu
    try:
        import imageio.v3 as iio
        lo, hi = MU_SWEEP[0], MU_SWEEP[-1]
        # re-run lo/hi to grab final frames for the montage
        frames = {}
        for mu in (lo, hi):
            env.seed(SEED); obs = env.reset()
            pbid, dofadr, pg, tg = _plate_handles(env)
            set_friction(env, mu, pg, tg)
            d = env.env.sim.data
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            settle_img = obs["agentview_image"].copy()
            d.qvel[dofadr:dofadr+3] = PUSH_VEL; env.env.sim.forward()
            for _ in range(SLIDE_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            frames[mu] = (settle_img, obs["agentview_image"].copy())
        top = np.concatenate([frames[lo][0], frames[hi][0]], axis=1)
        bot = np.concatenate([frames[lo][1], frames[hi][1]], axis=1)
        montage = np.concatenate([top, bot], axis=0)[::-1]  # flip: libero imgs are upside down
        path = os.path.join(OUT_DIR, "friction_montage.png")
        iio.imwrite(path, montage)
        print(f"Montage (rows: settle | final; cols: mu={lo} | mu={hi}) -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"(montage skipped: {e})")

    env.close()


if __name__ == "__main__":
    main()
