"""Collect LIBERO 'branch set' data for uncertainty quantification.

A *branch set* is one shared history + a distribution of futures. Each set is
anchored to a real LIBERO human **demonstration** (which completes the task), and
the other branches are physical **counterfactuals** that replay the demo's exact
actions under a swept hidden variable (object sliding friction). Every branch
shares the demo's initial state and action sequence, so the history is identical
up to the moment the object is contacted, then the futures diverge with friction.

This is the LIBERO port of mujoco_suite's push sweeps, but grounded in the
original demonstration data rather than a scripted push.

Per branch set (one demo):
  branch 0  : the demonstration -- replay actions at default physics. Reproduces
              the original successful trajectory (is_demonstration=True).
  branch 1..: counterfactuals -- same init + same actions, plate friction = mu.
              Friction only bites after contact, so the pre-contact history is
              shared and only the future (does the task still succeed? where does
              the object end up?) diverges.

Source demos: LIBERO's own dataset (download once):
  uv run python external/openpi/third_party/libero/benchmark_scripts/\
download_libero_datasets.py --datasets libero_goal

Output (per suite) is self-contained SIM DATA + a branch_manifest.json:
  <out>/<suite>/raw_videos/{agentview,wrist}/<eid>.mp4   sim renders
  <out>/<suite>/states/<eid>.npz                          eef/object/actions/success
  <out>/<suite>/annotation/<split>/<eid>.json             tags + metadata
  <out>/<suite>/latent_videos/.../<eid>.pt                (optional, --encode-latents)
  <out>/branch_manifest.json

Run:
  export PATH="$HOME/.local/bin:$PATH"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
    uv run --no-sync python scripts/collect_libero_branch_sets.py \
      --num-branch-sets 3 --out data/libero_branch_sets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

import h5py
import numpy as np

# Reuse the exact WM-format writers so optional latent export is byte-identical
# to preprocess_libero_for_wm / run_data_collection output.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess_libero_for_wm import (  # noqa: E402
    LatentEncoder,
    write_episode,
    write_sample_list,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# obs / state helpers (copied from run_data_collection to stay decoupled)
# --------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


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
    used = []
    for split in ("train", "val"):
        ann_dir = output_root / suite / "annotation" / split
        if ann_dir.is_dir():
            for p in ann_dir.glob("*.json"):
                if p.stem.isdigit():
                    used.append(int(p.stem))
    return (max(used) + 1) if used else 0


# --------------------------------------------------------------------------
# env / physics helpers
# --------------------------------------------------------------------------
def _build_env(suite, task_id, res):
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    task = get_benchmark(suite)().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), suite, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    return env, task


def _plate_handles(env, object_body):
    m = env.env.sim.model
    pbid = m.body_name2id(object_body)
    plate_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pbid]
    defaults = {
        "friction": [m.geom_friction[g].copy() for g in plate_geoms],
        "priority": [int(m.geom_priority[g]) for g in plate_geoms],
        "mass": float(m.body_mass[pbid]),
        "inertia": m.body_inertia[pbid].copy(),
        "rgba": [m.geom_rgba[g].copy() for g in plate_geoms],
        "matid": [int(m.geom_matid[g]) for g in plate_geoms],
    }
    return pbid, plate_geoms, defaults


def _set_plate_friction(env, mu, plate_geoms):
    """Set the plate's sliding friction = mu, raising its contact priority so it
    dominates the plate-table contact. The table is left untouched (the gripper
    contacts it during the push, so changing table friction would make the shared
    history mu-dependent)."""
    m = env.env.sim.model
    for g in plate_geoms:
        m.geom_friction[g][0] = mu
        m.geom_priority[g] = 1
    env.env.sim.forward()


def _set_plate_mass(env, pbid, mass_scale, defaults):
    """Scale the plate's mass (and inertia, to stay consistent) by mass_scale."""
    m = env.env.sim.model
    m.body_mass[pbid] = defaults["mass"] * mass_scale
    m.body_inertia[pbid] = defaults["inertia"] * mass_scale
    env.env.sim.forward()


def _restore_plate_physics(env, pbid, plate_geoms, defaults):
    """Restore LIBERO-default plate physics AND appearance (used for the
    demonstration branch and before each counterfactual); geom_friction/priority/
    mass/rgba/matid persist across env.reset(), so reset them explicitly."""
    m = env.env.sim.model
    for g, fr, pr, rgba, matid in zip(plate_geoms, defaults["friction"],
                                      defaults["priority"], defaults["rgba"],
                                      defaults["matid"]):
        m.geom_friction[g] = fr
        m.geom_priority[g] = pr
        m.geom_rgba[g] = rgba
        m.geom_matid[g] = matid
    m.body_mass[pbid] = defaults["mass"]
    m.body_inertia[pbid] = defaults["inertia"]
    env.env.sim.forward()


# Distinct, ordered tints keyed to the friction sweep (low mu -> high mu):
# blue, cyan, green, yellow, orange, red, magenta. Re-used cyclically if the
# sweep is longer.
_MU_PALETTE = [
    (0.20, 0.40, 1.00), (0.10, 0.80, 0.90), (0.20, 0.85, 0.30),
    (0.95, 0.80, 0.10), (0.95, 0.45, 0.10), (0.90, 0.15, 0.15),
    (0.80, 0.15, 0.65),
]


def _mu_to_rgba(mu, mu_sweep):
    """Map a mu value to a distinct RGBA tint by its rank in the sweep, so each
    friction counterfactual is visually identifiable by the object's colour."""
    order = sorted(set(mu_sweep))
    idx = order.index(mu) if mu in order else 0
    r, g, b = _MU_PALETTE[idx % len(_MU_PALETTE)]
    return np.array([r, g, b, 1.0], dtype=np.float32)


def _set_object_color(env, plate_geoms, rgba):
    """Override the manipulated object's surface colour. Detaches each geom from
    its material (matid=-1) so the flat rgba is what renders instead of the
    object's original texture."""
    m = env.env.sim.model
    for g in plate_geoms:
        m.geom_matid[g] = -1
        m.geom_rgba[g] = rgba
    env.env.sim.forward()


# --------------------------------------------------------------------------
# branch rollouts
# --------------------------------------------------------------------------
# Robot (arm + gripper) qpos/qvel indices in the LIBERO Panda state.
ROBOT_IDX = list(range(0, 9))


def _make_payload(agent_frames, wrist_frames, cart_list, grip_list, obj_pose, actions):
    return {
        "agent_rgb": np.stack(agent_frames, axis=0),
        "wrist_rgb": np.stack(wrist_frames, axis=0),
        "cart": np.stack(cart_list, axis=0),
        "grip": np.asarray(grip_list, dtype=np.float32),
        "obj_pose": np.stack(obj_pose, axis=0),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def _run_demo_branch(env, *, states, actions, object_body, plate_geoms, pbid_,
                     defaults, env_max_reward=1.0):
    """The demonstration branch: full-state playback of the recorded demo. Setting
    the entire recorded sim state each frame reproduces the original trajectory
    exactly -- robot AND object follow the human demo (so it completes the task),
    and it is rendered through the same pipeline as the counterfactuals. The robot
    joint trajectory here is identical to what the counterfactuals pin to."""
    env.reset()
    _restore_plate_physics(env, pbid_, plate_geoms, defaults)
    env.set_init_state(states[0])
    sim = env.env.sim
    m, d = sim.model, sim.data
    pbid = m.body_name2id(object_body)
    nq, nv = m.nq, m.nv
    start_xy = None                       # set from the object body after frame 0
    af, wf, cl, gl, op = [], [], [], [], []
    is_success, contact = False, None
    for i in range(len(states)):
        d.qpos[:] = states[i][1:1 + nq]
        d.qvel[:] = states[i][1 + nq:1 + nq + nv]
        sim.forward()
        if start_xy is None:
            start_xy = d.body_xpos[pbid][:2].copy()
        o = env.env._get_observations(force_update=True)
        _record_obs(o, af, wf, cl, gl)
        op.append(np.concatenate((d.body_xpos[pbid].copy(),
                                  d.body_xquat[pbid].copy())).astype(np.float32))
        if env.env._check_success():
            is_success = True
        if contact is None and float(np.linalg.norm(d.body_xpos[pbid][:2] - start_xy)) > 2e-3:
            contact = len(af) - 1
    disp = float(np.linalg.norm(d.body_xpos[pbid][:2] - start_xy))
    return _make_payload(af, wf, cl, gl, op, actions), is_success, contact, disp


def _run_counterfactual_branch(env, *, states, actions, mu, mass_scale, object_body,
                               plate_geoms, pbid_, defaults, obj_rgba=None,
                               env_max_reward=1.0):
    """A counterfactual branch: the robot is *kinematically pinned* to the demo's
    exact joint trajectory (so its motion is identical across all branches), while
    the object responds to a swept hidden variable -- plate friction (``mu``) and/or
    plate mass (``mass_scale``). Re-pins the robot qpos/qvel every physics substep
    so the controller never fights it."""
    env.reset()
    _restore_plate_physics(env, pbid_, plate_geoms, defaults)   # clear prior branch
    if mu is not None:
        _set_plate_friction(env, mu, plate_geoms)
    if mass_scale is not None:
        _set_plate_mass(env, pbid_, mass_scale, defaults)
    if obj_rgba is not None:
        _set_object_color(env, plate_geoms, obj_rgba)
    obs = env.set_init_state(states[0])
    sim = env.env.sim
    m, d = sim.model, sim.data
    pbid = m.body_name2id(object_body)
    qpos_d = states[:, 1:1 + m.nq]                       # state = [time, qpos, qvel]
    cdt = env.env.control_timestep
    nsub = int(round(cdt / m.opt.timestep))
    start_xy = d.body_xpos[pbid][:2].copy()
    af, wf, cl, gl, op = [], [], [], [], []

    def _snap():
        o = env.env._get_observations(force_update=True)
        _record_obs(o, af, wf, cl, gl)
        op.append(np.concatenate((d.body_xpos[pbid].copy(),
                                  d.body_xquat[pbid].copy())).astype(np.float32))
    _snap()
    is_success, contact = False, None
    for i in range(1, len(states)):
        q0, q1 = qpos_d[i - 1][ROBOT_IDX], qpos_d[i][ROBOT_IDX]
        qvel = (q1 - q0) / cdt
        for sub in range(nsub):
            frac = (sub + 1) / nsub
            d.qpos[ROBOT_IDX] = q0 + frac * (q1 - q0)
            d.qvel[ROBOT_IDX] = qvel
            sim.step()
        d.qpos[ROBOT_IDX] = q1
        d.qvel[ROBOT_IDX] = qvel
        sim.forward()
        _snap()
        if env.env._check_success():
            is_success = True
        if contact is None and float(np.linalg.norm(d.body_xpos[pbid][:2] - start_xy)) > 2e-3:
            contact = len(af) - 1
    disp = float(np.linalg.norm(d.body_xpos[pbid][:2] - start_xy))
    return _make_payload(af, wf, cl, gl, op, actions), is_success, contact, disp


def _resolve_demo_file(suite, task, demo_file):
    if demo_file:
        return demo_file
    from libero.libero import get_libero_path
    stem = pathlib.Path(task.bddl_file).stem            # e.g. push_..._stove
    return os.path.join(get_libero_path("datasets"), suite, f"{stem}_demo.hdf5")


def _detect_object_body(env, states):
    """Return (body_name, demo_displacement_m) of the free-joint body that moves
    most over the demo -- i.e. the object the policy manipulates. Used to pick the
    hidden-variable target automatically for an arbitrary task."""
    m = env.env.sim.model
    best = (None, -1.0)
    for j in range(m.njnt):
        if m.jnt_type[j] != 0:            # free joint only
            continue
        adr = m.jnt_qposadr[j]
        pos = states[:, 1 + adr:1 + adr + 3]      # state = [time, qpos, qvel]
        disp = float(np.linalg.norm(pos[-1] - pos[0]))
        if disp > best[1]:
            best = (m.body_id2name(m.jnt_bodyid[j]), disp)
    return best


def _process_task(args, task_id, out_root, encoder, next_id, manifest, all_ids):
    """Collect branch sets for one task; returns the updated next_id."""
    env, task = _build_env(args.suite, task_id, args.res)
    demo_file = _resolve_demo_file(args.suite, task, None)
    if not os.path.exists(demo_file):
        logger.warning("[task %d] no demo file (%s) -- skipping", task_id, demo_file)
        env.close()
        return next_id

    f = h5py.File(demo_file, "r")
    data = f["data"]
    demo_names = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
    demo_names = demo_names[args.demo_start_idx: args.demo_start_idx + args.num_branch_sets]

    # auto-detect the manipulated object from the first demo (or use override)
    states0 = np.asarray(data[demo_names[0]]["states"])
    if args.object_body:
        object_body, det_disp = args.object_body, None
    else:
        object_body, det_disp = _detect_object_body(env, states0)
    if det_disp is not None and det_disp < args.min_demo_disp:
        logger.warning("[task %d] %s: object '%s' moves only %.1fcm in the demo "
                       "(< %.0fcm) -- no manipulable sliding object, skipping",
                       task_id, task.language, object_body, det_disp * 100,
                       args.min_demo_disp * 100)
        f.close(); env.close()
        return next_id

    pbid, plate_geoms, defaults = _plate_handles(env, object_body)
    bddl_stem = pathlib.Path(task.bddl_file).stem
    logger.info("[task %d] %s | object=%s (demo moves %s) | %d demos",
                task_id, task.language, object_body,
                f"{det_disp*100:.1f}cm" if det_disp is not None else "override",
                len(demo_names))

    for demo_name in demo_names:
        demo = data[demo_name]
        states = np.asarray(demo["states"])
        actions = np.asarray(demo["actions"])
        bset = len([b for b in manifest["branch_sets"]])
        set_record = {"branch_set_id": bset, "task_id": task_id,
                      "task_name": task.language, "object_body": object_body,
                      "demo_name": demo_name, "num_actions": int(len(actions)),
                      "branches": []}
        # branch 0 = demonstration; then friction CFs; then mass CFs. Each spec is
        # (kind, hidden_var, value) where value is mu (friction) or mass_scale.
        branch_specs = [("demo", None, None)]
        branch_specs += [("cf", "object_sliding_friction", mu) for mu in args.mu_sweep]
        branch_specs += [("cf", "object_mass", ms) for ms in args.mass_sweep]
        for bidx, (kind, hvar, val) in enumerate(branch_specs):
            if kind == "demo":   # demonstration: full-state playback of the demo
                payload, succ, contact, disp = _run_demo_branch(
                    env, states=states, actions=actions, object_body=object_body,
                    plate_geoms=plate_geoms, pbid_=pbid, defaults=defaults)
            else:            # counterfactual: robot kinematically pinned to the demo
                # friction CFs vary mu at default mass; mass CFs vary mass at a
                # fixed low base friction (so the plate actually slides).
                cf_mu = val if hvar == "object_sliding_friction" else args.mass_base_mu
                cf_mass = val if hvar == "object_mass" else None
                # tint the object by mu so each friction counterfactual is visually
                # distinct (only for friction CFs, when enabled)
                obj_rgba = (_mu_to_rgba(val, args.mu_sweep)
                            if args.tint_by_mu and hvar == "object_sliding_friction"
                            else None)
                payload, succ, contact, disp = _run_counterfactual_branch(
                    env, states=states, actions=actions, mu=cf_mu, mass_scale=cf_mass,
                    object_body=object_body, plate_geoms=plate_geoms,
                    pbid_=pbid, defaults=defaults, obj_rgba=obj_rgba)
            payload["language"] = task.language
            payload["bddl"] = bddl_stem
            eid = str(next_id); next_id += 1

            is_demo = kind == "demo"
            mu_val = val if hvar == "object_sliding_friction" else None
            mass_val = val if hvar == "object_mass" else None
            write_episode(
                suite=args.suite, split="train", episode_id=eid, output_root=out_root,
                encoder=encoder, payload=payload, fps=args.fps, write_raw=True,
                extra_annotation={
                    "task_suite": args.suite, "task_id": task_id,
                    "task_name": task.language, "bddl": bddl_stem,
                    "branch_set_id": bset, "branch_id": bidx, "demo_name": demo_name,
                    "is_demonstration": is_demo,
                    "hidden_var": hvar,
                    "hidden_value": val,
                    "mu": mu_val,
                    "mass_scale": mass_val,
                    "is_success": bool(succ),
                    "contact_frame_idx": contact,
                    "object_displacement_m": disp,
                    "num_steps": int(payload["cart"].shape[0]),
                },
            )
            # self-contained sim-data record
            states_dir = out_root / args.suite / "states"
            states_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                states_dir / f"{eid}.npz",
                eef_cartesian=payload["cart"], gripper=payload["grip"],
                object_pose=payload["obj_pose"], actions=payload["actions"],
                hidden_var=(hvar or "none"), hidden_value=(np.nan if val is None else val),
                mu=(np.nan if mu_val is None else mu_val),
                mass_scale=(np.nan if mass_val is None else mass_val),
                is_demonstration=is_demo, is_success=bool(succ),
                contact_frame_idx=(-1 if contact is None else contact),
                object_displacement_m=disp, branch_set_id=bset, branch_id=bidx,
                demo_name=demo_name)
            all_ids.append(eid)
            set_record["branches"].append(
                {"episode_id": eid, "branch_id": bidx, "kind": kind,
                 "hidden_var": hvar, "hidden_value": val, "is_success": bool(succ),
                 "object_displacement_m": round(disp, 4), "contact_frame_idx": contact})
            tag = "DEMO" if is_demo else (f"mu={val}" if hvar == "object_sliding_friction"
                                          else f"mass x{val}")
            logger.info("  [b%d] %-10s success=%-5s obj_disp=%6.2fcm contact@%s eid=%s",
                        bidx, tag, succ, disp * 100, contact, eid)

        disps = [b["object_displacement_m"] for b in set_record["branches"]]
        n_succ = sum(b["is_success"] for b in set_record["branches"])
        set_record["disp_spread_cm"] = round((max(disps) - min(disps)) * 100, 2)
        set_record["num_success"] = int(n_succ)
        manifest["branch_sets"].append(set_record)
        logger.info("  set %d: %d/%d branches succeed; obj-disp spread = %.2f cm",
                    bset, n_succ, len(branch_specs), set_record["disp_spread_cm"])

    f.close()
    env.close()
    return next_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=5,
                    help="single task (default: push the plate to the front of the stove)")
    ap.add_argument("--task-ids", type=int, nargs="+", default=None,
                    help="run multiple tasks; overrides --task-id. Use -1 for ALL tasks "
                         "in the suite.")
    ap.add_argument("--object-body", default=None,
                    help="manipulated object body; default = auto-detect from the demo")
    ap.add_argument("--min-demo-disp", type=float, default=0.03,
                    help="skip a task if its object moves less than this (m) in the demo "
                         "(e.g. knob/button tasks with no sliding object)")
    ap.add_argument("--num-branch-sets", type=int, default=3,
                    help="how many demos PER TASK to use (each demo = one branch set)")
    ap.add_argument("--demo-start-idx", type=int, default=0)
    ap.add_argument("--mu-sweep", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.4, 0.8],
                    help="counterfactual object frictions")
    ap.add_argument("--mass-sweep", type=float, nargs="*", default=[],
                    help="counterfactual object mass SCALE factors (x default); empty=off")
    ap.add_argument("--mass-base-mu", type=float, default=0.1,
                    help="friction used for mass counterfactuals (sliding regime)")
    ap.add_argument("--tint-by-mu", action=argparse.BooleanOptionalAction, default=True,
                    help="tint the manipulated object a distinct colour per friction "
                         "value (--no-tint-by-mu to keep the original texture)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="data/libero_branch_sets")
    ap.add_argument("--encode-latents", action="store_true",
                    help="also write SVD latents + WM sample index (needs --svd-path)")
    ap.add_argument("--svd-path", default="external/stable-video-diffusion-img2vid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-history", type=int, default=6)
    ap.add_argument("--num-frames", type=int, default=5)
    args = ap.parse_args()

    # resolve the task list
    if args.task_ids is None:
        task_ids = [args.task_id]
    elif args.task_ids == [-1]:
        from libero.libero.benchmark import get_benchmark
        task_ids = list(range(get_benchmark(args.suite)().n_tasks))
    else:
        task_ids = args.task_ids

    out_root = pathlib.Path(args.out)
    logger.info("Suite=%s tasks=%s  friction=%s mass=%s", args.suite, task_ids,
                args.mu_sweep, args.mass_sweep)
    encoder = LatentEncoder(args.svd_path, device=args.device) if args.encode_latents else None
    manifest = {"suite": args.suite, "task_ids": task_ids, "mu_sweep": args.mu_sweep,
                "mass_sweep": args.mass_sweep, "branch_sets": []}
    all_ids: list[str] = []
    next_id = _next_episode_id(out_root, args.suite)

    for tid in task_ids:
        next_id = _process_task(args, tid, out_root, encoder, next_id, manifest, all_ids)

    if args.encode_latents:
        write_sample_list(out_root / args.suite, "train", all_ids,
                          args.num_history, args.num_frames, 1)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "branch_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # per-task summary
    by_task: dict = {}
    for s in manifest["branch_sets"]:
        by_task.setdefault(s["task_id"], []).append(s)
    logger.info("=" * 64)
    for tid, sets in sorted(by_task.items()):
        spreads = [s["disp_spread_cm"] for s in sets]
        succ = [s["num_success"] for s in sets]
        n_br = len(sets[0]["branches"]) if sets else 0
        logger.info("task %2d  %-44s  sets=%d  mean obj-spread=%.1fcm  mean succ=%.1f/%d",
                    tid, sets[0]["task_name"][:44], len(sets),
                    sum(spreads) / len(spreads), sum(succ) / len(succ), n_br)
    logger.info("Wrote %d episodes across %d branch sets -> %s",
                len(all_ids), len(manifest["branch_sets"]), out_root)


if __name__ == "__main__":
    main()
