"""Build a 2x2 uncertainty-controlled LIBERO dataset.

Realizes the design discussed in docs: a dataset whose scenarios have *known*
ground-truth uncertainty, decomposed into the two canonical kinds --

  variance (ALEATORIC, irreducible)  -- how widely an object's OUTCOME spreads
      when the hidden physical variable (sliding friction mu) is drawn at random.
      Measured per object by a friction probe sweep, then binned high/low.

  data (EPISTEMIC, reducible)        -- how many TRAINING episodes the object
      gets. Frequent objects are seen often (low epistemic); rare objects seldom
      (high epistemic). Set by `data_level` in the registry.

Each generated episode is a *distinct draw*: mu is sampled from the hidden-var
distribution and the robot is kinematically pinned to a demo (reusing the
counterfactual machinery of collect_libero_branch_sets). So "more data" means
more COVERAGE of the outcome distribution, not repeated identical trajectories.

Crucial subtlety (see the conceptual notes): epistemic uncertainty is ~ sigma/sqrt(N).
A flat count (e.g. 100 vs 10) applied to both variance rows does NOT give clean 2x2
corners -- 10 episodes is "small" for a high-variance object but "plenty" for a
low-variance one. `count_mode: coverage` compensates by setting N proportional to
sigma**2, so the estimation error matches within a data level and the two axes stay
independent. `count_mode: flat` keeps the raw counts (and the interaction).

Each object is rendered with a fixed colour (texture) so a scenario's set is
visually identifiable -- the same tint mechanism added to collect_libero_branch_sets.

Output (WM training format, same writers as preprocess_libero_for_wm):
  <out>/<suite>/raw_videos/{agentview,wrist}/<eid>.mp4
  <out>/<suite>/states/<eid>.npz
  <out>/<suite>/annotation/train/<eid>.json   (carries variance_level/data_level/mu)
  <out>/uq_manifest.json                       (per-object + per-cell tallies)

Run:
  export PATH="$HOME/.local/bin:$PATH"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
    uv run --no-sync python scripts/build_uq_dataset.py \
      --config configs/uq/objects_2x2.yaml --out data/uq_2x2
  # preview the sampling plan without generating anything:
  uv run --no-sync python scripts/build_uq_dataset.py --config ... --plan-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys

import h5py
import numpy as np
import yaml

# reuse the env / physics / replay / writer machinery
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from collect_libero_branch_sets import (  # noqa: E402
    _build_env,
    _detect_object_body,
    _plate_handles,
    _resolve_demo_file,
    _run_counterfactual_branch,
)
from preprocess_libero_for_wm import (  # noqa: E402
    LatentEncoder,
    write_episode,
    write_sample_list,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# fallback per-cell hues when an object omits `color`
_CELL_HUE = {
    ("high", "large"): [0.95, 0.80, 0.10],
    ("high", "small"): [0.90, 0.15, 0.15],
    ("low", "large"): [0.20, 0.40, 1.00],
    ("low", "small"): [0.20, 0.85, 0.30],
}


def _sample_mu(rng, hv, n):
    lo, hi = float(hv["low"]), float(hv["high"])
    if hv.get("distribution", "loguniform") == "loguniform":
        return np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))
    return rng.uniform(lo, hi, size=n)


def _load_demos(suite, task_id, res):
    """Open a task's env + demo file; return (env, task, object_body, demos)
    where demos is a list of (states, actions). Returns None if unusable."""
    env, task = _build_env(suite, task_id, res)
    demo_file = _resolve_demo_file(suite, task, None)
    if not pathlib.Path(demo_file).exists():
        logger.warning("  no demo file (%s) -- skipping", demo_file)
        env.close()
        return None
    f = h5py.File(demo_file, "r")
    data = f["data"]
    names = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
    demos = [(np.asarray(data[n]["states"]), np.asarray(data[n]["actions"]))
             for n in names]
    object_body, det_disp = _detect_object_body(env, demos[0][0])
    f.close()
    return env, task, object_body, det_disp, demos


def _measure_variance(env, object_body, demos, hv, num_mu, max_disp_cm):
    """Run the manipulated object through a friction probe sweep on the first demo
    and return (sigma_cm, range_cm, unstable). sigma = std of final object
    displacement (cm), the empirical aleatoric-variance estimate used to bin the
    object. Displacements above `max_disp_cm` indicate the object was ejected
    (sim blow-up under the kinematic pin), not real variance -- they are clipped
    and flagged so they don't dominate the coverage counts."""
    pbid, geoms, defaults = _plate_handles(env, object_body)
    states, actions = demos[0]
    mus = np.geomspace(float(hv["low"]), float(hv["high"]), num_mu)
    disps = []
    for mu in mus:
        _, _, _, disp = _run_counterfactual_branch(
            env, states=states, actions=actions, mu=float(mu), mass_scale=None,
            object_body=object_body, plate_geoms=geoms, pbid_=pbid,
            defaults=defaults, obj_rgba=None)
        disps.append(disp * 100.0)
    disps = np.asarray(disps)
    unstable = bool(np.any(disps > max_disp_cm))
    disps = np.clip(disps, 0.0, max_disp_cm)
    return float(np.std(disps)), float(np.ptp(disps)), unstable


def _assign_counts(objs, cfg):
    """Set obj['N'] from data_level, count_mode, and measured sigma."""
    dl = cfg["data_levels"]
    mode = cfg.get("count_mode", "coverage")
    ref = float(cfg.get("ref_sigma_cm", 8.0))
    nmax = int(cfg.get("max_count", 400))
    nmin = int(cfg.get("min_count", 4))
    for o in objs:
        base = int(dl[o["data_level"]])
        if mode == "coverage":
            scale = (o["sigma_cm"] / ref) ** 2 if ref > 0 else 1.0
            n = int(round(base * scale))
        else:  # flat
            n = base
        o["N"] = int(min(nmax, max(nmin, n)))


def _bin_variance(objs, probe_cfg):
    """Fill obj['variance_level'] for any object set to 'auto'."""
    auto = [o for o in objs if o.get("variance_level", "auto") == "auto"]
    if not auto:
        return
    if probe_cfg.get("split", "median") == "threshold":
        thr = float(probe_cfg.get("threshold_cm", 6.0))
    else:
        thr = float(np.median([o["sigma_cm"] for o in auto]))
    for o in auto:
        o["variance_level"] = "high" if o["sigma_cm"] >= thr else "low"
    logger.info("variance bin threshold = %.2f cm (split=%s)",
                thr, probe_cfg.get("split", "median"))


def _generate_object(env, o, demos, hv, rng, out_root, suite, fps, encoder,
                     next_id, all_ids):
    """Generate o['N'] episodes for one object; return updated next_id."""
    pbid, geoms, defaults = _plate_handles(env, o["object_body"])
    color = np.array(list(o["color"]) + [1.0], dtype=np.float32)
    mus = _sample_mu(rng, hv, o["N"])
    records = []
    for k in range(o["N"]):
        states, actions = demos[k % len(demos)]
        payload, succ, contact, disp = _run_counterfactual_branch(
            env, states=states, actions=actions, mu=float(mus[k]), mass_scale=None,
            object_body=o["object_body"], plate_geoms=geoms, pbid_=pbid,
            defaults=defaults, obj_rgba=color)
        payload["language"] = o["language"]
        payload["bddl"] = o["bddl_stem"]
        eid = str(next_id); next_id += 1
        ann = {
            "task_suite": suite, "task_id": o["task_id"], "task_name": o["language"],
            "bddl": o["bddl_stem"], "object_name": o["name"],
            "object_body": o["object_body"],
            "variance_level": o["variance_level"], "data_level": o["data_level"],
            "uncertainty_cell": f'{o["variance_level"]}_var/{o["data_level"]}_data',
            "hidden_var": hv["name"], "mu": float(mus[k]),
            "sigma_cm": round(o["sigma_cm"], 3),
            "draw_idx": k, "demo_idx": k % len(demos),
            "color_rgba": [float(c) for c in color],
            "is_success": bool(succ), "contact_frame_idx": contact,
            "object_displacement_m": disp,
            "num_steps": int(payload["cart"].shape[0]),
        }
        write_episode(suite=suite, split="train", episode_id=eid,
                      output_root=out_root, encoder=encoder, payload=payload,
                      fps=fps, write_raw=True, extra_annotation=ann)
        states_dir = out_root / suite / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            states_dir / f"{eid}.npz",
            eef_cartesian=payload["cart"], gripper=payload["grip"],
            object_pose=payload["obj_pose"], actions=payload["actions"],
            mu=float(mus[k]), variance_level=o["variance_level"],
            data_level=o["data_level"], object_name=o["name"],
            is_success=bool(succ), object_displacement_m=disp)
        all_ids.append(eid)
        records.append({"episode_id": eid, "mu": round(float(mus[k]), 4),
                        "is_success": bool(succ),
                        "object_displacement_m": round(disp, 4)})
    o["episodes"] = records
    o["num_success"] = int(sum(r["is_success"] for r in records))
    logger.info("  %-16s %s_var/%s_data  N=%d  succ=%d/%d  sigma=%.2fcm",
                o["name"], o["variance_level"], o["data_level"], o["N"],
                o["num_success"], o["N"], o["sigma_cm"])
    return next_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/uq/objects_2x2.yaml")
    ap.add_argument("--out", default="data/uq_2x2")
    ap.add_argument("--plan-only", action="store_true",
                    help="measure variance + print the sampling plan, generate nothing")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every per-object count (cheap dry runs, e.g. 0.1)")
    ap.add_argument("--encode-latents", action="store_true")
    ap.add_argument("--svd-path", default="external/stable-video-diffusion-img2vid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-history", type=int, default=6)
    ap.add_argument("--num-frames", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    res, fps = int(cfg.get("res", 96)), int(cfg.get("fps", 20))
    hv = cfg["hidden_var"]
    out_root = pathlib.Path(args.out)

    # ---- Phase A: measure variance for every object ----
    logger.info("=" * 64)
    logger.info("Phase A: measuring outcome variance (%d-mu probe over [%.3f,%.3f])",
                int(cfg["variance_probe"]["num_mu"]), hv["low"], hv["high"])
    objs = []
    for spec in cfg["objects"]:
        o = dict(spec)
        loaded = _load_demos(o["suite"], o["task_id"], res)
        if loaded is None:
            continue
        env, task, object_body, det_disp, demos = loaded
        o["object_body"] = o.get("object_body") or object_body
        o["language"] = task.language
        o["bddl_stem"] = pathlib.Path(task.bddl_file).stem
        o["color"] = o.get("color") or _CELL_HUE[(o.get("variance_level", "low"),
                                                  o["data_level"])]
        sigma, ptp, unstable = _measure_variance(
            env, o["object_body"], demos, hv, int(cfg["variance_probe"]["num_mu"]),
            float(cfg["variance_probe"].get("max_disp_cm", 60.0)))
        o["sigma_cm"], o["ptp_cm"], o["_env"], o["_demos"] = sigma, ptp, env, demos
        flag = "  [UNSTABLE: clipped, set object_body or drop]" if unstable else ""
        logger.info("  %-16s task %d  object=%s  sigma=%.2fcm (range %.2fcm)%s",
                    o["name"], o["task_id"], o["object_body"], sigma, ptp, flag)
        objs.append(o)

    if not objs:
        logger.error("no usable objects (missing demos?) -- nothing to do")
        return

    # ---- Phase B: bin variance + assign counts ----
    _bin_variance(objs, cfg["variance_probe"])
    _assign_counts(objs, cfg)
    if args.scale != 1.0:
        for o in objs:
            o["N"] = max(1, int(round(o["N"] * args.scale)))

    logger.info("=" * 64)
    logger.info("Sampling plan (count_mode=%s, scale=%g):", cfg.get("count_mode"), args.scale)
    cells: dict = {}
    for o in objs:
        cell = f'{o["variance_level"]}_var/{o["data_level"]}_data'
        cells.setdefault(cell, [0, 0])
        cells[cell][0] += 1; cells[cell][1] += o["N"]
        logger.info("  %-16s %-22s sigma=%5.2fcm  N=%d",
                    o["name"], cell, o["sigma_cm"], o["N"])
    logger.info("-" * 64)
    for cell, (nobj, nep) in sorted(cells.items()):
        logger.info("  CELL %-22s %d objects, %d episodes", cell, nobj, nep)
    total = sum(o["N"] for o in objs)
    logger.info("  TOTAL %d episodes across %d objects", total, len(objs))

    if args.plan_only:
        for o in objs:
            o["_env"].close()
        logger.info("plan-only: no episodes generated")
        return

    # ---- Phase C: generate ----
    logger.info("=" * 64)
    logger.info("Phase C: generating %d episodes", total)
    encoder = LatentEncoder(args.svd_path, device=args.device) if args.encode_latents else None
    suite = objs[0]["suite"]
    if any(o["suite"] != suite for o in objs):
        logger.warning("objects span multiple suites; writing all under '%s'", suite)
    all_ids: list[str] = []
    next_id = 0
    for o in objs:
        next_id = _generate_object(o["_env"], o, o["_demos"], hv, rng, out_root,
                                   suite, fps, encoder, next_id, all_ids)
        o["_env"].close()

    if args.encode_latents:
        write_sample_list(out_root / suite, "train", all_ids,
                          args.num_history, args.num_frames, 1)

    manifest = {
        "config": str(args.config), "suite": suite, "hidden_var": hv,
        "count_mode": cfg.get("count_mode"), "total_episodes": len(all_ids),
        "cells": {c: {"objects": v[0], "episodes": v[1]} for c, v in cells.items()},
        "objects": [{k: o[k] for k in (
            "name", "task_id", "object_body", "variance_level", "data_level",
            "sigma_cm", "ptp_cm", "N", "num_success", "color", "episodes")}
            for o in objs],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "uq_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("=" * 64)
    logger.info("Wrote %d episodes -> %s  (manifest: uq_manifest.json)",
                len(all_ids), out_root)


if __name__ == "__main__":
    main()
