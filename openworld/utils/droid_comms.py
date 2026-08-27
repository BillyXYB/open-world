"""File-based comms protocol for driving real DROID hardware from a
server-side world-model planner.

Mirrors the wire protocol already used in production by
``svd_ac_video_model/vidwm/evaluators/uq_data_collection.py`` exactly, so the
ROBOT side (``uq_data_collection/examples/droid/main.py``) needs zero code
changes to talk to a server built on this module:

    <comms_dir>/obs/droid_observations_instructions_{traj_idx}_{t_step}.npz
        npz key "obs": array of dicts (the robot's last 5 observations),
        each with left_image/right_image/wrist_image (uint8 RGB),
        cartesian_position/joint_position/gripper_position, instruction[,
        instruction_list].
    <comms_dir>/obs/trajectory_{traj_idx}_done.txt
        End-of-trajectory signal written by the robot (safety reset,
        timestep limit, or normal completion).
    <comms_dir>/pred/droid_predicted_actions_{traj_idx}_{t_step}.npz
        npz keys "action" (shape (5, 8)) and "text" (instruction string),
        written by the server for the robot to consume next.

``traj_idx`` starts at 1 (the robot increments its trajectory counter before
sending its first observation), not 0.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


def send_file_rsync(
    source_path: Any,
    destination_path: str,
    options: str = "-avzp",
) -> None:
    """Thin rsync wrapper -- port of
    ``svd_ac_video_model/vidwm/evaluators/utils/file_transfer_utils.py``.
    No-op destination validation: callers are responsible for passing a
    reachable rsync target (``user@host:/path/`` or a local directory)."""
    cmd = ["rsync", options, str(source_path), str(destination_path)]
    subprocess.run(cmd, check=True)


def poll_for_observation(
    comms_dir: Path,
    traj_idx: int,
    t_step: int,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.2,
) -> dict[str, Any]:
    """Block until the robot's obs npz or its trajectory-done signal appears.

    Returns ``{"done": True}`` if the done-signal file is seen first (the
    caller should break out of the per-trajectory loop). Otherwise returns
    ``{"done": False, "obs": <last-5 observation dicts array>}``.

    Raises ``TimeoutError`` if neither file appears within ``timeout_s``.
    """
    obs_dir = Path(comms_dir) / "obs"
    obs_path = obs_dir / f"droid_observations_instructions_{traj_idx}_{t_step}.npz"
    done_path = obs_dir / f"trajectory_{traj_idx}_done.txt"

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if done_path.exists():
            return {"done": True}
        if obs_path.exists():
            try:
                with np.load(obs_path, allow_pickle=True) as data:
                    return {"done": False, "obs": data["obs"]}
            except (EOFError, ValueError, OSError):
                # File may still be mid-write (rsync/copy race); retry.
                pass
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for {obs_path} or {done_path}"
    )


def write_predicted_action(
    comms_dir: Path,
    traj_idx: int,
    t_step: int,
    action: np.ndarray,
    text: str,
    rsync_dest: Optional[str] = None,
    rsync_options: str = "-avzp",
) -> Path:
    """Write ``pred/droid_predicted_actions_{traj_idx}_{t_step}.npz`` and
    delete the previous step's pred file (stale-file hygiene, matching
    ``uq_data_collection.py::send_action``). ``action`` must already be
    shaped ``(5, 8)`` -- this function does not reshape or validate it, so
    callers should assert the shape before calling.
    """
    pred_dir = Path(comms_dir) / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    file_path = pred_dir / f"droid_predicted_actions_{traj_idx}_{t_step}.npz"
    np.savez(file_path, action=np.asarray(action, dtype=np.float32), text=text)

    if t_step > 0:
        prev_path = pred_dir / f"droid_predicted_actions_{traj_idx}_{t_step - 1}.npz"
        prev_path.unlink(missing_ok=True)

    if rsync_dest is not None:
        send_file_rsync(file_path, rsync_dest, rsync_options)

    return file_path


def clear_comms_dir(comms_dir: Path) -> None:
    """Delete every obs/pred file (any traj_idx) and done signal, regardless
    of which trajectory they belong to -- port of
    ``svd_ac_video_model/bash_scripts/evaluate/uq_data_collect.sh``'s
    "clear stale comms/obs and comms/pred directories" step, which the old
    pipeline runs before every server launch. Call this once at server
    startup, before entering the trajectory loop: unlike
    ``cleanup_trajectory_files`` (which only cleans up a completed
    trajectory's own files), this also removes leftovers from a run that
    crashed or was interrupted mid-trajectory -- e.g. a stale
    ``droid_observations_instructions_1_0.npz`` from a previous session,
    which would otherwise be misread as the CURRENT run's first real
    observation.
    """
    comms_dir = Path(comms_dir)
    for subdir, pattern in (
        ("obs", "droid_observations_instructions_*.npz"),
        ("obs", "trajectory_*_done.txt"),
        ("pred", "droid_predicted_actions_*.npz"),
    ):
        for p in (comms_dir / subdir).glob(pattern):
            p.unlink(missing_ok=True)


class FileChannel:
    """Adapts the file-based comms functions above to the small
    ``poll_observation``/``send_action`` interface
    ``_rollout_trajectory_hardware`` uses, so it can be driven by either this
    or ``droid_comms_socket.SocketChannel`` without caring which transport is
    underneath. One instance per trajectory (mirrors one ``traj_idx``)."""

    def __init__(self, comms_dir: Path, traj_idx: int) -> None:
        self.comms_dir = Path(comms_dir)
        self.traj_idx = traj_idx

    def poll_observation(self, t_step: int, timeout_s: float, poll_interval_s: float) -> dict:
        return poll_for_observation(
            self.comms_dir, self.traj_idx, t_step,
            timeout_s=timeout_s, poll_interval_s=poll_interval_s)

    def send_action(self, t_step: int, action: np.ndarray, text: str) -> None:
        write_predicted_action(self.comms_dir, self.traj_idx, t_step, action, text)


def cleanup_trajectory_files(comms_dir: Path, traj_idx: int) -> None:
    """Delete every obs/pred file (and the done signal) belonging to one
    trajectory -- port of ``uq_data_collection.py::_cleanup_trajectory_files``.
    Call this once a trajectory ends, before starting the next ``traj_idx``.
    """
    comms_dir = Path(comms_dir)
    for pattern, subdir in (
        (f"droid_observations_instructions_{traj_idx}_*.npz", "obs"),
        (f"droid_predicted_actions_{traj_idx}_*.npz", "pred"),
    ):
        for p in (comms_dir / subdir).glob(pattern):
            p.unlink(missing_ok=True)
    done_path = comms_dir / "obs" / f"trajectory_{traj_idx}_done.txt"
    done_path.unlink(missing_ok=True)
