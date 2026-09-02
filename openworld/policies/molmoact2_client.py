"""Dependency-free TCP client for the MolmoAct2-DROID inference server
(``~/molmoact2/serve.py``), used as the ``molmoact2`` policy backend of
``scripts/run_droid_hardware_active_uq.py``.

Runs in the ``open-world`` env -- imports only ``socket``/``struct``/``pickle``/
``numpy``/``logging`` (+ optional ``PIL`` only when ``debug_dump_dir`` is set).
MolmoAct2 lives in a separate process/env (no torch, no lerobot here).
Wire format is byte-for-byte the length-prefixed pickle protocol
``serve.py`` speaks (same as ``/home/tennyyin/droid/droid/evaluation/molmoact2_wrapper.py``):

    [4 bytes big-endian uint32 = N][N bytes pickle payload]

Request (candidate mode):
    {"images": list[np.uint8 HxWx3], "state": np.float32(8,) [joint(7),grip],
     "task": str, "n_candidates": int(>1), "seed": int, "num_steps": int|None,
     "n_action_steps": int, "norm_tag": str}
Response:
    np.float32 (n_candidates, n_action_steps, 8) = [joint_pos(7), gripper]
"""

from __future__ import annotations

import logging
import pathlib
import pickle
import socket
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HEADER = struct.Struct(">I")

# Order the standalone (known-working) path sends cameras in:
# droid/evaluation/molmoact2_wrapper.py::_CAMERA_SERIALS = [hand_camera_id,
# varied_camera_1_id, varied_camera_2_id] = [wrist, 38872458, 31177322], and
# examples/droid/main.py maps right_camera_id=38872458, left_camera_id=31177322.
_DEFAULT_CAMERA_ORDER = ["wrist_image", "right_image", "left_image"]


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        chunk = conn.recv_into(view[pos:], n - pos)
        if not chunk:
            raise ConnectionResetError("MolmoAct2 server disconnected")
        pos += chunk
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> object:
    (length,) = _HEADER.unpack(_recv_exact(conn, _HEADER.size))
    return pickle.loads(_recv_exact(conn, length))  # noqa: S301


def _send_msg(conn: socket.socket, obj: object) -> None:
    data = pickle.dumps(obj, protocol=2)
    conn.sendall(_HEADER.pack(len(data)) + data)


class MolmoAct2Client:
    """One persistent TCP connection to ``serve.py``; one ``infer_candidates``
    round-trip per active-UQ decision."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 9999,
        norm_tag: str = "franka_droid",
        num_steps: Optional[int] = 10,
        n_action_steps: int = 15,
        camera_order: Optional[list[str]] = None,
        seed: int = 0,
        gripper_scale: float = 1.0,
        timeout_s: float = 20.0,
        debug_dump_dir: Optional[str] = None,
        debug_dump_n: int = 0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.norm_tag = norm_tag
        self.num_steps = None if num_steps is None else int(num_steps)
        self.n_action_steps = int(n_action_steps)
        self.camera_order = list(camera_order or _DEFAULT_CAMERA_ORDER)
        self.seed = int(seed)
        self.gripper_scale = float(gripper_scale)
        self.timeout_s = float(timeout_s)
        self._conn: Optional[socket.socket] = None
        # opt-in: for the first `debug_dump_n` calls, log what we send + what
        # comes back, and save a side-by-side PNG of the images. Confirms the
        # live model is getting good frames and producing real chunks.
        self._dump_dir = pathlib.Path(debug_dump_dir) if debug_dump_dir else None
        self._dump_n = int(debug_dump_n)
        self._call = 0

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.settimeout(self.timeout_s)
        self._conn.connect((self.host, self.port))

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _reconnect(self) -> None:
        self.close()
        self.connect()

    # ------------------------------------------------------------------
    def _build_msg(self, obs: dict, prompt: str, num_candidates: int) -> dict:
        images = []
        for key in self.camera_order:
            if key not in obs:
                raise KeyError(
                    f"MolmoAct2Client: obs is missing camera key {key!r} "
                    f"(have {sorted(k for k in obs if 'image' in k)})"
                )
            img = np.ascontiguousarray(np.asarray(obs[key], dtype=np.uint8))
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            images.append(img)

        joint = np.asarray(obs["joint_position"], dtype=np.float32).reshape(-1)[:7]
        grip = float(np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)[0])
        state = np.concatenate([joint, [grip]], axis=0).astype(np.float32)

        return {
            "images": images,
            "state": state,
            "task": str(prompt),
            "n_candidates": int(num_candidates),
            "seed": self.seed,
            "num_steps": self.num_steps,
            "n_action_steps": self.n_action_steps,
            "norm_tag": self.norm_tag,
        }

    def _debug_dump(self, msg: dict, out: np.ndarray) -> None:
        imgs = msg["images"]
        state = np.asarray(msg["state"], dtype=np.float32)
        drift = np.linalg.norm(out[0, :, :7] - state[None, :7], axis=1)
        logger.info(
            "[ma2 dump %d] task=%r  order=%s  img shapes/means=%s  state=%s",
            self._call, msg["task"], self.camera_order,
            [(i.shape, round(float(i.mean()), 1)) for i in imgs],
            np.round(state, 3).tolist())
        logger.info(
            "[ma2 dump %d] returned chunk[0] per-step joint drift from current: %s "
            "(gripper col: %s)",
            self._call, np.round(drift, 3).tolist(),
            np.round(out[0, :, 7], 3).tolist())
        try:
            from PIL import Image
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            h = max(i.shape[0] for i in imgs)
            tiles = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs]
            Image.fromarray(np.concatenate(tiles, axis=1).astype(np.uint8)).save(
                self._dump_dir / f"ma2_obs_{self._call:04d}.png")
        except Exception as e:  # pragma: no cover
            logger.warning("[ma2 dump %d] PNG save failed: %s", self._call, e)

    def infer_candidates(
        self, obs: dict, external_camera: str, prompt: str, num_candidates: int
    ) -> np.ndarray:
        """Return np.float32 (num_candidates, n_action_steps, 8) absolute
        joint-position + gripper candidate chunks. ``external_camera`` is
        accepted for signature parity with the openpi path but unused
        (MolmoAct2 consumes all cameras in ``camera_order``)."""
        del external_camera
        if self._conn is None:
            raise RuntimeError("MolmoAct2Client.connect() not called")
        if int(num_candidates) < 2:
            raise ValueError("MolmoAct2 candidate mode needs num_candidates >= 2")

        msg = self._build_msg(obs, prompt, int(num_candidates))
        try:
            _send_msg(self._conn, msg)
            out = _recv_msg(self._conn)
        except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError):
            self._reconnect()
            _send_msg(self._conn, msg)
            out = _recv_msg(self._conn)

        out = np.asarray(out, dtype=np.float32)
        if out.ndim != 3 or out.shape[0] != int(num_candidates) or out.shape[-1] != 8:
            raise RuntimeError(
                f"MolmoAct2 server returned {out.shape}, expected "
                f"({num_candidates}, n_action_steps, 8)"
            )
        if out.shape[1] < 9:
            raise RuntimeError(
                f"MolmoAct2 chunk has {out.shape[1]} steps; the active-UQ pipeline "
                "needs enough for wire_skip*(wire_len-1) (>=9); exact check is in "
                "run_droid_hardware_active_uq.py::_ma2_chunk_to_adapted"
            )

        if self._dump_dir is not None and self._call < self._dump_n:
            try:
                self._debug_dump(msg, out)
            except Exception as e:  # pragma: no cover
                logger.warning("[ma2 dump %d] failed: %s", self._call, e)
        self._call += 1
        return out
