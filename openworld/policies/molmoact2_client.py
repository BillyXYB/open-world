"""Dependency-free TCP client for the MolmoAct2-DROID inference server
(``~/molmoact2/serve.py``), used as the ``molmoact2`` policy backend of
``scripts/run_droid_hardware_active_uq.py``.

Runs in the ``open-world`` env -- imports only ``socket``/``struct``/``pickle``/
``numpy`` (no torch, no lerobot; MolmoAct2 lives in a separate process/env).
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

import pickle
import socket
import struct
from typing import Optional

import numpy as np

_HEADER = struct.Struct(">I")


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
    ) -> None:
        self.host = host
        self.port = int(port)
        self.norm_tag = norm_tag
        self.num_steps = None if num_steps is None else int(num_steps)
        self.n_action_steps = int(n_action_steps)
        self.camera_order = list(camera_order or ["wrist_image", "left_image", "right_image"])
        self.seed = int(seed)
        self.gripper_scale = float(gripper_scale)
        self.timeout_s = float(timeout_s)
        self._conn: Optional[socket.socket] = None

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
                "needs >= 9 (stride-2 x 5 with a prepended current row)"
            )
        return out
