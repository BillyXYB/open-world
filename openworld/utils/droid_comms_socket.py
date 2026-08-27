"""Direct socket-based comms for driving real DROID hardware, replacing
``droid_comms.py``'s file-based IPC to cut disk/rsync/polling latency. Kept
in a fully separate module so the file-based path stays untouched --
select between them via ``hardware.comms_mode`` in the collection config.

Wire protocol: ONE persistent websocket connection per trajectory, opened by
the robot at trajectory start and closed at trajectory end. Connection
lifetime IS the trajectory-done signal -- no separate in-band "done" file is
needed, unlike the file-based protocol's ``trajectory_{traj_idx}_done.txt``
(an explicit in-band ``{"done": True}`` message is also accepted, as a
belt-and-suspenders option, but the robot side in this pipeline just closes
the connection). Each round-trip is exactly one blocking send-then-recv on
each side:

    robot  -> server: packed {"obs": [obs_dict, ...]}             (last-5 observations, matching the file protocol's payload)
    server -> robot:  packed {"action": (5, 8) float32, "text": str}

Uses the same numpy-safe msgpack convention as ``openpi_client``'s
``websocket_client_policy``/``websocket_policy_server`` (arrays packed as
``{data: bytes, dtype, shape}``, no compression), reimplemented standalone
here so this module doesn't depend on the openpi submodule's internal
layout (its two vendored copies have already drifted slightly from each
other) and doesn't require adding ``openpi-client`` as a dependency of this
environment, which doesn't otherwise have it installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

import msgpack
import numpy as np
import websockets.sync.server
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

_NDARRAY_KEY = "__ndarray__"

# Real obs payloads (multi-frame, multi-camera uint8 images) routinely exceed
# websockets' default 1 MiB max_size -- both ends of this protocol must pass
# max_size=None. compression=None matches openpi_client's WebsocketClientPolicy
# (raw image bytes don't compress well and deflate just adds latency, which
# is the entire point of this module).
WS_MAX_SIZE = None
WS_COMPRESSION = None


def _pack_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_KEY: True, "data": obj.tobytes(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"droid_comms_socket: cannot serialize object of type {type(obj)}")


def _unpack_object_hook(obj: dict) -> Any:
    if obj.get(_NDARRAY_KEY):
        # np.frombuffer is read-only (backed by the immutable msgpack bytes);
        # copy so callers can freely index/mutate the result like any other
        # in-memory array, matching what np.load(...) from the file-based
        # path already hands them.
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"]).copy()
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_default, use_bin_type=True)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_object_hook, raw=False)


class SocketChannel:
    """Adapts one live websocket connection (one trajectory) to the same
    ``poll_observation``/``send_action`` interface
    ``droid_comms.FileChannel`` exposes, so ``_rollout_trajectory_hardware``
    can be driven by either transport without caring which one is underneath.
    """

    def __init__(self, connection) -> None:
        self._ws = connection

    def poll_observation(
        self, t_step: int, timeout_s: float, poll_interval_s: Optional[float] = None,
    ) -> dict:
        # poll_interval_s is accepted-but-unused: recv() blocks until a
        # message arrives (or times out), so there's no polling loop to
        # space out here -- kept only so callers can pass the exact same
        # kwargs as FileChannel.poll_observation without branching.
        del poll_interval_s
        try:
            data = self._ws.recv(timeout=timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for t_step={t_step}'s observation "
                "over the socket connection")
        except ConnectionClosed:
            return {"done": True}
        msg = unpack(data)
        if msg.get("done"):
            return {"done": True}
        return {"done": False, "obs": msg["obs"]}

    def send_action(self, t_step: int, action: np.ndarray, text: str) -> None:
        del t_step  # unused here (no per-step filename to manage, unlike FileChannel)
        self._ws.send(pack({"action": np.asarray(action, dtype=np.float32), "text": text}))


def serve_trajectories(
    host: str,
    port: int,
    handle_trajectory: Callable[[SocketChannel, int], None],
    start_traj_idx: int = 1,
    max_trajectories: Optional[int] = None,
) -> None:
    """Accept connections until ``max_trajectories`` complete successfully
    (or forever, if ``None``); each new connection is treated as one
    trajectory. ``handle_trajectory(channel, traj_idx)`` should run exactly
    the same per-trajectory logic ``main()`` already runs for the file-based
    path (candidate generation, WM scoring, episode export, etc.) -- this
    function only owns the connection/traj_idx bookkeeping and the shutdown
    condition; it does not interpret the trajectory's contents.

    Trajectories are processed strictly one at a time via a shared lock, even
    though ``websockets.sync.server`` dispatches each connection on its own
    thread by default -- this preserves the same single-flight semantics the
    file-based main() loop already has (one GPU, one decision_metrics.jsonl
    writer, one traj_idx counter) and guards against a stray/overlapping
    second connection racing the first. A trajectory that raises (including a
    mid-trajectory disconnect surfacing as ``ConnectionClosed``) is logged and
    does NOT count toward ``max_trajectories`` -- only a clean completion does.
    """
    lock = threading.Lock()
    state = {"next_traj_idx": start_traj_idx, "n_done": 0, "server": None}

    def _handler(connection) -> None:
        with lock:
            traj_idx = state["next_traj_idx"]
            state["next_traj_idx"] += 1
            logger.info("[socket] connection accepted -> traj_idx %d", traj_idx)
            channel = SocketChannel(connection)
            try:
                handle_trajectory(channel, traj_idx)
            except ConnectionClosed:
                logger.info("[traj %d] connection closed mid-trajectory", traj_idx)
                return
            except Exception:
                logger.exception("[traj %d] handler raised", traj_idx)
                return
            state["n_done"] += 1
            if max_trajectories is not None and state["n_done"] >= max_trajectories:
                logger.info(
                    "Reached max_trajectories=%d, shutting down socket server",
                    max_trajectories)
                if state["server"] is not None:
                    state["server"].shutdown()

    with websockets.sync.server.serve(
        _handler, host, port, max_size=WS_MAX_SIZE, compression=WS_COMPRESSION,
    ) as server:
        state["server"] = server
        logger.info("Socket comms server listening on %s:%d", host, port)
        server.serve_forever()
