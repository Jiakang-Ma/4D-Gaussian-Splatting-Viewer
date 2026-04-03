"""Training bridge — TCP socket server for real-time training data."""

import json
import socket
import struct
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingFrame:
    """One frame of training data received from the trainer."""
    step: int = 0
    means: Optional[np.ndarray] = None          # [N, 3]
    opacities: Optional[np.ndarray] = None       # [N, 1]
    velocities: Optional[np.ndarray] = None      # [N, 3]
    times: Optional[np.ndarray] = None           # [N, 1]
    durations: Optional[np.ndarray] = None       # [N, 1]
    grad_accum: Optional[np.ndarray] = None      # [N]
    relocation_indices: List[int] = field(default_factory=list)
    relocation_targets: Optional[np.ndarray] = None  # [M, 3]


def serialize_frame(frame: TrainingFrame) -> bytes:
    """Serialize a TrainingFrame to wire format (for trainer side)."""
    arrays = {}
    offset = 0
    layout = {}

    def add_array(name, arr):
        nonlocal offset
        if arr is None:
            return
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        arrays[name] = arr
        layout[name] = {
            "offset": offset,
            "shape": list(arr.shape),
            "dtype": "float32"
        }
        offset += arr.nbytes

    add_array("means", frame.means)
    add_array("opacities", frame.opacities)
    add_array("velocities", frame.velocities)
    add_array("times", frame.times)
    add_array("durations", frame.durations)
    add_array("grad_accum", frame.grad_accum)

    if frame.relocation_indices:
        idx_arr = np.array(frame.relocation_indices, dtype=np.int32)
        arrays["relocation_indices"] = idx_arr
        layout["relocation_indices"] = {
            "offset": offset,
            "shape": list(idx_arr.shape),
            "dtype": "int32"
        }
        offset += idx_arr.nbytes

    add_array("relocation_targets", frame.relocation_targets)

    n_gauss = frame.means.shape[0] if frame.means is not None else 0
    header = {
        "step": frame.step,
        "n_gaussians": n_gauss,
        "has_grad_accum": frame.grad_accum is not None,
        "n_relocations": len(frame.relocation_indices),
        "payload_layout": layout,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    # Wire format: [4 bytes header_len][header][payload]
    payload = b""
    for name in layout:
        payload += arrays[name].tobytes()

    return struct.pack("<I", len(header_bytes)) + header_bytes + payload


def deserialize_frame(data: bytes) -> TrainingFrame:
    """Deserialize wire-format bytes into a TrainingFrame."""
    header_len = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4:4 + header_len].decode("utf-8"))
    payload = data[4 + header_len:]
    layout = header.get("payload_layout", {})

    frame = TrainingFrame(step=header.get("step", 0))

    def read_array(name, dtype_str="float32"):
        if name not in layout:
            return None
        info = layout[name]
        dt = np.float32 if dtype_str == "float32" else np.int32
        if info.get("dtype") == "int32":
            dt = np.int32
        off = info["offset"]
        shape = info["shape"]
        nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
        return np.frombuffer(payload[off:off + nbytes], dtype=dt).reshape(shape).copy()

    frame.means = read_array("means")
    frame.opacities = read_array("opacities")
    frame.velocities = read_array("velocities")
    frame.times = read_array("times")
    frame.durations = read_array("durations")
    frame.grad_accum = read_array("grad_accum")

    idx = read_array("relocation_indices")
    frame.relocation_indices = idx.tolist() if idx is not None else []
    frame.relocation_targets = read_array("relocation_targets")

    return frame


class TrainingBridge:
    """TCP socket server that receives real-time training data."""

    def __init__(self, port: int = 6009):
        self.port = port
        self.connected = False
        self.current_step = 0
        self.latest_frame: Optional[TrainingFrame] = None
        self.relocation_highlight_until: float = 0.0

        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """Start the background receiver thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        """Background thread: accept connections and receive data."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind(("0.0.0.0", self.port))
            self._server_socket.listen(1)
            self._server_socket.settimeout(1.0)
        except OSError:
            self._running = False
            return

        while self._running:
            # Accept connection
            if self._client_socket is None:
                try:
                    self._client_socket, _ = self._server_socket.accept()
                    self._client_socket.settimeout(2.0)
                    with self._lock:
                        self.connected = True
                except socket.timeout:
                    continue
                except OSError:
                    break

            # Receive data
            try:
                raw = self._recv_exact(4)
                if raw is None:
                    self._disconnect()
                    continue
                header_len = struct.unpack("<I", raw)[0]
                header_bytes = self._recv_exact(header_len)
                if header_bytes is None:
                    self._disconnect()
                    continue
                header = json.loads(header_bytes.decode("utf-8"))

                # Calculate payload size
                layout = header.get("payload_layout", {})
                payload_size = 0
                for info in layout.values():
                    dt = np.float32 if info.get("dtype", "float32") == "float32" else np.int32
                    nbytes = int(np.prod(info["shape"])) * np.dtype(dt).itemsize
                    end = info["offset"] + nbytes
                    payload_size = max(payload_size, end)

                payload = self._recv_exact(payload_size) if payload_size > 0 else b""
                if payload is None and payload_size > 0:
                    self._disconnect()
                    continue

                # Reconstruct full packet for deserialize
                packet = struct.pack("<I", header_len) + header_bytes + (payload or b"")
                frame = deserialize_frame(packet)

                with self._lock:
                    self.latest_frame = frame
                    self.current_step = frame.step
                    if frame.relocation_indices:
                        self.relocation_highlight_until = time.time() + 0.5

            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                self._disconnect()
            except Exception:
                self._disconnect()

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = self._client_socket.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except (socket.timeout, OSError):
                return None
        return buf

    def _disconnect(self):
        if self._client_socket:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
        with self._lock:
            self.connected = False

    def poll(self) -> Optional[TrainingFrame]:
        """Non-blocking: return latest frame and clear it."""
        with self._lock:
            frame = self.latest_frame
            self.latest_frame = None
            return frame

    def has_relocation_highlight(self) -> bool:
        return time.time() < self.relocation_highlight_until

    def stop(self):
        self._running = False
        self._disconnect()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
