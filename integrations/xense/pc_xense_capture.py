#!/usr/bin/env python3
"""Persistent Xense capture service for a GPU workstation.

The service deliberately has no ROS dependency. A NUC controls it through SSH,
while local commands reach the daemon through a Unix socket.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SOCKET = Path.home() / ".cache/franka_xense_capture/control.sock"
DEFAULT_ROOT = Path.home() / "Documents/VLA_RL_dataset/franka_remote_tactile"
FEATURE_NAMES = [
    "sample_valid", "force_valid", "force_fx", "force_fy", "force_fz",
    "force_mx", "force_my", "force_mz", "force_mag", "force_tan",
    "force_norm", "torque_mag", "depth_mean", "depth_max", "shear_mean",
    "shear_max", "deformation_mean", "deformation_max",
]


def sanitize(value: str) -> str:
    result = "".join(c if c.isalnum() or c in "_-" else "_" for c in value)
    return result.strip("_") or "tactile"


def feature_values(depth: Any, force: Any, resultant: Any) -> list[float]:
    depth_arr = np.asarray(depth, dtype=np.float64)
    force_arr = np.asarray(force, dtype=np.float64)
    resultant_arr = np.asarray(resultant, dtype=np.float64).reshape(-1)
    wrench = np.zeros(6, dtype=np.float64)
    wrench[: min(6, resultant_arr.size)] = resultant_arr[:6]
    valid = float(np.all(np.isfinite(wrench)))
    wrench = np.nan_to_num(wrench, nan=0.0, posinf=0.0, neginf=0.0)
    finite_depth = depth_arr[np.isfinite(depth_arr)]
    finite_force = np.nan_to_num(force_arr, nan=0.0, posinf=0.0, neginf=0.0)
    shear = (
        np.linalg.norm(finite_force[..., :2], axis=-1)
        if finite_force.ndim >= 3 and finite_force.shape[-1] >= 2
        else np.zeros(1)
    )
    deformation = np.abs(finite_depth) if finite_depth.size else np.zeros(1)
    linear, torque = wrench[:3], wrench[3:6]
    return [
        1.0, valid, *[float(x) for x in wrench], float(np.linalg.norm(linear)),
        float(np.linalg.norm(linear[:2])), float(abs(linear[2])),
        float(np.linalg.norm(torque)),
        float(finite_depth.mean()) if finite_depth.size else 0.0,
        float(finite_depth.max()) if finite_depth.size else 0.0,
        float(shear.mean()), float(shear.max()), float(deformation.mean()),
        float(deformation.max()),
    ]


class CaptureDaemon:
    def __init__(self, socket_path: Path, root: Path, serials: list[str], names: list[str]) -> None:
        self.socket_path = socket_path
        self.root = root
        self.serials = serials
        self.names = names
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.sensors: list[dict[str, Any]] = []
        self.session: dict[str, Any] | None = None
        self.started_ns = time.time_ns()

    def open_sensors(self) -> None:
        from xensesdk import Sensor

        available = sorted(str(x) for x in Sensor.scanSerialNumber())
        serials = self.serials or available
        if not serials:
            raise RuntimeError("No Xense sensors detected")
        names = list(self.names)
        defaults = ["left_tip", "right_tip"]
        while len(names) < len(serials):
            names.append(defaults[len(names)] if len(names) < len(defaults) else f"tactile_{len(names)+1}")
        for serial, raw_name in zip(serials, names):
            if serial not in available:
                raise RuntimeError(f"Xense {serial} unavailable; detected={available}")
            sensor = Sensor.create(serial, infer_mode="fast")
            runtime = {
                "serial": serial,
                "name": sanitize(raw_name),
                "sensor": sensor,
                "ring": collections.deque(maxlen=100),
                "frames": 0,
                "raw_reads": 0,
                "errors": 0,
                "consecutive_errors": 0,
                "last_error": "",
                "last_error_ns": None,
                "last_hash": None,
                "first_ns": None,
                "last_ns": None,
                "gpu": bool(getattr(sensor, "infer_engine_using_gpu", False)),
            }
            self.sensors.append(runtime)
            threading.Thread(target=self.capture_loop, args=(runtime,), daemon=True).start()

    def capture_loop(self, runtime: dict[str, Any]) -> None:
        from xensesdk import Sensor

        outputs = (Sensor.OutputType.Raw, Sensor.OutputType.Depth, Sensor.OutputType.Force,
                   Sensor.OutputType.ForceResultant, Sensor.OutputType.TimeStamp)
        while not self.stop_event.is_set():
            try:
                raw, depth, force, resultant, sensor_time = runtime["sensor"].selectSensorInfo(*outputs)
                runtime["raw_reads"] += 1
                raw = np.ascontiguousarray(raw)
                frame_hash = zlib.adler32(memoryview(raw))
                if frame_hash == runtime["last_hash"]:
                    continue
                runtime["last_hash"] = frame_hash
                captured_ns = time.time_ns()
                ok, encoded = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                sample = {
                    "pc_time_ns": captured_ns,
                    "sensor_time": float(sensor_time),
                    "source_frame": runtime["frames"],
                    "feature_names": FEATURE_NAMES,
                    "values": feature_values(depth, force, resultant),
                    "jpeg": encoded.tobytes(),
                }
                with self.lock:
                    runtime["ring"].append(sample)
                    runtime["frames"] += 1
                    runtime["consecutive_errors"] = 0
                    runtime["first_ns"] = runtime["first_ns"] or captured_ns
                    runtime["last_ns"] = captured_ns
                    if self.session is not None:
                        self.write_sample(self.session, runtime, sample)
            except Exception as exc:
                runtime["errors"] += 1
                runtime["consecutive_errors"] += 1
                runtime["last_error"] = f"{type(exc).__name__}: {exc}"
                runtime["last_error_ns"] = time.time_ns()
                time.sleep(0.01)

    def sensor_status(self) -> list[dict[str, Any]]:
        now_ns = time.time_ns()
        sensors = []
        for runtime in self.sensors:
            samples = list(runtime["ring"])
            timestamps = [int(sample["pc_time_ns"]) for sample in samples]
            age_sec = (
                (now_ns - int(runtime["last_ns"])) / 1e9
                if runtime["last_ns"] is not None
                else None
            )
            if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
                rate_hz = (len(timestamps) - 1) / ((timestamps[-1] - timestamps[0]) / 1e9)
            else:
                rate_hz = 0.0
            healthy = age_sec is not None and age_sec <= 2.0 and rate_hz >= 1.0
            sensors.append({
                "name": runtime["name"],
                "serial": runtime["serial"],
                "frames": runtime["frames"],
                "rate_hz": rate_hz,
                "last_sample_age_sec": age_sec,
                "healthy": healthy,
                "errors": runtime["errors"],
                "consecutive_errors": runtime["consecutive_errors"],
                "last_error": runtime["last_error"],
                "gpu": runtime["gpu"],
            })
        return sensors

    def write_sample(self, session: dict[str, Any], runtime: dict[str, Any], sample: dict[str, Any]) -> None:
        cutoff = int(session["pc_start_ns"]) - int(session["pre_roll_ns"])
        if int(sample["pc_time_ns"]) < cutoff:
            return
        name = runtime["name"]
        seen = session["seen"][name]
        identity = (int(sample["pc_time_ns"]), int(sample["source_frame"]))
        if identity in seen:
            return
        seen.add(identity)
        index = session["counts"][name]
        image_rel = Path(name) / "images" / f"sample_{index:06d}.jpg"
        image_path = session["path"] / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(sample["jpeg"])
        row = {key: value for key, value in sample.items() if key != "jpeg"}
        row["sample_index"] = index
        row["image_path"] = str(image_rel)
        handle = session["handles"][name]
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        session["counts"][name] += 1

    def start_session(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.session is not None:
                raise RuntimeError("A tactile episode is already recording")
            unhealthy = [sensor for sensor in self.sensor_status() if not sensor["healthy"]]
            if unhealthy:
                details = ", ".join(
                    f"{sensor['name']} age={sensor['last_sample_age_sec']} "
                    f"rate={sensor['rate_hz']:.2f}Hz error={sensor['last_error'] or 'none'}"
                    for sensor in unhealthy
                )
                raise RuntimeError(f"Xense stream is stale; reconnect/restart required: {details}")
            episode_index = int(request["episode_index"])
            session_id = str(request.get("session_id") or uuid.uuid4().hex)
            dataset_name = sanitize(str(request.get("dataset_name") or "franka_dataset"))
            path = self.root / dataset_name / f"episode_{episode_index:06d}_{session_id}"
            path.mkdir(parents=True, exist_ok=False)
            pc_start_ns = int(request.get("pc_start_ns") or time.time_ns())
            session = {
                "path": path,
                "session_id": session_id,
                "episode_index": episode_index,
                "dataset_name": dataset_name,
                "task": str(request.get("task") or ""),
                "pc_start_ns": pc_start_ns,
                "nuc_start_ns": int(request.get("nuc_start_ns") or 0),
                "nuc_minus_pc_offset_ns": int(request.get("nuc_minus_pc_offset_ns") or 0),
                "clock_rtt_ns": int(request.get("clock_rtt_ns") or 0),
                "pre_roll_ns": int(float(request.get("pre_roll_sec", 0.25)) * 1e9),
                "counts": {r["name"]: 0 for r in self.sensors},
                "handles": {},
                "seen": {r["name"]: set() for r in self.sensors},
            }
            for runtime in self.sensors:
                samples_path = path / runtime["name"] / "samples.jsonl"
                samples_path.parent.mkdir(parents=True, exist_ok=True)
                session["handles"][runtime["name"]] = samples_path.open("a", encoding="utf-8")
            self.session = session
            for runtime in self.sensors:
                for sample in list(runtime["ring"]):
                    self.write_sample(session, runtime, sample)
            self.write_manifest(session, active=True)
            return {"ok": True, "active": True, "session_id": session_id, "path": str(path),
                    "episode_index": episode_index, "counts": dict(session["counts"])}

    def write_manifest(self, session: dict[str, Any], active: bool) -> dict[str, Any]:
        now_ns = time.time_ns()
        payload = {
            "format": "franka_remote_xense_v1", "active": active,
            "session_id": session["session_id"], "episode_index": session["episode_index"],
            "dataset_name": session["dataset_name"], "task": session["task"],
            "pc_start_ns": session["pc_start_ns"], "pc_stop_ns": None if active else now_ns,
            "nuc_start_ns": session["nuc_start_ns"],
            "nuc_minus_pc_offset_ns": session["nuc_minus_pc_offset_ns"],
            "clock_rtt_ns": session["clock_rtt_ns"], "feature_names": FEATURE_NAMES,
            "sensors": [
                {"name": r["name"], "serial": r["serial"], "gpu": r["gpu"],
                 "samples": session["counts"][r["name"]]}
                for r in self.sensors
            ],
        }
        (session["path"] / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    def stop_session(self, discard: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                return {"ok": False, "error": "not recording"}
            session = self.session
            self.session = None
            for handle in session["handles"].values():
                handle.close()
            manifest = self.write_manifest(session, active=False)
            result = {"ok": True, "active": False, "path": str(session["path"]),
                      "session_id": session["session_id"], "episode_index": session["episode_index"],
                      "counts": dict(session["counts"]), "manifest": manifest}
            if discard:
                import shutil
                shutil.rmtree(session["path"], ignore_errors=True)
                result["discarded"] = True
            return result

    def status(self) -> dict[str, Any]:
        with self.lock:
            sensors = self.sensor_status()
            return {"ok": True, "healthy": bool(sensors) and all(s["healthy"] for s in sensors),
                    "pid": os.getpid(), "active": self.session is not None,
                    "uptime_sec": (time.time_ns() - self.started_ns) / 1e9, "sensors": sensors,
                    "session": None if self.session is None else {
                        "path": str(self.session["path"]), "episode_index": self.session["episode_index"],
                        "counts": dict(self.session["counts"]),
                    }}

    def recent_samples(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return recent feature rows without consuming capture data or JPEGs."""
        start_ns = int(request.get("start_ns", 0))
        end_ns = int(request.get("end_ns", time.time_ns()))
        max_samples = int(request.get("max_samples", 500))
        if start_ns < 0 or end_ns < start_ns:
            raise ValueError(f"invalid time range: start_ns={start_ns}, end_ns={end_ns}")
        if max_samples < 1 or max_samples > 5000:
            raise ValueError("max_samples must be in [1, 5000]")

        with self.lock:
            sensors: dict[str, list[dict[str, Any]]] = {}
            for runtime in self.sensors:
                selected = [
                    {
                        key: value
                        for key, value in sample.items()
                        if key != "jpeg"
                    }
                    for sample in runtime["ring"]
                    if start_ns <= int(sample["pc_time_ns"]) < end_ns
                ][-max_samples:]
                sensors[runtime["name"]] = selected
            return {
                "ok": True,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "sensors": sensors,
            }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("command", "status"))
        if command == "status":
            return self.status()
        if command == "recent":
            return self.recent_samples(request)
        if command == "start":
            return self.start_session(request)
        if command == "stop":
            return self.stop_session(False)
        if command == "discard":
            return self.stop_session(True)
        if command == "shutdown":
            self.stop_event.set()
            return {"ok": True}
        return {"ok": False, "error": f"unknown command: {command}"}

    def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server.listen(8)
        server.settimeout(0.5)
        while not self.stop_event.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    raw = b""
                    while not raw.endswith(b"\n"):
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        raw += chunk
                    response = self.handle(json.loads(raw.decode("utf-8")))
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                connection.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
        server.close()
        self.socket_path.unlink(missing_ok=True)
        if self.session is not None:
            self.stop_session(False)
        for runtime in self.sensors:
            try:
                runtime["sensor"].release()
            except Exception:
                pass


def send_request(socket_path: Path, request: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    client.connect(str(socket_path))
    client.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode())
    response = b""
    while not response.endswith(b"\n"):
        chunk = client.recv(65536)
        if not chunk:
            break
        response += chunk
    client.close()
    return json.loads(response.decode("utf-8"))


def ensure_daemon(args: argparse.Namespace) -> None:
    try:
        status = send_request(args.socket, {"command": "status"})
        if status.get("healthy"):
            print(json.dumps(status))
            return
        if status.get("active"):
            raise SystemExit(
                "Xense daemon is recording but its sensor stream is stale; "
                "stop or discard the episode before restarting"
            )
        send_request(args.socket, {"command": "shutdown"})
        deadline = time.monotonic() + 10.0
        while args.socket.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
    except (OSError, ValueError):
        pass
    log = Path.home() / ".cache/franka_xense_capture/daemon.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "daemon", "--socket", str(args.socket),
               "--root", str(args.root), "--serials", args.serials, "--names", args.names]
    environment = dict(os.environ)
    nvidia_root = Path.home() / "Documents/lerobot/.venv/lib/python3.13/site-packages/nvidia"
    nvidia_libs = sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
    if nvidia_libs:
        old_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(nvidia_libs + ([old_path] if old_path else []))
    with log.open("ab", buffering=0) as handle:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True, env=environment)
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        try:
            result = send_request(args.socket, {"command": "status"})
            result["log"] = str(log)
            print(json.dumps(result))
            return
        except (OSError, ValueError):
            time.sleep(0.5)
    raise SystemExit(f"Xense daemon did not start; inspect {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "ensure", "daemon", "status", "recent", "start", "stop", "discard",
            "shutdown",
        ],
    )
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--serials", default="")
    parser.add_argument("--names", default="left_tip,right_tip")
    parser.add_argument("--request-json", default="{}")
    args = parser.parse_args()
    args.socket = args.socket.expanduser()
    args.root = args.root.expanduser()
    if args.command == "ensure":
        ensure_daemon(args)
        return
    if args.command == "daemon":
        daemon = CaptureDaemon(args.socket, args.root,
                               [x for x in args.serials.split(",") if x],
                               [x for x in args.names.split(",") if x])
        signal.signal(signal.SIGTERM, lambda *_: daemon.stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: daemon.stop_event.set())
        daemon.open_sensors()
        daemon.serve()
        return
    request = json.loads(args.request_json)
    request["command"] = args.command
    result = send_request(args.socket, request, timeout=30.0)
    print(json.dumps(result))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
