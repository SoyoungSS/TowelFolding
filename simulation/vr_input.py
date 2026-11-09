#!/usr/bin/env python3
"""Forward existing WebXR controller input to local Newton, without ROS or video."""

import argparse
import json
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teleop/src"))


def valid_pose(pose):
    pose = np.asarray(pose)
    return (pose.shape == (4, 4) and np.isfinite(pose).all()
            and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=0.02)
            and abs(np.linalg.det(pose[:3, :3]) - 1) < 0.02)


class ControllerPackets:
    """Cumulative press counts survive dropped datagrams; poses remain latest-only."""

    BUTTONS = ("left_squeeze_ctrl_state", "right_squeeze_ctrl_state", "right_aButton", "right_bButton")

    def __init__(self):
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.buttons = [False] * 4
        self.counts = [0] * 4

    def encode(self, telemetry, tracked, now):
        states = [bool(getattr(telemetry.tele_state, name)) for name in self.BUTTONS]
        if tracked:
            for i, pressed in enumerate(states):
                if pressed and not self.buttons[i]:
                    self.counts[i] += 1
        self.buttons = states
        self.sequence += 1
        return json.dumps({
            "version": 1, "session": self.session, "sequence": self.sequence, "time": now,
            "tracked": bool(tracked), "presses": self.counts,
            "triggers": [bool(telemetry.tele_state.left_trigger_state), bool(telemetry.tele_state.right_trigger_state)],
            "poses": [telemetry.head_pose.tolist(), telemetry.left_arm_pose.tolist(), telemetry.right_arm_pose.tolist()],
        }, allow_nan=False).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds (connection smoke test)")
    args = parser.parse_args()
    cert, key = ROOT / "teleop/cert.pem", ROOT / "teleop/key.pem"
    if not cert.is_file() or not key.is_file():
        parser.error("Missing VR TLS certificate; run: bash scripts/create_certificate.sh")
    # Fail clearly instead of launching a second Vuer process over an existing VR server.
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 8012))
    from televuer import TeleVuerWrapper

    wrapper = TeleVuerWrapper(False, False, (1, 1, 3), None, cert_file=str(cert), key_file=str(key),
                             stream_images=False)
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    packets = ControllerPackets()
    start = time.monotonic()
    event_count, errors, last_event = 0, 0, -np.inf
    print(f"Simulation VR input -> 127.0.0.1:{args.port}; no ROS commands or image stream.", flush=True)
    print("Open https://<host-IP>:8012?ws=wss://<host-IP>:8012 and enter Virtual Reality.", flush=True)
    print("Release both triggers once, then hold a trigger to move its arm; squeeze toggles its gripper.", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            while not stopped and (args.duration is None or time.monotonic() - start < args.duration):
                if not wrapper.tvuer.process.is_alive():
                    raise RuntimeError("VR web server exited; check TLS certificate and port 8012")
                now = time.monotonic()
                snapshot = wrapper.controller_debug_snapshot()
                if snapshot["events"] != event_count:
                    if snapshot["errors"] == errors:
                        last_event = now
                    event_count, errors = snapshot["events"], snapshot["errors"]
                raw_poses = (wrapper.tvuer.head_pose, wrapper.tvuer.left_arm_pose, wrapper.tvuer.right_arm_pose)
                tracked = (now - last_event < 0.25 and snapshot["left_seen"] and snapshot["right_seen"]
                           and all(valid_pose(pose) for pose in raw_poses))
                data = wrapper.get_motion_state_data()
                if all(valid_pose(pose) for pose in (data.head_pose, data.left_arm_pose, data.right_arm_pose)):
                    sock.sendto(packets.encode(data, tracked, now), ("127.0.0.1", args.port))
                time.sleep(max(0, 0.02 - (time.monotonic() - now)))
    finally:
        wrapper.tvuer.process.terminate()
        wrapper.tvuer.process.join(timeout=3)
        if wrapper.tvuer.process.is_alive():
            wrapper.tvuer.process.kill()
            wrapper.tvuer.process.join()


if __name__ == "__main__":
    main()
