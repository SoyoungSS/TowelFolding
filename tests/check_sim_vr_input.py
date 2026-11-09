"""Opt-in live HTTPS/WebSocket-to-UDP check using the existing VR environment.

Run: make test-vr-input
Requires free port 8012 and existing teleop TLS files; never starts ROS nodes.
"""

import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import aiohttp
import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


async def check(sock, process):
    loop = asyncio.get_running_loop()
    async with aiohttp.ClientSession() as client:
        deadline = time.monotonic() + 8
        while True:
            try:
                ws = await client.ws_connect("https://127.0.0.1:8012", ssl=False)
                break
            except aiohttp.ClientError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.1)
        try:
            initial = await ws.receive(timeout=3)
            first = msgpack.unpackb(initial.data, raw=False)
            # Only controller nodes are sent, with no ImageBackground/video payload.
            received = [first]
            head = np.eye(4)
            head[:3, 3] = [0, 1.5, 0]
            left, right = head.copy(), head.copy()
            left[:3, 3] = [-0.15, 1.2, -0.3]
            right[:3, 3] = [0.15, 1.2, -0.3]
            await ws.send_bytes(msgpack.packb({"etype": "CAMERA_MOVE", "value": {"camera": {"matrix": head.flatten(order="F").tolist()}}}))
            value = {"left": left.flatten(order="F").tolist(), "right": right.flatten(order="F").tolist(),
                     "leftState": {"trigger": True, "squeeze": True}, "rightState": {"trigger": False}}
            for _ in range(3):
                await ws.send_bytes(msgpack.packb({"etype": "CONTROLLER_MOVE", "value": value}))
                await asyncio.sleep(0.04)
            valid = None
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                raw, _ = await asyncio.wait_for(loop.sock_recvfrom(sock, 16384), 1)
                data = json.loads(raw)
                if data["tracked"]:
                    valid = data
                    break
            assert valid is not None, "No tracked controller packet"
            assert valid["triggers"] == [True, False]
            assert valid["presses"][:2] == [1, 0]
            poses = np.asarray(valid["poses"])
            np.testing.assert_allclose(poses[1, :3, 3], [0.45, 0.15, 0.15], atol=1e-6)
            await asyncio.sleep(0.4)
            newest = valid
            while True:
                try:
                    raw, _ = sock.recvfrom(16384)
                    newest = json.loads(raw)
                except BlockingIOError:
                    break
            assert not newest["tracked"], "Stale WebXR events must stop tracking despite bridge heartbeat"
            while True:
                try:
                    response = await ws.receive(timeout=0.1)
                except asyncio.TimeoutError:
                    break
                if response.type == aiohttp.WSMsgType.BINARY:
                    received.append(msgpack.unpackb(response.data, raw=False))
            text = repr(received)
            assert "MotionControllers" in text, text
            assert "ImageBackground" not in text and "WebRTCVideoPlane" not in text
        finally:
            await ws.close()


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        process = subprocess.Popen([sys.executable, "-I", "-u", str(ROOT / "simulation/vr_input.py"),
                                    "--port", str(sock.getsockname()[1]), "--duration", "12"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            asyncio.run(check(sock, process))
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
            print(output)
    print("PASS: input-only WebXR, coordinate conversion, button edges, stale-event timeout and shutdown")


if __name__ == "__main__":
    main()
