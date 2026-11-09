"""Local VR input, Cartesian IK and independent assisted towel grasps."""

import json
from pathlib import Path
import socket
import sys
import time

import newton
import newton.ik as ik
import numpy as np
import warp as wp
import yaml
from robot_model import OPEN_FRACTION, CLOSED_FRACTION, CAPTURE_FRACTION, is_arm_joint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teleop/VR_controller"))
from pose_mapping import (
    _head_yaw_rotation, _limit_increment, _conservative_deadband, _rotate_xy,
    robot_frame_position_delta, robot_frame_orientation_delta,
)


def decode_packet(raw, now, timeout):
    """Reject malformed, non-rigid, stale or future inputs before updating state."""
    try:
        data = json.loads(raw)
        if data["version"] != 1 or not isinstance(data["session"], str) or not 1 <= len(data["session"]) <= 64:
            return None
        if type(data["sequence"]) is not int or data["sequence"] < 0:
            return None
        stamp = float(data["time"])
        if not np.isfinite(stamp) or not -0.02 <= now - stamp <= timeout:
            return None
        if type(data["tracked"]) is not bool or len(data["triggers"]) != 2:
            return None
        if any(type(value) is not bool for value in data["triggers"]):
            return None
        if len(data["presses"]) != 4 or any(type(n) is not int or n < 0 for n in data["presses"]):
            return None
        poses = np.asarray(data["poses"], dtype=float)
        if poses.shape != (3, 4, 4) or not np.isfinite(poses).all():
            return None
        if not np.allclose(poses[:, 3, :], [[0, 0, 0, 1]] * 3, atol=1e-4):
            return None
        for pose in poses:
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.02) or abs(np.linalg.det(rotation) - 1) > 0.02:
                return None
        data["poses"], data["time"] = poses, stamp
        return data
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


class VRMapper:
    """Existing head/room mapping with clutch re-anchoring and press counters."""

    def __init__(self, config):
        self.config = config
        self.fine = False
        self.reset()

    def reset(self):
        self.previous = None
        self.reference = None
        self.armed = [False, False]
        self.residual = np.zeros((2, 2, 3))

    def consume(self, packet):
        previous = self.previous
        self.previous = packet
        position, rotation = np.zeros((2, 3)), np.zeros((2, 3))
        active, presses = [False, False], [False] * 4
        if previous is None or packet["session"] != previous["session"]:
            self.reference = _head_yaw_rotation(packet["poses"][0])
            self.armed = [not held for held in packet["triggers"]]
            self.residual.fill(0)
            return position, rotation, active, presses
        dt = packet["time"] - previous["time"]
        if not 0 < dt <= 0.08:
            self.armed = [False, False]
            self.residual.fill(0)
            return position, rotation, active, presses
        presses = [bool((new - old) % 2) for new, old in zip(packet["presses"], previous["presses"])]
        presses[2] = packet["presses"][2] > previous["presses"][2]
        if presses[3] and not presses[2]:
            self.fine = not self.fine
            self.residual.fill(0)
        cfg = self.config
        mode = "fine" if self.fine else "fast"
        gain = np.asarray(cfg.get("position_axis_scale", [1, 1, 1])) * cfg.get("position_axis_sign", [1, 1, 1])
        yaw = np.deg2rad(cfg.get("translation_yaw_offset_deg", 0))
        for side in range(2):
            held = packet["triggers"][side]
            if not held:
                self.armed[side] = True
                self.residual[side].fill(0)
                continue
            if not self.armed[side]:
                continue
            active[side] = True
            if not previous["triggers"][side]:
                continue  # Re-clutch at the actual robot pose without a position jump.
            current, old = packet["poses"][side + 1], previous["poses"][side + 1]
            head, old_head = packet["poses"][0], previous["poses"][0]
            if cfg.get("reference_frame", "head") == "head":
                dp = robot_frame_position_delta(current, old, head, old_head, self.reference)
            else:
                dp = current[:3, 3] - old[:3, 3] + head[:3, 3] - old_head[:3, 3]
            dr = robot_frame_orientation_delta(current, old, head, old_head, self.reference)
            if np.linalg.norm(dp) > 0.05 or np.linalg.norm(dr) > 0.20:
                active[side] = self.armed[side] = False
                self.residual[side].fill(0)
                continue
            dp = _limit_increment(dp, True, 0.05, cfg.get(f"{mode}_position_speed_mps", 0.3), dt)
            dr = _limit_increment(dr, True, 0.20, cfg.get(f"{mode}_orientation_speed_radps", 1.5), dt)
            dp = _rotate_xy(dp * gain * cfg.get(f"{mode}_position_scale", 1), np.cos(yaw), np.sin(yaw))
            dr *= cfg.get(f"{mode}_orientation_scale", 1)
            position[side] = _conservative_deadband(dp, self.residual[side, 0], 0.0004, True)
            rotation[side] = _conservative_deadband(dr, self.residual[side, 1], 0.002, True)
        return position, rotation, active, presses


class VRTeleop:
    def __init__(self, scene, port=8766, clock=time.monotonic):
        self.scene, self.clock = scene, clock
        self.config = scene.config["vr"]
        for name in ("timeout", "max_target_lead"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f"vr.{name} must be positive and finite")
        mapping = ROOT / self.config["mapping_config"]
        self.mapper = VRMapper(yaml.safe_load(mapping.read_text()))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("127.0.0.1", port))
        except OSError:
            self.sock.close()
            raise
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.body_ids = scene.fold.body_ids
        self.tcp = scene.fold.tcp
        self.arm_indices = [[i for name, i in scene.q_index.items() if name.startswith(f"arm_{side}_") and is_arm_joint(name)]
                            for side in ("l", "r")]
        self.enabled = True
        self.solver = None
        self.packet = None
        self.status = "Waiting for VR"
        self.reset()

    def actual_poses(self):
        bodies = self.scene.state_0.body_q.numpy()
        positions = np.array([wp.transform_point(wp.transform(*bodies[i]), tcp)
                              for i, tcp in zip(self.body_ids, self.tcp)])
        return positions, bodies[self.body_ids, 3:].copy()

    def reset(self):
        self.mapper.reset()
        self.positions, self.rotations = self.actual_poses()
        self.active = [False, False]
        self.closed = [self.scene.gripper_opening(side, targets=True) < 0.25 for side in ("l", "r")]
        self.pending = [None, None]
        self.homing = False
        self.packet = None
        self.status = "Release triggers to engage VR"

    def suspend(self, message="VR stopped", release=False):
        self.scene.targets[:] = self.scene.state_0.joint_q.numpy()
        self.mapper.reset()
        self.active = [False, False]
        self.pending = [None, None]
        self.homing = False
        self.status = message
        if release:
            self.scene.release_grasps()

    def set_enabled(self, enabled):
        self.suspend(release=True)
        self.enabled = enabled
        if enabled:
            self.scene.fold.stop("VR control")
            self.reset()

    def poll(self):
        now = self.clock()
        latest = None
        for _ in range(256):
            try:
                raw, _ = self.sock.recvfrom(16384)
            except BlockingIOError:
                break
            packet = decode_packet(raw, now, self.config["timeout"])
            if packet is None:
                continue
            previous = latest if latest is not None else self.packet
            if previous is not None:
                if packet["session"] == previous["session"]:
                    if packet["sequence"] <= previous["sequence"] or packet["time"] <= previous["time"]:
                        continue
                    if any(a < b for a, b in zip(packet["presses"], previous["presses"])):
                        continue
                elif now - previous["time"] <= self.config["timeout"]:
                    continue
            latest = packet
        if latest is not None:
            self.packet = latest
        return latest

    def update(self, paused=False):
        packet = self.poll()
        if not self.enabled:
            return
        now = self.clock()
        if paused or self.packet is None or not self.packet["tracked"] or now - self.packet["time"] > self.config["timeout"]:
            self.suspend("Paused" if paused else "Waiting for VR tracking")
            return
        if self.homing:
            if np.max(np.abs(self.scene.state_0.joint_q.numpy() - self.scene.home)) < 0.005:
                self.homing = False
                self.mapper.reset()
            return
        if packet is not None:
            dp, dr, active, presses = self.mapper.consume(packet)
            if presses[2]:
                self.suspend(release=True)
                # Simulation home is the scene's initial pose, independent of hardware presets.
                self.scene.targets[:] = self.scene.home
                self.closed = [False, False]
                self.homing = True
                self.status = "Returning home"
                return
            actual, rotations = self.actual_poses()
            for side in range(2):
                if not active[side] or not self.active[side]:
                    self.positions[side] = actual[side]
                    self.rotations[side] = rotations[side]
                if not active[side]:
                    self.scene.targets[self.arm_indices[side]] = self.scene.state_0.joint_q.numpy()[self.arm_indices[side]]
                else:
                    self.positions[side] += dp[side]
                    yaw, pitch, roll = dr[side]
                    delta = wp.quat_rpy(float(roll), float(pitch), float(yaw))
                    orientation = wp.quat(*self.rotations[side])
                    if self.config["lock_tool_twist"]:
                        axis = np.asarray(wp.quat_rotate(orientation, wp.vec3(0, 0, 1)))
                        q = np.asarray(delta)
                        projection = axis * np.dot(q[:3], axis)
                        twist = np.r_[projection, q[3]]
                        if np.linalg.norm(twist) > 1e-8:
                            delta = wp.mul(delta, wp.quat_inverse(wp.quat(*(twist / np.linalg.norm(twist)))))
                    self.rotations[side] = wp.normalize(wp.mul(delta, orientation))
                if presses[side]:
                    self.closed[side] = not self.closed[side]
                    self.scene.set_gripper_opening(("l", "r")[side], CLOSED_FRACTION if self.closed[side] else OPEN_FRACTION)
                    if self.closed[side]:
                        self.pending[side] = self.scene.sim_time + 2.0
                    else:
                        self.release(side)
                        self.pending[side] = None
            self.active = active
            if any(held and not ready for held, ready in zip(packet["triggers"], self.mapper.armed)):
                self.status = "Release triggers to resume"
            else:
                self.status = "Fine" if self.mapper.fine else "Fast"
        if any(self.active):
            self.solve()
        self.update_grasps()

    def solve(self):
        scene = self.scene
        actual, rotations = self.actual_poses()
        for i in range(2):
            if not self.active[i]:
                self.positions[i], self.rotations[i] = actual[i], rotations[i]
                continue
            self.positions[i] = np.clip(self.positions[i],
                                        [0.05, -0.45, scene.config["table"]["height"] + 0.004], [0.65, 0.45, 1.05])
            lag = self.positions[i] - actual[i]
            length = np.linalg.norm(lag)
            if length > self.config["max_target_lead"]:
                self.positions[i] = actual[i] + lag * self.config["max_target_lead"] / length
            cosine = np.clip(abs(np.dot(rotations[i], self.rotations[i])), 0, 1)
            angle = 2 * np.arccos(cosine)
            if angle > 0.25:
                self.rotations[i] = wp.quat_slerp(wp.quat(*rotations[i]), wp.quat(*self.rotations[i]), float(0.25 / angle))
        if self.solver is None:
            # Prioritize the grasp point when wrist orientation competes with reach.
            self.position_objectives = [ik.IKObjectivePosition(body, tcp, wp.zeros(1, dtype=wp.vec3), weight=3.0)
                                        for body, tcp in zip(self.body_ids, self.tcp)]
            self.rotation_objectives = [ik.IKObjectiveRotation(body, wp.quat_identity(),
                                        wp.array([[0, 0, 0, 1]], dtype=wp.vec4), weight=0.3) for body in self.body_ids]
            limits = ik.IKObjectiveJointLimit(scene.model.joint_limit_lower, scene.model.joint_limit_upper, weight=10)
            self.q = wp.array(scene.home[None], dtype=float)
            self.solver = ik.IKSolver(scene.model, n_problems=1,
                                      objectives=[*self.position_objectives, *self.rotation_objectives, limits],
                                      lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            self.graph = None
            if scene.model.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    self.solver.step(self.q, self.q, iterations=24)
                self.graph = capture.graph
        for i in range(2):
            self.position_objectives[i].target_positions.assign([self.positions[i]])
            self.rotation_objectives[i].target_rotations.assign([self.rotations[i]])
        self.q.assign(scene.state_0.joint_q.numpy()[None])
        if self.graph is None:
            self.solver.step(self.q, self.q, iterations=24)
        else:
            wp.capture_launch(self.graph)
        q = self.q.numpy()[0]
        if not np.isfinite(q).all():
            self.suspend("IK failed; release triggers")
            return
        for side, prefix in enumerate(("arm_l_", "arm_r_")):
            if self.active[side]:
                for name, index in scene.q_index.items():
                    if name.startswith(prefix) and is_arm_joint(name):
                        scene.set_joint_target(name, float(q[index]))

    def release(self, side):
        owners = self.scene.grasp_owners.numpy()
        owners[owners == self.body_ids[side]] = -1
        self.assign_owners(owners)

    def assign_owners(self, owners):
        flags = self.scene.free_particle_flags.copy()
        flags[owners >= 0] &= ~int(newton.ParticleFlags.ACTIVE)
        self.scene.grasp_owners.assign(owners)
        self.scene.model.particle_flags.assign(flags)

    def update_grasps(self):
        scene = self.scene
        for side, name in enumerate(("l", "r")):
            if self.pending[side] is None:
                continue
            if scene.sim_time > self.pending[side]:
                self.pending[side] = None
                continue
            if scene.gripper_opening(name) > CAPTURE_FRACTION:
                continue
            body = self.body_ids[side]
            inverse = wp.transform_inverse(wp.transform(*scene.state_0.body_q.numpy()[body]))
            local = np.array([wp.transform_point(inverse, wp.vec3(*p)) for p in scene.state_0.particle_q.numpy()])
            owners = scene.grasp_owners.numpy()
            tcp = np.asarray(self.tcp[side])
            eligible = (owners < 0) & np.all(np.abs(local - tcp) <= [0.022, 0.007, 0.010], axis=1)
            patch = np.flatnonzero(eligible)
            if not len(patch):
                continue
            patch = patch[np.argsort(np.linalg.norm(local[patch] - tcp, axis=1))[:4]]
            offsets = scene.grasp_offsets.numpy()
            offsets[patch] = local[patch]
            owners[patch] = body
            scene.grasp_offsets.assign(offsets)
            self.assign_owners(owners)
            self.pending[side] = None

    def close(self):
        self.suspend(release=True)
        self.sock.close()
