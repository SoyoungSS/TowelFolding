"""Simulation VR regressions; run with the towel-newton Conda Python."""

import copy
import importlib.util
import json
from pathlib import Path
import socket
import unittest

if importlib.util.find_spec("newton") is None:
    raise unittest.SkipTest("Requires the towel-newton environment")

import newton
import newton.examples
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("towel_scene", ROOT / "simulation/towel_scene.py")
scene_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_module)
from vr_control import VRMapper, VRTeleop, decode_packet
from vr_input import ControllerPackets


def packet(stamp=10.0, sequence=1):
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, :3, 3] = [0.15, 0.15, 0.45]
    poses[2, :3, 3] = [0.15, -0.15, 0.45]
    return dict(version=1, session="test", sequence=sequence, time=stamp, poses=poses,
                presses=[0, 0, 0, 0], triggers=[False, False], tracked=True)


def wire(data):
    return json.dumps({**data, "poses": np.asarray(data["poses"]).tolist()}).encode()


class MappingTests(unittest.TestCase):
    def test_packet_rejects_invalid_or_stale_tracking(self):
        self.assertIsNotNone(decode_packet(wire(packet()), 10.1, 0.25))
        for raw in (b"[]", b"bad", b"null", wire(packet(9)), wire(packet(11))):
            self.assertIsNone(decode_packet(raw, 10.1, 0.25))
        for name, value in (("tracked", 1), ("sequence", True), ("triggers", [True]),
                            ("presses", [0, -1, 0, 0]), ("time", float("nan"))):
            data = packet()
            data[name] = value
            self.assertIsNone(decode_packet(wire(data), 10.1, 0.25))
        for bad in (float("nan"), 2.0):
            data = packet()
            data["poses"][1, 0, 0] = bad
            self.assertIsNone(decode_packet(wire(data), 10.1, 0.25))

    def test_clutch_mapping_stale_samples_and_button_counters(self):
        mapper = VRMapper(yaml.safe_load((ROOT / "teleop/VR_controller/config/gripper_pd.yaml").read_text()))
        data = packet()

        def consume():
            data["time"] += 0.02
            data["sequence"] += 1
            return mapper.consume(copy.deepcopy(data))

        consume()
        data["triggers"][0] = True
        data["poses"][1, 0, 3] += 0.2
        np.testing.assert_array_equal(consume()[0], 0)
        data["poses"][1, 0, 3] += 0.004
        dp, _, active, _ = consume()
        np.testing.assert_allclose(dp[0], [0.004, 0, 0])
        np.testing.assert_array_equal(dp[1], 0)
        self.assertEqual(active, [True, False])
        data["triggers"][0] = False
        data["poses"][1, 0, 3] += 1
        np.testing.assert_array_equal(consume()[0], 0)
        data["triggers"][0] = True
        consume()
        data["time"] += 0.2
        data["poses"][1, 0, 3] += 0.03
        self.assertFalse(any(consume()[2]))
        self.assertFalse(any(consume()[2]))  # A held trigger cannot re-arm after a stall.
        data["triggers"][0] = False
        consume()
        data["triggers"][0] = True
        consume()
        data["presses"][3] += 1
        self.assertTrue(consume()[3][3])
        self.assertTrue(mapper.fine)
        self.assertFalse(any(consume()[3]))
        data["presses"][0] += 2
        self.assertFalse(consume()[3][0])
        data["poses"][1, 0, 3] += 0.1
        self.assertFalse(consume()[2][0])

    def test_input_encoder_squeeze_is_edge_triggered(self):
        from types import SimpleNamespace

        states = {name: False for name in ControllerPackets.BUTTONS}
        state = SimpleNamespace(**states, left_trigger_state=False, right_trigger_state=False)
        data = SimpleNamespace(head_pose=np.eye(4), left_arm_pose=np.eye(4), right_arm_pose=np.eye(4), tele_state=state)
        encoder = ControllerPackets()
        encoder.encode(data, True, 1)
        state.left_squeeze_ctrl_state = True
        first = json.loads(encoder.encode(data, True, 2))
        held = json.loads(encoder.encode(data, True, 3))
        self.assertEqual(first["presses"], [1, 0, 0, 0])
        self.assertEqual(held["presses"], first["presses"])
        self.assertNotIn("image", held)


class Stream:
    def __init__(self, scene):
        self.scene = scene
        self.now = 10.0
        self.data = packet()
        scene.vr = VRTeleop(scene, port=0, clock=lambda: self.now)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def step(self):
        self.now += self.scene.frame_dt
        self.data["time"] = self.now
        self.data["sequence"] += 1
        self.sock.sendto(wire(self.data), ("127.0.0.1", self.scene.vr.port))
        self.scene.step()

    def close(self):
        self.sock.close()
        self.scene.vr.close()


class SimulationVRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("Requires CUDA access")
        wp.set_device("cuda:0")

    def setUp(self):
        self.viewer = newton.viewer.ViewerNull(num_frames=0)
        args = newton.examples.default_args()
        args.config = ROOT / "simulation/scene.yaml"
        self.scene = scene_module.TowelScene(self.viewer, args)
        self.stream = Stream(self.scene)

    def tearDown(self):
        self.stream.close()
        self.viewer.close()

    def test_live_transport_per_arm_motion_timeout_pause_and_home(self):
        scene, stream = self.scene, self.stream
        for _ in range(90):
            stream.step()
        initial = scene.state_0.joint_q.numpy()
        np.testing.assert_array_equal(initial, scene.home)
        tcp = scene.vr.actual_poses()[0].copy()
        stream.data["triggers"] = [True, False]
        stream.step()
        for _ in range(90):
            stream.data["poses"][1, 0, 3] += 0.0005
            stream.step()
        self.assertGreater(scene.vr.actual_poses()[0][0, 0] - tcp[0, 0], 0.015)
        np.testing.assert_allclose(scene.state_0.joint_q.numpy()[scene.vr.arm_indices[1]], initial[scene.vr.arm_indices[1]], atol=1e-6)
        stream.data["triggers"] = [False, False]
        stream.step()
        stopped = scene.state_0.joint_q.numpy().copy()
        for _ in range(10):
            stream.step()
        np.testing.assert_allclose(scene.state_0.joint_q.numpy(), stopped, atol=1e-6)
        stream.data["triggers"] = [True, False]
        stream.step()
        stream.now += 1
        scene.step()
        self.assertFalse(any(scene.vr.active))
        self.assertIn("Waiting", scene.vr.status)
        np.testing.assert_allclose(scene.targets, scene.state_0.joint_q.numpy(), atol=1e-6)
        stream.step()
        self.assertFalse(any(scene.vr.active))
        stream.data["triggers"] = [False, False]
        stream.step()
        stream.data["triggers"] = [True, False]
        stream.step()
        self.assertTrue(scene.vr.active[0])
        scene.vr.update(paused=True)
        stream.step()
        self.assertFalse(any(scene.vr.active))
        stream.data["triggers"] = [False, False]
        stream.step()
        stream.data["presses"][2] += 1
        stream.step()
        self.assertTrue(scene.vr.homing)
        np.testing.assert_array_equal(scene.targets, scene.home)
        np.testing.assert_array_equal(scene.targets, initial)
        for _ in range(180):
            stream.step()
        np.testing.assert_allclose(scene.state_0.joint_q.numpy(), scene.home, atol=0.005)
        scene.reset()
        self.assertIsNone(scene.vr.mapper.previous)
        self.assertTrue(np.all(scene.grasp_owners.numpy() < 0))
        scene.vr.set_enabled(False)
        scene.set_gripper_opening("l", scene_module.CLOSED_FRACTION)
        for _ in range(90):
            stream.step()
        scene.vr.set_enabled(True)
        self.assertTrue(scene.vr.closed[0])
        stream.step()
        stream.data["presses"][0] += 1
        stream.step()
        self.assertAlmostEqual(scene.gripper_opening("l", targets=True), scene_module.OPEN_FRACTION, places=6)

    def test_both_wrists_follow_tool_axis_twist(self):
        scene, stream = self.scene, self.stream
        for _ in range(90):
            stream.step()
        positions, orientations = scene.vr.actual_poses()
        initial = [Rotation.from_quat(q) for q in orientations]
        axes = [r.apply([0, 0, 1]) for r in initial]
        stream.data["triggers"] = [True, True]
        stream.step()
        for _ in range(90):
            for side in range(2):
                increment = Rotation.from_rotvec(axes[side] * 0.0025).as_matrix()
                stream.data["poses"][side + 1, :3, :3] = increment @ stream.data["poses"][side + 1, :3, :3]
            stream.step()
        for _ in range(90):
            stream.step()
        actual_positions, actual_orientations = scene.vr.actual_poses()
        for side in range(2):
            target_delta = (initial[side].inv() * Rotation.from_quat(scene.vr.rotations[side])).as_rotvec()
            actual_delta = (initial[side].inv() * Rotation.from_quat(actual_orientations[side])).as_rotvec()
            self.assertGreater(target_delta[2], 0.14, "Tool-axis input was filtered")
            self.assertGreater(actual_delta[2], 0.10, "Wrist did not follow tool-axis rotation")
        np.testing.assert_allclose(actual_positions, positions, atol=0.01)

    def test_controller_driven_fold_and_independent_release(self):
        scene, stream = self.scene, self.stream
        for _ in range(90):
            stream.step()
        points = scene.state_0.particle_q.numpy()
        pick = np.array([points[p].mean(axis=0) for p in scene.fold.patches])
        pick[:, 2] = scene.config["table"]["height"] + 0.004
        place = np.array([points[p].mean(axis=0) for p in scene.fold.far_patches])
        place[:, 2] = scene.config["table"]["height"] + 0.012
        stream.data["triggers"] = [True, True]
        stream.step()

        def move(destination, max_frames=1200):
            for _ in range(max_frames):
                desired, orientations = scene.vr.positions, scene.vr.rotations
                for side in range(2):
                    delta = destination[side] - desired[side]
                    delta *= min(1, 0.001 / max(np.linalg.norm(delta), 1e-9))
                    stream.data["poses"][side + 1, :3, 3] += delta
                    error = Rotation.from_quat(orientations[side]).inv().as_rotvec()
                    error *= min(1, 0.005 / max(np.linalg.norm(error), 1e-9))
                    stream.data["poses"][side + 1, :3, :3] = Rotation.from_rotvec(error).as_matrix() @ stream.data["poses"][side + 1, :3, :3]
                stream.step()
                if np.linalg.norm(scene.vr.actual_poses()[0] - destination, axis=1).max() < 0.006:
                    return
            self.fail(f"VR goal unreachable: actual={scene.vr.actual_poses()}, destination={destination}, "
                      f"target={scene.vr.positions}, rotations={scene.vr.rotations}, active={scene.vr.active}, "
                      f"status={scene.vr.status}")

        move(pick + [0, 0, 0.09])
        move(pick)
        stream.data["presses"][0] += 1
        stream.data["presses"][1] += 1
        for _ in range(90):
            stream.step()
        owners = scene.grasp_owners.numpy()
        for body in scene.fold.body_ids:
            self.assertGreater(np.count_nonzero(owners == body), 0, "VR close failed to grasp nearby towel")
        # Feed a slow fold arc as controller movements, without invoking AutoFold.
        origin = scene.vr.actual_poses()[0].copy()
        peak = 0.0
        for frame in range(1, 481):
            u = frame / 480
            u = u * u * (3 - 2 * u)
            destination = origin * (1 - u) + (place + [0, 0, 0.025]) * u
            # The longer rotary gripper has less upward reach with a vertical wrist.
            destination[:, 2] += 0.085 * np.sin(np.pi * u)
            move(destination)
            peak = max(peak, float(scene.state_0.particle_q.numpy()[owners >= 0, 2].min()))
        self.assertGreater(peak, scene.config["table"]["height"] + 0.07)
        points = scene.state_0.particle_q.numpy()
        for i in np.flatnonzero(owners >= 0):
            body = wp.transform(*scene.state_0.body_q.numpy()[owners[i]])
            np.testing.assert_allclose(points[i], wp.transform_point(body, wp.vec3(*scene.grasp_offsets.numpy()[i])), atol=1e-6)
        move(place)
        stream.data["presses"][0] += 1
        stream.step()
        owners = scene.grasp_owners.numpy()
        self.assertEqual(np.count_nonzero(owners == scene.fold.body_ids[0]), 0)
        self.assertGreater(np.count_nonzero(owners == scene.fold.body_ids[1]), 0)
        stream.data["presses"][1] += 1
        for _ in range(90):
            stream.step()
        move(place + [0, 0, 0.08])
        for _ in range(180):
            stream.step()
        self.assertTrue(np.all(scene.grasp_owners.numpy() < 0))
        np.testing.assert_array_equal(scene.model.particle_flags.numpy(), scene.free_particle_flags)
        metrics = scene.fold.measure()
        print(f"VR-driven fold: {metrics}", flush=True)
        self.assertLess(metrics["mean_pair_error_m"], scene.config["fold"]["pair_tolerance"])
        self.assertLess(metrics["footprint_ratio"], 0.68)
        scene.test_final()


if __name__ == "__main__":
    unittest.main(verbosity=2)
