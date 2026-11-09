"""Run with the towel-newton Conda environment; other environments skip these checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
HAS_NEWTON = importlib.util.find_spec("newton") is not None
if HAS_NEWTON:
    import newton
    import newton.examples
    import numpy as np
    import warp as wp

    spec = importlib.util.spec_from_file_location("towel_scene", ROOT / "simulation/towel_scene.py")
    scene_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene_module)


@unittest.skipUnless(HAS_NEWTON, "Requires the towel-newton environment")
class GeometryTests(unittest.TestCase):
    def test_new_robot_assets_and_rotary_grippers(self):
        source = ROOT / "dual_arm/urdf/dual_arm.urdf"
        original = source.read_bytes()
        robot = scene_module.robot_description()
        self.assertEqual(original, source.read_bytes())
        self.assertEqual(len(robot.findall("joint[@type='revolute']")), 16)
        self.assertEqual(len(robot.findall("joint[@type='prismatic']")), 0)
        links = {link.get("name") for link in robot.findall("link")}
        children = [joint.find("child").get("link") for joint in robot.findall("joint")]
        self.assertEqual(len(children), len(set(children)))
        self.assertEqual(links - set(children), {"world"})
        for mesh in robot.findall(".//mesh"):
            self.assertTrue(Path(mesh.get("filename")).is_file())
            self.assertTrue(Path(mesh.get("filename")).is_relative_to(ROOT / "dual_arm/meshes"))
        for side in ("l", "r"):
            joint = robot.find(f"joint[@name='arm_{side}_joint_finger']")
            self.assertEqual(joint.get("type"), "revolute")
            self.assertEqual(joint.find("axis").get("xyz"), "1 0 0")
            for part in ("gripper", "finger", "d405"):
                link = robot.find(f"link[@name='arm_{side}_link_{part}']")
                for geometry in ("visual", "collision"):
                    self.assertEqual(Path(link.find(f"{geometry}/geometry/mesh").get("filename")).name,
                                     f"arm_{side}_link_{part}.STL")

    def test_rectangular_towel_area_and_winding(self):
        vertices, indices = scene_module.towel_mesh((0.3, 0.4), (20, 28))
        vertices = np.asarray(vertices)
        faces = vertices[np.asarray(indices).reshape(-1, 3)]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        self.assertEqual(len(vertices), 609)
        self.assertTrue((normals[:, 2] > 0).all())
        self.assertAlmostEqual(np.linalg.norm(normals, axis=1).sum() / 2, 0.12)


@unittest.skipUnless(HAS_NEWTON, "Requires the towel-newton environment")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("Requires CUDA access")
        wp.set_device("cuda:0")

    def test_camera_views_follow_wrist_and_deforming_towel(self):
        viewer = newton.viewer.ViewerNull(num_frames=0)
        args = newton.examples.default_args()
        args.config = ROOT / "simulation/scene.yaml"
        scene = scene_module.TowelScene(viewer, args)
        try:
            rig = scene.cameras
            self.assertTrue(rig.update(scene))
            initial_poses = rig.poses.copy()
            initial_rgb = rig.rgba.numpy().copy()
            initial_depth = rig.depth.numpy().copy()
            self.assertEqual(initial_rgb.shape, (3, 240, 320, 4))
            self.assertFalse(rig.update(scene))
            for frame in initial_rgb:
                rgb = frame[:, :, :3].astype(float)
                cloth = (rgb[:, :, 1] > rgb[:, :, 0] * 1.5) & (rgb[:, :, 2] > rgb[:, :, 0] * 1.5)
                self.assertGreater(np.count_nonzero(cloth), 100)
                self.assertGreater(float(rgb.std()), 10)
            scene.set_joint_target("arm_l_joint_2", 0.2)
            scene.set_joint_target("arm_r_joint_2", -0.2)
            for _ in range(120):
                scene.step()
            self.assertTrue(rig.update(scene))
            np.testing.assert_allclose(rig.poses[0], initial_poses[0], atol=1e-6)
            for index in (1, 2):
                self.assertGreater(np.linalg.norm(rig.poses[index, :3] - initial_poses[index, :3]), 0.01)
                body = wp.transform(*scene.state_0.body_q.numpy()[rig.body_ids[index]])
                relative = wp.transform_multiply(wp.transform_inverse(body), wp.transform(*rig.poses[index]))
                np.testing.assert_allclose(list(relative), list(rig.mounts[index]), atol=1e-6)
                self.assertFalse(np.array_equal(initial_rgb[index], rig.rgba.numpy()[index]))
            depth = rig.depth.numpy()
            self.assertTrue(np.isfinite(depth).all())
            self.assertTrue((depth >= 0).all())
            # The fixed head camera must see the towel descend onto the table.
            before = initial_rgb[0].astype(float)
            after = rig.rgba.numpy()[0].astype(float)
            cloth = ((before[:, :, 1] > before[:, :, 0] * 1.5)
                     & (after[:, :, 1] > after[:, :, 0] * 1.5))
            self.assertGreater(np.count_nonzero(cloth), 100)
            self.assertGreater(float(np.median((depth[0] - initial_depth[0])[cloth])), 0.02)
            with tempfile.TemporaryDirectory() as directory:
                rig.save(scene, Path(directory))
                calibration = json.loads((Path(directory) / "cameras.json").read_text())
                self.assertEqual(set(calibration["cameras"]), set(rig.NAMES))
                for name in rig.NAMES:
                    self.assertTrue((Path(directory) / f"{name}.png").is_file())
                    saved = np.load(Path(directory) / f"{name}_depth.npy")
                    self.assertEqual(saved.shape, (240, 320))
            scene.reset()
            self.assertTrue(rig.update(scene))
            np.testing.assert_allclose(rig.poses, initial_poses, atol=1e-6)
        finally:
            viewer.close()

    def test_motion_grippers_contact_and_reset(self):
        viewer = newton.viewer.ViewerNull(num_frames=0)
        args = newton.examples.default_args()
        args.config = ROOT / "simulation/scene.yaml"
        scene = scene_module.TowelScene(viewer, args)
        try:
            initial_body = scene.state_0.body_q.numpy().copy()
            self.assertGreater(scene.fold.tcp_positions()[:, 2].min(), scene.config["table"]["height"] + 0.04)
            scene.set_joint_target("arm_l_joint_2", 0.2)
            scene.set_joint_target("arm_r_joint_2", -0.2)
            scene.set_gripper_opening("l", 0.0)
            scene.set_gripper_opening("r", 1.0)
            for _ in range(120):
                scene.step()
            q = scene.state_0.joint_q.numpy()
            for name, value in {
                "arm_l_joint_2": 0.2, "arm_r_joint_2": -0.2,
                "arm_l_joint_finger": 0.0, "arm_r_joint_finger": 1.5708,
            }.items():
                self.assertAlmostEqual(q[scene.q_index[name]], value, places=5)
            self.assertGreater(np.abs(scene.state_0.body_q.numpy() - initial_body).max(), 0.01)
            self.assertAlmostEqual(scene.gripper_opening("l"), 0.0, places=5)
            self.assertAlmostEqual(scene.gripper_opening("r"), 1.0, places=5)
            scene.test_final()
            with self.assertRaises(ValueError):
                scene.set_joint_target("arm_l_joint_2", float("nan"))
            scene.set_joint_target("arm_l_joint_2", 100.0)
            self.assertAlmostEqual(scene.targets[scene.q_index["arm_l_joint_2"]], 1.5708, places=5)

            scene.reset()
            self.assertEqual(scene.sim_time, 0.0)
            np.testing.assert_allclose(scene.state_0.joint_q.numpy(), scene.home, atol=1e-6)
            np.testing.assert_allclose(scene.state_0.particle_q.numpy(), scene.model.particle_q.numpy())
            for _ in range(90):
                scene.step()
            np.testing.assert_allclose(scene.state_0.joint_q.numpy(), scene.home, atol=1e-6)
            scene.test_final()
        finally:
            viewer.close()

    def test_automatic_fold_release_stop_and_reset(self):
        viewer = newton.viewer.ViewerNull(num_frames=0)
        args = newton.examples.default_args()
        args.config = ROOT / "simulation/scene.yaml"
        args.auto_fold = True
        scene = scene_module.TowelScene(viewer, args)
        try:
            # A closed gripper far from the corners must not attach anything.
            for side in ("l", "r"):
                scene.set_gripper_opening(side, scene_module.CLOSED_FRACTION)
            for _ in range(90):
                scene.step()
            self.assertFalse(scene.fold.attach())
            self.assertTrue(np.all(scene.grasp_owners.numpy() == -1))
            scene.fold.start_fold()
            saw_grasp = False
            peak_height = 0.0
            for frame in range(3600):
                scene.step()
                if scene.fold.phase == "fold":
                    owners = scene.grasp_owners.numpy()
                    self.assertEqual(np.count_nonzero(owners >= 0), 4)
                    points = scene.state_0.particle_q.numpy()
                    bodies = scene.state_0.body_q.numpy()
                    offsets = scene.grasp_offsets.numpy()
                    for i in np.flatnonzero(owners >= 0):
                        expected = wp.transform_point(wp.transform(*bodies[owners[i]]), wp.vec3(*offsets[i]))
                        np.testing.assert_allclose(points[i], expected, atol=1e-6)
                    peak_height = max(peak_height, float(points[owners >= 0, 2].max()))
                    if not saw_grasp:
                        # Stop freezes the actual pose and releases every pinned node.
                        scene.fold.stop()
                        self.assertFalse(scene.fold.active)
                        self.assertTrue(np.all(scene.grasp_owners.numpy() == -1))
                        np.testing.assert_array_equal(scene.model.particle_flags.numpy(), scene.free_particle_flags)
                        np.testing.assert_allclose(scene.targets, scene.state_0.joint_q.numpy())
                        self.assertTrue(scene.fold.attach())
                        scene.reset()
                        self.assertTrue(np.all(scene.grasp_owners.numpy() == -1))
                        np.testing.assert_array_equal(scene.model.particle_flags.numpy(), scene.free_particle_flags)
                        scene.fold.start_fold()
                        saw_grasp = True
                if frame % 60 == 0:
                    points = scene.state_0.particle_q.numpy()
                    self.assertTrue(np.isfinite(points).all())
                    self.assertGreater(float(points[:, 2].min()), scene.config["table"]["height"] - 0.01)
                if scene.fold.phase in ("done", "failed", "stopped"):
                    break
            self.assertTrue(saw_grasp)
            self.assertGreater(peak_height, scene.config["table"]["height"] + 0.08)
            self.assertEqual(scene.fold.phase, "done", scene.fold.message)
            self.assertEqual(scene.fold.history, [*scene.fold.ACTIVE, "done"])
            self.assertLess(scene.fold.metrics["mean_pair_error_m"], 0.025)
            self.assertLess(scene.fold.metrics["footprint_ratio"], 0.60)
            self.assertEqual(scene.config["towel"]["size"], [0.30, 0.30])
            self.assertEqual(scene.model.particle_count, 441)
            scene.cameras.update(scene)
            rgb = scene.cameras.rgba.numpy()[0, :, :, :3].astype(float)
            cloth = (rgb[:, :, 1] > 1.5 * rgb[:, :, 0]) & (rgb[:, :, 2] > 1.5 * rgb[:, :, 0])
            self.assertGreater(np.count_nonzero(cloth), 1500, "Folded towel back side disappeared from head camera")
            np.testing.assert_array_equal(scene.model.particle_flags.numpy(), scene.free_particle_flags)
            scene.test_final()
            scene.reset()
            self.assertEqual(scene.fold.phase, "idle")
            self.assertEqual(scene.fold.history, [])
            self.assertTrue(np.all(scene.grasp_owners.numpy() == -1))
            np.testing.assert_allclose(scene.state_0.particle_q.numpy(), scene.model.particle_q.numpy())
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
