#!/usr/bin/env python3
"""ENCOS dual arms and rotary grippers with a Newton VBD towel.

The robot is rate-limited kinematics; contact affects the towel, not the robot.
No ROS nodes or real hardware interfaces are used.
"""

from pathlib import Path
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import newton
import newton.examples
import newton.ik as ik
from newton.sensors import SensorCamera
import numpy as np
import warp as wp
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_model import (robot_description, GRASP_POINTS, OPEN_FRACTION, CLOSED_FRACTION,
                         CAPTURE_FRACTION, is_arm_joint)


def towel_mesh(size, cells):
    """Regular triangle sheet; material density is specified by total area."""
    nx, ny = cells
    if nx < 2 or ny < 2 or min(size) <= 0:
        raise ValueError("Towel requires positive dimensions and at least 2 cells per axis")
    vertices = [(x * size[0] / nx, y * size[1] / ny, 0.0)
                for y in range(ny + 1) for x in range(nx + 1)]
    triangles = []
    for y in range(ny):
        for x in range(nx):
            a = y * (nx + 1) + x
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            triangles.extend((a, b, c, b, d, c))
    return vertices, triangles


@wp.kernel
def advance_joints(
    q: wp.array[float], target: wp.array[float], speed: wp.array[float],
    dt: float, next_q: wp.array[float], next_qd: wp.array[float],
):
    i = wp.tid()
    delta = wp.clamp(target[i] - q[i], -speed[i] * dt, speed[i] * dt)
    next_q[i] = q[i] + delta
    next_qd[i] = delta / dt


@wp.kernel
def move_grasped_particles(
    owners: wp.array[int], offsets: wp.array[wp.vec3], bodies: wp.array[wp.transform],
    previous_bodies: wp.array[wp.transform], dt: float,
    positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    body = owners[i]
    if body >= 0:
        point = wp.transform_point(bodies[body], offsets[i])
        previous = wp.transform_point(previous_bodies[body], offsets[i])
        positions[i] = point
        velocities[i] = (point - previous) / dt


class AutoFold:
    """A dual-arm half-fold using assisted corner grasps."""

    ACTIVE = ("settle", "approach", "descend", "close", "fold", "lower", "open", "retreat", "settling")

    def __init__(self, scene):
        self.scene = scene
        self.phase = "idle"
        self.message = ""
        self.ik_solver = None
        self.body_ids = [scene.model.body_label.index(f"dual_arm/arm_{s}_link_gripper") for s in ("l", "r")]
        self.tcp = [wp.vec3(*point) for point in GRASP_POINTS]
        nx, ny = scene.config["towel"]["cells"]
        self.patches = [[ny * (nx + 1), ny * (nx + 1) + 1], [0, 1]]
        self.far_patches = [[i // (nx + 1) * (nx + 1) + nx - i % (nx + 1) for i in patch]
                            for patch in self.patches]
        self.history = []
        self.metrics = None

    @property
    def active(self):
        return self.phase in self.ACTIVE

    def tcp_positions(self):
        bodies = self.scene.state_0.body_q.numpy()
        return np.array([wp.transform_point(wp.transform(*bodies[i]), tcp) for i, tcp in zip(self.body_ids, self.tcp)])

    def enter(self, phase):
        self.phase = phase
        self.elapsed = 0.0
        self.start = self.tcp_positions()
        self.history.append(phase)
        print(f"Fold: {phase}", flush=True)

    def start_fold(self):
        cfg = self.scene.config["fold"]
        for name in ("approach_height", "lift_height", "fold_duration", "pair_tolerance"):
            if not np.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError(f"fold.{name} must be positive and finite")
        self.scene.reset()
        self.history = []
        self.metrics = None
        self.message = ""
        self.enter("settle")

    def stop(self, message="Stopped"):
        self.scene.release_grasps()
        self.scene.targets[:] = self.scene.state_0.joint_q.numpy()
        self.phase = "stopped"
        self.message = message
        print(f"Fold stopped: {message}", flush=True)

    def init_ik(self):
        model = self.scene.model
        self.positions = [ik.IKObjectivePosition(i, tcp, wp.zeros(1, dtype=wp.vec3), weight=3.0)
                          for i, tcp in zip(self.body_ids, self.tcp)]
        self.rotations = [ik.IKObjectiveRotation(
            i, wp.quat_identity(), wp.array([[0, 0, 0, 1]], dtype=wp.vec4), weight=0.3,
        ) for i in self.body_ids]
        limits = ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=10.0)
        self.ik_q = wp.array(self.scene.home[None], dtype=float)
        self.ik_solver = ik.IKSolver(
            model, n_problems=1, objectives=[*self.positions, *self.rotations, limits],
            lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_graph = None
        if model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_q, self.ik_q, iterations=24)
            self.ik_graph = capture.graph

    def move(self, destination, duration, arc=0.0):
        if self.ik_solver is None:
            self.init_ik()
        u = min(self.elapsed / duration, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        target = self.start * (1.0 - u) + destination * u
        target[:, 2] += arc * np.sin(np.pi * u)
        for objective, point in zip(self.positions, target):
            objective.target_positions.assign([point])
        if self.ik_graph is None:
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=24)
        else:
            wp.capture_launch(self.ik_graph)
        q = self.ik_q.numpy()[0]
        if not np.isfinite(q).all():
            self.stop("IK produced non-finite joint targets")
            return False
        for name, index in self.scene.q_index.items():
            if is_arm_joint(name):
                self.scene.set_joint_target(name, float(q[index]))
        error = np.linalg.norm(self.tcp_positions() - destination, axis=1).max()
        if self.elapsed > duration + 12.0:
            self.stop(f"Unreachable or stalled target ({error * 1000:.1f} mm)")
            return False
        return self.elapsed >= duration and error < 0.006

    def attach(self):
        scene = self.scene
        if getattr(self, "external_grasps", False):
            owners = scene.grasp_owners.numpy()
            return all(np.any(owners == body) for body in self.body_ids)
        points = scene.state_0.particle_q.numpy()
        bodies = scene.state_0.body_q.numpy()
        owners = np.full(scene.model.particle_count, -1, dtype=np.int32)
        offsets = np.zeros((scene.model.particle_count, 3), dtype=np.float32)
        for side, body, patch, tcp in zip(("l", "r"), self.body_ids, self.patches, self.tcp):
            if scene.gripper_opening(side) > CAPTURE_FRACTION:
                return False
            inverse = wp.transform_inverse(wp.transform(*bodies[body]))
            local = np.array([wp.transform_point(inverse, wp.vec3(*points[i])) for i in patch])
            # Only capture cloth already inside the finger pads; never teleport a corner.
            inside = np.all(np.abs(local - np.asarray(tcp)) <= [0.022, 0.007, 0.008], axis=1)
            if not np.all(inside):
                return False
            owners[patch] = body
            offsets[patch] = local
        flags = scene.free_particle_flags.copy()
        flags[owners >= 0] &= ~int(newton.ParticleFlags.ACTIVE)
        scene.grasp_owners.assign(owners)
        scene.grasp_offsets.assign(offsets)
        scene.model.particle_flags.assign(flags)
        return True

    def measure(self):
        scene = self.scene
        nx, ny = scene.config["towel"]["cells"]
        grid = scene.state_0.particle_q.numpy().reshape(ny + 1, nx + 1, 3)
        columns = np.arange(nx // 2)
        pairs = np.linalg.norm(grid[:, columns, :2] - grid[:, nx - columns, :2], axis=2)
        return {"mean_pair_error_m": float(pairs.mean()),
                "max_pair_error_m": float(pairs.max()),
                "footprint_ratio": float(np.ptp(grid[:, :, 0]) / scene.config["towel"]["size"][0])}

    def update(self):
        if not self.active:
            return
        self.elapsed += self.scene.frame_dt
        scene = self.scene
        height = scene.config["table"]["height"]
        cfg = scene.config["fold"]
        if self.phase == "settle" and self.elapsed >= 1.0:
            points = scene.state_0.particle_q.numpy()
            if np.ptp(points[:, 2]) > 0.015:
                self.stop("Towel must start flat on the table")
                return
            self.pick = np.array([points[p].mean(axis=0) for p in self.patches])
            self.pick[:, 2] = height + 0.004
            self.place = np.array([points[p].mean(axis=0) for p in self.far_patches])
            self.place[:, 2] = height + 0.012
            if self.ik_solver is not None:
                self.ik_q.assign(scene.state_0.joint_q.numpy()[None])
            self.enter("approach")
        elif self.phase == "approach":
            if self.move(self.pick + [0, 0, cfg["approach_height"]], 4.0):
                self.enter("descend")
        elif self.phase == "descend":
            if self.move(self.pick, 3.0):
                for side in ("l", "r"):
                    scene.set_gripper_opening(side, CLOSED_FRACTION)
                self.enter("close")
        elif self.phase == "close" and self.elapsed >= 1.2:
            if not self.attach():
                self.stop("Corners are not between both closed grippers")
                return
            self.enter("fold")
        elif self.phase == "fold":
            destination = self.place + [0, 0, 0.025]
            if self.move(destination, cfg["fold_duration"], arc=cfg["lift_height"]):
                self.enter("lower")
        elif self.phase == "lower":
            if self.move(self.place, 2.0):
                if not getattr(self, "external_grasps", False):
                    scene.release_grasps()
                for side in ("l", "r"):
                    scene.set_gripper_opening(side, OPEN_FRACTION)
                self.enter("open")
        elif self.phase == "open" and self.elapsed >= 1.2:
            self.enter("retreat")
        elif self.phase == "retreat":
            destination = self.place + [0, 0, 0.08]
            if self.move(destination, 2.5):
                self.enter("settling")
        elif self.phase == "settling" and self.elapsed >= 3.0:
            self.metrics = self.measure()
            success = (self.metrics["mean_pair_error_m"] < cfg["pair_tolerance"]
                       and self.metrics["max_pair_error_m"] < 2 * cfg["pair_tolerance"]
                       and self.metrics["footprint_ratio"] < 0.68)
            self.enter("done" if success else "failed")
            self.message = f"Mean alignment error: {1000 * self.metrics['mean_pair_error_m']:.1f} mm"
            print(f"Fold result: {self.metrics}", flush=True)


class CameraRig:
    """Three pinhole RGB-D cameras rigidly mounted to the robot links."""

    NAMES = ("head", "left_wrist", "right_wrist")
    LABEL = "Cameras: 0 Head | 1 Left wrist | 2 Right wrist"

    def __init__(self, builder, config):
        self.config = config
        self.width, self.height = config["resolution"]
        if min(self.width, self.height) <= 0 or config["fps"] <= 0:
            raise ValueError("Camera resolution and rate must be positive")
        if not 0 < config["vertical_fov_deg"] < 180:
            raise ValueError("Camera vertical FOV must be between 0 and 180 degrees")
        self.body_ids, self.mounts = [], []
        for name in self.NAMES:
            mount = config[name]
            body = builder.body_label.index(f"dual_arm/{mount['parent']}")
            eye, target, up = (np.asarray(mount[key], dtype=float) for key in ("position", "target", "up"))
            # OpenGL camera frame: -Z forward, +Y up, +X right.
            back = eye - target
            if np.linalg.norm(back) < 1e-8:
                raise ValueError(f"Camera {name} position and target must differ")
            back /= np.linalg.norm(back)
            right = np.cross(up, back)
            if np.linalg.norm(right) < 1e-8:
                raise ValueError(f"Camera {name} up must not be parallel to its viewing direction")
            right /= np.linalg.norm(right)
            rotation = wp.quat_from_matrix(wp.mat33(np.column_stack((right, np.cross(back, right), back))))
            transform = wp.transform(wp.vec3(*eye), rotation)
            self.body_ids.append(body)
            self.mounts.append(transform)
        self.sensor = None
        self.timestamp = -np.inf

    def world_poses(self, state):
        bodies = state.body_q.numpy()
        return np.asarray([list(wp.transform_multiply(wp.transform(*bodies[i]), mount))
                           for i, mount in zip(self.body_ids, self.mounts)], dtype=np.float32)

    def update(self, scene, force=False):
        if not force and 0 <= scene.sim_time - self.timestamp < 1.0 / self.config["fps"] - 1e-8:
            return False
        if self.sensor is None:
            self.sensor = SensorCamera(scene.model)
            # A folded sheet exposes its back side; both sides must remain visible.
            self.sensor.default_render_config.enable_backface_culling = False
            self.sensor.create_default_light()
            self.rays = SensorCamera.compute_camera_rays_pinhole(
                self.width, self.height, camera_fov=np.deg2rad(self.config["vertical_fov_deg"]),
                device=scene.model.device,
            )
            self.transforms = wp.empty(3, dtype=wp.transform, device=scene.model.device)
            self.worlds = wp.zeros(3, dtype=wp.int32, device=scene.model.device)
            self.color = self.sensor.allocate_image_color(3, self.width, self.height)
            self.depth = self.sensor.allocate_image_depth(3, self.width, self.height)
        self.poses = self.world_poses(scene.state_0)
        self.transforms.assign(self.poses)
        scene.model.bvh_refit_shapes(scene.state_0)
        scene.model.bvh_refit_particles(scene.state_0)
        self.sensor.update(scene.state_0, self.transforms, self.rays, world_indices=self.worlds,
                           color_image=self.color, forward_depth_image=self.depth)
        self.rgba = SensorCamera.Utils.to_rgba_from_color(self.color)
        self.timestamp = scene.sim_time
        return True

    def save(self, scene, directory):
        from PIL import Image

        self.update(scene, force=True)
        directory.mkdir(parents=True, exist_ok=True)
        rgb, depth = self.rgba.numpy(), self.depth.numpy()
        focal = self.height / (2 * np.tan(np.deg2rad(self.config["vertical_fov_deg"]) / 2))
        metadata = {
            "simulation_time_s": self.timestamp, "resolution": [self.width, self.height],
            "intrinsics": [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]],
            "pose_convention": "camera to world; xyz + quaternion xyzw; -Z forward, +Y up",
            "depth_convention": "positive forward-axis depth in meters; 0 means no hit",
            "cameras": {},
        }
        for i, name in enumerate(self.NAMES):
            Image.fromarray(rgb[i, :, :, :3]).save(directory / f"{name}.png")
            np.save(directory / f"{name}_depth.npy", depth[i])
            metadata["cameras"][name] = {"world_pose": self.poses[i].tolist(), **self.config[name]}
        (directory / "cameras.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved RGB-D camera snapshots: {directory}", flush=True)


class TowelScene:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.config = yaml.safe_load(Path(args.config).read_text())
        cfg = self.config
        sim = cfg["simulation"]
        if min(sim["fps"], sim["substeps"], sim["iterations"]) <= 0:
            raise ValueError("Simulation rates and iteration counts must be positive")
        self.frame_dt = 1.0 / sim["fps"]
        self.substeps = int(sim["substeps"])
        self.dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.reset_requested = False
        self.vr = None

        robot = robot_description()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            ET.tostring(robot, encoding="unicode"), floating=False,
            enable_self_collisions=False, collapse_fixed_joints=False,
            joint_damping_mode="passive",
        )
        builder.body_flags = [int(newton.BodyFlags.KINEMATIC)] * builder.body_count
        self.cameras = CameraRig(builder, cfg["cameras"]) if cfg.get("cameras") and not getattr(args, "no_cameras", False) else None
        self.q_index = {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[i]
            for i, name in enumerate(builder.joint_label)
            if builder.joint_dof_dim[i][0] + builder.joint_dof_dim[i][1] == 1
        }
        if len(self.q_index) != 16:
            raise ValueError(f"Expected 14 arm and 2 rotary finger joints, got {self.q_index}")
        self.limits = {}
        for joint in robot.findall("joint"):
            if joint.get("type") in ("revolute", "prismatic"):
                limit = joint.find("limit")
                self.limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))

        self.home = np.zeros(len(builder.joint_q), dtype=np.float32)
        self.speeds = np.full_like(self.home, cfg["robot"]["joint_speed"])
        for side, key in (("l", "left"), ("r", "right")):
            if len(cfg["robot"][key]) != 7:
                raise ValueError(f"robot.{key} must contain seven joint angles")
            for index, value in enumerate(cfg["robot"][key], 1):
                self.home[self.q_index[f"arm_{side}_joint_{index}"]] = value
            name = f"arm_{side}_joint_finger"
            i = self.q_index[name]
            self.home[i] = self.gripper_angle(side, cfg["robot"]["gripper_opening"])
            self.speeds[i] = cfg["robot"]["finger_speed"]
        for name, (lower, upper) in self.limits.items():
            if not lower <= self.home[self.q_index[name]] <= upper:
                raise ValueError(f"Initial position outside URDF limits: {name}")
        if np.any(self.speeds <= 0):
            raise ValueError("Robot joint speeds must be positive")
        builder.joint_q[:] = self.home.tolist()
        self.targets = self.home.copy()

        table = cfg["table"]
        tx, ty = table["center"]
        sx, sy = table["size"]
        height, thickness = table["height"], table["thickness"]
        if min(sx, sy, height, thickness) <= 0 or thickness >= height:
            raise ValueError("Invalid table dimensions")
        surface = newton.ModelBuilder.ShapeConfig(mu=table["friction"], ke=1.0e4, kd=100.0)
        builder.add_shape_box(
            -1, xform=wp.transform((tx, ty, height - thickness / 2), wp.quat_identity()),
            hx=sx / 2, hy=sy / 2, hz=thickness / 2,
            cfg=surface, color=(0.55, 0.59, 0.62), label="table_top",
        )
        for x in (tx - sx / 2 + 0.045, tx + sx / 2 - 0.045):
            for y in (ty - sy / 2 + 0.045, ty + sy / 2 - 0.045):
                builder.add_shape_box(
                    -1, xform=wp.transform((x, y, (height - thickness) / 2), wp.quat_identity()),
                    hx=0.025, hy=0.025, hz=(height - thickness) / 2,
                    cfg=surface, color=(0.20, 0.23, 0.25), label="table_leg",
                )
        # The scene mount is added in memory; the source URDF has no stand.
        builder.add_shape_box(
            -1, xform=wp.transform((-0.07, 0.0, 0.45), wp.quat_identity()),
            hx=0.045, hy=0.06, hz=0.45, color=(0.25, 0.28, 0.30), label="robot_stand",
        )
        builder.add_ground_plane()

        towel = cfg["towel"]
        vertices, triangles = towel_mesh(towel["size"], towel["cells"])
        wx, wy = towel["size"]
        cx, cy = towel["center"]
        if towel["mass"] <= 0 or towel["particle_radius"] <= 0 or towel["drop_height"] <= 0:
            raise ValueError("Towel mass, radius and drop height must be positive")
        builder.add_cloth_mesh(
            vertices=[wp.vec3(*v) for v in vertices], indices=triangles,
            pos=wp.vec3(cx - wx / 2, cy - wy / 2, height + towel["drop_height"]),
            rot=wp.quat_identity(), vel=wp.vec3(0.0), scale=1.0,
            density=towel["mass"] / (wx * wy),
            tri_ke=towel["stretch_stiffness"], tri_ka=towel["area_stiffness"],
            tri_kd=towel["damping"], edge_ke=towel["bending_stiffness"], edge_kd=0.0,
            particle_radius=towel["particle_radius"], color=(0.12, 0.65, 0.55), label="towel",
        )
        builder.color()
        # All rigid bodies are prescribed/static; only cloth-rigid contacts are solved.
        builder.shape_flags = [int(flags) & ~int(newton.ShapeFlags.COLLIDE_SHAPES) for flags in builder.shape_flags]
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = towel["friction"]
        self.target_q = wp.array(self.targets, dtype=float)
        self.speed_q = wp.array(self.speeds, dtype=float)
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.004)
        self.contacts = self.pipeline.contacts()
        self.free_particle_flags = self.model.particle_flags.numpy().copy()
        self.grasp_owners = wp.array(np.full(self.model.particle_count, -1, dtype=np.int32), dtype=int)
        self.grasp_offsets = wp.zeros(self.model.particle_count, dtype=wp.vec3)
        self.fold = AutoFold(self)
        self.viewer.set_model(self.model)
        self.reset()
        self.viewer.set_camera(wp.vec3(1.45, -1.5, 1.45), -24.0, 132.0)
        if hasattr(self.viewer, "set_reset_callback"):
            self.viewer.set_reset_callback(lambda: setattr(self, "reset_requested", True))
        print(f"Loaded dual arms: 14 arm joints, 2 grippers; towel: {self.model.particle_count} vertices, "
              f"{wx:.2f} x {wy:.2f} m, {towel['mass']:.3f} kg.", flush=True)

    def reset(self):
        self.graph = None
        if self.cameras:
            self.cameras.timestamp = -np.inf
        self.release_grasps()
        self.fold.phase = "idle"
        self.fold.message = ""
        self.fold.metrics = None
        self.fold.history = []
        self.targets[:] = self.home
        self.target_q.assign(self.targets)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.config["simulation"]["iterations"],
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True, particle_self_contact_margin=0.006,
            particle_self_contact_gap=0.004,
            particle_topological_contact_filter_threshold=2,
        )
        self.sim_time = 0.0
        self.reset_requested = False
        # An even number of swaps leaves graph input/output buffers unchanged.
        if self.model.device.is_cuda and self.substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        if self.vr is not None:
            self.vr.reset()

    def release_grasps(self):
        self.grasp_owners.fill_(-1)
        self.model.particle_flags.assign(self.free_particle_flags)

    def set_joint_target(self, name, value):
        if not np.isfinite(value):
            raise ValueError("Joint target must be finite")
        lower, upper = self.limits[name]
        self.targets[self.q_index[name]] = np.clip(value, lower, upper)

    def set_gripper_opening(self, side, opening):
        """Set normalized jaw opening: 0 closed, 1 at the URDF's angular limit."""
        self.set_joint_target(f"arm_{side}_joint_finger", self.gripper_angle(side, opening))

    def gripper_angle(self, side, opening):
        if not np.isfinite(opening):
            raise ValueError("Gripper opening must be finite")
        lower, upper = self.limits[f"arm_{side}_joint_finger"]
        return float(np.clip(opening, 0, 1)) * (lower if side == "l" else upper)

    def gripper_opening(self, side, targets=False):
        q = self.targets if targets else self.state_0.joint_q.numpy()
        return float(q[self.q_index[f"arm_{side}_joint_finger"]] / self.gripper_angle(side, 1.0))

    def step(self):
        if self.reset_requested:
            self.reset()
        if self.vr is not None:
            self.vr.update()
        self.fold.update()
        self.target_q.assign(self.targets)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(advance_joints, dim=len(self.home), inputs=[
                self.state_0.joint_q, self.target_q, self.speed_q, self.dt,
                self.state_1.joint_q, self.state_1.joint_qd,
            ])
            newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            wp.launch(move_grasped_particles, dim=self.model.particle_count, inputs=[
                self.grasp_owners, self.grasp_offsets, self.state_1.body_q, self.state_0.body_q, self.dt,
                self.state_0.particle_q, self.state_0.particle_qd,
            ])
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            wp.launch(move_grasped_particles, dim=self.model.particle_count, inputs=[
                self.grasp_owners, self.grasp_offsets, self.state_1.body_q, self.state_0.body_q, self.dt,
                self.state_1.particle_q, self.state_1.particle_qd,
            ])
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.cameras and not isinstance(self.viewer, newton.viewer.ViewerNull):
            if self.cameras.update(self):
                self.viewer.log_image(CameraRig.LABEL, self.cameras.rgba)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        if self.vr is not None:
            changed, enabled = ui.checkbox("VR control", self.vr.enabled)
            if changed:
                self.vr.set_enabled(enabled)
            ui.text(f"VR: {self.vr.status}")
            owners = self.grasp_owners.numpy()
            held = [int(np.count_nonzero(owners == i)) for i in self.fold.body_ids]
            ui.text(f"Held nodes: L {held[0]} / R {held[1]}")
        if ui.button("Reset + fold"):
            if self.vr is not None:
                self.vr.set_enabled(False)
            self.fold.start_fold()
        ui.same_line()
        if ui.button("Stop"):
            if self.vr is not None:
                self.vr.set_enabled(False)
            self.fold.stop()
        ui.text(f"Fold: {self.fold.phase}")
        if self.fold.message:
            ui.text_wrapped(self.fold.message)
        if ui.button("Reset scene"):
            self.reset_requested = True
        ui.same_line()
        if ui.button("Home"):
            if self.vr is not None:
                self.vr.set_enabled(False)
            self.fold.stop()
            self.targets[:] = self.home
        for side, label in (("l", "Left arm"), ("r", "Right arm")):
            if ui.collapsing_header(label):
                for index in range(1, 8):
                    name = f"arm_{side}_joint_{index}"
                    lo, hi = self.limits[name]
                    changed, value = ui.slider_float(
                        f"J{index}##{side}", float(self.targets[self.q_index[name]]), lo, hi,
                    )
                    if changed:
                        if self.vr is not None and self.vr.enabled:
                            self.vr.set_enabled(False)
                        if self.fold.active:
                            self.fold.stop("Manual control")
                        self.set_joint_target(name, value)
        for side, label in (("l", "Left opening (%)"), ("r", "Right opening (%)")):
            value = self.gripper_opening(side, targets=True) * 100
            ui.text(label)
            ui.set_next_item_width(-1)
            changed, opening = ui.slider_float(f"##opening_{side}", value, 0.0, 100.0)
            if changed:
                if self.vr is not None and self.vr.enabled:
                    self.vr.set_enabled(False)
                if self.fold.active:
                    self.fold.stop("Manual control")
                self.set_gripper_opening(side, opening / 100)

    def test_final(self):
        points = self.state_0.particle_q.numpy()
        velocity = self.state_0.particle_qd.numpy()
        q = self.state_0.joint_q.numpy()
        assert np.isfinite(points).all() and np.isfinite(velocity).all(), "Non-finite towel state"
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "Non-finite robot state"
        assert abs(float(self.model.particle_mass.numpy().sum()) - self.config["towel"]["mass"]) < 1e-5
        for name, (lo, hi) in self.limits.items():
            assert lo - 1e-5 <= q[self.q_index[name]] <= hi + 1e-5, name
        assert points[:, 2].min() > self.config["table"]["height"] - 0.01, "Towel penetrated table"
        assert np.max(np.abs(velocity)) < 5.0, "Unstable towel velocity"
        if self.sim_time >= 1.0 and not self.fold.active:
            assert np.median(points[:, 2]) < self.config["table"]["height"] + 0.025, "Towel did not settle"
        if getattr(self.args, "auto_fold", False):
            assert self.fold.phase == "done", f"Fold did not complete: {self.fold.phase}: {self.fold.message}"
            assert np.all(self.grasp_owners.numpy() == -1), "Grasps were not released"
        print(f"PASS: finite states, arm/finger limits, towel mass and table contact ({self.sim_time:.2f}s).")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scene.yaml"))
    parser.add_argument("--screenshot", type=Path, help="Save the final rendered frame")
    parser.add_argument("--camera-output", type=Path, help="Save final head and wrist RGB-D frames and calibration")
    parser.add_argument("--no-cameras", action="store_true", help="Disable camera rendering and housings")
    parser.add_argument("--auto-fold", action="store_true", help="Reset and perform one dual-arm half-fold")
    parser.add_argument("--vr", action="store_true", help="Receive local VR motion and assisted gripper commands")
    parser.add_argument("--vr-port", type=int, default=8766, help="Loopback UDP motion port")
    parser.add_argument("--vr-python", type=Path, help="Start input-only WebXR using this existing VR Python environment")
    parser.set_defaults(device="cuda:0", num_frames=180)
    viewer, args = newton.examples.init(parser)
    scene, vr_process = None, None
    try:
        if args.vr and args.auto_fold:
            raise ValueError("Choose --vr or --auto-fold as the initial controller")
        if args.vr_python and not args.vr:
            raise ValueError("--vr-python requires --vr")
        scene = TowelScene(viewer, args)
        if args.vr:
            from vr_control import VRTeleop

            scene.vr = VRTeleop(scene, args.vr_port)
            if args.vr_python:
                vr_process = subprocess.Popen([
                    str(args.vr_python.absolute()), "-I", "-u", str(Path(__file__).with_name("vr_input.py")),
                    "--port", str(scene.vr.port),
                ], cwd=ROOT)
        if args.camera_output and scene.cameras is None:
            raise ValueError("--camera-output requires enabled cameras")
        if args.auto_fold:
            scene.fold.start_fold()
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(scene.gui, position="panel")
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        while viewer.is_running():
            frame_start = time.monotonic()
            if vr_process is not None and vr_process.poll() is not None:
                raise RuntimeError(f"VR input process exited ({vr_process.returncode}); check its terminal output")
            if scene.reset_requested:
                scene.reset()
            if viewer.should_step():
                scene.step()
            elif scene.vr is not None:
                scene.vr.update(paused=True)
            scene.render()
            if scene.vr is not None:
                time.sleep(max(0, scene.frame_dt - (time.monotonic() - frame_start)))
        if args.test:
            scene.test_final()
        if args.camera_output:
            scene.cameras.save(scene, args.camera_output)
        if args.screenshot:
            from PIL import Image

            if not hasattr(viewer, "get_frame"):
                raise ValueError("--screenshot requires the OpenGL viewer")
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
    finally:
        if scene is not None and scene.vr is not None:
            scene.vr.close()
        if vr_process is not None:
            vr_process.terminate()
            try:
                vr_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vr_process.kill()
                vr_process.wait()
        viewer.close()


if __name__ == "__main__":
    main()
