"""Headless, explicit-action adapter around the existing Newton towel scene.

All states are SI world coordinates. Actions/states use 14 radians followed by
two normalized gripper openings. Local assisted capture is driven by gripper
commands and geometry in step(), identically for expert and policy execution.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import hashlib
from importlib.metadata import version

import newton
import numpy as np
import warp as wp
import yaml

from towel_scene import TowelScene, ROOT, towel_mesh


JOINT_ORDER = [f"arm_{s}_joint_{i}" for s in ("l", "r") for i in range(1, 8)] + [
    "arm_l_joint_finger", "arm_r_joint_finger"]


class NewtonTowelEnv:
    def __init__(self, config=ROOT / "simulation/scene.yaml", device="cuda:0"):
        wp.init()
        wp.set_device(device)
        self.base_config = yaml.safe_load(Path(config).read_text())
        self.workspace = TemporaryDirectory(prefix="towel-env-")
        self.viewer = newton.viewer.ViewerNull(num_frames=0)
        self.scene = None
        self.episode_id = -1

    def reset(self, seed=0, scene_config=None):
        """Reset particles, grasps, clock and expert. Seed is reserved for randomization.

        scene_config is an optional complete config, not a partial nested patch.
        No implicit randomization is performed in this first milestone.
        """
        cfg = deepcopy(self.base_config if scene_config is None else scene_config)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        if self.scene is None or cfg != self.scene.config:
            path = Path(self.workspace.name) / "scene.yaml"
            path.write_text(yaml.safe_dump(cfg))
            self.scene = TowelScene(self.viewer, SimpleNamespace(config=path, no_cameras=False))
        else:
            self.scene.reset()
        # A fresh expert also removes stale IK graphs/history after reset.
        from towel_scene import AutoFold
        self.scene.fold = AutoFold(self.scene)
        self.scene.fold.external_grasps = True
        self.frame_id = 0
        self.close_frames = {"l": 0, "r": 0}
        self.episode_id += 1
        self.indices = [self.scene.q_index[n] for n in JOINT_ORDER]
        return self.get_state()

    def _ordered(self, q):
        result = np.asarray(q)[self.indices].copy()
        for i, side in ((14, "l"), (15, "r")):
            result[i] /= self.scene.gripper_angle(side, 1.)
        return result

    def expert_action(self):
        """Choose current expert target; grasp capture is handled by step()."""
        self.scene.fold.update()
        return self._ordered(self.scene.targets)

    def start_expert(self):
        # Unlike AutoFold.start_fold(), no hidden scene reset occurs here.
        if self.frame_id != 0:
            raise RuntimeError("Start expert immediately after reset")
        self.scene.fold.enter("settle")

    def step(self, action):
        """Advance exactly one scene frame, without hidden expert updates/rendering."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Expected finite action of shape (16,)")
        for name, value in zip(JOINT_ORDER[:14], action[:14]):
            self.scene.set_joint_target(name, float(value))
        for side, value in zip(("l", "r"), action[14:]):
            self.scene.set_gripper_opening(side, float(value))
        self.scene.target_q.assign(self.scene.targets)
        self._update_grasps()
        if self.scene.graph is not None:
            wp.capture_launch(self.scene.graph)
        else:
            self.scene.simulate()
        self.scene.sim_time += self.scene.frame_dt
        self.frame_id += 1
        return self.get_state()

    def _update_grasps(self):
        """Local assisted capture, shared by expert and action-only execution."""
        from robot_model import CAPTURE_FRACTION, GRASP_POINTS
        s = self.scene
        owners = s.grasp_owners.numpy().copy()
        offsets = s.grasp_offsets.numpy().copy()
        points = s.state_0.particle_q.numpy()
        bodies = s.state_0.body_q.numpy()
        for side, tcp in zip(("l", "r"), GRASP_POINTS):
            body = s.model.body_label.index(f"dual_arm/arm_{side}_link_gripper")
            commanded = s.targets[s.q_index[f"arm_{side}_joint_finger"]] / s.gripper_angle(side, 1.)
            if commanded > CAPTURE_FRACTION:
                owners[owners == body] = -1
                self.close_frames[side] = 0
                continue
            self.close_frames[side] += 1
            if self.close_frames[side] * s.frame_dt < 1.0:
                continue
            if s.gripper_opening(side) > CAPTURE_FRACTION or np.any(owners == body):
                continue
            inverse = wp.transform_inverse(wp.transform(*bodies[body]))
            local = np.array([wp.transform_point(inverse, wp.vec3(*p)) for p in points])
            capture = (owners < 0) & np.all(np.abs(local - np.asarray(tcp)) <= [0.022, 0.007, 0.008], axis=1)
            candidates = np.flatnonzero(capture)
            # Small fingertip patch: avoid rigidly pinning a broad cloth strip.
            candidates = candidates[np.argsort(np.linalg.norm(local[candidates] - np.asarray(tcp), axis=1))[:2]]
            owners[candidates] = body
            offsets[candidates] = local[candidates]
        flags = s.free_particle_flags.copy()
        flags[owners >= 0] &= ~int(newton.ParticleFlags.ACTIVE)
        s.grasp_owners.assign(owners)
        s.grasp_offsets.assign(offsets)
        s.model.particle_flags.assign(flags)

    def get_state(self):
        s = self.scene
        state = {
            "joint_state": self._ordered(s.state_0.joint_q.numpy()),
            "body_pose": s.state_0.body_q.numpy().copy(),
            "cloth_points": s.state_0.particle_q.numpy().copy(),
            "cloth_velocity": s.state_0.particle_qd.numpy().copy(),
            "grasp_owner": s.grasp_owners.numpy().copy(),
            "sim_time": float(s.sim_time), "frame_id": self.frame_id,
            "episode_id": self.episode_id, "phase": s.fold.phase,
        }
        for key in ("joint_state", "body_pose", "cloth_points", "cloth_velocity"):
            if not np.isfinite(state[key]).all():
                raise RuntimeError(f"Non-finite {key}")
        return state

    def metadata(self):
        _, triangles = towel_mesh(self.scene.config["towel"]["size"], self.scene.config["towel"]["cells"])
        return {"schema_version": 1, "seed": self.seed, "joint_order": JOINT_ORDER,
                "versions": {p: version(p) for p in ("newton", "warp-lang", "numpy")},
                "urdf_sha256": hashlib.sha256((ROOT / "dual_arm/urdf/dual_arm.urdf").read_bytes()).hexdigest(),
                "action_units": "14 radians, 2 normalized openings",
                "pose_convention": "world xyz + quaternion xyzw; meters; Z up",
                "body_labels": list(self.scene.model.body_label),
                "cloth_triangles": np.asarray(triangles).reshape(-1, 3).tolist(),
                "config": self.scene.config,
                "limitations": ["kinematic robot; no cloth reaction on arms",
                    "local geometry-based assisted attachment; not friction-only grasping",
                    "seed recorded; no implicit randomization"]}

    def close(self):
        self.viewer.close()
        self.workspace.cleanup()


def save_episode(directory, states, actions, metadata, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray([s[k] for s in states]) for k in states[0]}
    # T+1 states and T commands: action[t] maps state[t] to state[t+1].
    arrays["action"] = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "result.json").write_text(json.dumps(result, indent=2))
