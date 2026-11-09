"""学习步骤 02：机器人动力学、关节命令、桌面及三路相机冒烟验证。"""
from pathlib import Path
import json
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/isaac/step02"
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT / "simulation/scene.yaml").read_text())
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux, Gf
    from scipy.spatial.transform import Rotation
    from PIL import Image

    world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/20)
    add_reference_to_stage(str(ROOT / "outputs/isaac/step01/dual_arm.usd"), "/World/Robot")
    stage = world.stage
    robot = world.scene.add(SingleArticulation(prim_path="/World/Robot", name="dual_arm",
                                              position=np.array([0., 0., 1.])))
    robot.set_default_state(position=np.array([0., 0., 1.]))
    table = cfg["table"]
    world.scene.add(FixedCuboid(prim_path="/World/Table", name="table",
        position=np.array([*table["center"], table["height"]-table["thickness"]/2]),
        scale=np.array([*table["size"], table["thickness"]]), color=np.array([0.55, 0.48, 0.38])))
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1000)
    # Camera 的 USD 坐标：-Z 向前、+Y 向上，与原工程配置一致。
    cameras = {}
    for name in ("head", "left_wrist", "right_wrist"):
        c = cfg["cameras"][name]
        matches = [p for p in stage.Traverse() if p.GetName() == c["parent"]]
        assert len(matches) == 1, (name, matches)
        p, target, up = (np.array(c[k], dtype=float) for k in ("position", "target", "up"))
        z = p-target; z /= np.linalg.norm(z)
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
        cam = Camera(prim_path=str(matches[0].GetPath())+"/LearningCamera", name=name,
                     resolution=(320, 240), frequency=20)
        cam.set_local_pose(translation=p, orientation=q[[3, 0, 1, 2]], camera_axes="usd")
        cameras[name] = cam
    world.reset()
    robot.set_world_pose(position=np.array([0., 0., 1.]))
    names = robot.dof_names
    order = json.loads((ROOT / "outputs/isaac/step01/urdf_audit.json").read_text())["policy_joint_order"]
    assert set(names) == set(order), names
    home = np.zeros(len(names))
    for side, key in (("l", "left"), ("r", "right")):
        for i, value in enumerate(cfg["robot"][key], 1):
            home[names.index(f"arm_{side}_joint_{i}")] = value
        home[names.index(f"arm_{side}_joint_finger")] = (1 if side == "r" else -1)*np.pi/4
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros_like(home))
    controller = robot.get_articulation_controller()
    controller.set_gains(kps=np.full(len(names), 10000.), kds=np.full(len(names), 1000.))
    for cam in cameras.values():
        cam.initialize()
        cam.set_clipping_range(0.01, 10.)
        cam.set_focal_length(12.)
        cam.set_vertical_aperture(24*np.tan(np.deg2rad(65/2)))
    max_error = 0.
    for step in range(240):
        target = home.copy()
        target[names.index("arm_l_joint_1")] += 0.05*np.sin(2*np.pi*step/240)
        controller.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=step % 6 == 0)
        actual = robot.get_joint_positions()
        assert np.isfinite(actual).all(), "Non-finite joint state"
        max_error = max(max_error, float(np.max(np.abs(actual-target))))
    image_results = {}
    for name, cam in cameras.items():
        rgba = cam.get_rgba()
        assert rgba.shape == (240, 320, 4), (name, rgba.shape)
        rgb = rgba[:, :, :3].astype(np.uint8)
        Image.fromarray(rgb).save(OUT / f"{name}.png")
        image_results[name] = {"shape": list(rgb.shape), "std": float(rgb.std())}
        assert rgb.std() > 1, f"Camera {name} has no useful variation"
    stage.GetRootLayer().Export(str(OUT / "scene.usda"))
    report = {"status": "passed" if max_error < 0.1 else "tracking_needs_tuning", "physics_steps": 240, "physics_dt": 1/120,
              "dof_names": names, "final_positions": actual.tolist(),
              "max_tracking_error_rad": max_error, "cameras": image_results,
              "limits": "Robot smoke test only; towel and grasping are not implemented."}
    (OUT / "result.json").write_text(json.dumps(report, indent=2))
    print("STEP02:", json.dumps(report), flush=True)
finally:
    app.close()
