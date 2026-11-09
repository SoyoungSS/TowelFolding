"""Render a Newton trajectory in Isaac Sim, with NO simulated display replicas.

All visual link poses and cloth points come directly from the recorded world
state. This bypasses importer-realigned link frames and any PhysX integration.
"""
import argparse
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", type=Path, default=ROOT / "outputs/bridge/episode_000")
parser.add_argument("--output", type=Path, default=ROOT / "outputs/bridge/replay_000")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--max-frames", type=int, default=0, help="0 renders complete trajectory")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "result.json").unlink(missing_ok=True)
data = np.load(args.episode / "trajectory.npz", allow_pickle=False)
meta = json.loads((args.episode / "metadata.json").read_text())
if meta.get("urdf_sha256"):
    assert hashlib.sha256((ROOT / "dual_arm/urdf/dual_arm.urdf").read_bytes()).hexdigest() == meta["urdf_sha256"], "Source URDF changed since recording"
cfg = meta["config"]
sim_fps = cfg["simulation"]["fps"]
if args.fps <= 0 or sim_fps % args.fps:
    raise ValueError("Replay fps must divide the recorded simulation fps")
frame_indices = list(range(0, len(data["sim_time"]), sim_fps // args.fps))
if frame_indices[-1] != len(data["sim_time"])-1:
    frame_indices.append(len(data["sim_time"])-1)
if args.max_frames:
    frame_indices = frame_indices[:args.max_frames]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})
writers = {}
try:
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux, UsdShade, UsdPhysics, Gf, Vt, Sdf
    from scipy.spatial.transform import Rotation
    import trimesh
    from PIL import Image, ImageDraw

    world = World(stage_units_in_meters=1., physics_dt=1/args.fps, rendering_dt=1/args.fps)
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500.)
    UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(25, -30, 0))

    def material(name, color, roughness=0.6):
        mat = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def pose_ops(prim):
        x = UsdGeom.Xformable(prim)
        return x.AddTranslateOp(), x.AddOrientOp()

    def set_pose(ops, pose):
        ops[0].Set(Gf.Vec3d(*[float(v) for v in pose[:3]]))
        # FK quaternions can drift slightly from unit norm after GPU float32
        # operations. USD matrices assume unit quaternions; normalize at the bridge.
        q = np.asarray(pose[3:],dtype=np.float64).copy()
        q /= np.linalg.norm(q)
        ops[1].Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*[float(v) for v in q[:3]])))

    # Sibling link Xforms hold world poses; visuals retain original URDF origins.
    robot_xml = ET.parse(ROOT / "dual_arm/urdf/dual_arm.urdf").getroot()
    labels = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(meta["body_labels"])}
    links = {}
    for link in robot_xml.findall("link"):
        name = link.get("name")
        if name not in labels:
            raise ValueError(f"Missing Newton link: {name}")
        xf = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}")
        links[name] = (labels[name], pose_ops(xf.GetPrim()))
        for j, visual in enumerate(link.findall("visual")):
            mesh_xml = visual.find("geometry/mesh")
            if mesh_xml is None:
                raise ValueError("This robot replay expects mesh visuals")
            asset = ROOT / "dual_arm" / mesh_xml.get("filename").removeprefix("package://dual_arm/")
            loaded = trimesh.load(asset, force="mesh", process=False)
            mesh = UsdGeom.Mesh.Define(stage, f"{xf.GetPath()}/visual_{j}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(loaded.vertices, dtype=np.float32)))
            mesh.CreateFaceVertexCountsAttr([3]*len(loaded.faces))
            mesh.CreateFaceVertexIndicesAttr(np.asarray(loaded.faces).ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            origin = visual.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            set_pose(pose_ops(mesh.GetPrim()), np.r_[xyz, Rotation.from_euler("xyz", rpy).as_quat()])
            if mesh_xml.get("scale"):
                UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3f(*np.fromstring(mesh_xml.get("scale"), sep=" ")))
            color = visual.find("material/color")
            rgb = np.fromstring(color.get("rgba"), sep=" ")[:3] if color is not None else [0.4]*3
            bind(mesh.GetPrim(), material(f"{name}_{j}", rgb))

    def box(name, center, size, color):
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateSizeAttr(1.)
        x = UsdGeom.XformCommonAPI(cube)
        x.SetTranslate(Gf.Vec3d(*center))
        x.SetScale(Gf.Vec3f(*size))
        bind(cube.GetPrim(), material(name, color))

    t = cfg["table"]
    box("Table", [*t["center"], t["height"]-t["thickness"]/2], [*t["size"], t["thickness"]], [0.55,0.59,0.62])
    for i, (x, y) in enumerate(( (x,y) for x in (-1,1) for y in (-1,1) )):
        box(f"TableLeg{i}", [t["center"][0]+x*(t["size"][0]/2-.045),
             t["center"][1]+y*(t["size"][1]/2-.045), (t["height"]-t["thickness"])/2],
            [.05,.05,t["height"]-t["thickness"]], [.2,.23,.25])
    box("Stand", [-.07,0,.45], [.09,.12,.9], [.25,.28,.30])
    box("Floor", [0,0,-.025], [5,5,.05], [.25,.25,.25])
    cloth = UsdGeom.Mesh.Define(stage, "/World/Towel")
    faces = np.asarray(meta["cloth_triangles"], dtype=np.int32)
    cloth.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(data["cloth_points"][0], dtype=np.float32)))
    cloth.CreateFaceVertexCountsAttr([3]*len(faces))
    cloth.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    cloth.CreateSubdivisionSchemeAttr("none")
    cloth.CreateDoubleSidedAttr(True)
    cloth.CreateOrientationAttr("rightHanded")
    bind(cloth.GetPrim(), material("Towel", [.025,.55,.38], .95))

    def look_at(position, target, up):
        z = np.asarray(position)-np.asarray(target); z /= np.linalg.norm(z)
        x = np.cross(up,z); x /= np.linalg.norm(x)
        q = Rotation.from_matrix(np.column_stack([x,np.cross(z,x),z])).as_quat()
        return q[[3,0,1,2]]

    cameras = {}
    camera_mounts = {}
    for name in ("head", "left_wrist", "right_wrist", "overview"):
        c = cfg["cameras"].get(name)
        if c:
            path = f"/World/Robot/{c['parent']}/Camera"
            p, target, up = (np.array(c[k], dtype=float) for k in ("position","target","up"))
        else:
            path = "/World/Overview"
            p, target, up = np.array([.95,-.65,1.15]), np.array([.35,0,.58]), np.array([0.,0.,1.])
        cam = Camera(prim_path=path, name=name, resolution=(320,240))
        cam.set_local_pose(translation=p, orientation=look_at(p,target,up), camera_axes="usd")
        cameras[name] = cam
        camera_mounts[name] = (p, look_at(p,target,up))

    # Display geometry MUST NOT have PhysX body/collision/articulation schemas.
    forbidden = (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI)
    assert not [(str(p.GetPath()), str(api)) for p in stage.Traverse() for api in forbidden if p.HasAPI(api)]
    world.reset()
    time_annotators = {}
    calibration = {}
    for name, cam in cameras.items():
        cam.initialize()
        cam.set_clipping_range(.01,10.)
        # Pinhole intrinsics depend on aperture/focal ratio. Use conventional
        # USD camera magnitudes to avoid observed tiny-aperture RTX artifacts.
        lens = UsdGeom.Camera(cam.prim)
        fov = cfg["cameras"]["vertical_fov_deg"] if name != "overview" else 50.
        aperture = 24*np.tan(np.deg2rad(fov/2))
        lens.GetFocalLengthAttr().Set(12.)
        lens.GetVerticalApertureAttr().Set(float(aperture))
        lens.GetHorizontalApertureAttr().Set(float(aperture*4/3))
        focal_pixels = 240/(2*np.tan(np.deg2rad(fov/2)))
        calibration[name] = {"resolution":[320,240], "vertical_fov_deg": fov,
                             "intrinsics":[[focal_pixels,0,160],[0,focal_pixels,120],[0,0,1]],
                             "pose_convention":"world camera xyz + xyzw; -Z forward, +Y up"}
    for name, cam in cameras.items():
        p, q = camera_mounts[name]
        cam.set_local_pose(translation=p, orientation=q, camera_axes="usd")
        time_annotators[name] = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
        time_annotators[name].attach([cam.get_render_product_path()])
    rep.orchestrator.set_capture_on_play(False)

    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    def writer(name, width, height):
        command = [str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-an",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(args.output / f"{name}.mp4")]
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    for name in cameras:
        writers[name] = writer(name,320,240)
    writers["multiview"] = writer("multiview",640,480)

    manifests = []
    previous_rgb = {}
    changes = {n: 0 for n in cameras}
    max_pose_error = 0.
    max_camera_error = 0.
    checkpoint_phases = set()
    transform_cache = UsdGeom.XformCache()
    for output_frame, index in enumerate(frame_indices):
        for name, (body_index, ops) in links.items():
            set_pose(ops, data["body_pose"][index, body_index])
        points = np.asarray(data["cloth_points"][index], dtype=np.float32)
        cloth.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        cloth.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],dtype=np.float32)))
        # Render twice: consume a complete update after the visual state is set.
        # No physics attributes exist on these replicas; nothing integrates them.
        # Camera's legacy callback tries to interpolate a PhysX simulation clock.
        # We capture raw annotators synchronously instead, with Newton timestamps.
        for cam in cameras.values():
            cam.pause()
        rep.orchestrator.step(rt_subframes=2, delta_time=1/args.fps, pause_timeline=False)
        if output_frame == 0:
            # RTX annotators need several startup renders before their first buffer.
            for _ in range(8):
                rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=False)
        transform_cache.Clear()
        if output_frame % 50 == 0:
            for name, (body_index, _) in links.items():
                actual = transform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/Robot/{name}"))
                actual_pos = np.array(actual.ExtractTranslation())
                target_pose = data["body_pose"][index,body_index]
                max_pose_error = max(max_pose_error,float(np.max(np.abs(actual_pos-target_pose[:3]))))
                expected_rot = Rotation.from_quat(target_pose[3:]).as_matrix()
                max_pose_error = max(max_pose_error,float(np.max(np.abs(np.array(actual)[:3,:3].T-expected_rot))))
            np.testing.assert_allclose(np.array(cloth.GetPointsAttr().Get()),points,atol=1e-7)
            for name in ("head","left_wrist","right_wrist"):
                c = cfg["cameras"][name]
                body_pose = data["body_pose"][index,labels[c["parent"]]]
                expected_pos = body_pose[:3]+Rotation.from_quat(body_pose[3:]).apply(camera_mounts[name][0])
                actual = transform_cache.GetLocalToWorldTransform(cameras[name].prim)
                max_camera_error = max(max_camera_error,float(np.max(np.abs(np.array(actual.ExtractTranslation())-expected_pos))))
            if output_frame == 0:
                print(f"FRAME0 link_error={max_pose_error} camera_error={max_camera_error}",flush=True)

        rgbs, render_frames, camera_poses = {}, {}, {}
        for name, cam in cameras.items():
            frame = time_annotators[name].get_data()
            render_frame = {"numerator":int(frame["referenceTimeNumerator"]),
                            "denominator":int(frame["referenceTimeDenominator"])}
            if manifests:
                previous = manifests[-1]["camera_reference_times"][name]
                assert Fraction(**render_frame) > Fraction(**previous), (name,"stale camera frame")
            rgb = cam.get_rgba()[:,:,:3].astype(np.uint8)
            assert rgb.shape == (240,320,3) and rgb.std()>1, name
            if name in previous_rgb and not np.array_equal(rgb,previous_rgb[name]):
                changes[name] += 1
            previous_rgb[name] = rgb.copy()
            writers[name].stdin.write(rgb.tobytes())
            rgbs[name], render_frames[name] = rgb, render_frame
            pos, quat = cam.get_world_pose(camera_axes="usd")
            camera_poses[name] = np.r_[pos,quat[[1,2,3,0]]].tolist()
        assert len({Fraction(**v) for v in render_frames.values()}) == 1, "Camera captures are not synchronous"
        phase = str(data["phase"][index])
        panels = []
        for name in cameras:
            im = Image.fromarray(rgbs[name])
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,320,26),fill="black")
            draw.text((5,3), f"{name} | t={data['sim_time'][index]:.2f}s | {phase}", fill="white")
            panels.append(np.asarray(im))
        mosaic = np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
        writers["multiview"].stdin.write(mosaic.tobytes())
        if phase not in checkpoint_phases or output_frame == len(frame_indices)-1:
            Image.fromarray(mosaic).save(args.output / f"checkpoint_{phase}.png")
            for name in cameras:
                Image.fromarray(rgbs[name]).save(args.output / f"{name}_{phase}.png")
            checkpoint_phases.add(phase)
        manifests.append({"output_frame":output_frame,"source_index":index,
                          "frame_id":int(data["frame_id"][index]),"sim_time":float(data["sim_time"][index]),
                          "phase":phase,"camera_reference_times":render_frames,
                          "camera_world_poses":camera_poses,
                          "action_index":index if index < len(data["action"]) else None})
        if output_frame % 100 == 0:
            print(f"REPLAY {output_frame}/{len(frame_indices)} t={data['sim_time'][index]:.2f} phase={phase}",flush=True)
    for proc in writers.values():
        proc.stdin.close()
        assert proc.wait(timeout=60)==0, "Video encoding failed"
    writers.clear()
    assert max_pose_error < 1e-5 and max_camera_error < 1e-5, (max_pose_error,max_camera_error)
    if len(frame_indices)>20:
        assert all(v>0 for v in changes.values()),changes
    stage.GetRootLayer().Export(str(args.output / "final_scene.usdc"))
    report = {"status":"passed","rendered_frames":len(manifests),"video_fps":args.fps,
              "source_duration_s":float(data["sim_time"][frame_indices[-1]]),
              "max_link_pose_component_error":max_pose_error,"max_camera_position_error_m":max_camera_error,
              "changed_frames":changes,"physics_replicas":False,
              "phase_checkpoints":sorted(checkpoint_phases),
              "limitations":"Offline replay, not live IPC or an ACT-ready aligned RGB-action dataset"}
    (args.output / "frames.json").write_text(json.dumps(manifests,indent=2))
    (args.output / "calibration.json").write_text(json.dumps(calibration,indent=2))
    (args.output / "result.json").write_text(json.dumps(report,indent=2))
    print("PHASE2 PASS:",json.dumps(report),flush=True)
except BaseException:
    import traceback
    (args.output / "result.json").write_text(json.dumps({"status":"failed", "traceback":traceback.format_exc()},indent=2))
    traceback.print_exc()
    raise
finally:
    for proc in writers.values():
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    app.close()
