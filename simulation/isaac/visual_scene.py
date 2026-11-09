"""Shared pure-display scene for live Newton state (Isaac imports occur after app startup)."""
from pathlib import Path
from fractions import Fraction
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def build_scene(meta, initial_points, fps=20):
    from types import SimpleNamespace
    cfg = meta["config"]
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux, UsdShade, UsdPhysics, Gf, Vt, Sdf
    from scipy.spatial.transform import Rotation
    import trimesh
    from PIL import Image, ImageDraw

    world = World(stage_units_in_meters=1., physics_dt=1/fps, rendering_dt=1/fps)
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
        return x.AddTranslateOp(), x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

    def set_pose(ops, pose):
        ops[0].Set(Gf.Vec3d(*[float(v) for v in pose[:3]]))
        # FK quaternions can drift slightly from unit norm after GPU float32
        # operations. USD matrices assume unit quaternions; normalize at the bridge.
        q = np.asarray(pose[3:],dtype=np.float64).copy()
        q /= np.linalg.norm(q)
        ops[1].Set(Gf.Quatd(float(q[3]), Gf.Vec3d(*[float(v) for v in q[:3]])))

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
    cloth.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(initial_points, dtype=np.float32)))
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
        cam = Camera(prim_path=path, name=name, resolution=(640,480))
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
        focal_pixels = 480/(2*np.tan(np.deg2rad(fov/2)))
        calibration[name] = {"resolution":[640,480], "vertical_fov_deg": fov,
                             "intrinsics":[[focal_pixels,0,160],[0,focal_pixels,120],[0,0,1]],
                             "pose_convention":"world camera xyz + xyzw; -Z forward, +Y up"}
    for name, cam in cameras.items():
        p, q = camera_mounts[name]
        cam.set_local_pose(translation=p, orientation=q, camera_axes="usd")
        time_annotators[name] = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
        time_annotators[name].attach([cam.get_render_product_path()])
    rep.orchestrator.set_capture_on_play(False)

    return VisualScene(SimpleNamespace(world=world, stage=stage, links=links, cloth=cloth,
        cameras=cameras, time_annotators=time_annotators, calibration=calibration,
        labels=labels, mounts=camera_mounts, set_pose=set_pose, rep=rep, cfg=cfg, fps=fps))


class VisualScene:
    def __init__(self, parts):
        self.__dict__.update(vars(parts))
        self.first = True
        self.previous_times = None
        self.max_pose_error = 0.
        self.max_camera_error = 0.

    def update(self, state):
        from pxr import UsdGeom, Vt
        from scipy.spatial.transform import Rotation
        bodies = np.asarray(state["body_pose"], dtype=np.float32)
        for _, (index, ops) in self.links.items():
            self.set_pose(ops, bodies[index])
        points = np.asarray(state["cloth_points"], dtype=np.float32)
        self.cloth.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        self.cloth.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32)))
        cache = UsdGeom.XformCache()
        for name, (index, _) in self.links.items():
            matrix = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"/World/Robot/{name}")))
            self.max_pose_error = max(self.max_pose_error, float(np.max(np.abs(matrix[3,:3]-bodies[index,:3]))),
                float(np.max(np.abs(matrix[:3,:3].T-Rotation.from_quat(bodies[index,3:]).as_matrix()))))
        for name in ("head", "left_wrist", "right_wrist"):
            c = self.cfg["cameras"][name]
            body = bodies[self.labels[c["parent"]]]
            expected = body[:3]+Rotation.from_quat(body[3:]).apply(self.mounts[name][0])
            actual = cache.GetLocalToWorldTransform(self.cameras[name].prim).ExtractTranslation()
            self.max_camera_error = max(self.max_camera_error,float(np.max(np.abs(np.asarray(actual)-expected))))
        if self.max_pose_error > 2e-5 or self.max_camera_error > 2e-5:
            raise RuntimeError(f"Visual pose mismatch: {self.max_pose_error}, {self.max_camera_error}")

    def capture(self):
        for camera in self.cameras.values():
            camera.pause()
        self.rep.orchestrator.step(rt_subframes=2, delta_time=1/self.fps, pause_timeline=False)
        if self.first:
            for _ in range(8):
                self.rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=False)
            self.first = False
        rgbs, refs = {}, {}
        for name, camera in self.cameras.items():
            t = self.time_annotators[name].get_data()
            refs[name] = [int(t["referenceTimeNumerator"]), int(t["referenceTimeDenominator"])]
            if self.previous_times and Fraction(*refs[name]) <= Fraction(*self.previous_times[name]):
                raise RuntimeError(f"Stale camera capture: {name}")
            rgb = camera.get_rgba()[:,:,:3].astype(np.uint8)
            if rgb.shape != (480,640,3) or rgb.std() <= 1:
                raise RuntimeError(f"Invalid image: {name}")
            rgbs[name] = rgb.copy()
        if len({Fraction(*r) for r in refs.values()}) != 1:
            raise RuntimeError("Camera capture times differ")
        self.previous_times = refs
        return rgbs, refs
