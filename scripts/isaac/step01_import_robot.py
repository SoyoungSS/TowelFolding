"""学习步骤 01：解析原始 URDF，启动 Isaac Sim，导出机器人 USD。

使用 Isaac Sim 5.1 自带/所在环境的 Python 执行；原始 URDF 不作修改。
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/isaac/step01"
OUT.mkdir(parents=True, exist_ok=True)
robot = ET.parse(ROOT / "dual_arm/urdf/dual_arm.urdf")
for mesh in robot.findall(".//mesh"):
    relative = mesh.attrib["filename"].removeprefix("package://dual_arm/")
    path = ROOT / "dual_arm" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh.set("filename", str(path))
joints = [j.attrib["name"] for j in robot.findall("joint") if j.attrib["type"] != "fixed"]
expected = [f"arm_{s}_joint_{i}" for s in ("l", "r") for i in range(1, 8)]
expected += [f"arm_{s}_joint_finger" for s in ("l", "r")]
assert set(joints) == set(expected), joints
robot.write(OUT / "resolved.urdf", encoding="utf-8", xml_declaration=True)
(OUT / "urdf_audit.json").write_text(json.dumps({
    "source": "dual_arm/urdf/dual_arm.urdf", "mesh_count": len(robot.findall(".//mesh")),
    "policy_joint_order": expected, "urdf_joint_order": joints,
    "note": "Policy order is explicit; do not use imported array order implicitly."
}, indent=2))
print("STEP01: URDF assets verified", flush=True)

# 必须先创建 SimulationApp，随后才能导入 omni/pxr 等运行时模块。
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})
try:
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
    assert ok
    config.merge_fixed_joints = False
    config.fix_base = True
    config.make_default_prim = True
    config.create_physics_scene = True
    config.import_inertia_tensor = True
    config.self_collision = False
    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(OUT / "resolved.urdf"),
        import_config=config, dest_path=str(OUT / "dual_arm.usd"),
    )
    assert ok, "URDF import command failed"
    for _ in range(10):
        app.update()
    assert (OUT / "dual_arm.usd").is_file()
    (OUT / "import_result.json").write_text(json.dumps({
        "status": "imported", "prim_path": prim_path,
        "usd": str(OUT / "dual_arm.usd"),
        "limitations": "Import only; no dynamics, camera or towel validation yet."
    }, indent=2))
    print("STEP01: USD export completed", flush=True)
finally:
    app.close()
