"""Shared model loading for Newton and the isolated RViz model viewer."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "dual_arm"
BASE_HEIGHT = 1.0
# Contact centers at the mesh fingertips with approximately 6 mm jaw clearance.
GRASP_POINTS = ((0.0, 0.002, -0.136), (0.0, -0.002, -0.136))
OPEN_FRACTION = 0.5
CLOSED_FRACTION = 0.04
CAPTURE_FRACTION = 0.055


def robot_description(mesh_uris=False):
    """Resolve the new source URDF and add a fixed scene mount without modifying it."""
    robot = ET.parse(MODEL_DIR / "urdf/dual_arm.urdf").getroot()
    for mesh in robot.findall(".//mesh"):
        uri = mesh.get("filename")
        prefix = "package://dual_arm/"
        if not uri.startswith(prefix):
            raise ValueError(f"Unexpected mesh URI: {uri}")
        path = MODEL_DIR / uri.removeprefix(prefix)
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", path.as_uri() if mesh_uris else str(path))
    ET.SubElement(robot, "link", name="world")
    mount = ET.SubElement(robot, "joint", name="simulation_base_mount", type="fixed")
    ET.SubElement(mount, "parent", link="world")
    ET.SubElement(mount, "child", link="arm_base_link")
    ET.SubElement(mount, "origin", xyz=f"0 0 {BASE_HEIGHT}", rpy="0 0 0")
    return robot


def is_arm_joint(name):
    return name in {f"arm_{side}_joint_{i}" for side in ("l", "r") for i in range(1, 8)}
