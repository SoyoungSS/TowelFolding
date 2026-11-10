#!/usr/bin/env python3
"""Build a source/asset delivery archive without local outputs or credentials."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", "outputs", ".runtime", ".venv"}


def files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.startswith(".venv-") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Delivery must not contain symlinks: {relative}")
        if not path.is_file() or path == MANIFEST:
            continue
        if path.suffix in {".pem", ".key", ".pyc", ".pyo"} or path.name == "imgui.ini":
            continue
        if relative.parts[0] in {"simulation", "dual_arm", "teleop", "scripts", "tests", "environments", "docs", "learning"} or path.name in {"README.md", "Makefile", "THIRD_PARTY_NOTICES.md", ".gitignore", "run_collect.sh"}:
            yield path


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    robot = ET.parse(ROOT / "dual_arm/urdf/dual_arm.urdf").getroot()
    for mesh in robot.findall(".//mesh"):
        uri = mesh.get("filename")
        if not uri.startswith("package://dual_arm/"):
            raise ValueError(f"Unexpected mesh URI: {uri}")
        asset = ROOT / "dual_arm" / uri.removeprefix("package://dual_arm/")
        if not asset.resolve().is_relative_to(ROOT / "dual_arm") or not asset.is_file():
            raise ValueError(f"Missing or external mesh: {uri}")
    current = hashes()
    if args.verify:
        expected = json.loads(MANIFEST.read_text())["sha256"]
        changed = sorted(k for k in set(current) | set(expected) if current.get(k) != expected.get(k))
        if changed:
            raise SystemExit("Files differ from delivery manifest:\n" + "\n".join(changed))
        print(f"PASS: {len(current)} file hashes and all URDF mesh paths")
        return
    MANIFEST.write_text(json.dumps({"format": 1, "sha256": current}, indent=2) + "\n")
    archive = ROOT.parent / "newton_towel_sim.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in [*files(), MANIFEST]:
            output.add(path, arcname=str(Path("newton_towel_sim") / path.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Created {archive} ({archive.stat().st_size / 1024**2:.1f} MiB), with SHA-256 sidecar")


if __name__ == "__main__":
    main()
