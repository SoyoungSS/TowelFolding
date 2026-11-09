"""CPU-only acceptance of phase 1/2 artifacts, including full video decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", type=Path, default=ROOT / "outputs/bridge/episode_000")
parser.add_argument("--replay", type=Path, default=ROOT / "outputs/bridge/replay_000")
args = parser.parse_args()
d = np.load(args.episode / "trajectory.npz", allow_pickle=False)
result1 = json.loads((args.episode / "result.json").read_text())
result2 = json.loads((args.replay / "result.json").read_text())
frames = json.loads((args.replay / "frames.json").read_text())
assert result1["status"] == result2["status"] == "passed"
assert len(d["joint_state"]) == len(d["action"])+1
np.testing.assert_allclose(np.diff(d["sim_time"]),1/60,atol=1e-10)
assert np.array_equal(d["frame_id"],np.arange(len(d["frame_id"])))
assert np.isfinite(d["body_pose"]).all() and np.isfinite(d["cloth_points"]).all()
assert (d["grasp_owner"][0] == -1).all() and (d["grasp_owner"][-1] == -1).all()
assert (d["grasp_owner"]>=0).sum(1).max() == 4
assert len(frames) == result2["rendered_frames"]
assert frames[0]["source_index"] == 0
assert frames[-1]["source_index"] == len(d["sim_time"])-1
assert frames[-1]["phase"] == "done" and frames[-1]["action_index"] is None
for i, frame in enumerate(frames):
    j = frame["source_index"]
    assert abs(frame["sim_time"]-d["sim_time"][j]) < 1e-10
    assert frame["phase"] == str(d["phase"][j])
    refs = frame["camera_reference_times"]
    assert len({Fraction(**r) for r in refs.values()}) == 1
    if i:
        assert frames[i-1]["source_index"] < j
        assert all(Fraction(**refs[n]) > Fraction(**frames[i-1]["camera_reference_times"][n]) for n in refs)

bindir = Path(sys.executable).parent
def check_video(name):
    path = args.replay / f"{name}.mp4"
    command = [str(bindir/"ffprobe"),"-v","error","-select_streams","v:0","-count_frames",
               "-show_entries","stream=codec_name,width,height,nb_read_frames,r_frame_rate",
               "-of","json",str(path)]
    stream = json.loads(subprocess.check_output(command))["streams"][0]
    assert stream["codec_name"] == "h264", stream
    assert int(stream["nb_read_frames"]) == len(frames), stream
    assert Fraction(stream["r_frame_rate"]) == result2["video_fps"], stream
    assert [stream["width"],stream["height"]] == ([640,480] if name=="multiview" else [320,240])
    subprocess.run([str(bindir/"ffmpeg"),"-v","error","-xerror","-i",str(path),"-f","null","-"],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    return name,stream

with ThreadPoolExecutor(max_workers=5) as pool:
    videos = dict(pool.map(check_video,["head","left_wrist","right_wrist","overview","multiview"]))
cloth_pixels = {}
for phase in ("settle","done"):
    im = np.asarray(Image.open(args.replay / f"head_{phase}.png")).astype(float)
    mask = (im[:,:,1] > im[:,:,0]*1.3) & (im[:,:,2] > im[:,:,0]*1.3)
    cloth_pixels[phase] = int(mask.sum())
    assert mask.sum()>100, (phase,"head camera cannot see the colored towel")
report = {"status":"passed","source_states":len(d["sim_time"]),"replay_frames":len(frames),
          "videos":videos,"head_cloth_pixels":cloth_pixels,
          "checks":["T+1/T alignment","monotonic physical time","grasps held and released",
                    "source frame mapping","fresh synchronized camera captures","all videos fully decoded",
                    "head view sees towel at start and finish"]}
(args.replay / "acceptance.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
