"""Launch Newton physics + a live Isaac Sim UI in separate Python environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "simulation"))
sys.path.insert(0,str(ROOT / "simulation/isaac"))
from bridge_transport import send_message, receive_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--paused",action="store_true",help="Start with physics paused")
    parser.add_argument("--exit-on-done",action="store_true")
    parser.add_argument("--max-updates",type=int,default=0)
    parser.add_argument("--reset-after",type=int,default=0,help="One automatic reset for integration checks")
    parser.add_argument("--fps",type=int,default=20)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--timeout",type=float,default=120.)
    parser.add_argument("--newton-python",type=Path,default=Path.home()/"miniforge3/envs/towel-newton/bin/python")
    parser.add_argument("--output",type=Path,default=None)
    args = parser.parse_args()
    if args.fps<=0 or 60 % args.fps or args.max_updates<0 or args.reset_after<0:
        parser.error("fps must divide 60; update counts must be non-negative")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / f"outputs/bridge/live_{stamp}"
    output.mkdir(parents=True,exist_ok=True)
    (output/"result.json").unlink(missing_ok=True)
    status, failure = "closed", None
    worker = connection = app = executor = None
    request_id = 0
    updates, resets = 0, 0
    state = visual = None
    started = time.monotonic()
    with TemporaryDirectory(prefix="towel-live-") as runtime, (output/"newton.log").open("w") as worker_log, (output/"frames.jsonl").open("w") as frames_log:
        try:
            path = Path(runtime)/"physics.sock"
            worker = subprocess.Popen([str(args.newton_python),"-I","-u",str(ROOT/"scripts/isaac/live_newton_worker.py"),
                "--socket",str(path),"--seed",str(args.seed)],stdout=worker_log,stderr=subprocess.STDOUT,cwd=ROOT)
            deadline = time.monotonic()+args.timeout
            while not path.exists():
                if worker.poll() is not None:
                    raise RuntimeError(f"Newton worker exited; see {output/'newton.log'}")
                if time.monotonic()>deadline:
                    raise TimeoutError("Newton socket startup timed out")
                time.sleep(.05)
            connection = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            connection.settimeout(args.timeout)
            connection.connect(str(path))
            ready = receive_message(connection)
            state, meta = ready["state"], ready["metadata"]
            (output/"metadata.json").write_text(json.dumps(meta,indent=2))
            print(f"LIVE Newton connected pid={worker.pid}; output={output}",flush=True)

            from isaacsim import SimulationApp
            app = SimulationApp({"headless":args.headless,"width":1280,"height":720,"window_width":1440,"window_height":900})
            from visual_scene import build_scene
            from PIL import Image
            visual = build_scene(meta,np.asarray(state["cloth_points"],dtype=np.float32),args.fps)
            executor = ThreadPoolExecutor(max_workers=1)
            request_id = 0
            pending = None
            flags = {"playing":not args.paused,"reset":False,"step":False,"quit":False,"save":False}
            dirty = True
            terminal = False
            terminal_validated = False
            auto_reset_done = False
            phases_seen = set()
            previous_state = None
            labels = {}
            camera_window = control_window = None
            providers = {}

            def rpc(command):
                send_message(connection,command)
                response = receive_message(connection)
                if response.get("request_id") != command["request_id"]:
                    raise RuntimeError("RPC response id mismatch")
                return response

            def request(command):
                nonlocal request_id
                request_id += 1
                return executor.submit(rpc,{"command":command,"request_id":request_id,
                    "episode_id":state["episode_id"],"frame_id":state["frame_id"],"steps":60//args.fps})

            if not args.headless:
                import omni.ui as ui
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("Isaac viewport is unavailable")
                viewport.camera_path = "/World/Overview"
                control_window = ui.Window("Newton Live Controls",width=360,height=265,position_x=20,position_y=60)
                def toggle():
                    flags["playing"] = not flags["playing"]
                def mark(key):
                    flags[key] = True
                with control_window.frame:
                    with ui.VStack(spacing=6):
                        ui.Label("Newton physics + Isaac RTX | LOCKSTEP")
                        labels["state"] = ui.Label("Starting...")
                        labels["time"] = ui.Label("")
                        with ui.HStack(height=28):
                            ui.Button("Play / Pause",clicked_fn=toggle)
                            ui.Button("Step",clicked_fn=lambda:mark("step"))
                        with ui.HStack(height=28):
                            if not args.collect:
                                ui.Button("Reset + Fold",clicked_fn=lambda:mark("reset"))
                            ui.Button("Save views",clicked_fn=lambda:mark("save"))
                        ui.Button("Show / hide cameras",height=28,clicked_fn=lambda:setattr(camera_window,"visible",not camera_window.visible))
                        ui.Button("Exit both processes",height=28,clicked_fn=lambda:mark("quit"))
                        ui.Label("Recording one episode. Wait for DATASET SAVED." if args.collect else "Finished: Reset + Fold to run again.")
                camera_window = ui.Window("Newton RGB Cameras",width=985,height=285,visible=False)
                with camera_window.frame:
                    with ui.HStack():
                        for name in ("head","left_wrist","right_wrist"):
                            with ui.VStack(width=320):
                                ui.Label(name,height=20)
                                providers[name] = ui.ByteImageProvider()
                                ui.ImageWithProvider(providers[name],width=320,height=240)

            collector = None
            if args.collect:
                from collect_dataset import Collector
                collector = Collector(output / "dataset", args.fps)
            last_rgbs = None
            while app.is_running() and not flags["quit"]:
                cycle_start = time.monotonic()
                if pending is not None:
                    if not pending.done():
                        app.update()
                        time.sleep(.005)
                        continue
                    response = pending.result()
                    pending = None
                    if collector is not None and response.get("action") is not None:
                        collector.add(state, last_rgbs, response["action"])
                    new_state = response["state"]
                    if previous_state is not None:
                        if new_state["episode_id"] == previous_state["episode_id"]:
                            if new_state["frame_id"] <= previous_state["frame_id"]:
                                raise RuntimeError("Physics did not advance")
                        else:
                            if new_state["episode_id"] != previous_state["episode_id"]+1 or new_state["frame_id"] != 0:
                                raise RuntimeError("Invalid synchronized reset")
                            resets += 1
                            phases_seen.clear()
                    state = new_state
                    terminal = response["terminal"]
                    terminal_validated = False
                    dirty = True
                if dirty:
                    visual.update(state)
                    rgbs, refs = visual.capture()
                    updates += 1
                    last_rgbs = rgbs
                    frames_log.write(json.dumps({"update":updates,"episode_id":state["episode_id"],
                        "frame_id":state["frame_id"],"sim_time":state["sim_time"],"phase":state["phase"],
                        "camera_reference_times":refs})+"\n")
                    frames_log.flush()
                    for name, provider in providers.items():
                        import cv2
                        preview = cv2.resize(rgbs[name], (320,240))
                        rgba = np.dstack([preview,np.full((240,320),255,dtype=np.uint8)])
                        provider.set_bytes_data(rgba.ravel().tolist(),[320,240])
                    if state["phase"] not in phases_seen or terminal:
                        for name,rgb in rgbs.items():
                            Image.fromarray(rgb).save(output/f"ep{state['episode_id']}_{name}_{state['phase']}.png")
                        phases_seen.add(state["phase"])
                    if "state" in labels:
                        labels["state"].text = f"Episode {state['episode_id']} | {state['phase']} | frame {state['frame_id']}"
                        labels["time"].text = f"Newton t={state['sim_time']:.2f}s | captured {updates}"
                    if updates%50==0 or terminal:
                        print(f"LIVE update={updates} ep={state['episode_id']} frame={state['frame_id']} t={state['sim_time']:.2f} {state['phase']}",flush=True)
                    previous_state = state
                    dirty = False
                    if not args.headless and updates == 1:
                        import omni.renderer_capture
                        omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str((output/"ui_started.png").resolve()))
                    (output/"live_status.json").write_text(json.dumps({"isaac_pid":os.getpid(),"worker_pid":worker.pid,
                        "updates":updates,"episode_id":state["episode_id"],"frame_id":state["frame_id"],
                        "sim_time":state["sim_time"],"phase":state["phase"],"headless":args.headless},indent=2))
                if terminal and not terminal_validated:
                    terminal_validated = True
                    flags["playing"] = False
                    result = {"status":"passed" if state["phase"]=="done" else "failed",
                        "mode":"live lockstep","updates":updates,"resets":resets,
                        "episode_id":state["episode_id"],"frame_id":state["frame_id"],"sim_time":state["sim_time"],
                        "phase":state["phase"],"metrics":response["metrics"],
                        "max_link_pose_component_error":visual.max_pose_error,
                        "max_camera_position_error_m":visual.max_camera_error,
                        "ui":not args.headless,"worker_pid":worker.pid}
                    if collector is not None:
                        collector.finish(state["phase"] == "done")
                    (output/"completed_episode.json").write_text(json.dumps(result,indent=2))
                    if not args.headless:
                        import omni.renderer_capture
                        omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str((output/"ui_completed.png").resolve()))
                    print("LIVE EPISODE:",json.dumps(result),flush=True)
                    if args.headless or args.exit_on_done:
                        status = result["status"]
                        break
                if args.max_updates and updates>=args.max_updates:
                    status = "passed"
                    break
                if args.reset_after and updates>=args.reset_after and not auto_reset_done:
                    flags["reset"] = True
                    auto_reset_done = True
                if flags["save"] and last_rgbs:
                    for name,rgb in last_rgbs.items():
                        Image.fromarray(rgb).save(output/f"snapshot_{name}.png")
                    flags["save"] = False
                if flags["reset"] and args.collect:
                    print("Collection is one fixed episode; restart the shell script for a new run.", flush=True)
                    flags["reset"] = False
                if flags["reset"]:
                    flags["reset"] = False
                    flags["playing"] = True
                    pending = request("reset")
                elif (flags["playing"] or flags["step"]) and not terminal:
                    flags["step"] = False
                    pending = request("advance")
                app.update()
                time.sleep(max(0.,1/args.fps-(time.monotonic()-cycle_start)))
        except BaseException:
            status, failure = "failed", traceback.format_exc()
            traceback.print_exc()
        finally:
            if "collector" in locals() and collector is not None:
                collector.close()
            if connection is not None:
                # Never race a final quit packet with an outstanding RPC.
                if executor is not None:
                    executor.shutdown(wait=True,cancel_futures=True)
                try:
                    send_message(connection,{"command":"quit","request_id":request_id+1})
                    receive_message(connection)
                except (OSError,EOFError,ValueError):
                    pass
                connection.close()
            if worker is not None:
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    try:
                        worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
            summary = {"status":status,"updates":updates,"resets":resets,"wall_time_s":time.monotonic()-started,
                       "last_state":{k:state[k] for k in ("episode_id","frame_id","sim_time","phase")} if state else None,
                       "worker_stopped":worker is None or worker.poll() is not None,"headless":args.headless,"error":failure}
            if visual:
                summary.update(max_link_pose_component_error=visual.max_pose_error,max_camera_position_error_m=visual.max_camera_error)
            (output/"result.json").write_text(json.dumps(summary,indent=2))
            print("LIVE FINISHED:",json.dumps(summary),flush=True)
            if app is not None:
                app.close()
    if failure:
        raise RuntimeError(f"Live bridge failed; see {output/'result.json'}")


if __name__ == "__main__":
    main()
