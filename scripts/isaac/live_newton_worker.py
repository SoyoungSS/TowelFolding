"""Own all physics state; advance only on acknowledged requests from Isaac UI."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))
from bridge_transport import send_message, receive_message
from newton_env import NewtonTowelEnv


def serve(connection, env, seed, scene_config=None):
    state = env.reset(seed, scene_config=scene_config)
    env.start_expert()
    state = env.get_state()
    def reply(request_id):
        return {"type": "state", "request_id": request_id, "state": state,
                "terminal": state["phase"] in ("done", "failed", "stopped"),
                "metrics": env.scene.fold.metrics, "phase_history": env.scene.fold.history}
    ready = reply(0)
    ready["metadata"] = env.metadata()
    send_message(connection, ready)
    last_request_id = 0
    while True:
        request = receive_message(connection)
        request_id = request["request_id"]
        if request_id != last_request_id+1:
            raise ValueError("Non-sequential request id")
        if request["command"] == "quit":
            send_message(connection, {"type":"bye", "request_id":request_id})
            return
        if (request["episode_id"],request["frame_id"]) != (state["episode_id"],state["frame_id"]):
            raise ValueError("Client acknowledged wrong physics state")
        action = None
        if request["command"] == "reset":
            env.reset(seed, scene_config=scene_config)
            env.start_expert()
            state = env.get_state()
        elif request["command"] == "advance":
            count = request.get("steps", 3)
            if not isinstance(count,int) or not 1 <= count <= 60:
                raise ValueError("steps must be an integer from 1 to 60")
            original_dt = env.scene.frame_dt
            env.scene.frame_dt = original_dt * count
            try:
                action = env.expert_action()
            finally:
                env.scene.frame_dt = original_dt
            for _ in range(count):
                state = env.step(action)
        else:
            raise ValueError("Unknown command")
        last_request_id = request_id
        response = reply(request_id)
        response["action"] = action
        send_message(connection, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--recovery-plan", type=Path)
    parser.add_argument("--towel-offset", nargs=2, type=float, default=(0., 0.))
    args = parser.parse_args()
    if any(not math.isfinite(x) or abs(x) > .02 for x in args.towel_offset):
        parser.error("Towel displacement must be within ±20 mm per axis")
    if args.recovery_plan and any(args.towel_offset):
        parser.error("Recovery requires the original towel position")
    env = None
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(args.socket))
            listener.listen(1)
            print(f"WORKER listening {args.socket}",flush=True)
            with listener.accept()[0] as connection:
                connection.settimeout(None)  # UI can remain paused indefinitely
                env = NewtonTowelEnv()
                if args.recovery_plan:
                    from recovery_collection import serve_recovery
                    serve_recovery(connection, env, args.seed, args.recovery_plan)
                else:
                    config = deepcopy(env.base_config)
                    config["towel"]["center"] = [
                        center + offset
                        for center, offset in zip(config["towel"]["center"], args.towel_offset)
                    ]
                    serve(connection, env, args.seed, config)
    except EOFError:
        print("WORKER peer closed; shutting down",flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
