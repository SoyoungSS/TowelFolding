"""Record one headless Newton expert fold for deterministic Isaac replay."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))
from newton_env import NewtonTowelEnv, save_episode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "outputs/bridge/episode_000")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=3600)
args = parser.parse_args()
env = NewtonTowelEnv()
try:
    initial = env.reset(args.seed)
    # Verify explicit-action and reset semantics before producing the episode.
    try:
        env.step(np.full(16, np.nan))
        raise AssertionError("Invalid action accepted")
    except ValueError:
        pass
    command = initial["joint_state"].copy()
    command[0] += 0.05
    moved = env.step(command)
    assert moved["frame_id"] == 1 and moved["sim_time"] > 0
    assert moved["joint_state"][0] > initial["joint_state"][0]
    reset = env.reset(args.seed)
    np.testing.assert_allclose(reset["cloth_points"], initial["cloth_points"], atol=1e-7)
    np.testing.assert_allclose(reset["body_pose"], initial["body_pose"], atol=1e-7)
    assert reset["frame_id"] == 0 and reset["sim_time"] == 0
    assert (reset["grasp_owner"] == -1).all()
    env.start_expert()
    states, actions = [env.get_state()], []
    for frame in range(args.max_frames):
        action = env.expert_action()
        actions.append(action)
        state = env.step(action)
        states.append(state)
        if frame % 300 == 0:
            print(f"frame={frame} t={state['sim_time']:.2f} phase={state['phase']}", flush=True)
        if state["phase"] in ("done", "failed", "stopped"):
            break
    env.scene.args.auto_fold = True
    result = {"status": "passed" if env.scene.fold.phase == "done" else "failed",
              "phase": env.scene.fold.phase, "metrics": env.scene.fold.metrics,
              "phase_history": env.scene.fold.history, "frames": len(actions),
              "duration_s": states[-1]["sim_time"],
              "reset_and_explicit_action_checks": "passed",
              "state_action_alignment": "T+1 states, T actions; action[t] advances state[t]",
              "expert_grasp_events": "recorded in post-step grasp_owner; privileged expert side effects"}
    save_episode(args.output, states, actions, env.metadata(), result)
    env.scene.test_final()
    assert len(states) == len(actions)+1
    print(f"PHASE1 PASS: {args.output}; {result}", flush=True)
finally:
    env.close()
