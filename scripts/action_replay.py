"""Record expert targets and verify replay without any expert updates."""
import sys
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))
from newton_env import NewtonTowelEnv, save_episode

def canonical(x):
    y = x[[0,1,2,3,4,5,6,14,7,8,9,10,11,12,13,15]].copy()
    y[[7,15]] = (y[[7,15]] < .1).astype(np.float32)
    return y

out = ROOT / 'outputs/action_replay/episode_000'
env = NewtonTowelEnv()
try:
    states = [env.reset(0)]
    env.start_expert()
    actions = []
    for _ in range(3600):
        a = env.expert_action()
        actions.append(a)
        states.append(env.step(a))
        if env.scene.fold.phase in ('done','failed','stopped'):
            break
    assert env.scene.fold.phase == 'done', env.scene.fold.message
    expert_metrics = env.scene.fold.measure()
    save_episode(out, states, actions, env.metadata(), {'expert_metrics': expert_metrics})
    np.savez_compressed(out / 'canonical_samples.npz', **{
        'observation.state': np.array([canonical(s['joint_state']) for s in states[:-1]]),
        'action': np.array([canonical(a) for a in actions]),
        'timestamp': np.array([s['sim_time'] for s in states[:-1]])})
    env.reset(0)
    def forbidden(*a, **kw):
        raise AssertionError('Expert called during replay')
    env.scene.fold.update = forbidden
    env.scene.fold.attach = forbidden
    max_error = 0.
    captured = False
    for i, a in enumerate(actions):
        s = env.step(a)
        captured |= bool(np.any(s['grasp_owner'] >= 0))
        max_error = max(max_error, float(np.max(np.abs(s['cloth_points'] - states[i+1]['cloth_points']))))
    metrics = env.scene.fold.measure()
    cfg = env.scene.config['fold']
    assert captured and np.all(s['grasp_owner'] == -1)
    assert metrics['mean_pair_error_m'] < cfg['pair_tolerance']
    assert metrics['max_pair_error_m'] < 2 * cfg['pair_tolerance']
    assert metrics['footprint_ratio'] < .68
    result = dict(status='passed', frames=len(actions), duration_s=s['sim_time'],
                  expert_metrics=expert_metrics, replay_metrics=metrics,
                  max_cloth_error_m=max_error, expert_disabled=True,
                  limitations=['NPZ intermediate; RGB and LeRobot export are next stage',
                               'local assisted grasp; kinematic arms; fixed initial scene'])
    (out / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    env.close()
