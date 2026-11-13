"""Evaluate regrasp continuations after an ACT grasp in Newton."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulation'))
from act_io import canonical_state, fold_verdict, internal_action, measure_fold
from newton_env import NewtonTowelEnv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--adapt-to-cloth', action='store_true',
                        help='Generate a new expert path after the cloth settles')
    parser.add_argument('--use-corner-height', action='store_true',
                        help='Plan a 3D approach to the measured corners after clearance')
    parser.add_argument('--frame', type=int, help='Use this recorded bilateral-grasp frame for adaptive runs')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.use_corner_height and not args.adapt_to_cloth:
        parser.error('--use-corner-height requires --adapt-to-cloth')
    if args.frame is not None and not args.adapt_to_cloth:
        parser.error('--frame requires --adapt-to-cloth')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    with np.load(args.rollout / 'trajectory.npz') as z:
        actions, recorded_owners, recorded_joints = z['action'], z['grasp_owner'], z['joint_state']
    with np.load(args.reference) as z:
        reference = z['expert_target']
    first_close = int(np.flatnonzero(np.all(reference[:, [7, 15]] >= .5, axis=1))[0])
    first_release = first_close + int(np.flatnonzero(np.all(reference[first_close:, [7, 15]] < .5, axis=1))[0])
    seed = json.loads((args.rollout / 'evaluation.json').read_text())['seed']
    env = NewtonTowelEnv(device=args.device)

    def step(action):
        env.step(action)
        return env.step(action)

    try:
        grasp_frames = [i for i, owners in enumerate(recorded_owners[:-1])
                        if len(np.unique(owners[owners >= 0])) == 2]
        if args.adapt_to_cloth and not grasp_frames:
            parser.error('The rollout must include a bilateral grasp')
        candidates = [(grasp_frames[0], None, 10), (grasp_frames[0] + 3, None, 10),
                      (292, None, 20)] if args.adapt_to_cloth else [
            (292, 260, 45), (292, 240, 45), (292, 260, 75),
            (292, 220, 45), (295, 245, 45), (292, 240, 75),
        ]
        if args.frame is not None:
            if args.frame not in grasp_frames:
                parser.error('--frame must identify a recorded bilateral grasp')
            candidates = [(args.frame, None, 10)]
        for trial, (frame, start, wait) in enumerate(candidates * args.repeats, start=1):
            env.reset(seed)
            for action in actions[:frame]:
                initial = step(action)
            owners = initial['grasp_owner']
            if not np.array_equal(owners, recorded_owners[frame]) or not np.any(owners>=0):
                continue
            opened = initial['joint_state'].copy()
            opened[14:] = .5
            seq = [opened.copy() for _ in range(wait)]
            phases = ['regrasp_open'] * wait
            if start is None:
                clearance = recorded_joints[150].copy()
                clearance[14:] = .5
                for fraction in np.linspace(0, 1, 46)[1:]:
                    u = fraction * fraction * (3 - 2 * fraction)
                    seq.append(opened + u * (clearance - opened))
                    phases.append('regrasp_clearance')
            ever = [False, False]
            bodies = env.scene.fold.body_ids
            for command in seq:
                state = step(command)
            if start is None:
                fold = env.scene.fold
                if args.use_corner_height:
                    for _ in range(60):
                        seq.append(clearance.copy())
                        phases.append('regrasp_settle')
                        state = step(clearance)
                    points = state['cloth_points']
                    height = env.scene.config['table']['height']
                    fold.pick = np.array([points[patch].mean(axis=0) for patch in fold.patches])
                    fold.pick[:, 2] = np.maximum(fold.pick[:, 2], height + .004)
                    fold.place = np.array([points[patch].mean(axis=0) for patch in fold.far_patches])
                    fold.place[:, 2] = height + .012
                    fold.init_ik()
                    fold.ik_q.assign(env.scene.state_0.joint_q.numpy()[None])
                    fold.enter('approach')
                else:
                    fold.enter('settle')
                for _ in range(1200):
                    # One expert update per action, two 60 Hz physics steps.
                    fold.elapsed += env.scene.frame_dt
                    command = env.expert_action()
                    seq.append(command.copy())
                    phases.append(fold.phase if fold.active else 'settling')
                    state = step(command)
                    ever = [v or bool(np.any(state['grasp_owner'] == b)) for v, b in zip(ever, bodies)]
                    if not fold.active:
                        break
            else:
                for i in range(start, len(reference)):
                    command = internal_action(reference[i])
                    seq.append(command)
                    phases.append('correct' if i < first_close else 'fold_closed' if i < first_release else 'release')
                    state = step(command)
                    ever = [v or bool(np.any(state['grasp_owner'] == b)) for v, b in zip(ever, bodies)]
            metrics = measure_fold(state['cloth_points'], env.scene.config)
            success,checks = fold_verdict(metrics,env.scene.config,ever,np.all(state['grasp_owner']<0))
            result = dict(trial=trial, frame=frame, expert_start=start, open_wait_frames=wait, success=success,
                          checks=checks,metrics=metrics)
            print(json.dumps(result),flush=True)
            with (root / 'results.jsonl').open('a') as log:
                log.write(json.dumps(result) + '\n')
            if success:
                name = root / f'regrasp{frame}_from{start}_wait{wait}_trial{trial}'
                np.savez_compressed(name.with_suffix('.npz'),
                    prefix=np.stack([canonical_state(a) for a in actions[:frame]]),
                    recovery=np.stack([canonical_state(a) for a in seq]),phases=phases,
                    joint_state=initial['joint_state'],cloth_points=initial['cloth_points'],
                    grasp_owner=owners,goal='fold')
                name.with_suffix('.json').write_text(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
