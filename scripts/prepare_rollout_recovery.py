"""Build a full-fold correction from an ACT rollout state."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_grasp_plans import (
    NewtonTowelEnv, canonical_state, step_control,
)
from act_io import fold_verdict, measure_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--expert-start-frame', type=int,
                        help='Replay the expert approach from this frame before closing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.rollout / 'evaluation.json').read_text())
    with np.load(args.rollout / 'trajectory.npz') as trace:
        if not 1 <= args.frame <= len(trace['canonical_policy_action']):
            parser.error('Frame must follow an executed policy action')
        actions = np.stack([canonical_state(action) for action in trace['action'][:args.frame]])
        joints = trace['joint_state'][args.frame].copy()
        cloth = trace['cloth_points'][args.frame].copy()
    initialization = evaluation['initialization']
    if initialization:
        with np.load(initialization['plan']) as plan:
            actions = np.concatenate([plan['prefix'], actions])
    with np.load(args.reference) as reference:
        expert = reference['expert_target'].copy()
    close_frames = np.flatnonzero(np.all(expert[:, [7, 15]] >= .5, axis=1))
    close_frame = int(close_frames[0])
    if args.expert_start_frame is not None and not 0 <= args.expert_start_frame < close_frame:
        parser.error('Expert approach must start before the first close command')
    approach = expert[close_frame].copy()
    approach[[7, 15]] = 0

    from recovery_collection import reproduce_prefix
    env = NewtonTowelEnv()
    try:
        env.reset(evaluation['seed'])
        state, joint_error, cloth_error = reproduce_prefix(env, actions, joints, cloth)
        recovery = []
        # Hold the open approach target only until the joints have reached it.
        if args.expert_start_frame is None:
            for _ in range(150):
                actual = canonical_state(state['joint_state'])
                actual[[7, 15]] = 0
                if np.max(np.abs(actual - approach)) < 1e-4:
                    break
                recovery.append(approach.copy())
                state = step_control(env, approach)
            else:
                raise RuntimeError('Correction did not reach the reference pose')
        else:
            for target in expert[args.expert_start_frame:close_frame]:
                recovery.append(target.copy())
                state = step_control(env, target)
        phases = ['correct'] * len(recovery)
        bodies = [env.scene.model.body_label.index(f'dual_arm/arm_{side}_link_gripper')
                  for side in 'lr']
        ever = [False, False]
        for target in expert[close_frame:]:
            recovery.append(target)
            phases.append('fold_closed' if np.all(target[[7, 15]] >= .5) else 'release')
            state = step_control(env, target)
            owners = np.asarray(state['grasp_owner'])
            ever = [previous or bool(np.any(owners == body))
                    for previous, body in zip(ever, bodies)]
        metrics = measure_fold(state['cloth_points'], env.scene.config)
        success, checks = fold_verdict(metrics, env.scene.config, ever, np.all(owners < 0))
        result = dict(success=success, checks=checks, metrics=metrics, frame=args.frame,
                      expert_start_frame=args.expert_start_frame,
                      joint_error=joint_error, cloth_error_m=cloth_error,
                      correction_frames=phases.count('correct'), recovery_frames=len(recovery),
                      source_rollout=str(args.rollout.resolve()), seed=evaluation['seed'])
        print(json.dumps(result, indent=2), flush=True)
        if not success:
            raise RuntimeError('Correction failed the full fold criterion')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        np.savez_compressed(args.output, prefix=actions, joint_state=joints, cloth_points=cloth,
                            recovery=np.asarray(recovery), phases=np.asarray(phases), goal='fold')
        args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
