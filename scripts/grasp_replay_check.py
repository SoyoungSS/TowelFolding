"""Separate arm and gripper errors by replaying recorded commands, without a policy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))
from act_io import internal_action, grasp_verdict
from newton_env import NewtonTowelEnv
from recovery_collection import reproduce_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.rollout / 'trajectory.npz') as trajectory:
        predicted = trajectory['canonical_policy_action'].copy()
        recorded_states = trajectory['joint_state'].copy()
    with np.load(args.plan) as plan:
        prefix, joints, cloth = plan['prefix'], plan['joint_state'], plan['cloth_points']
        expert = plan['recovery'].copy()
    expert = expert[np.minimum(np.arange(len(predicted)), len(expert) - 1)]
    variants = {'policy_replay': predicted.copy(), 'expert_replay': expert.copy(),
                'policy_arms_expert_grippers': predicted.copy(),
                'expert_arms_policy_grippers': expert.copy()}
    variants['policy_arms_expert_grippers'][:, [7, 15]] = expert[:, [7, 15]]
    variants['expert_arms_policy_grippers'][:, [7, 15]] = predicted[:, [7, 15]]
    env = NewtonTowelEnv()
    results = {}
    try:
        for name, actions in variants.items():
            env.reset(0)
            reproduce_prefix(env, prefix, joints, cloth)
            bodies = [env.scene.model.body_label.index(f'dual_arm/arm_{s}_link_gripper') for s in 'lr']
            held = maximum = 0
            ever = [False, False]
            error = 0.0
            for index, action in enumerate(actions):
                for _ in range(2):
                    state = env.step(internal_action(action))
                    owners = np.asarray(state['grasp_owner'])
                    grasped = [bool(np.any(owners == body)) for body in bodies]
                    ever = [a or b for a, b in zip(ever, grasped)]
                    held = held + 1 if all(grasped) else 0
                    maximum = max(maximum, held)
                if name == 'policy_replay':
                    error = max(error, float(np.max(np.abs(np.asarray(state['joint_state']) - recorded_states[index + 1]))))
            results[name] = dict(success=grasp_verdict(held)[0], ever_grasped=ever,
                                 final_both_held_s=held / 60, max_both_held_s=maximum / 60)
            if name == 'policy_replay':
                results[name]['replayed_joint_max_error'] = error
                if error > 1e-4:
                    raise ValueError('Replay does not reproduce recorded control')
        if not results['expert_replay']['success']:
            raise ValueError('Reference plan failed its grasp goal')
    finally:
        env.close()
    args.output.write_text(json.dumps(dict(
        rollout=str(args.rollout.resolve()), plan=str(args.plan.resolve()), results=results,
        limitation='Recorded action replay and label substitution; no new closed-loop policy success'), indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
