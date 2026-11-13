"""Build and physically validate short grasp-correction demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))
from act_io import canonical_state, grasp_verdict, internal_action
from newton_env import NewtonTowelEnv


def step_control(env, target):
    for _ in range(2):
        state = env.step(internal_action(target))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.reference) as data:
        recorded = data['expert_target'].copy()
    variants = [
        ('left_yaw', {0: .02}, 'train'),
        ('left_pitch', {1: -.02}, 'train'),
        ('bilateral', {1: .02, 8: .02}, 'train'),
        ('heldout', {0: -.015, 8: .015}, 'test'),
    ]
    target = recorded[275].copy()
    target[[7, 15]] = 0
    results = []
    env = NewtonTowelEnv()
    try:
        for name, offsets, split in variants:
            state = env.reset(0)
            env.expert_action = lambda: (_ for _ in ()).throw(RuntimeError('Expert disabled'))
            prefix = list(recorded[:275])
            for action in prefix:
                state = step_control(env, action)
            offset_target = target.copy()
            for index, offset in offsets.items():
                offset_target[index] += offset
            for fraction in np.linspace(1 / 12, 1, 12):
                action = target + fraction * (offset_target - target)
                prefix.append(action)
                state = step_control(env, action)
            for _ in range(6):
                prefix.append(offset_target.copy())
                state = step_control(env, offset_target)
            if np.any(np.asarray(state['grasp_owner']) >= 0):
                raise ValueError('Prefix unexpectedly captured cloth')
            initial_joint = np.asarray(state['joint_state']).copy()
            initial_cloth = np.asarray(state['cloth_points']).copy()
            initial = canonical_state(initial_joint)
            initial[[7, 15]] = 0
            recovery = [initial + f * (target - initial) for f in np.linspace(1 / 12, 1, 12)]
            closed = target.copy()
            closed[[7, 15]] = 1
            recovery.extend([closed.copy() for _ in range(60)])
            bodies = [env.scene.model.body_label.index(f'dual_arm/arm_{s}_link_gripper') for s in 'lr']
            held = 0
            for action in recovery:
                for _ in range(2):
                    state = env.step(internal_action(action))
                    owners = np.asarray(state['grasp_owner'])
                    held = held + 1 if all(np.any(owners == b) for b in bodies) else 0
            success, checks = grasp_verdict(held)
            result = dict(name=name, split=split, success=success, checks=checks,
                          final_both_held_s=held / 60, offsets=offsets,
                          prefix_controls=len(prefix), recovery_controls=len(recovery))
            results.append(result)
            if not success:
                raise RuntimeError(f'Grasp demonstration failed: {result}')
            np.savez_compressed(args.output / f'{name}.npz', prefix=np.asarray(prefix),
                recovery=np.asarray(recovery), phases=np.array(['correct'] * 12 + ['close_hold'] * 60),
                joint_state=initial_joint, cloth_points=initial_cloth, goal=np.array('grasp'))
            print(json.dumps(result), flush=True)
    finally:
        env.close()
        (args.output / 'validation.json').write_text(json.dumps(dict(
            reference=str(args.reference.resolve()),
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            plans=results), indent=2))


if __name__ == '__main__':
    main()
