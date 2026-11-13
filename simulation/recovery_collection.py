"""Replay a policy prefix and collect only the recorded recovery continuation."""
import numpy as np

from act_io import fold_verdict, grasp_verdict, internal_action, measure_fold
from bridge_transport import receive_message, send_message


def reproduce_prefix(env, prefix, expected_joint_state, expected_cloth, *, expected_owners=None):
    for action in prefix:
        for _ in range(2):
            state = env.step(internal_action(action))
            if expected_owners is None and np.any(np.asarray(state['grasp_owner']) >= 0):
                raise ValueError('Initialization prefix unexpectedly grasped cloth')
    if expected_owners is not None and not np.array_equal(state['grasp_owner'], expected_owners):
        raise ValueError('Initialization prefix did not reproduce the grasped particles')
    joint_error = float(np.max(np.abs(np.asarray(state['joint_state']) - expected_joint_state)))
    cloth_error = float(np.max(np.abs(np.asarray(state['cloth_points']) - expected_cloth)))
    if joint_error > 1e-4 or cloth_error > 1e-4:
        raise ValueError(f'Prefix did not reproduce the near-miss: joint={joint_error}, cloth={cloth_error}')
    return state, joint_error, cloth_error


def serve_recovery(connection, env, seed, plan_path):
    with np.load(plan_path, allow_pickle=False) as plan:
        prefix = plan['prefix']
        recovery = plan['recovery']
        phases = plan['phases']
        expected_joint_state = plan['joint_state']
        expected_cloth = plan['cloth_points']
        expected_owners = plan['grasp_owner'] if 'grasp_owner' in plan else None
        goal = str(plan['goal']) if 'goal' in plan else 'fold'
    if goal not in ('grasp', 'fold'):
        raise ValueError('Unknown recovery goal')
    for actions in (prefix, recovery):
        if actions.ndim != 2 or actions.shape[1] != 16 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError('Recovery plan must contain finite, nonempty 16-dimensional action rows')
    if phases.shape != (len(recovery),):
        raise ValueError('Recovery phases must align with actions')

    env.reset(seed)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Expert calls are disabled during recorded recovery')
    env.start_expert = env.expert_action = forbidden
    for method in ('update', 'attach', 'move', 'start_fold', 'enter'):
        setattr(env.scene.fold, method, forbidden)

    def advance(action):
        nonlocal held_frames, max_held_frames
        target = internal_action(action)
        for _ in range(2):
            snapshot = env.step(target)
            owners = np.asarray(snapshot['grasp_owner'])
            both_holding = all(np.any(owners == body_id) for body_id in body_ids)
            held_frames = held_frames + 1 if both_holding else 0
            max_held_frames = max(max_held_frames, held_frames)
            for side, body_id in enumerate(body_ids):
                if np.any(owners == body_id):
                    ever_grasped[side] = True
                    if first_grasp[side] is None:
                        first_grasp[side] = snapshot['sim_time'] - start_time
        return snapshot, target

    body_ids = [env.scene.model.body_label.index(f'dual_arm/arm_{side}_link_gripper') for side in ('l', 'r')]
    ever_grasped = [False, False]
    first_grasp = [None, None]
    start_time = len(prefix) / 30
    state, joint_error, cloth_error = reproduce_prefix(
        env, prefix, expected_joint_state, expected_cloth, expected_owners=expected_owners)
    held_frames = max_held_frames = 0

    index = 0
    diagnostics = {
        'prefix_controls': len(prefix), 'prefix_joint_max_error': joint_error,
        'prefix_cloth_max_error_m': cloth_error, 'expert_calls': 0,
        'ever_grasped': ever_grasped, 'first_grasp_s': first_grasp,
    }
    def reply(request_id, action=None):
        snapshot = dict(state, frame_id=2 * index, sim_time=index / 30)
        metrics = measure_fold(state['cloth_points'], env.scene.config)
        success, checks = fold_verdict(metrics, env.scene.config, ever_grasped, np.all(np.asarray(state['grasp_owner']) < 0))
        if goal == 'grasp':
            success, checks = grasp_verdict(held_frames)
        terminal = index == len(recovery)
        snapshot['phase'] = ('done' if success else 'failed') if terminal else str(phases[index])
        return {'type': 'state', 'request_id': request_id, 'state': snapshot, 'action': action,
                'terminal': terminal, 'metrics': metrics, 'recovery': dict(
                    diagnostics, checks=checks, goal=goal, final_both_held_s=held_frames / 60,
                    max_both_held_s=max_held_frames / 60)}

    metadata = env.metadata()
    metadata['recovery_plan'] = str(plan_path.resolve())
    metadata['recovery_goal'] = goal
    send_message(connection, dict(reply(0), metadata=metadata))
    last_request_id = 0
    while True:
        request = receive_message(connection)
        request_id = request['request_id']
        if request_id != last_request_id + 1:
            raise ValueError('Non-sequential recovery request')
        if request['command'] == 'quit':
            send_message(connection, {'type': 'bye', 'request_id': request_id})
            return
        if (request['command'] != 'advance' or request['steps'] != 2
                or request['frame_id'] != 2 * index or request['episode_id'] != state['episode_id']
                or index >= len(recovery)):
            raise ValueError('Invalid recovery advance request')
        state, action = advance(recovery[index])
        index += 1
        last_request_id = request_id
        send_message(connection, reply(request_id, action))
