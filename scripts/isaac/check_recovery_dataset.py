"""Replay a collected recovery dataset and compare each pre-action observation."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq

from act_live import connect_worker
from act_io import canonical_state
from bridge_transport import receive_message, send_message

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--finish-interrupted', action='store_true',
                        help='Finalize flushed data after replay verifies success')
    parser.add_argument('--newton-python', type=Path,
                        default=Path.home() / 'miniforge3/envs/towel-newton/bin/python')
    args = parser.parse_args()
    collection = args.collection.resolve()
    dataset = collection / 'dataset'
    success_path = dataset / 'SUCCESS.json'
    metadata = json.loads((collection / 'metadata.json').read_text())
    table = pq.read_table(dataset / 'data/chunk-000/file-000.parquet')
    if success_path.exists():
        success = json.loads(success_path.read_text())
        if len(table) != success['frames'] or success['fps'] != 30:
            raise ValueError('Recovery sample count or frame rate mismatch')
    elif not args.finish_interrupted:
        raise ValueError('Missing dataset completion marker')
    else:
        frames = (collection / 'frames.jsonl').read_text().splitlines()
        if len(frames) != len(table) + 1 or json.loads(frames[-1])['phase'] != 'done':
            raise ValueError('Interrupted collection lacks a successful terminal observation')
    if not np.allclose(table['timestamp'].to_numpy(), np.arange(len(table)) / 30, atol=1e-6):
        raise ValueError('Recovery timestamps are not synchronized')
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix='towel-recovery-check-') as temporary:
        tmp = Path(temporary)
        plan = tmp / 'dataset_plan.npz'
        with np.load(metadata['recovery_plan'], allow_pickle=False) as original:
            if not np.array_equal(np.asarray(original['recovery'], dtype=np.float32), actions):
                raise ValueError('Collected labels differ from planned recovery actions')
            content = dict(original)
        content['recovery'] = actions
        np.savez_compressed(plan, **content)
        endpoint = tmp / 'physics.sock'
        with (collection / 'replay_verification.log').open('w') as log:
            process = subprocess.Popen([
                str(args.newton_python), '-I', '-u', str(ROOT / 'scripts/isaac/live_newton_worker.py'),
                '--socket', str(endpoint), '--seed', str(metadata['seed']), '--recovery-plan', str(plan),
            ], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            try:
                connection, reply = connect_worker(endpoint, process, 120)
                max_state_error = 0.0
                with connection:
                    for i, recorded_state in enumerate(states):
                        state = reply['state']
                        if state['frame_id'] != 2 * i or reply['terminal']:
                            raise ValueError('Physics state index does not match dataset row')
                        error = float(np.max(np.abs(canonical_state(state['joint_state']) - recorded_state)))
                        max_state_error = max(error, max_state_error)
                        if error > 1e-4:
                            raise ValueError(f'Row {i} state mismatch: {error}')
                        send_message(connection, {'command': 'advance', 'request_id': i + 1,
                            'episode_id': state['episode_id'], 'frame_id': state['frame_id'], 'steps': 2})
                        reply = receive_message(connection)
                    if not reply['terminal'] or reply['state']['phase'] != 'done':
                        raise ValueError(
                            f"Replayed dataset did not achieve its recorded goal: "
                            f"metrics={reply['metrics']}, checks={reply['recovery']['checks']}")
                    send_message(connection, {'command': 'quit', 'request_id': len(states) + 1})
                    if receive_message(connection)['type'] != 'bye':
                        raise ValueError('Recovery worker did not acknowledge shutdown')
                if process.wait(timeout=20):
                    raise RuntimeError('Recovery worker exited with an error')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
    result = dict(reply['recovery'], success=True, frames=len(states),
                  max_state_error=max_state_error, metrics=reply['metrics'])
    if not success_path.exists():
        from collect_dataset import Collector
        Collector.from_written_episode(dataset, 30).finish(True)
        result['metadata_finalized_after_interruption'] = True
    (collection / 'replay_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
