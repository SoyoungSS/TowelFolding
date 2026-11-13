"""Repeat independent ACT rollouts without relaxing the physics acceptance rules."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def accepted(evaluation, result):
    return (result.get('status') == 'completed'
            and result.get('all_workers_stopped') is True
            and evaluation.get('status') == 'completed'
            and evaluation.get('goal') == 'fold'
            and evaluation.get('initialization') is None
            and evaluation.get('task_success') is True
            and evaluation.get('expert_calls') == 0
            and evaluation.get('expert_guards_enabled') is True
            and set(evaluation.get('checks', {})) == {
                'both_grippers_captured_cloth', 'cloth_released', 'mean_pair_error',
                'max_pair_error', 'footprint_ratio'}
            and all(evaluation['checks'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--action-steps', type=int, default=30)
    parser.add_argument('--duration', type=float, default=29.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(args.checkpoint.resolve()), requested=args.repeats,
                  action_steps=args.action_steps, runs=[], passed=False,
                  scope='Independent nominal-scene runs; not unseen-position generalization')
    for index in range(args.repeats):
        folder = output / f'run_{index:03d}'
        command = [str(ROOT / 'run_act_live.sh'), '--checkpoint', report['checkpoint'],
                   '--output', str(folder), '--headless', '--exit-on-done',
                   '--duration', str(args.duration), '--action-steps', str(args.action_steps),
                   '--seed', str(args.seed + index), '--goal', 'fold']
        with (output / f'run_{index:03d}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        def read(name):
            path = folder / name
            return json.loads(path.read_text()) if path.exists() else {}
        evaluation, result = read('evaluation.json'), read('result.json')
        passed = process.returncode == 0 and accepted(evaluation, result)
        report['runs'].append(dict(seed=args.seed + index, passed=passed,
                                   returncode=process.returncode, evaluation=evaluation, result=result))
        report['successes'] = sum(row['passed'] for row in report['runs'])
        report['passed'] = len(report['runs']) == args.repeats and report['successes'] == args.repeats
        temporary = output / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / 'report.json')
        print(f'Rollout {index + 1}/{args.repeats}: passed={passed}, '
              f'metrics={evaluation.get("metrics")}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
