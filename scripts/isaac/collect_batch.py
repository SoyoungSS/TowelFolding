"""Collect full-fold episodes sequentially and index successful datasets."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--newton-python', type=Path, required=True)
    parser.add_argument('--offset-mm', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted batch without replacing completed episodes')
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.offset_mm <= 20 or args.seed < 0:
        parser.error('Choose at least one episode, an offset range of 0–20 mm and a nonnegative seed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest = output / 'batch.json'
    batch = dict(requested=args.episodes, seed=args.seed, offset_mm=args.offset_mm, episodes=[])
    if args.resume:
        previous = json.loads(manifest.read_text())
        if any(previous[key] != batch[key] for key in ('requested', 'seed', 'offset_mm')):
            parser.error('Resume requires the original episode count, seed and offset range')
        if [row['index'] for row in previous['episodes']] != list(range(len(previous['episodes']))):
            parser.error('Manifest episode indices are not consecutive')
        batch = previous

    def save_manifest():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(batch, indent=2))
        temporary.replace(manifest)

    save_manifest()
    rng = random.Random(args.seed)
    bridge = Path(__file__).with_name('live_bridge.py')
    for index in range(args.episodes):
        folder = output / f'episode_{index:03d}'
        offset = [0., 0.] if index == 0 else [
            rng.uniform(-args.offset_mm, args.offset_mm) / 1000 for _ in range(2)
        ]
        if index < len(batch['episodes']):
            continue
        if folder.exists():
            # Preserve incomplete recordings for diagnostics; never mix their
            # frames with a restarted physical episode.
            folder.rename(output / f'interrupted_{index:03d}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
        folder.mkdir()
        print(f'采集第 {index + 1}/{args.episodes} 轮，位置偏移(m)={offset}', flush=True)
        command = [sys.executable, '-u', str(bridge), '--collect', '--fps', '30',
                   '--headless', '--exit-on-done', '--max-updates', '1800',
                   '--newton-python', str(args.newton_python), '--seed', str(args.seed + index),
                   '--towel-offset', *(f'{value:.12f}' for value in offset), '--output', str(folder)]
        with (folder / 'console.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        completed = folder / 'completed_episode.json'
        result = json.loads(completed.read_text()) if completed.exists() else {}
        dataset = folder / 'dataset'
        marker = dataset / 'SUCCESS.json'
        passed = run.returncode == 0 and result.get('status') == 'passed' and marker.is_file()
        entry = dict(index=index, offset_m=offset, passed=passed, returncode=run.returncode,
                     metrics=result.get('metrics'), log=str((folder / 'console.log').relative_to(output)))
        if passed:
            entry['dataset'] = str(dataset.relative_to(output))
            entry['frames'] = json.loads(marker.read_text())['frames']
        batch['episodes'].append(entry)
        save_manifest()
        print(f'第 {index + 1} 轮：{"通过" if passed else "未通过"}。{folder}', flush=True)
    count = sum(episode['passed'] for episode in batch['episodes'])
    print(f'采集完成：{count}/{args.episodes} 通过。清单：{manifest}', flush=True)
    return 0 if count else 1


if __name__ == '__main__':
    raise SystemExit(main())
