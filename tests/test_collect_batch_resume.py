import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('collect_batch',
    Path(__file__).resolve().parents[1] / 'scripts/isaac/collect_batch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_completed_data_and_random_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'batch.json').write_text(json.dumps(dict(requested=3, seed=12, offset_mm=3.,
                episodes=[dict(index=0, passed=False)])))
            interrupted = root / 'episode_001'
            interrupted.mkdir()
            (interrupted / 'partial').write_text('keep me')
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=1)
            argv = ['collect_batch', '--episodes', '3', '--seed', '12', '--offset-mm', '3',
                    '--newton-python', '/unused/python', '--output', directory, '--resume']
            with patch('sys.argv', argv), patch.object(module.subprocess, 'run', side_effect=run):
                self.assertEqual(module.main(), 1)
            self.assertEqual(len(commands), 2)
            rng = random.Random(12)
            for command in commands:
                i = command.index('--towel-offset')
                actual = list(map(float, command[i + 1:i + 3]))
                for value in actual:
                    self.assertAlmostEqual(value, rng.uniform(-3, 3) / 1000, places=11)
            preserved = list(root.glob('interrupted_001_*/partial'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(preserved[0].read_text(), 'keep me')
            self.assertEqual(len(json.loads((root / 'batch.json').read_text())['episodes']), 3)

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'batch.json').write_text(json.dumps(
                dict(requested=3, seed=12, offset_mm=3., episodes=[])))
            argv = ['collect_batch', '--episodes', '3', '--seed', '13',
                    '--newton-python', '/unused/python', '--output', directory, '--resume']
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                module.main()


if __name__ == '__main__':
    unittest.main()
