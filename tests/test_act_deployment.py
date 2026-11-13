import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulation'))
from act_deployment import load_deployment


class DeploymentTests(unittest.TestCase):
    def test_absent_registry_leaves_explicit_fallback_available(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(load_deployment(Path(folder)))

    def test_changed_or_missing_weights_do_not_silently_select_another_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'outputs/verified/checkpoint'
            checkpoint.mkdir(parents=True)
            weights = checkpoint / 'model.safetensors'
            weights.write_bytes(b'verified weights')
            registry = root / 'training/act/deployment.json'
            registry.parent.mkdir(parents=True)
            registry.write_text(json.dumps(dict(checkpoint='outputs/verified/checkpoint',
                checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                action_steps=30, duration_s=29)))
            selected = load_deployment(root)
            self.assertEqual(selected['checkpoint'], checkpoint)
            self.assertEqual(selected['action_steps'], 30)
            weights.write_bytes(b'other weights')
            with self.assertRaises(ValueError):
                load_deployment(root)
            weights.unlink()
            with self.assertRaises(FileNotFoundError):
                load_deployment(root)


if __name__ == '__main__':
    unittest.main()
