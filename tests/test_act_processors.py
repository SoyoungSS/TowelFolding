"""Fine-tuning must not reinterpret weights in a new normalization system."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/act'))
from common import training_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig


class TrainingProcessorTests(unittest.TestCase):
    def test_finetuning_preserves_input_and_output_coordinates(self):
        config = ACTConfig(device='cpu',
            input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(16,)),
                            'observation.environment_state': PolicyFeature(type=FeatureType.ENV, shape=(2,))},
            output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(16,))})
        stats = {key: {'mean': torch.ones(size), 'std': torch.full((size,), 2.)}
                 for key, size in [('observation.state', 16), ('action', 16),
                                   ('observation.environment_state', 2)]}
        raw = {'observation.state': torch.arange(16).float(),
               'observation.environment_state': torch.zeros(2)}
        pre, post = training_processors(config, stats)
        expected = pre(copy.deepcopy(raw))['observation.state']
        action = torch.ones(1, 16)
        expected_action = post(action.clone())
        with tempfile.TemporaryDirectory() as directory:
            pre.save_pretrained(directory, config_filename='policy_preprocessor.json')
            post.save_pretrained(directory, config_filename='policy_postprocessor.json')
            changed = {key: {name: value * 100 for name, value in values.items()}
                       for key, values in stats.items()}
            restored_pre, restored_post = training_processors(config, changed, Path(directory))
            torch.testing.assert_close(restored_pre(copy.deepcopy(raw))['observation.state'], expected)
            torch.testing.assert_close(restored_post(action.clone()), expected_action)
            fresh_pre, _ = training_processors(config, changed)
            self.assertFalse(torch.allclose(fresh_pre(copy.deepcopy(raw))['observation.state'], expected))


if __name__ == '__main__':
    unittest.main()
