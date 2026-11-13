import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/act'))
from evaluate_closed_loop import accepted


class AcceptanceTests(unittest.TestCase):
    def test_only_completed_autonomous_full_folds_pass(self):
        result = dict(status='completed', all_workers_stopped=True)
        evaluation = dict(status='completed', goal='fold', initialization=None,
                          task_success=True, expert_calls=0, expert_guards_enabled=True,
                          checks={key: True for key in ('both_grippers_captured_cloth',
                              'cloth_released', 'mean_pair_error', 'max_pair_error', 'footprint_ratio')})
        self.assertTrue(accepted(evaluation, result))
        for key, value in [('goal', 'grasp'), ('initialization', 'expert-plan'),
                           ('expert_calls', 1), ('task_success', False), ('checks', {}),
                           ('checks', {'mean_pair_error': False}), ('status', 'interrupted')]:
            changed = copy.deepcopy(evaluation)
            changed[key] = value
            self.assertFalse(accepted(changed, result), key)
        self.assertFalse(accepted(evaluation, dict(status='completed', all_workers_stopped=False)))


if __name__ == '__main__':
    unittest.main()
