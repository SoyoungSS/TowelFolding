"""Hand-calculated checks for the diagnostic report's units and event counts."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training/act'))
from evaluate_fit import summarize,ARM_INDICES

class FitMetricsTests(unittest.TestCase):
    def test_stage_errors_and_gripper_events(self):
        target=np.zeros((3,16),np.float32);pred=target.copy()
        target[:,7]=[0,1,1];pred[:,7]=[1,0,1]
        pred[0,ARM_INDICES]=1;pred[1,ARM_INDICES]=3;pred[2,ARM_INDICES]=2
        result=summarize(pred,target,target.copy(),['reach','reach','close'])
        self.assertEqual(result['all']['arm_mae_rad'],2.)
        self.assertEqual(result['by_phase']['reach']['arm_mae_rad'],2.)
        left=result['all']['grippers']['left']
        self.assertEqual((left['false_closed'],left['missed_closed'],left['closed_recall']),(1,1,.5))
        events=result['gripper_transition_frames']['left']
        self.assertEqual(events['expert'],{'close':[1],'open':[]})
        self.assertEqual(events['predicted'],{'close':[0,2],'open':[1]})

    def test_misaligned_phase_log_rejected(self):
        a=np.zeros((3,16))
        with self.assertRaises(ValueError):summarize(a,a,a,['reach'])

if __name__=='__main__':unittest.main()
