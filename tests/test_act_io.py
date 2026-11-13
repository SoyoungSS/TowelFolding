"""Check the control boundary and independent fold verdict without a GPU."""
import base64
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'simulation'))
from act_io import canonical_state,internal_action,finite_vector,measure_fold,fold_verdict,grasp_verdict,CAMERAS
from bridge_transport import MAX_MESSAGE

class ActBoundaryTests(unittest.TestCase):
    def test_grasp_requires_continuous_half_second(self):
        self.assertFalse(grasp_verdict(29)[0])
        self.assertTrue(grasp_verdict(30)[0])
        self.assertFalse(grasp_verdict(0)[0])

    def test_joint_order_and_gripper_encoding(self):
        actual=np.r_[np.arange(14)/10,.5,.04].astype(np.float32)
        canonical=canonical_state(actual)
        np.testing.assert_array_equal(canonical[:7],actual[:7])
        np.testing.assert_array_equal(canonical[8:15],actual[7:14])
        np.testing.assert_array_equal(canonical[[7,15]],[0,1])
        np.testing.assert_allclose(internal_action(canonical),actual)
        canonical[[7,15]]=[.5,.499]
        np.testing.assert_allclose(internal_action(canonical)[14:],[.04,.5])

    def test_invalid_targets_rejected(self):
        for value in (np.zeros(15),np.zeros((1,16)),np.full(16,np.nan),np.full(16,np.inf)):
            with self.assertRaises(ValueError):finite_vector(value)

    def test_fold_verdict_requires_grasp_release_and_geometry(self):
        config={'towel':{'cells':[20,20],'size':[.3,.3]},'fold':{'pair_tolerance':.025}}
        x,y=np.meshgrid(np.linspace(-.15,.15,21),np.linspace(-.15,.15,21))
        points=np.stack([x,y,np.zeros_like(x)],axis=-1)
        flat=measure_fold(points,config)
        self.assertFalse(fold_verdict(flat,config,[True,True],True)[0])
        points[:,:,0]=np.abs(points[:,:,0])
        folded=measure_fold(points,config)
        self.assertTrue(fold_verdict(folded,config,[True,True],True)[0])
        self.assertFalse(fold_verdict(folded,config,[False,True],True)[0])
        self.assertFalse(fold_verdict(folded,config,[True,True],False)[0])

    def test_three_lossless_live_images_fit_transport(self):
        rgb=np.arange(480*640*3,dtype=np.uint8).reshape(480,640,3)
        encoded=base64.b64encode(rgb.tobytes()).decode('ascii')
        payload={'command':'infer','request_id':1,'episode_id':0,'frame_id':0,
                 'joint_state':[0.]*16,'images':{n:encoded for n in CAMERAS}}
        self.assertLess(len(json.dumps(payload).encode()),MAX_MESSAGE)
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(encoded),np.uint8).reshape(rgb.shape),rgb)

if __name__=='__main__':unittest.main()
