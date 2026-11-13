"""Pure ACT/robot conversion and evaluation; no planner or privileged policy input."""
import numpy as np

CAMERAS=('head','left_wrist','right_wrist')
GRASP_HOLD_FRAMES = 30  # 0.5 s at the 60 Hz physics rate
CANONICAL_NAMES=([f'left_joint_{i}' for i in range(1,8)]+['left_gripper_closed']+
                 [f'right_joint_{i}' for i in range(1,8)]+['right_gripper_closed'])


def finite_vector(value):
    array=np.asarray(value,dtype=np.float32)
    if array.shape!=(16,) or not np.isfinite(array).all():raise ValueError('Expected finite 16D vector')
    return array


def canonical_state(internal):
    x=finite_vector(internal)
    result=x[[0,1,2,3,4,5,6,14,7,8,9,10,11,12,13,15]].copy()
    result[[7,15]]=(result[[7,15]]<.1).astype(np.float32)
    return result


def internal_action(canonical):
    from robot_model import OPEN_FRACTION,CLOSED_FRACTION
    x=finite_vector(canonical)
    result=x[[0,1,2,3,4,5,6,8,9,10,11,12,13,14,7,15]].copy()
    result[14:]=np.where(x[[7,15]]>=.5,CLOSED_FRACTION,OPEN_FRACTION)
    return result


def measure_fold(points,config):
    nx,ny=config['towel']['cells']
    grid=np.asarray(points).reshape(ny+1,nx+1,3)
    if not np.isfinite(grid).all():raise ValueError('Non-finite cloth geometry')
    columns=np.arange(nx//2)
    distances=np.linalg.norm(grid[:,columns,:2]-grid[:,nx-columns,:2],axis=2)
    return {'mean_pair_error_m':float(distances.mean()),'max_pair_error_m':float(distances.max()),
            'footprint_ratio':float(np.ptp(grid[:,:,0])/config['towel']['size'][0])}


def fold_verdict(metrics,config,ever_grasped,released):
    tolerance=config['fold']['pair_tolerance']
    checks={'both_grippers_captured_cloth':all(ever_grasped),'cloth_released':bool(released),
            'mean_pair_error':metrics['mean_pair_error_m']<tolerance,
            'max_pair_error':metrics['max_pair_error_m']<2*tolerance,
            'footprint_ratio':metrics['footprint_ratio']<.68}
    return all(checks.values()),checks


def grasp_verdict(held_frames):
    checks = {'both_grippers_holding_for_half_second': held_frames >= GRASP_HOLD_FRAMES}
    return all(checks.values()), checks
