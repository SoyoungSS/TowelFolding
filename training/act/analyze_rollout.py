"""Audit a completed live rollout and locate grasp failures using recorded truth."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
from act_io import canonical_state,measure_fold,CAMERAS
from robot_model import GRASP_POINTS,CAPTURE_FRACTION


def rotate_xyzw(quaternion,vectors):
    """Rotate broadcast-compatible vectors by normalized xyzw quaternions."""
    q=np.asarray(quaternion)
    if not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-5):raise ValueError('Non-unit quaternion')
    cross=2*np.cross(q[...,:3],vectors)
    return vectors+q[...,3:]*cross+np.cross(q[...,:3],cross)


def longest_true(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return int(lengths.max()) if len(lengths) else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    args=parser.parse_args();folder=args.rollout.resolve()
    result=json.loads((folder/'evaluation.json').read_text())
    final=json.loads((folder/'result.json').read_text())
    metadata=json.loads((folder/'metadata.json').read_text())
    rows=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    z=np.load(folder/'trajectory.npz');count=len(rows)
    if not(final['status']=='completed' and final['all_workers_stopped'] and final['error'] is None):raise ValueError('Run did not complete cleanly')
    if result['expert_calls']!=0 or not result['expert_guards_enabled']:raise ValueError('Expert isolation failed')
    if len(z['joint_state'])!=count+1 or result['policy_steps']!=count:raise ValueError('State/action count mismatch')
    if z['action'].shape!=(count,16) or z['canonical_policy_action'].shape!=(count,16):raise ValueError('Wrong action array shape')
    if not all(np.isfinite(z[k]).all() for k in z.files):raise ValueError('Nonfinite trajectory')
    if not np.array_equal(z['frame_id'],np.arange(0,2*count+1,2)):raise ValueError('Nonsequential state frames')
    if not np.allclose(np.diff(z['sim_time']),1/30):raise ValueError('Wrong state period')
    for name in CAMERAS:
        if result['video_counts'][name]!=count+1:raise ValueError('Recorded video verification count mismatch')
    previous=Fraction(-1)
    for i,row in enumerate(rows):
        refs=[Fraction(*v) for v in row['camera_reference_times'].values()]
        if len(refs)!=3 or len(set(refs))!=1 or refs[0]<=previous:raise ValueError('Invalid camera reference times')
        previous=refs[0]
        if row['observation_frame_id']!=2*i or row['post_step_frame_id']!=2*i+2 or row['expert_calls']!=0:raise ValueError('Frame/expert invariant failed')
        if not np.allclose(z['action'][i],row['applied_internal_action']):raise ValueError('Action log mismatch')
        if not np.allclose(z['canonical_policy_action'][i],row['policy_action']):raise ValueError('Prediction log mismatch')
    recomputed=measure_fold(z['cloth_points'][-1],metadata['config'])
    if any(not np.isclose(v,result['metrics'][k],atol=1e-7) for k,v in recomputed.items()):raise ValueError('Final geometry mismatch')
    grips={}
    capture_half_lengths = np.array([.022, .007, .008])
    for side,name,j,opening,tcp in zip(('l','r'),('left','right'),(7,15),(14,15),GRASP_POINTS):
        body=metadata['body_labels'].index(f'dual_arm/arm_{side}_link_gripper')
        poses=z['body_pose'][:,body,:]
        inverse=poses[:,None,3:].copy();inverse[...,:3]*=-1
        local=rotate_xyzw(inverse,z['cloth_points']-poses[:,None,:3])
        delta=local-np.asarray(tcp)
        unowned = z['grasp_owner'] < 0
        candidates=np.all(np.abs(delta)<=capture_half_lengths,axis=-1)&unowned
        box_ratios = np.max(np.abs(delta) / capture_half_lengths, axis=-1)
        box_ratios[~unowned] = np.inf
        best_frame, best_particle = np.unravel_index(box_ratios.argmin(), box_ratios.shape)
        best_ratio = box_ratios[best_frame, best_particle]
        closest_box = None
        if np.isfinite(best_ratio):
            closest_box = {
                'required_scale': float(best_ratio),
                'time_s': float(z['sim_time'][best_frame]),
                'local_offset_mm': (delta[best_frame, best_particle] * 1000).tolist(),
            }
        nearest=np.linalg.norm(delta,axis=-1).min(1)
        close=z['canonical_policy_action'][:,j]>=.5
        first=int(np.flatnonzero(close)[0]) if close.any() else None
        grips[name]={'close_command_frames':int(close.sum()),'first_close_sim_time_s':float(z['sim_time'][first]) if first is not None else None,
            'max_continuous_close_command_s':longest_true(close)/30,
            'prediction_min_max':[float(z['canonical_policy_action'][:,j].min()),float(z['canonical_policy_action'][:,j].max())],
            'actual_jaw_capture_ready_frames':int((z['joint_state'][:,opening]<=CAPTURE_FRACTION).sum()),
            'frames_with_unowned_particles_in_capture_box':int(candidates.any(1).sum()),
            'closest_particle_to_capture_box':closest_box,
            'min_tcp_to_cloth_distance_m':float(nearest.min()),
            'time_of_min_tcp_to_cloth_distance_s':float(z['sim_time'][nearest.argmin()]),
            'final_tcp_to_cloth_distance_m':float(nearest[-1]),
            'ever_grasped':result['ever_grasped'][0 if name=='left' else 1]}
    diagnostic={'audit_passed':True,'rollout':str(folder),'checkpoint':result['checkpoint'],
        'task_success':result['task_success'],'metrics':result['metrics'],'grippers':grips,
        'policy_steps':count,'expert_calls':result['expert_calls'],
        'truth_usage':'Recorded cloth truth used only for analysis, never supplied to the actor',
        'sampling_note':'Capture geometry evaluated at recorded 30Hz states; physics runs at 60Hz',
        'capture_box_half_lengths_mm':(capture_half_lengths * 1000).tolist()}
    fit=Path(result['checkpoint']).parent/'fit_evaluation/predictions.npz'
    if fit.exists():
        reference=np.load(fit);expert=reference['expert_state'];ids=[*range(7),*range(8,15)]
        states=np.stack([canonical_state(x) for x in z['joint_state']])
        std=np.maximum(expert[:,ids].std(0),.05)
        distances=np.sqrt(np.mean(((states[:,None,ids]-expert[None,:,ids])/std)**2,axis=-1))
        nearest=distances.argmin(1);diagnostic['nearest_expert_arm_state']={}
        for i in sorted({0,min(150,count),min(300,count),min(600,count),count}):
            k=int(nearest[i]);diagnostic['nearest_expert_arm_state'][str(i)]={'sim_time_s':float(z['sim_time'][i]),
                'expert_frame':k,'expert_time_s':k/30,'normalized_distance':float(distances[i,k])}
        first=z['canonical_policy_action'][0];offline=reference['predicted'][0]
        runtime=json.loads((Path(result['checkpoint'])/'runtime.json').read_text())
        start_frame=runtime.get('training_start_frame',0)
        diagnostic['initial_observation_comparison']={
            'actual_state_max_error':float(np.abs(states[0]-expert[0]).max()),
            'live_arm_prediction_mae_against_expert_rad':float(np.abs(first[ids]-reference['expert_target'][0,ids]).mean()),
            'live_vs_offline_arm_prediction_mae_rad':float(np.abs(first[ids]-offline[ids]).mean()),
            'training_start_frame':start_frame,
            'live_arm_mae_against_training_start_target_rad':float(np.abs(first[ids]-reference['expert_target'][start_frame,ids]).mean())}
    (folder/'rollout_analysis.json').write_text(json.dumps(diagnostic,indent=2))
    print(json.dumps(diagnostic,indent=2))

if __name__=='__main__':main()
