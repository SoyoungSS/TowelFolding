"""Evaluate z=0 first-action precision by demonstration phase, without control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ARM_INDICES=[*range(7),*range(8,15)]
GRIPPER_INDICES=[7,15]


def summarize(predicted,targets,states,phases):
    if predicted.shape!=targets.shape or states.shape!=targets.shape or targets.shape[1:]!=(16,):
        raise ValueError('Expected aligned N x 16 prediction, target and state arrays')
    if len(phases)!=len(targets) or not all(np.isfinite(x).all() for x in (predicted,targets,states)):
        raise ValueError('Invalid phases or nonfinite data')
    def subset(indices):
        pred=predicted[indices];target=targets[indices];state=states[indices]
        error=np.abs(pred[:,ARM_INDICES]-target[:,ARM_INDICES])
        grips={}
        for side,j in zip(('left','right'),GRIPPER_INDICES):
            positive=target[:,j]>=.5;closed=pred[:,j]>=.5
            grips[side]={'mae':float(np.abs(pred[:,j]-target[:,j]).mean()),
                'expert_closed_frames':int(positive.sum()),'predicted_closed_frames':int(closed.sum()),
                'true_closed':int((positive&closed).sum()),'false_closed':int((~positive&closed).sum()),
                'missed_closed':int((positive&~closed).sum()),
                'closed_recall':float(closed[positive].mean()) if positive.any() else None}
        return {'frames':len(indices),'arm_mae_rad':float(error.mean()),'arm_mae_deg':float(np.rad2deg(error.mean())),
                'arm_error_p95_rad':float(np.percentile(error,95)),
                'expert_target_state_delta_mae_rad':float(np.abs(target[:,ARM_INDICES]-state[:,ARM_INDICES]).mean()),
                'grippers':grips}
    result={'all':subset(np.arange(len(targets))),
            'by_phase':{phase:subset(np.flatnonzero(np.asarray(phases)==phase)) for phase in dict.fromkeys(phases)}}
    result['first_frame']={'arm_mae_rad':float(np.abs(predicted[0,ARM_INDICES]-targets[0,ARM_INDICES]).mean()),
        'arm_max_error_rad':float(np.abs(predicted[0,ARM_INDICES]-targets[0,ARM_INDICES]).max()),
        'prediction':predicted[0].tolist(),'expert_target':targets[0].tolist()}
    result['gripper_transition_frames']={}
    for name,j in zip(('left','right'),GRIPPER_INDICES):
        result['gripper_transition_frames'][name]={}
        for key,values in [('expert',targets[:,j]),('predicted',predicted[:,j])]:
            closed=values>=.5
            result['gripper_transition_frames'][name][key]={
                'close':np.flatnonzero(closed&~np.r_[False,closed[:-1]]).tolist(),
                'open':np.flatnonzero(~closed&np.r_[False,closed[:-1]]).tolist()}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recovery', action='store_true', help='Evaluate the recorded recovery episode')
    parser.add_argument('--cached-predictions',type=Path,help='Reuse prior aligned first-action predictions; no inference')
    args=parser.parse_args();checkpoint=args.checkpoint.resolve()
    runtime=json.loads((checkpoint/'runtime.json').read_text())
    source_key, copy_key = ('recovery_source_dataset', 'recovery_training_copy') if args.recovery else ('source_dataset', 'training_copy')
    if source_key not in runtime:
        parser.error('Checkpoint has no recorded recovery dataset')
    source=Path(runtime[source_key]);phase_path=source.parent/'frames.jsonl'
    rows=[json.loads(s) for s in phase_path.read_text().splitlines()]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();chunk_l1=None
    if args.cached_predictions:
        z=np.load(args.cached_predictions)
        predicted=z['predicted'];targets=z['expert_target'];states=z['expert_state']
        mode='cached first-action predictions; no new inference'
    else:
        import torch
        from torch.utils.data import DataLoader
        from common import load_dataset,observations_only
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition,transition_to_policy_action
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for saved CUDA checkpoint')
        torch.set_num_threads(2)
        dataset=load_dataset(runtime[copy_key])
        model=ACTPolicy.from_pretrained(checkpoint,local_files_only=True).eval()
        def forbidden(*unused):raise AssertionError('Expert-action encoder used in evaluation')
        model.model.vae_encoder.register_forward_pre_hook(forbidden)
        pre=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_preprocessor.json',local_files_only=True)
        post=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_postprocessor.json',local_files_only=True,
            to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
        predictions=[];target_rows=[];state_rows=[];error_sum=0.;valid_count=0;offset=0
        with torch.inference_mode():
            for raw in DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0):
                n=len(raw['observation.state'])
                if not np.allclose(raw['timestamp'].numpy(),[r['sim_time'] for r in rows[offset:offset+n]],atol=1e-5):
                    raise ValueError('Phase log and dataset timestamps do not align')
                target_rows.append(raw['action'][:,0].numpy().copy());state_rows.append(raw['observation.state'].numpy().copy())
                batch=pre(raw);pred=model.predict_action_chunk(observations_only(batch))
                mask=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
                error_sum+=float((pred-batch['action']).abs()[mask].sum());valid_count+=int(mask.sum())
                predictions.append(post(pred)[:,0].cpu().numpy());offset+=n
        predicted=np.concatenate(predictions);targets=np.concatenate(target_rows);states=np.concatenate(state_rows)
        chunk_l1=error_sum/valid_count;mode='all training frames, z=0, no actions passed to model'
    count=len(targets)
    if len(rows)!=count+1 or [r['frame_id'] for r in rows[:count]]!=list(range(0,2*count,2)):
        raise ValueError('Unexpected phase/trajectory frame layout')
    result=summarize(predicted,targets,states,[r['phase'] for r in rows[:count]])
    np.savez_compressed(args.output/'predictions.npz',predicted=predicted,expert_target=targets,expert_state=states)
    result.update(checkpoint=str(checkpoint),mode=mode,frames=count,
        cached_predictions=str(args.cached_predictions.resolve()) if args.cached_predictions else None,
        checkpoint_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        phase_log=str(phase_path),phase_note='Observation phase from collector log, never supplied to policy',
        normalized_chunk_l1=chunk_l1,wall_time_s=time.monotonic()-started,
        limitation='training episode only; not held-out validation or closed-loop success')
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(args.output),'arm_mae_rad':result['all']['arm_mae_rad'],
        'first_frame_mae_rad':result['first_frame']['arm_mae_rad'],'normalized_chunk_l1':chunk_l1,
        'phases':{k:v['arm_mae_rad'] for k,v in result['by_phase'].items()}},indent=2),flush=True)

if __name__=='__main__':main()
