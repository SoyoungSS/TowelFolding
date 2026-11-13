"""Serve ACT inference on live RGB and state only; no dataset or expert calls."""
import argparse,base64,json,socket,sys,time
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import Resize
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition,transition_to_policy_action
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
from bridge_transport import send_message,receive_message
from act_io import CAMERAS,CANONICAL_NAMES,canonical_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--action-steps', type=int, default=1)
    p.add_argument('--fold-checkpoint', type=Path)
    p.add_argument('--switch-step', type=int)
    a=p.parse_args();torch.set_num_threads(2)
    if a.action_steps < 1:
        p.error('--action-steps must be positive')
    if (a.fold_checkpoint is None) != (a.switch_step is None):
        p.error('--fold-checkpoint and --switch-step must be used together')
    if a.switch_step is not None and (a.switch_step <= 0 or a.switch_step % a.action_steps):
        p.error('Switch step must be a positive action chunk boundary')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for live ACT inference')
    expected_keys = [f'observation.images.{name}' for name in CAMERAS]

    def load_policy(checkpoint):
        runtime=json.loads((checkpoint/'runtime.json').read_text())
        if runtime['camera_order']!=expected_keys or runtime['state_names']!=CANONICAL_NAMES or runtime['action_names']!=CANONICAL_NAMES:
            raise ValueError('Checkpoint camera/joint order does not match this environment')
        if runtime['fps']!=30 or runtime['resize']!={'height':240,'width':320,'antialias':True,'center_crop':False}:
            raise ValueError('Checkpoint preprocessing incompatible')
        policy=ACTPolicy.from_pretrained(checkpoint,local_files_only=True)
        if not 1 <= a.action_steps <= policy.config.chunk_size:
            raise ValueError('Invalid number of actions to execute per chunk')
        if policy.config.temporal_ensemble_coeff is not None and a.action_steps != 1:
            raise ValueError('Temporal ensembling requires a single action per prediction')
        policy.config.n_action_steps = a.action_steps
        policy.eval()
        policy.reset()
        def forbidden(*unused):raise AssertionError('VAE expert-action encoder invoked during inference')
        policy.model.vae_encoder.register_forward_pre_hook(forbidden)
        pre=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_preprocessor.json',local_files_only=True)
        post=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_postprocessor.json',local_files_only=True,
            to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
        return policy, pre, post

    policy, pre, post = load_policy(a.checkpoint)
    fold_policy = load_policy(a.fold_checkpoint) if a.fold_checkpoint else None
    resize=Resize((240,320),antialias=True)
    last_request=0;last_frame=-2;episode=None;calls=0
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(a.socket));listener.listen(1)
            with listener.accept()[0] as connection:
                send_message(connection,{'type':'ready','checkpoint':str(a.checkpoint),'fps':30,'camera_order':expected_keys,
                    'fold_checkpoint':str(a.fold_checkpoint) if a.fold_checkpoint else None, 'switch_step':a.switch_step,
                    'expert_actions_available':False,'n_action_steps':a.action_steps,'parameters':sum(v.numel() for v in policy.parameters())})
                while True:
                    request=receive_message(connection);rid=request['request_id']
                    if rid!=last_request+1:raise ValueError('Non-sequential policy request')
                    if request['command']=='quit':send_message(connection,{'type':'bye','request_id':rid});return
                    if request['command']!='infer':raise ValueError('Only live inference supported')
                    if episode is None:episode=request['episode_id']
                    if request['episode_id']!=episode or request['frame_id']!=last_frame+2:raise ValueError('Stale or out-of-order observation')
                    start=time.monotonic()
                    batch={'observation.state':torch.from_numpy(canonical_state(request['joint_state']))}
                    for name in CAMERAS:
                        raw=base64.b64decode(request['images'][name],validate=True)
                        if len(raw)!=480*640*3:raise ValueError('RGB byte count mismatch')
                        rgb=np.frombuffer(raw,dtype=np.uint8).reshape(480,640,3).copy()
                        batch[f'observation.images.{name}']=resize(torch.from_numpy(rgb).permute(2,0,1).float()/255.)
                    if fold_policy is not None and calls == a.switch_step:
                        policy, pre, post = fold_policy
                        print(f'Activated fold checkpoint at control step {calls}', flush=True)
                    normalized=pre(batch)
                    with torch.no_grad():action=post(policy.select_action(normalized)).detach().cpu().numpy()[0]
                    if action.shape!=(16,) or not np.isfinite(action).all():raise ValueError('Invalid policy target')
                    calls+=1;last_request=rid;last_frame=request['frame_id']
                    send_message(connection,{'type':'action','request_id':rid,'episode_id':episode,'frame_id':last_frame,
                        'policy_action':action,'inference_ms':(time.monotonic()-start)*1000,'policy_calls':calls,
                        'expert_encoder_used':False})
    finally:a.socket.unlink(missing_ok=True)

if __name__=='__main__':main()
