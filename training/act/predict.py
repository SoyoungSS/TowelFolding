"""Load a saved official ACT checkpoint and predict one observed frame offline."""
import argparse,json,contextlib,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition,transition_to_policy_action
from common import load_dataset,observations_only

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--frame',type=int,default=100)
args=parser.parse_args()
checkpoint=args.checkpoint.resolve()
runtime=json.loads((checkpoint/'runtime.json').read_text())
dataset=load_dataset(runtime['training_copy'])
if not 0<=args.frame<len(dataset):parser.error('帧号超出范围')
with contextlib.redirect_stdout(sys.stderr):
    policy=ACTPolicy.from_pretrained(checkpoint,local_files_only=True)
policy.eval()
pre=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_preprocessor.json',local_files_only=True)
post=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_postprocessor.json',local_files_only=True,
    to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
raw=next(iter(DataLoader(Subset(dataset,[args.frame]),batch_size=1)))
batch=pre(observations_only(raw))
with torch.no_grad():
    policy.reset();normalized=policy.predict_action_chunk(batch);unnormalized=post(normalized)
assert unnormalized.shape==(1,30,16) and torch.isfinite(unnormalized).all()
result={'frame':args.frame,'normalized_shape':list(normalized.shape),'unnormalized_shape':list(unnormalized.shape),
        'first_target':unnormalized[0,0].cpu().tolist(),'expert_action_not_passed_to_model':True,
        'note':'offline prediction only; not sent to any robot or simulator'}
print(json.dumps(result,indent=2))
