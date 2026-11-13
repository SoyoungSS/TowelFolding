"""Shared official ACT dataset and preprocessing setup for stage 1."""
from pathlib import Path
import json
import torch
from torchvision.transforms import Resize
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors

CHUNK=30
HEIGHT,WIDTH=240,320
TASK='fold a towel with both arms'

def training_processors(config, dataset_stats, init_checkpoint=None):
    """Keep a pretrained policy's coordinate system when adding demonstrations."""
    if init_checkpoint is None:
        return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    pre = PolicyProcessorPipeline.from_pretrained(init_checkpoint,
        config_filename='policy_preprocessor.json', local_files_only=True)
    post = PolicyProcessorPipeline.from_pretrained(init_checkpoint,
        config_filename='policy_postprocessor.json', local_files_only=True,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return pre, post

def load_dataset(root):
    return LeRobotDataset(repo_id='local/towel_fold_stage1',root=Path(root),
        delta_timestamps={'action':[i/30 for i in range(CHUNK)]},
        video_backend='pyav',image_transforms=Resize((HEIGHT,WIDTH),antialias=True))

def make_config(dataset,device):
    features=dataset_to_policy_features(dataset.features)
    inputs={k:v for k,v in features.items() if v.type!=FeatureType.ACTION}
    for k,v in inputs.items():
        if v.type==FeatureType.VISUAL:v.shape=(3,HEIGHT,WIDTH)
    return ACTConfig(input_features=inputs,output_features={'action':features['action']},
        chunk_size=CHUNK,n_action_steps=1,device=device,push_to_hub=False)

def observations_only(batch):
    return {k:v for k,v in batch.items() if k.startswith('observation.')}

def validate_dataset(dataset):
    checks=[]
    for i in [0,min(100,len(dataset)-1),len(dataset)-2,len(dataset)-1]:
        item=dataset[i]
        assert item['observation.state'].shape==(16,)
        assert item['action'].shape==(CHUNK,16)
        assert item['action_is_pad'].shape==(CHUNK,)
        assert item['task']==TASK,(item['task'],TASK)
        for k in dataset.meta.camera_keys:
            assert item[k].shape==(3,HEIGHT,WIDTH),(k,item[k].shape)
            assert torch.isfinite(item[k]).all() and 0<=item[k].min()<=item[k].max()<=1
        assert torch.isfinite(item['action']).all()
        valid=min(CHUNK,len(dataset)-i)
        assert (~item['action_is_pad']).sum().item()==valid
        assert not item['action_is_pad'][0]
        # Current chunk target corresponds to the exact current scalar row.
        assert torch.allclose(item['action'][0],dataset.hf_dataset[i]['action'])
        checks.append({'index':i,'valid_steps':valid,'task':item['task']})
    return {'frames':len(dataset),'episodes':dataset.num_episodes,'fps':dataset.fps,
            'camera_keys':list(dataset.meta.camera_keys),'resized_image_shape':[3,HEIGHT,WIDTH],
            'chunk_shape':[CHUNK,16],'boundary_checks':checks,'official_class':type(dataset).__name__}
