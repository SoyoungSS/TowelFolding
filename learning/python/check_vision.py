"""Lesson 11: feature interventions with fixed states, no retraining."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import cv2
import numpy as np
import torch
from torch import nn
import pyarrow.parquet as pq
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multimodal_policy import VisualTeachingPolicy

torch.manual_seed(11);torch.set_num_threads(2)
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先采集数据')
p=found[-1].parent
train_dir=p.parent/'learning_decoder_train'
if not (train_dir/'teaching_policy.pt').exists():raise SystemExit('请先运行 ./learning/scripts/learn_decoder_train.sh 完成第十课')
report=json.loads((train_dir/'result.json').read_text())
assert Path(report['dataset']).resolve()==p.resolve(),'模型与当前数据不一致'
starts=report['train_frames'];saved=torch.load(train_dir/'teaching_policy.pt',map_location='cpu',weights_only=True)
rows=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
state=torch.tensor(rows['observation.state'],dtype=torch.float32)[starts]
s=(state-saved['state_mean'])/saved['state_std']
K=saved['config']['chunk_size']
a=torch.tensor(rows['action'],dtype=torch.float32)
target=torch.stack([a[t:t+K] for t in starts])
y=(target-saved['action_mean'])/saved['action_std']
weight_path=ROOT/'.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth'
if not weight_path.exists():raise SystemExit('请先运行 ./learning/scripts/learn_vision.sh')
assert hashlib.sha256(weight_path.read_bytes()).hexdigest().startswith('f37072fd')
backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True))
backbone=nn.Sequential(*list(backbone.children())[:-2]);backbone.eval();backbone.requires_grad_(False)
transform=ResNet18_Weights.IMAGENET1K_V1.transforms();features=[]
for name in ('head','left_wrist','right_wrist'):
    cap=cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4'))
    frames={}
    try:
        for t in range(max(starts)+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'{name}第{t}帧解码失败')
            if t in starts:frames[t]=transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()
    with torch.no_grad():features.append(backbone(torch.stack([frames[t] for t in starts])).flatten(2).transpose(1,2))
f=torch.stack(features,dim=1)
model=VisualTeachingPolicy();model.load_state_dict(saved['model']);model.eval()
parameters_before=[v.detach().clone() for v in model.parameters()]
# All cases keep the actual robot-state input unchanged. Features change only.
cases={'baseline':f,'repeat_baseline':f.clone(),'other_time_images':f.roll(1,0),
       'zero_visual_features':torch.zeros_like(f),'shuffle_grid':f[:,:,torch.randperm(49),:]}
for i,name in enumerate(('head','left_wrist','right_wrist')):
    altered=f.clone();altered[:,i]=0
    cases[f'zero_{name}']=altered
results=[];predictions={}
with torch.no_grad():
    base=model(f,s)
    for name,feature in cases.items():
        pred=model(feature,s)
        assert torch.isfinite(pred).all()
        predictions[name]=pred
        results.append({'case':name,'normalized_mse_to_original_expert':float((pred-y).square().mean()),
                        'normalized_mean_abs_prediction_change':float((pred-base).abs().mean()),
                        'normalized_max_abs_prediction_change':float((pred-base).abs().max())})
assert results[1]['normalized_max_abs_prediction_change']<1e-6
assert abs(results[0]['normalized_mse_to_original_expert']-report['final_loss'])<1e-5,'基线与第十课结果不一致，请核对模型/数据'
assert all(torch.equal(old,new) for old,new in zip(parameters_before,model.parameters()))
assert all(v.grad is None for v in model.parameters())
out=p.parent/'learning_vision_check';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
labels=[r['case'] for r in results]
axes[0].barh(labels,[r['normalized_mean_abs_prediction_change'] for r in results]);axes[0].invert_yaxis()
axes[0].set(xlabel='Mean absolute prediction change (normalized units)',title='Sensitivity relative to unchanged input')
axes[1].barh(labels,[r['normalized_mse_to_original_expert'] for r in results]);axes[1].invert_yaxis()
axes[1].set(xlabel='MSE against original expert labels',title='Diagnostic mismatch, not counterfactual ground truth')
fig.suptitle('Robot states fixed in every case; frozen trained model; feature-level interventions')
fig.savefig(out/'vision_check.png',dpi=150);plt.close(fig)
(out/'result.json').write_text(json.dumps({'dataset':str(p),'train_frames':starts,'cases':results,
    'parameters_unchanged':True,'no_parameter_gradients':True,'baseline_reproduced':True,
    'limitations':['same eight training examples; no generalization or closed-loop evaluation',
                   'zero feature vectors are not black images; tokens and their embeddings remain',
                   'other-time image/state combinations may be physically inconsistent',
                   'original expert labels are not valid ground truth for every counterfactual scene',
                   'sensitivity does not establish correct semantic use of vision']},indent=2))
print('所有情况保持实际关节状态不变；模型不更新。')
for r in results:print(f'{r["case"]:24s} 预测变化MAE={r["normalized_mean_abs_prediction_change"]:.6f} 原标签MSE={r["normalized_mse_to_original_expert"]:.6f}')
print('重复基线无变化、复现第十课基线误差、参数未改变：检查通过。')
print('预测变化说明受视觉特征影响；不证明理解毛巾或泛化。')
print('清零的是ResNet特征，不是原始图像；替换图像可能与状态不一致。')
print('图表：',out/'vision_check.png')
