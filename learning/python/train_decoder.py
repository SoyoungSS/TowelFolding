"""Lesson 10: fit eight multimodal examples with a small encoder-decoder."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multimodal_policy import VisualTeachingPolicy

torch.manual_seed(10);torch.set_num_threads(2)
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先运行采集脚本')
p=found[-1].parent
info=json.loads((p/'meta/info.json').read_text());assert info['total_episodes']==1
rows=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
state=torch.tensor(rows['observation.state'],dtype=torch.float32)
action=torch.tensor(rows['action'],dtype=torch.float32)
K=30;B=8
assert len(action)>=K+B
starts=torch.linspace(0,len(action)-K,B).long()
target=torch.stack([action[t:t+K] for t in starts]);state=state[starts]
smean=state.mean(0);sstd=state.std(0,unbiased=False).clamp_min(.05)
amean=target.mean((0,1));astd=target.std((0,1),unbiased=False).clamp_min(.05)
s=(state-smean)/sstd;y=(target-amean)/astd
weight_path=ROOT/'.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth'
if not weight_path.exists():raise SystemExit('请先运行 ./learning/scripts/learn_vision.sh 下载视觉权重')
assert hashlib.sha256(weight_path.read_bytes()).hexdigest().startswith('f37072fd')
backbone=resnet18(weights=None)
backbone.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True))
backbone=nn.Sequential(*list(backbone.children())[:-2]);backbone.eval();backbone.requires_grad_(False)
transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
features=[]
print('提取8个观测的三路视觉特征：ResNet冻结，仅提取一次。',flush=True)
for name in ('head','left_wrist','right_wrist'):
    cap=cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4'))
    frames={}
    try:
        for t in range(int(starts[-1])+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'{name}第{t}帧读取失败')
            if t in starts.tolist():frames[t]=transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()
    with torch.no_grad():f=backbone(torch.stack([frames[int(t)] for t in starts]))
    features.append(f.flatten(2).transpose(1,2))
f=torch.stack(features,dim=1) # [8,3,49,512]
assert f.shape==(B,3,49,512) and not f.requires_grad

model=VisualTeachingPolicy()
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
criterion=nn.MSELoss()
with torch.no_grad():before=model(f,s).clone();initial=criterion(before,y).item()
original={name:v.detach().clone() for name,v in model.named_parameters()}
losses=[initial]
print('视觉特征',tuple(f.shape),'状态',tuple(s.shape),'动作标签',tuple(y.shape),flush=True)
print('本课只有训练样本，没有验证集；教学网络没有CVAE。',flush=True)
for step in range(1,301):
    model.train();prediction=model(f,s);loss=criterion(prediction,y)
    optimizer.zero_grad();loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():current=criterion(model(f,s),y).item()
    assert np.isfinite(current)
    losses.append(current)
    if step%50==0:print(f'更新{step}次，训练归一化MSE={current:.6f}',flush=True)
model.eval()
with torch.no_grad():after=model(f,s)*astd+amean
assert losses[-1]<initial, '本次未降低训练误差，请检查数据和输出'
changed={}
for group in ('visual_projection','state_projection','policy.encoder','policy.action_queries','policy.decoder','policy.action_head'):
    changed[group]=any(not torch.equal(original[n],v) for n,v in model.named_parameters() if n.startswith(group+'.'))
assert all(changed.values())
out=p.parent/'learning_decoder_train';out.mkdir(exist_ok=True)
checkpoint={'model':model.state_dict(),'state_mean':smean,'state_std':sstd,'action_mean':amean,'action_std':astd,
            'config':{'chunk_size':K,'hidden_dim':128,'heads':4,'ffn_dim':256,'visual_weights':'resnet18-f37072fd','preprocessing':'ImageNet1K V1 transforms','visual_backbone_frozen':True}}
torch.save(checkpoint,out/'teaching_policy.pt')
saved=torch.load(out/'teaching_policy.pt',map_location='cpu',weights_only=True)
restored=VisualTeachingPolicy();restored.load_state_dict(saved['model']);restored.eval()
with torch.no_grad():reloaded=restored(f,(state-saved['state_mean'])/saved['state_std'])*saved['action_std']+saved['action_mean']
assert torch.allclose(after,reloaded,atol=1e-6)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
axes[0].plot(losses);axes[0].set(xlabel='Parameter updates',ylabel='Normalized action MSE',yscale='log',title='Same eight training examples')
b=3;j=0;t=np.arange(K)/info['fps']
axes[1].plot(t,target[b,:,j],label='Expert target',linewidth=3)
axes[1].plot(t,(before*astd+amean)[b,:,j],label='Before training',linestyle='--')
axes[1].plot(t,after[b,:,j],label='After training',linestyle=':')
axes[1].set(xlabel='Time from observation (s)',ylabel='Left joint 1 target (rad)',title=f'Training frame {int(starts[b])}')
axes[1].legend();fig.suptitle('Frozen ResNet + trainable projections / encoder / queries / decoder / action head')
fig.savefig(out/'training.png',dpi=150);plt.close(fig)
report={'dataset':str(p),'train_frames':starts.tolist(),'batch_size':B,'updates':300,'initial_loss':initial,'final_loss':losses[-1],
        'trainable_parameters':sum(v.numel() for v in model.parameters()),'updated_groups':changed,'reload_check':'passed',
        'limitations':['eight fixed training examples only; no validation or closed-loop test','teaching encoder-decoder without CVAE; not standard ACT','ImageNet center crop; frozen backbone; no evidence that image input is necessary for this fit','checkpoint excludes backbone weights; reconstruction requires the matching pretrained ResNet file']}
(out/'result.json').write_text(json.dumps(report,indent=2))
print(f'训练误差 {initial:.6f} → {losses[-1]:.6f}；保存重载预测一致。')
print('更新参数的模块：',changed)
print('图表：',out/'training.png')
print('模型：',out/'teaching_policy.pt')
print('没有连接机器人；低训练损失不代表新场景成功，也不证明模型依赖图像。')
