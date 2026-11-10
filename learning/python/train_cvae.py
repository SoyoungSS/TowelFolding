"""Lesson 13: monitor posterior and answer-free paths during CVAE fitting."""
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
from cvae_demo import TeachingCVAE,standard_normal_kl

torch.manual_seed(13);torch.set_num_threads(2)
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先采集示范')
p=found[-1].parent;previous=p.parent/'learning_decoder_train'
if not (previous/'teaching_policy.pt').exists():raise SystemExit('请先运行第十课 ./learning/scripts/learn_decoder_train.sh')
old_report=json.loads((previous/'result.json').read_text())
assert Path(old_report['dataset']).resolve()==p.resolve()
saved=torch.load(previous/'teaching_policy.pt',map_location='cpu',weights_only=True)
starts=old_report['train_frames'];B=len(starts);K=30
rows=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
a=torch.tensor(rows['action'],dtype=torch.float32)
state=torch.tensor(rows['observation.state'],dtype=torch.float32)[starts]
s=(state-saved['state_mean'])/saved['state_std']
target=torch.stack([a[t:t+K] for t in starts]);assert target.shape==(B,K,16)
y=(target-saved['action_mean'])/saved['action_std']
weights=ROOT/'.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth'
if not weights.exists():raise SystemExit('请先完成第五课')
assert hashlib.sha256(weights.read_bytes()).hexdigest().startswith('f37072fd')
backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
backbone=nn.Sequential(*list(backbone.children())[:-2]);backbone.eval();backbone.requires_grad_(False)
transform=ResNet18_Weights.IMAGENET1K_V1.transforms();features=[]
print('准备固定观测：沿用第十课训练过的投影层并冻结，ResNet同样冻结。',flush=True)
for name in ('head','left_wrist','right_wrist'):
    cap=cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4'));frames={}
    try:
        for t in range(max(starts)+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'{name}第{t}帧解码失败')
            if t in starts:frames[t]=transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    finally:cap.release()
    with torch.no_grad():features.append(backbone(torch.stack([frames[t] for t in starts])).flatten(2).transpose(1,2))
f=torch.stack(features,1)
frontend=VisualTeachingPolicy();frontend.load_state_dict(saved['model']);frontend.eval();frontend.requires_grad_(False)
with torch.no_grad():
    baseline=frontend(f,s)
    assert abs(float((baseline-y).square().mean())-old_report['final_loss'])<1e-5
    obs=frontend.encode_observations(f,s).detach()
assert obs.shape==(B,148,128) and not obs.requires_grad
model=TeachingCVAE();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
padding=torch.zeros(B,K,dtype=torch.bool);beta=10.
# Fixed noise makes monitored curves comparable; training uses new noise each update.
monitor_noise=torch.randn(B,32)
history=[]
def measure(step):
    model.eval()
    with torch.no_grad():
        mu,logvar=model.posterior(obs,y,padding)
        posterior_prediction=model.decode(obs,mu+(logvar*.5).exp()*monitor_noise)
        posterior_l1=(posterior_prediction-y).abs().mean()
        kl=standard_normal_kl(mu,logvar)
        calls=model.posterior_calls
        zero_prediction=model.predict(obs)
        assert calls==model.posterior_calls,'推理错误地调用了读取答案的分支'
        zero_l1=(zero_prediction-y).abs().mean()
    record={'step':step,'posterior_sample_l1':float(posterior_l1),'kl':float(kl),'weighted_kl':float(beta*kl),
            'total_with_fixed_noise':float(posterior_l1+beta*kl),'zero_latent_l1':float(zero_l1)}
    assert all(np.isfinite(v) for v in record.values())
    history.append(record)
    return zero_prediction
initial_prediction=measure(0)
original={n:p.detach().clone() for n,p in model.named_parameters()}
for step in range(1,301):
    model.train()
    mu,logvar=model.posterior(obs,y,padding)
    z=mu+(logvar*.5).exp()*torch.randn_like(mu)
    prediction=model.decode(obs,z)
    loss=(prediction-y).abs().mean()+beta*standard_normal_kl(mu,logvar)
    optimizer.zero_grad();loss.backward();optimizer.step()
    if step%10==0:
        final_prediction=measure(step)
        if step%50==0:
            r=history[-1]
            print(f'更新{step:3d}：有答案分支L1={r["posterior_sample_l1"]:.5f}，KL={r["kl"]:.5f}，z=0无答案L1={r["zero_latent_l1"]:.5f}',flush=True)
assert history[-1]['total_with_fixed_noise']<history[0]['total_with_fixed_noise']
changed={group:any(not torch.equal(original[n],p) for n,p in model.named_parameters() if n.startswith(group+'.')) for group in ('posterior_encoder','distribution','latent_projection','policy')}
assert all(changed.values())
out=p.parent/'learning_cvae_train';out.mkdir(exist_ok=True)
checkpoint={'model':model.state_dict(),'frontend_model':saved['model'],'state_mean':saved['state_mean'],'state_std':saved['state_std'],
            'action_mean':saved['action_mean'],'action_std':saved['action_std'],'config':{'beta':beta,'chunk_size':30,'latent_dim':32,'visual_weights':'resnet18-f37072fd','frontend_source':'lesson10 trained projections, now frozen'}}
torch.save(checkpoint,out/'teaching_cvae.pt')
restored=TeachingCVAE();restored.load_state_dict(torch.load(out/'teaching_cvae.pt',map_location='cpu',weights_only=True)['model']);restored.eval()
with torch.no_grad():assert torch.allclose(restored.predict(obs),final_prediction,atol=1e-6)
fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
steps=[r['step'] for r in history]
for key,label in [('posterior_sample_l1','Posterior sample (answers used)'),('zero_latent_l1','z=0 inference (no answers)')]:axes[0].plot(steps,[r[key] for r in history],label=label)
axes[0].set(xlabel='Updates',ylabel='Normalized action L1',yscale='log',title='Same eight training examples');axes[0].legend(fontsize=8)
axes[1].plot(steps,[r['kl'] for r in history],label='KL')
axes[1].plot(steps,[r['weighted_kl'] for r in history],label='10 x KL')
axes[1].set(xlabel='Updates',ylabel='KL contribution',yscale='log',title='Regularization, not action accuracy');axes[1].legend()
b=3;j=0;t=np.arange(K)/json.loads((p/'meta/info.json').read_text())['fps']
axes[2].plot(t,target[b,:,j],label='Expert',linewidth=2)
for name,pred in [('Before, z=0',initial_prediction),('After, z=0',final_prediction)]:
    raw=pred*saved['action_std']+saved['action_mean'];axes[2].plot(t,raw[b,:,j],label=name,linestyle='--')
axes[2].set(xlabel='Time from observation (s)',ylabel='Left joint 1 target (rad)',title='No-answer prediction on a training sample');axes[2].legend(fontsize=8)
fig.suptitle('Teaching CVAE fitting; frozen observation frontend; no validation or robot execution')
fig.savefig(out/'training.png',dpi=150);plt.close(fig)
(out/'result.json').write_text(json.dumps({'dataset':str(p),'train_frames':starts,'beta':beta,'updates':300,'initial':history[0],'final':history[-1],
    'updated_groups':changed,'reload_zero_latent_check':'passed','history':history,
    'limitations':['same eight training examples, not a validation/test set','frozen frontend previously learned from the same examples','simplified teaching CVAE, not standard ACT','fixed-noise posterior evaluation is one sample, not an expectation','low KL alone does not establish posterior collapse','no independent-scene or closed-loop success test']},indent=2))
print('z=0推理不调用有答案分支；保存重载一致：检查通过。')
print('图表：',out/'training.png')
print('全部曲线都是训练样本上的指标，不是验证集成绩。')
