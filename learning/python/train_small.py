"""Lesson 3: fit eight state-to-action-chunk examples on CPU."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.set_num_threads(2)
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先运行 ./run_collect.sh 采集一条示范')
p=found[-1].parent
info=json.loads((p/'meta/info.json').read_text())
assert info['total_episodes']==1, '本课只支持单条示范'
d=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
s=torch.tensor(d['observation.state'],dtype=torch.float32)
a=torch.tensor(d['action'],dtype=torch.float32)
K=30;B=8
if len(a)<K+B:raise SystemExit('示范长度不足')
starts=torch.linspace(0,len(a)-K,B).long()
x=s[starts]
y=torch.stack([a[t:t+K] for t in starts])
# Statistics belong to these training examples only. Clamp near-constant dimensions.
xmean=x.mean(0);xstd=x.std(0,unbiased=False).clamp_min(.05)
ymean=y.mean((0,1));ystd=y.std((0,1),unbiased=False).clamp_min(.05)
xn=(x-xmean)/xstd;yn=(y-ymean)/ystd

def create_model():
    return nn.Sequential(nn.Linear(16,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,K*16), nn.Unflatten(1,(K,16)))
model=create_model()
optimizer=torch.optim.Adam(model.parameters(),lr=.001)
loss_fn=nn.MSELoss()
print('数据：',p)
print('仅在8个固定训练样本上拟合；没有验证集，不证明泛化，不连接机器人。')
print('输入',tuple(xn.shape),'标签',tuple(yn.shape))
print(model)
print('参数量：',sum(v.numel() for v in model.parameters()))
with torch.no_grad():
    before_normalized=model(xn).clone()
    initial=loss_fn(before_normalized,yn).item()
losses=[initial]
for step in range(1,301):
    model.train()
    prediction=model(xn)               # Forward: predict future actions.
    loss=loss_fn(prediction,yn)         # Compare with expert answers.
    optimizer.zero_grad()              # Clear gradients from the previous update.
    loss.backward()                    # Compute how each parameter affects loss.
    optimizer.step()                   # Adjust parameters.
    with torch.no_grad():current=loss_fn(model(xn),yn).item()
    losses.append(current)
    if step==1 or step%50==0:print(f'更新{step:3d}次后，归一化动作MSE = {current:.6f}')
model.eval()
with torch.no_grad():after=model(xn)*ystd+ymean
before=before_normalized*ystd+ymean
out=p.parent/'learning_train'
out.mkdir(exist_ok=True)
weights=out/'small_network.pt'
torch.save({'model':model.state_dict(),'xmean':xmean,'xstd':xstd,'ymean':ymean,'ystd':ystd,'chunk_size':K},weights)
# Verify that saved weights and normalization reproduce the same prediction.
saved=torch.load(weights,map_location='cpu',weights_only=True)
restored=create_model();restored.load_state_dict(saved['model']);restored.eval()
with torch.no_grad():reloaded=restored((x-saved['xmean'])/saved['xstd'])*saved['ystd']+saved['ymean']
assert torch.allclose(reloaded,after,atol=1e-6)
assert losses[-1]<initial*.2, '未达到本课的小样本拟合检查，请查看输出'
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].plot(losses);axes[0].set(xlabel='Parameter updates',ylabel='Normalized action MSE',title='Training error on the same 8 examples',yscale='log')
b=3;j=0;time=np.arange(K)/info['fps']
axes[1].plot(time,y[b,:,j],label='Expert target',linewidth=3)
axes[1].plot(time,before[b,:,j],label='Before training',linestyle='--')
axes[1].plot(time,after[b,:,j],label='After training',linestyle=':')
axes[1].set(xlabel='Time from observation (s)',ylabel='Left joint 1 target (rad)',title=f'Training example at frame {int(starts[b])}')
axes[1].legend()
fig.savefig(out/'training.png',dpi=150);plt.close(fig)
report={'dataset':str(p),'batch_size':B,'chunk_size':K,'training_frames':starts.tolist(),'updates':300,'parameters':sum(v.numel() for v in model.parameters()),'initial_loss':initial,'final_loss':losses[-1],'reload_check':'passed','evaluation':'training samples only; no generalization test'}
(out/'result.json').write_text(json.dumps(report,indent=2))
print(f'\n初始误差：{initial:.6f} → 最终误差：{losses[-1]:.6f}')
print('保存并重新加载模型：预测一致，检查通过。')
print('图表：',out/'training.png')
print('模型：',weights)
print('注意：此网络没有图像输入，不是ACT，也没有在仿真中闭环验证。')
