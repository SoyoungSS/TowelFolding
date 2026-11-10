"""Lesson 4: held-out temporal segment, not an independent-episode benchmark."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0);torch.set_num_threads(2)
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先采集一条成功示范')
p=found[-1].parent
info=json.loads((p/'meta/info.json').read_text())
assert info['total_episodes']==1
rows=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
s=torch.tensor(rows['observation.state'],dtype=torch.float32)
a=torch.tensor(rows['action'],dtype=torch.float32)
K=30;n=len(a);split=int(n*.7)
assert split>=K+8 and n-split>=K+8,'数据太短'
train_ids=torch.linspace(0,split-K,8).long()
val_ids=torch.linspace(split,n-K,8).long()
assert int(train_ids.max())+K<=int(val_ids.min()),'动作块跨越划分边界'
def samples(ids):return s[ids],torch.stack([a[t:t+K] for t in ids])
x,y=samples(train_ids);vx,vy=samples(val_ids)
# Fit every preprocessing statistic on the training set only.
xmean=x.mean(0);xstd=x.std(0,unbiased=False).clamp_min(.05)
ymean=y.mean((0,1));ystd=y.std((0,1),unbiased=False).clamp_min(.05)
x=(x-xmean)/xstd;vx=(vx-xmean)/xstd
y=(y-ymean)/ystd;vy=(vy-ymean)/ystd
model=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,K*16),nn.Unflatten(1,(K,16)))
optimizer=torch.optim.Adam(model.parameters(),lr=.001)
criterion=nn.MSELoss();history=[];best=float('inf');best_step=0
out=p.parent/'learning_validation';out.mkdir(exist_ok=True)
def evaluate():
    model.eval()
    with torch.no_grad():return criterion(model(x),y).item(),criterion(model(vx),vy).item()
def checkpoint(step):
    torch.save({'model':model.state_dict(),'xmean':xmean,'xstd':xstd,'ymean':ymean,'ystd':ystd,'chunk_size':K,'step':step},out/'best_validation.pt')
train,val=evaluate();history.append([0,train,val]);best=val;checkpoint(0)
print('训练起始帧：',train_ids.tolist())
print('验证起始帧：',val_ids.tolist())
print(f'分界帧{split}；所有训练动作块在分界前结束。归一化仅用训练样本计算。')
for step in range(1,601):
    model.train();prediction=model(x);loss=criterion(prediction,y)
    optimizer.zero_grad();loss.backward();optimizer.step()
    train,val=evaluate();history.append([step,train,val])
    if val<best:best=val;best_step=step;checkpoint(step)
    if step%100==0:print(f'更新{step:3d}次：训练MSE={train:.6f}，验证MSE={val:.6f}')
# Verify evaluation does not change model parameters.
params=[v.detach().clone() for v in model.parameters()]
evaluate()
assert all(torch.equal(old,new) for old,new in zip(params,model.parameters()))
h=np.array(history)
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
ax.plot(h[:,0],h[:,1],label='Training: 8 examples from first 70%')
ax.plot(h[:,0],h[:,2],label='Validation: 8 examples from last 30%')
ax.axvline(best_step,color='gray',linestyle=':',label=f'Lowest validation loss: step {best_step}')
ax.set(xlabel='Parameter updates',ylabel='MSE in training-normalized units',yscale='log',title='One trajectory, two non-overlapping temporal segments')
ax.legend();fig.savefig(out/'validation.png',dpi=150);plt.close(fig)
report={'dataset':str(p),'split_frame':split,'train_starts':train_ids.tolist(),'validation_starts':val_ids.tolist(),'chunk_size':K,'updates':600,'initial_train_loss':float(h[0,1]),'initial_validation_loss':float(h[0,2]),'final_train_loss':train,'final_validation_loss':val,'best_validation_step':best_step,'best_validation_loss':best,'evaluation_preserves_weights':True,'limitation':'Same episode temporal holdout; different task phases; not independent-episode generalization or closed-loop success'}
(out/'result.json').write_text(json.dumps(report,indent=2))
(out/'losses.json').write_text(json.dumps(history))
print(f'最低验证误差：{best:.6f}，更新次数：{best_step}（0表示训练前）')
print('图表：',out/'validation.png')
print('本课只在同一轨迹的不同阶段比较误差。误差差距可能来自过拟合、阶段差异或缺少图像；不能单凭此图判断原因。')
print('验证集用于选择模型，不是最终测试集；没有检验新初态或机器人闭环成功率。')
