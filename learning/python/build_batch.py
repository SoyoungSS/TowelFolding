"""Lesson 2: build ACT-shaped tensors, without training a network."""
import sys
from pathlib import Path
import json
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path)
args=parser.parse_args()
if args.dataset is None:
    found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
    if not found:raise SystemExit('请先运行 ./run_collect.sh 采集一条成功示范')
    args.dataset=found[-1].parent
p=args.dataset.resolve()
info=json.loads((p/'meta/info.json').read_text())
if info['total_episodes']!=1:raise SystemExit('本课只处理单条episode')
d=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
s=torch.tensor(d['observation.state'],dtype=torch.float32)
a=torch.tensor(d['action'],dtype=torch.float32)
n=len(a);K=30
if n<2:raise SystemExit('数据至少需要两帧')
starts=[min(100,n-2),n-2]  # One ordinary sample and one ending sample.
images={}
for name in ('head','left_wrist','right_wrist'):
    cap=cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4'))
    frames={}
    try:
        for i in range(max(starts)+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'{name}第{i}帧无法解码')
            if i in starts:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                frames[i]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.
    finally:cap.release()
    images[f'observation.images.{name}']=torch.stack([frames[t] for t in starts])
chunks=[];masks=[]
for t in starts:
    idx=torch.arange(t,t+K)
    masks.append(idx>=n)
    chunks.append(a[idx.clamp(max=n-1)])
batch={**images,'observation.state':s[starts], 'action':torch.stack(chunks),'action_is_pad':torch.stack(masks)}
assert batch['action'].shape==(2,K,16)
assert not batch['action_is_pad'][1,:2].any() and batch['action_is_pad'][1,2:].all()
assert torch.equal(batch['action'][0,0],a[starts[0]])
assert batch['observation.images.head'].shape==(2,3,480,640)
print(f'数据：{p}\n本课不创建网络、不训练、不控制机器人。\nB=2（两个训练样本），K=30（每个样本的未来动作步数），动作维度=16\n')
for key,value in batch.items():
    print(f'{key:40s} shape={str(tuple(value.shape)):20s} dtype={value.dtype}')
print('\n注意：本课图像仅做RGB转通道优先和除以255；正式训练还需使用选定模型的完整预处理。')
for b,t in enumerate(starts):
    valid=int((~batch['action_is_pad'][b]).sum())
    print(f'\n样本{b}：观测来自原始第{t}帧，时间{d["timestamp"][t]:.3f}s')
    print(f'  有效动作{valid}步，填充{K-valid}步；第一个标签就是action[{t}]，不是state[{t}]。')
    print('  左关节1当前实际值：',round(float(s[t,0]),5))
    print('  左关节1前5个目标：',batch['action'][b,:5,0].tolist())
    print('  前5个填充标记：',batch['action_is_pad'][b,:5].tolist())
# A deliberately wrong dummy prediction: zeros, not a real model.
pred=torch.zeros_like(batch['action'])
valid=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
loss=(pred-batch['action']).abs()[valid].mean()
print(f'\n损失演示：假装网络全部预测0，忽略填充后平均绝对误差={loss.item():.6f}')
print('这个值混合了关节弧度和夹爪0/1，仅用于理解误差计算；正式训练需匹配归一化。')
print('\n练习：解释[2,3,480,640]和[2,30,16]；为什么末尾28步不能参与损失？')
