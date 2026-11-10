"""Lesson 6: prepare spatial and state tokens; no attention or training yet."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/collector_deps'))
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frame',type=int,default=240)
parser.add_argument('--attention',action='store_true',help='Run lesson 7 after preparing tokens')
parser.add_argument('--encoder',action='store_true',help='Run lesson 8 encoder block demo')
parser.add_argument('--decoder',action='store_true',help='Run lesson 9 action decoder demo')
parser.add_argument('--cvae',action='store_true',help='Run lesson 12 CVAE demo')
args=parser.parse_args()
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先采集成功示范')
p=found[-1].parent;info=json.loads((p/'meta/info.json').read_text())
assert info['total_episodes']==1,'本课只支持一条episode'
if not 0<=args.frame<info['total_frames']:raise SystemExit('帧号超出范围')
weights=ROOT/'.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth'
if not weights.exists():raise SystemExit('请先运行 ./learning/scripts/learn_vision.sh 下载权重并完成上一课')
assert hashlib.sha256(weights.read_bytes()).hexdigest().startswith('f37072fd')
torch.manual_seed(0);torch.set_num_threads(2)
backbone=resnet18(weights=None)
backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));backbone.eval()
extractor=nn.Sequential(*list(backbone.children())[:-2])
transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
images=[];inputs=[]
cameras=('head','left_wrist','right_wrist')
for name in cameras:
    cap=cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4'))
    try:
        for _ in range(args.frame+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'{name}视频解码失败')
    finally:cap.release()
    x=transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    inputs.append(x)
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    images.append((x*std+mean).permute(1,2,0).numpy().clip(0,1))
with torch.no_grad():feature=extractor(torch.stack(inputs))
assert feature.shape==(3,512,7,7)
# Shared projection and embeddings are deliberately untrained teaching modules.
D=128
projection=nn.Linear(512,D)
state_projection=nn.Linear(16,D)
camera_embedding=nn.Embedding(3,D)
state_type=nn.Parameter(torch.zeros(1,1,D))
rows=pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
state=torch.tensor(rows['observation.state'][args.frame],dtype=torch.float32)[None,None,:]
# A deterministic row/column sinusoidal encoding, with 64 numbers for each axis.
def axis_encoding(values):
    frequencies=10000**(-torch.arange(0,D//2,2,dtype=torch.float32)/(D//2))
    angle=values[:,None].float()*frequencies
    return torch.stack((angle.sin(),angle.cos()),dim=-1).flatten(1)
grid_row=torch.arange(7).repeat_interleave(7)
grid_col=torch.arange(7).repeat(7)
pos=torch.cat((axis_encoding(grid_row),axis_encoding(grid_col)),dim=-1)
with torch.no_grad():
    flattened=feature.flatten(2).transpose(1,2) # [camera, row*column, channel]
    content=projection(flattened)
    visual=content+pos[None,:,:]+camera_embedding(torch.arange(3))[:,None,:]
    visual=visual.reshape(1,147,D)
    state_token=state_projection(state)+state_type
    sequence=torch.cat((state_token,visual),dim=1)
assert flattened.shape==(3,49,512) and sequence.shape==(1,148,D)
assert torch.equal(flattened[0,17],feature[0,:,2,3])
assert not torch.equal(pos[0],pos[1])
assert torch.allclose(sequence[:,0],state_token[:,0])
assert torch.isfinite(sequence).all()
# Changing the order preserves token contents but changes which sequence positions hold them.
without_position=content[0].mean(0)
permutation=torch.arange(48,-1,-1)
assert torch.allclose(without_position,content[0,permutation].mean(0),atol=1e-6)
out=p.parent/'learning_tokens';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
for c,ax in enumerate(axes):
    ax.imshow(images[c],extent=(0,7,7,0))
    for edge in range(8):
        ax.axhline(edge,color='white',alpha=.65,linewidth=.6)
        ax.axvline(edge,color='white',alpha=.65,linewidth=.6)
    for r in range(7):
        for col in range(7):
            ax.text(col+.5,r+.5,str(1+c*49+r*7+col),ha='center',va='center',fontsize=8,color='black',bbox={'facecolor':'white','alpha':.65,'edgecolor':'none','pad':.3})
    ax.set_title(f'{cameras[c]}: token {1+c*49} to {(c+1)*49}')
    ax.set(xlim=(0,7),ylim=(7,0),xlabel='Column index',ylabel='Row index')
fig.suptitle('Token 0 = robot state. Grid locations are feature anchors, not isolated image patches.')
fig.savefig(out/'tokens.png',dpi=150);plt.close(fig)
index=[{'token':0,'type':'robot_state'}]
for c,name in enumerate(cameras):
    for r in range(7):
        for col in range(7):index.append({'token':1+c*49+r*7+col,'camera':name,'row':r,'column':col})
report={'dataset':str(p),'frame':args.frame,'feature_shape':list(feature.shape),'flattened_shape':list(flattened.shape),'sequence_shape':list(sequence.shape),'hidden_dim':D,'token_index':index,'training':False,'limitations':['projection and camera embeddings randomly initialized','state not normalized in this shape-only lesson','no Transformer or action prediction','sinusoidal encoding added directly for teaching; not exact ACT implementation','grid anchors have overlapping receptive fields']}
(out/'result.json').write_text(json.dumps(report,indent=2))
print('准备阶段：整理Transformer输入；下方按所选课程继续计算，不控制机器人。')
print('三路特征图：',tuple(feature.shape))
print('每路展开后：',tuple(flattened.shape),'= 3路 × 49个位置 × 每位置512个特征')
print('共享线性投影：512 → 128；投影参数量：',sum(v.numel() for v in projection.parameters()))
print('状态投影：16 → 128；状态token：',tuple(state_token.shape))
print('视觉token：',tuple(visual.shape),'；完整序列：',tuple(sequence.shape))
print('顺序：状态0；head 1～49；left_wrist 50～98；right_wrist 99～147。')
print('例如：head行2列3 → token',1+2*7+3,'（行列从0开始）')
print('注意：序列已拼接，但没有注意力计算，信息尚未相互融合。')
print('128是教学宽度；投影和相机编码未训练；图像使用上一课的224中心裁剪。')
print('图表：',out/'tokens.png')

if args.attention:
    from attention_demo import demonstrate
    demonstrate(sequence, index, p.parent/'learning_attention')

if args.encoder:
    from encoder_demo import demonstrate_encoder
    demonstrate_encoder(sequence, p.parent/'learning_encoder')

if args.decoder:
    from decoder_demo import demonstrate_decoder
    demonstrate_decoder(sequence, rows['action'], args.frame, info['fps'], p.parent/'learning_decoder')

if args.cvae:
    from cvae_demo import demonstrate_cvae
    demonstrate_cvae(sequence, rows['action'], args.frame, info['fps'], p.parent/'learning_cvae')
