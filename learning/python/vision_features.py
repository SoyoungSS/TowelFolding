"""Lesson 5: inspect a pretrained ResNet-18 feature extractor on one image."""
from pathlib import Path
import sys,json,argparse,hashlib,os
ROOT=Path(__file__).resolve().parents[2]
import cv2
import numpy as np
import torch
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frame',type=int,default=240)
args=parser.parse_args()
found=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
if not found:raise SystemExit('请先采集成功示范')
p=found[-1].parent
info=json.loads((p/'meta/info.json').read_text())
if not 0<=args.frame<info['total_frames']:raise SystemExit('帧号超出数据范围')
weights_path=ROOT/'.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth'
if not weights_path.exists():raise SystemExit('缺少官方权重，请查看README第五课下载命令。')
assert hashlib.sha256(weights_path.read_bytes()).hexdigest().startswith('f37072fd'),'权重校验失败，请重新下载'
cap=cv2.VideoCapture(str(p/'videos/observation.images.head/chunk-000/file-000.mp4'))
try:
    for _ in range(args.frame+1):
        ok,bgr=cap.read()
        if not ok:raise RuntimeError('视频解码失败')
finally:cap.release()
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
torch.manual_seed(0);torch.set_num_threads(2)
model=resnet18(weights=None)
model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
model.eval()
preprocess=ResNet18_Weights.IMAGENET1K_V1.transforms()
x=preprocess(Image.fromarray(rgb)).unsqueeze(0)
features={}
with torch.no_grad():
    z=model.relu(model.bn1(model.conv1(x)));features['stem']=z
    z=model.maxpool(z);features['maxpool']=z
    for name in ('layer1','layer2','layer3','layer4'):
        z=getattr(model,name)(z);features[name]=z
    vector=torch.flatten(model.avgpool(z),1)
assert x.shape==(1,3,224,224) and z.shape==(1,512,7,7) and vector.shape==(1,512)
assert all(torch.isfinite(v).all() for v in features.values())
out=p.parent/'learning_vision';out.mkdir(exist_ok=True)
# Reconstruct the actual cropped RGB seen by the network for visual comparison.
mean=torch.tensor([.485,.456,.406])[:,None,None]
std=torch.tensor([.229,.224,.225])[:,None,None]
cropped=(x[0]*std+mean).permute(1,2,0).numpy().clip(0,1)
fig,axes=plt.subplots(2,5,figsize=(13,6),layout='constrained')
axes[0,0].imshow(rgb);axes[0,0].set_title('Original RGB (640 x 480)')
axes[1,0].imshow(cropped);axes[1,0].set_title('Actual input crop (224 x 224)')
for col,ch in enumerate((0,1,2,3),start=1):
    for row,name in enumerate(('stem','layer4')):
        im=features[name][0,ch].numpy();ax=axes[row,col]
        ax.imshow(im,cmap='viridis');ax.set_title(f'{name} channel {ch}\n{im.shape[0]} x {im.shape[1]}')
for ax in axes.flat:ax.axis('off')
fig.suptitle('Feature activations: each panel has its own color scale; not an attention map')
fig.savefig(out/'features.png',dpi=140);plt.close(fig)
report={'dataset':str(p),'frame':args.frame,'input_shape':list(x.shape),'feature_shapes':{k:list(v.shape) for k,v in features.items()},'pooled_vector_shape':list(vector.shape),'backbone_parameters':sum(v.numel() for name,v in model.named_parameters() if not name.startswith('fc.')),'weights':'ResNet18 ImageNet1K V1','training':False}
(out/'result.json').write_text(json.dumps(report,indent=2))
print('本课只做CPU前向推理，不训练、不连接机器人。')
print('读取头部相机第',args.frame,'帧')
print('原始RGB：',rgb.shape,'→ 网络输入：',tuple(x.shape))
for name,v in features.items():print(f'{name:10s}: {tuple(v.shape)}')
print('全局平均池化后的向量：',tuple(vector.shape))
print('特征骨干参数量（不含分类层）：',report['backbone_parameters'])
print('图表：',out/'features.png')
print('注意：特征通道不是物体类别；热区不是抓取点，也不是注意力解释。')
print('224中心裁剪与池化向量用于教学；正式ACT需按选定实现处理图像并保留空间特征。')
