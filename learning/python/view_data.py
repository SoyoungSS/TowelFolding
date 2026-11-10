"""Create a local interactive lesson from one collected episode."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.runtime/collector_deps'))
import cv2
import numpy as np
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path)
args = parser.parse_args()
if args.dataset is None:
    datasets = sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'), key=lambda p:p.stat().st_mtime)
    if not datasets:
        raise SystemExit('没有成功采集的数据，请先运行 ./run_collect.sh')
    args.dataset = datasets[-1].parent
p = args.dataset.resolve()
info = json.loads((p/'meta/info.json').read_text())
if info['total_episodes'] != 1:
    raise SystemExit('此入门工具只读取单条episode数据集。')
d = pq.read_table(p/'data/chunk-000/file-000.parquet').to_pydict()
n = len(d['timestamp'])
state = np.asarray(d['observation.state']); action = np.asarray(d['action'])
assert state.shape == action.shape == (n,16)
assert np.isfinite(state).all() and np.isfinite(action).all()
out = p.parent/'learning_view'
(out/'frames').mkdir(parents=True, exist_ok=True)
caps = [cv2.VideoCapture(str(p/f'videos/observation.images.{name}/chunk-000/file-000.mp4')) for name in ('head','left_wrist','right_wrist')]
try:
    for i in range(n):
        images=[]
        for c in caps:
            ok, image = c.read()
            if not ok: raise RuntimeError(f'视频在第{i}帧提前结束')
            images.append(cv2.resize(image,(320,240)))
        if not cv2.imwrite(str(out/f'frames/{i:05d}.jpg'),cv2.hconcat(images)):
            raise RuntimeError('不能写入预览图片')
finally:
    for c in caps:c.release()
names = info['features']['observation.state']['names']
payload = json.dumps(dict(state=state.tolist(), action=action.tolist(), time=d['timestamp'], names=names, fps=info['fps']),ensure_ascii=False)
html = '''<!doctype html><meta charset="utf-8"><title>第一课：读懂机器人示范数据</title>
<style>body{font:17px system-ui;max-width:1100px;margin:24px auto;padding:0 20px;color:#172b3a;background:#f5f7fa}section{background:white;padding:20px;margin:16px 0;border-radius:12px}img{width:100%}input[type=range]{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left}button,select{font:inherit;padding:6px}canvas{width:100%;height:240px}.names{display:flex;justify-content:space-around}.note{color:#526575}</style>
<h1>第一课：当前看到了什么，专家准备怎么动？</h1>
<p>图像和 state 是执行前的观测；action 是随后一个控制周期的目标。这里播放的是已经记录的数据，不会控制机器人。</p>
<section><div class="names"><b>head：整体</b><b>left_wrist：左腕</b><b>right_wrist：右腕</b></div><img id="picture"><p id="position"></p><button id="play">播放</button> <button id="prev">上一帧</button> <button id="next">下一帧</button><input id="slider" type="range" min="0" value="0"></section>
<section><h2>观察一个关节随时间的变化</h2><select id="joint"></select><p>蓝线：实际状态 state　橙线：目标命令 action　竖线：当前帧</p><canvas id="plot"></canvas><p id="value"></p><p class="note">横轴：仿真时间（秒）；纵轴：关节角（弧度），夹爪则为0/1。目标与实际存在差异是正常现象；仅凭差异大小不能判断折叠是否成功。</p></section>
<section><h2>当前帧的16个数字</h2><table><thead><tr><th>下标（从0开始）</th><th>名称</th><th>实际 state</th><th>目标 action</th><th>目标－实际</th></tr></thead><tbody id="rows"></tbody></table><p>下标7、15是左右夹爪：0=打开，1=关闭。其余为关节弧度；1弧度约57.3°。夹爪状态为阈值转换后的二值量，不是连续开度。</p></section>
<section><h2>请完成三个练习</h2><ol><li>拖动时间条，找出“接近、闭合、抬起、松手”各发生在哪些帧。</li><li>选择左右夹爪（下标7、15），观察目标关闭与实际关闭是否同时发生。</li><li>选择 left_joint_1，比较实际与目标曲线。说明为什么 action 不能直接复制 state。</li></ol><p>提示：同一帧的三路画面是不同视角；“下一帧”的时间增加约1/30秒。播放速度仅用于查看，不代表真实执行速度。</p></section>
<script>
const D=PAYLOAD;let frame=0,timer=null;const $=id=>document.getElementById(id);$('slider').max=D.time.length-1;
D.names.forEach((name,i)=>{$('joint').add(new Option(i+'：'+name,i))});
function draw(){const c=$('plot');c.width=1000;c.height=240;const ctx=c.getContext('2d'),j=+$('joint').value,all=D.state.map(x=>x[j]).concat(D.action.map(x=>x[j]));let low=Math.min(...all),high=Math.max(...all);const pad=Math.max((high-low)*.15,.02);low-=pad;high+=pad;const X=i=>55+i/(D.time.length-1)*925,Y=v=>205-(v-low)/(high-low)*180;ctx.strokeStyle='#bcc6cf';ctx.strokeRect(55,25,925,180);ctx.font='13px sans-serif';ctx.fillStyle='#172b3a';ctx.fillText(high.toFixed(3),0,30);ctx.fillText(low.toFixed(3),0,205);ctx.fillText('0 s',55,228);ctx.fillText(D.time.at(-1).toFixed(2)+' s',925,228);for(const [a,color] of [[D.state,'#1976d2'],[D.action,'#ef7c20']]){ctx.beginPath();ctx.strokeStyle=color;a.forEach((r,i)=>i?ctx.lineTo(X(i),Y(r[j])):ctx.moveTo(X(i),Y(r[j])));ctx.stroke()}ctx.beginPath();ctx.strokeStyle='#333';ctx.moveTo(X(frame),25);ctx.lineTo(X(frame),205);ctx.stroke();$('value').textContent=`当前 ${D.names[j]}：实际 ${D.state[frame][j].toFixed(4)}，目标 ${D.action[frame][j].toFixed(4)}`}
function update(){frame=+$('slider').value;$('picture').src='frames/'+String(frame).padStart(5,'0')+'.jpg';$('position').textContent=`第 ${frame} 帧（下标从0开始），仿真时间 ${D.time[frame].toFixed(3)} 秒，共 ${D.time.length} 条记录`; $('rows').innerHTML=D.names.map((name,j)=>`<tr><td>${j}</td><td>${name}</td><td>${D.state[frame][j].toFixed(4)}</td><td>${D.action[frame][j].toFixed(4)}</td><td>${(D.action[frame][j]-D.state[frame][j]).toFixed(4)}</td></tr>`).join('');draw()}
$('slider').oninput=update;$('joint').onchange=draw;$('prev').onclick=()=>{$('slider').value=Math.max(0,frame-1);update()};$('next').onclick=()=>{$('slider').value=Math.min(D.time.length-1,frame+1);update()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return}if(frame===D.time.length-1){$('slider').value=0;update()}$('play').textContent='暂停';timer=setInterval(()=>{if(frame===D.time.length-1){$('play').click();return}$('next').click()},1000/D.fps)};update();
</script>'''.replace('PAYLOAD',payload)
(out/'index.html').write_text(html)
print(f'数据：{p}\n记录数：{n}，频率：{info["fps"]} Hz\n学习页面：{out/"index.html"}')
