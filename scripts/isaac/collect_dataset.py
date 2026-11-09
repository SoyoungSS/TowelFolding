"""Single-episode LeRobot v3 writer for synchronized live captures."""
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ('head', 'left_wrist', 'right_wrist')
NAMES = [f'{side}_joint_{i}' for side in ('left','right') for i in range(1,8)]
NAMES = NAMES[:7]+['left_gripper_closed']+NAMES[7:]+['right_gripper_closed']

def canonical(x):
    y = np.asarray(x, dtype=np.float32)[[0,1,2,3,4,5,6,14,7,8,9,10,11,12,13,15]].copy()
    y[[7,15]] = (y[[7,15]] < .1).astype(np.float32)
    return y

class Collector:
    def __init__(self, root, fps):
        self.root, self.fps = Path(root), fps
        self.rows, self.writers = [], {}
        for name in CAMERAS:
            p = self.root/f'videos/observation.images.{name}/chunk-000/file-000.mp4'
            p.parent.mkdir(parents=True, exist_ok=True)
            w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640,480))
            if not w.isOpened():
                raise RuntimeError(f'Cannot encode {p}')
            self.writers[name] = w
        self.finished = False

    def add(self, state, images, action):
        if self.finished:
            raise RuntimeError('Episode already finalized')
        i = len(self.rows)
        if abs(state['sim_time'] - i/self.fps) > 1e-5:
            raise ValueError('Nonuniform observation timestamp')
        for name, w in self.writers.items():
            rgb = images[name]
            assert rgb.shape == (480,640,3)
            w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.rows.append({'observation.state':canonical(state['joint_state']).tolist(),
                          'action':canonical(action).tolist(), 'timestamp':i/self.fps,
                          'frame_index':i, 'episode_index':0, 'index':i, 'task_index':0})

    def close(self):
        for w in self.writers.values():
            w.release()

    def finish(self, success):
        self.close()
        if not success:
            (self.root/'FAILED.json').write_text(json.dumps({'success':False,'frames':len(self.rows)}))
            self.finished = True
            return
        n = len(self.rows)
        assert n > 0
        data = self.root/'data/chunk-000/file-000.parquet'
        data.parent.mkdir(parents=True,exist_ok=True)
        schema = pa.schema([(k,pa.list_(pa.float32(),16)) for k in ('observation.state','action')]+[
            ('timestamp',pa.float32())]+[(k,pa.int64()) for k in ('frame_index','episode_index','index','task_index')])
        pq.write_table(pa.Table.from_pylist(self.rows,schema=schema),data)
        meta = self.root/'meta'
        (meta/'episodes/chunk-000').mkdir(parents=True,exist_ok=True)
        task = 'fold a towel with both arms'
        pq.write_table(pa.Table.from_pylist([{'task_index':0,'task':task}]),meta/'tasks.parquet')
        features = {k:{'dtype':'float32','shape':[16],'names':NAMES} for k in ('observation.state','action')}
        stats = {}
        ep = {'episode_index':0,'tasks':[task],'length':n,'data/chunk_index':0,'data/file_index':0,
              'dataset_from_index':0,'dataset_to_index':n,'meta/episodes/chunk_index':0,'meta/episodes/file_index':0}
        for k in schema.names:
            if k not in features:
                features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
            a=np.array([r[k] for r in self.rows]).reshape(n,-1)
            stats[k]={key:val.tolist() for key,val in {'min':a.min(0),'max':a.max(0),'mean':a.mean(0),'std':a.std(0),'count':np.array([n])}.items()}
        for name in CAMERAS:
            k=f'observation.images.{name}'
            cap=cv2.VideoCapture(str(self.root/f'videos/{k}/chunk-000/file-000.mp4'))
            total=0; sums=np.zeros(3); squares=np.zeros(3); lo=np.ones(3); hi=np.zeros(3);frames=0
            while True:
                ok,bgr=cap.read()
                if not ok:break
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float64)/255
                total+=rgb.shape[0]*rgb.shape[1];frames+=1
                sums+=rgb.sum((0,1));squares+=(rgb*rgb).sum((0,1));lo=np.minimum(lo,rgb.min((0,1)));hi=np.maximum(hi,rgb.max((0,1)))
            cap.release()
            assert frames==n,(name,frames,n)
            stats[k]={key:val.reshape(3,1,1).tolist() for key,val in {'min':lo,'max':hi,'mean':sums/total,'std':np.sqrt(np.maximum(squares/total-(sums/total)**2,0))}.items()}
            stats[k]['count']=[n]
            features[k]={'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],
                         'info':{'video.height':480,'video.width':640,'video.fps':self.fps,'video.codec':'mpeg4','video.pix_fmt':'yuv420p','video.channels':3,'has_audio':False,'is_depth_map':False}}
            for suffix,value in [('chunk_index',0),('file_index',0),('from_timestamp',0.),('to_timestamp',n/self.fps)]:ep[f'videos/{k}/{suffix}']=value
        for k,v in stats.items():
            for name,value in v.items():ep[f'stats/{k}/{name}']=value
        pq.write_table(pa.Table.from_pylist([ep]),meta/'episodes/chunk-000/file-000.parquet')
        info={'codebase_version':'v3.0','fps':self.fps,'features':features,'total_episodes':1,'total_frames':n,'total_tasks':1,
              'chunks_size':1000,'data_files_size_in_mb':100,'video_files_size_in_mb':200,'robot_type':'encos_dual_arm_towel',
              'splits':{'train':'0:1'},'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
        (meta/'info.json').write_text(json.dumps(info,indent=2))
        (meta/'stats.json').write_text(json.dumps(stats,indent=2))
        (meta/'towel_schema.json').write_text(json.dumps({'state_names':NAMES,'action_names':NAMES,'gripper_encoding':'0=open, 1=closed','source':'Newton fixed scene; assisted grasp; uniform simulation time'},indent=2))
        (self.root/'SUCCESS.json').write_text(json.dumps({'frames':n,'fps':self.fps,'success':True},indent=2))
        self.finished=True
        print(f'DATASET SAVED: {self.root}',flush=True)
