"""Copy one collected episode into a standard-loader-compatible training dataset."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(source:Path,destination:Path):
    source=source.resolve();destination=destination.resolve()
    if source==destination or source in destination.parents:raise ValueError('训练副本必须位于原始数据目录外')
    if not (source/'SUCCESS.json').exists():raise ValueError('只接受含SUCCESS.json的采集数据')
    if destination.exists():raise FileExistsError(f'训练副本已存在：{destination}')
    info=json.loads((source/'meta/info.json').read_text())
    if info['total_episodes']!=1 or info['fps']!=30:raise ValueError('阶段1只支持单条30Hz示范')
    hashes={str(f.relative_to(source)):digest(f) for f in source.rglob('*') if f.is_file()}
    shutil.copytree(source,destination)
    tasks=pd.read_parquet(destination/'meta/tasks.parquet')
    changes=[]
    if 'task' in tasks.columns:
        # LeRobot 0.4.4 reads task text from the pandas index.
        tasks=tasks.set_index('task');tasks.to_parquet(destination/'meta/tasks.parquet')
        changes.append('tasks.parquet: move task text to pandas index for LeRobot 0.4.4')
    table=pq.read_table(destination/'data/chunk-000/file-000.parquet')
    if len(table)!=info['total_frames']:raise ValueError('实际数据条数与info.json不一致')
    if len(table)<30:raise ValueError('至少需要30帧才能构造当前动作块')
    values=table.to_pydict()
    for key in ['observation.state','action']:
        array=np.asarray(values[key])
        if array.shape!=(len(table),16) or not np.isfinite(array).all():raise ValueError(f'{key}形状或数值无效')
        if not np.isin(array[:,[7,15]],[0,1]).all():raise ValueError(f'{key}夹爪编码不为0/1')
    if not np.allclose(values['timestamp'],np.arange(len(table))/30,atol=1e-5):raise ValueError('时间戳不是均匀30Hz')
    if values['frame_index']!=list(range(len(table))) or set(values['episode_index'])!={0}:raise ValueError('帧号或episode编号异常')
    for rel,checksum in hashes.items():
        if digest(source/rel)!=checksum:raise AssertionError(f'原始数据发生变化：{rel}')
        if rel!='meta/tasks.parquet' and digest(destination/rel)!=checksum:raise AssertionError(f'副本非预期变化：{rel}')
    report={'source':str(source),'training_copy':str(destination),'source_sha256':hashes,'changes':changes,'total_frames':len(table),'source_unchanged':True}
    (destination.parent/'data_preparation.json').write_text(json.dumps(report,indent=2))
    return report
