"""Run a live ACT-controlled Newton scene with Isaac rendering, no expert fallback."""
import argparse,base64,hashlib,json,os,socket,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'simulation/isaac'))
from bridge_transport import send_message,receive_message
from act_io import CAMERAS
from act_deployment import load_deployment


def connect_worker(path,process,timeout):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if process.poll() is not None:raise RuntimeError(f'Worker exited with {process.returncode}; check worker log')
        if time.monotonic()>deadline:raise TimeoutError(f'Worker startup timeout: {path.name}')
        time.sleep(.05)
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    connection.settimeout(timeout);connection.connect(str(path))
    return connection,receive_message(connection)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--initial-plan', type=Path)
    p.add_argument('--fold-checkpoint', type=Path)
    p.add_argument('--switch-step', type=int)
    p.add_argument('--goal', choices=['fold', 'grasp'], default='fold')
    p.add_argument('--newton-python',type=Path,default=Path.home()/'miniforge3/envs/towel-newton/bin/python')
    p.add_argument('--policy-python',type=Path,default=ROOT/'.runtime/act-env311/bin/python')
    p.add_argument('--output',type=Path)
    p.add_argument('--duration',type=float)
    p.add_argument('--action-steps', type=int,
                   help='Number of actions to execute from each predicted chunk before replanning')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--exit-on-done',action='store_true')
    p.add_argument('--timeout',type=float,default=120.)
    a=p.parse_args()
    if a.checkpoint is None:
        try:
            selected = load_deployment(ROOT)
        except (OSError, ValueError, KeyError) as error:
            p.error(str(error))
        if selected is not None:
            a.checkpoint = selected['checkpoint']
            if a.action_steps is None:
                a.action_steps = selected['action_steps']
            if a.duration is None:
                a.duration = selected['duration_s']
        else:
            choices=sorted(ROOT.glob('outputs/act_stage1/*/SUCCESS.json'),key=lambda v:v.stat().st_mtime)
            if not choices:p.error('No passed stage1 model found; run ./run_act_stage1.sh first')
            a.checkpoint=choices[-1].parent/'checkpoint'
    if a.duration is None:
        a.duration = 30.
    if a.action_steps is None:
        a.action_steps = 30
    if not np.isfinite(a.duration) or a.duration<=0 or a.duration>120:p.error('duration must be in (0,120] seconds')
    a.checkpoint=a.checkpoint.resolve()
    runtime=json.loads((a.checkpoint/'runtime.json').read_text())
    if not 1 <= a.action_steps <= runtime['chunk_size']:
        p.error('action-steps must be between 1 and the checkpoint chunk size')
    if (a.fold_checkpoint is None) != (a.switch_step is None):
        p.error('--fold-checkpoint and --switch-step must be used together')
    if a.switch_step is not None and not (
        0 < a.switch_step < a.duration * 30 and a.switch_step % a.action_steps == 0
    ):
        p.error('Switch step must be a chunk boundary within the rollout')
    training_meta=Path(runtime['source_dataset']).parent/'metadata.json'
    if not training_meta.exists():p.error('Training-scene metadata not found beside source dataset')
    expected=json.loads(training_meta.read_text())
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output=(a.output or ROOT/f'outputs/act_rollout/{stamp}_{os.getpid()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    print(f'ACT LIVE output={output}; checkpoint={a.checkpoint}',flush=True)
    processes=[];connections=[];app=executor=None;writers={};state=None
    states=[];actions=[];predicted_actions=[];inference_times=[];request_id=0;updates=0
    evaluation=None;error=None;response=None;completed=False;pending=None
    started=time.monotonic()
    with TemporaryDirectory(prefix='towel-act-live-') as tmp, (output/'newton.log').open('w') as physics_log, (output/'policy.log').open('w') as policy_log, (output/'frames.jsonl').open('w') as journal:
        try:
            physics_path=Path(tmp)/'physics.sock';policy_path=Path(tmp)/'policy.sock'
            physics_command = [str(a.newton_python),'-I','-u',str(ROOT/'scripts/isaac/act_newton_worker.py'),
                '--socket',str(physics_path),'--seed',str(a.seed),'--duration',str(a.duration), '--goal', a.goal]
            if a.initial_plan:
                physics_command.extend(['--initial-plan', str(a.initial_plan.resolve())])
            physics_process=subprocess.Popen(physics_command,stdout=physics_log,stderr=subprocess.STDOUT,cwd=ROOT)
            processes.append(physics_process)
            physics,ready=connect_worker(physics_path,physics_process,a.timeout);connections.append(physics)
            state=ready['state'];meta=ready['metadata'];states.append(state)
            expected_config = expected['config']
            if runtime.get('collection_batch'):
                # Batch training varies only the initial towel centre. Evaluate
                # against the nominal scene, including when the first success was offset.
                expected_config = dict(expected_config, towel=dict(
                    expected_config['towel'], center=meta['config']['towel']['center']))
            if meta['urdf_sha256']!=expected['urdf_sha256'] or meta['config']!=expected_config:
                raise ValueError('Current physics/camera configuration differs from the collected training scene')
            (output/'metadata.json').write_text(json.dumps(meta,indent=2))
            (output/'training_scene.json').write_text(json.dumps(expected,indent=2))
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            env.update(HF_HOME=str(ROOT/'.runtime/act_hf'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TORCH_HOME=str(ROOT/'.runtime/torch'))
            policy_command = [str(a.policy_python), '-u', str(ROOT/'training/act/live_policy_worker.py'),
                '--socket', str(policy_path), '--checkpoint', str(a.checkpoint), '--action-steps', str(a.action_steps)]
            if a.fold_checkpoint:
                a.fold_checkpoint = a.fold_checkpoint.resolve()
                fold_runtime = json.loads((a.fold_checkpoint/'runtime.json').read_text())
                if fold_runtime['source_dataset'] != runtime['source_dataset']:
                    raise ValueError('The checkpoints must use the same training scene')
                policy_command.extend(['--fold-checkpoint', str(a.fold_checkpoint), '--switch-step', str(a.switch_step)])
            policy_process = subprocess.Popen(policy_command, stdout=policy_log, stderr=subprocess.STDOUT,
                                             cwd=ROOT, env=env)
            processes.append(policy_process)
            policy,policy_ready=connect_worker(policy_path,policy_process,a.timeout);connections.append(policy)
            if policy_ready['n_action_steps'] != a.action_steps:
                raise ValueError('Policy worker action horizon does not match the requested setting')
            (output/'policy_ready.json').write_text(json.dumps(policy_ready,indent=2))
            print('Physics and ACT connected; expert guards enabled. Starting Isaac window.',flush=True)
            from isaacsim import SimulationApp
            app=SimulationApp({'headless':a.headless,'width':1280,'height':720,'window_width':1440,'window_height':900})
            from visual_scene import build_scene
            from PIL import Image
            import cv2
            visual=build_scene(meta,np.asarray(state['cloth_points'],dtype=np.float32),30)
            (output/'videos').mkdir()
            for name in CAMERAS:
                writer=cv2.VideoWriter(str(output/f'videos/{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
                if not writer.isOpened():raise RuntimeError(f'Cannot encode {name}')
                writers[name]=writer
            executor=ThreadPoolExecutor(max_workers=1)
            flags={'playing':True,'quit':False};labels={};providers={};camera_window=None;last_rgbs=None
            def refresh_previews(images):
                for name,provider in providers.items():
                    preview=cv2.resize(images[name],(320,240))
                    rgba=np.dstack((preview,np.full((240,320),255,np.uint8)))
                    provider.set_bytes_data(rgba.ravel().tolist(),[320,240])
            def toggle_cameras():
                camera_window.visible=not camera_window.visible
                if camera_window.visible and last_rgbs is not None:refresh_previews(last_rgbs)
            if not a.headless:
                import omni.ui as ui
                from omni.kit.viewport.utility import get_active_viewport
                viewport=get_active_viewport()
                if viewport is None:raise RuntimeError('No active Isaac viewport')
                viewport.camera_path='/World/Overview'
                controls=ui.Window('ACT policy evaluation',width=400,height=245,position_x=20,position_y=60)
                def toggle():flags['playing']=not flags['playing']
                def quit_all():flags['quit']=True
                with controls.frame:
                    with ui.VStack(spacing=8):
                        ui.Label('LIVE ACT | Expert disabled | 30 Hz sim time')
                        labels['state']=ui.Label('Starting')
                        labels['capture']=ui.Label('No grasps yet')
                        labels['outcome']=ui.Label('Running independent evaluation')
                        ui.Button('Play / Pause',height=28,clicked_fn=toggle)
                        ui.Button('Show / hide cameras',height=28,clicked_fn=toggle_cameras)
                        ui.Button('Exit all processes',height=28,clicked_fn=quit_all)
                camera_window=ui.Window('ACT RGB observations',width=985,height=285,visible=False)
                with camera_window.frame:
                    with ui.HStack():
                        for name in CAMERAS:
                            with ui.VStack(width=320):
                                ui.Label(name,height=20);providers[name]=ui.ByteImageProvider()
                                ui.ImageWithProvider(providers[name],width=320,height=240)
            dirty=True;refs=None
            def infer_and_advance(rid,snapshot,images,reference_times):
                # The actor sees only live RGB and robot proprioception, never cloth truth or expert labels.
                payload={'command':'infer','request_id':rid,'episode_id':snapshot['episode_id'],'frame_id':snapshot['frame_id'],
                    'joint_state':snapshot['joint_state'],'images':{n:base64.b64encode(images[n].tobytes()).decode('ascii') for n in CAMERAS}}
                send_message(policy,payload);pred=receive_message(policy)
                if (pred['request_id'],pred['episode_id'],pred['frame_id'])!=(rid,snapshot['episode_id'],snapshot['frame_id']):raise ValueError('Policy observation acknowledgement mismatch')
                send_message(physics,{'command':'advance','request_id':rid,'episode_id':snapshot['episode_id'],
                    'frame_id':snapshot['frame_id'],'steps':2,'policy_action':pred['policy_action']})
                result=receive_message(physics)
                if result['request_id']!=rid or result['state']['frame_id']!=snapshot['frame_id']+2:raise ValueError('Physics lockstep mismatch')
                result['prediction']=pred;result['observation_frame_id']=snapshot['frame_id'];result['camera_reference_times']=reference_times
                return result
            while app.is_running() and not flags['quit']:
                if pending is not None:
                    if not pending.done():app.update();time.sleep(.002);continue
                    response=pending.result();pending=None
                    predicted_actions.append(response['prediction']['policy_action'])
                    actions.append(response['action']);inference_times.append(response['prediction']['inference_ms'])
                    journal.write(json.dumps({'observation_frame_id':response['observation_frame_id'],
                        'post_step_frame_id':response['state']['frame_id'],'camera_reference_times':response['camera_reference_times'],
                        'policy_action':response['prediction']['policy_action'],'applied_internal_action':response['action'],
                        'inference_ms':response['prediction']['inference_ms'],'expert_calls':response['expert_calls'],
                        'ever_grasped':response['ever_grasped'],'metrics':response['metrics']})+'\n');journal.flush()
                    state=response['state'];states.append(state);dirty=True
                    if response['terminal']:flags['playing']=False
                if dirty:
                    visual.update(state);all_rgbs,all_refs=visual.capture();updates+=1
                    last_rgbs={name:all_rgbs[name] for name in CAMERAS}
                    refs={name:all_refs[name] for name in CAMERAS}
                    for name,rgb in last_rgbs.items():
                        writers[name].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        if updates==1 or (response and response['terminal']):
                            Image.fromarray(rgb).save(output/f'{name}_{"final" if updates>1 else "initial"}.png')
                    if providers and camera_window.visible:refresh_previews(last_rgbs)
                    if labels:
                        labels['state'].text=f't={state["sim_time"]:.2f}s | policy steps={len(actions)}'
                        if response:labels['capture'].text=f'Grasped L/R: {response["ever_grasped"]} | expert calls: {response["expert_calls"]}'
                    (output/'live_status.json').write_text(json.dumps({'physics_pid':physics_process.pid,'policy_pid':policy_process.pid,
                        'isaac_pid':os.getpid(),'updates':updates,'frame_id':state['frame_id'],'sim_time':state['sim_time'],
                        'policy_steps':len(actions),'expert_calls':response['expert_calls'] if response else 0},indent=2))
                    if updates%100==0:print(f'ACT LIVE t={state["sim_time"]:.2f}s steps={len(actions)} grasp={response["ever_grasped"]}',flush=True)
                    if not a.headless and updates==1:
                        import omni.renderer_capture
                        omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(output/'ui_started.png'))
                    dirty=False
                if response and response['terminal'] and not completed:
                    for writer in writers.values():writer.release()
                    # Decode every recorded stream to check finalized video frame counts.
                    video_counts={}
                    for name in CAMERAS:
                        cap=cv2.VideoCapture(str(output/f'videos/{name}.mp4'));count=0
                        while True:
                            ok,_=cap.read()
                            if not ok:break
                            count+=1
                        cap.release();video_counts[name]=count
                        if count!=updates:raise RuntimeError(f'Video count mismatch {name}: {count} != {updates}')
                    arrays={k:np.asarray([s[k] for s in states]) for k in ('joint_state','cloth_points','body_pose','grasp_owner','sim_time','frame_id')}
                    arrays['action']=np.asarray(actions,dtype=np.float32);arrays['canonical_policy_action']=np.asarray(predicted_actions,dtype=np.float32)
                    np.savez_compressed(output/'trajectory.npz',**arrays)
                    evaluation={'status':'completed','task_success':response['task_success'],'metrics':response['metrics'],'checks':response['checks'],
                        'duration_s':state['sim_time'],'requested_duration_s':a.duration,'seed':a.seed,'policy_steps':len(actions),
                        'goal':a.goal,'initialization':meta.get('initialization'),
                        'final_both_held_s':response['final_both_held_s'],'max_both_held_s':response['max_both_held_s'],
                        'captured_states':updates,'video_counts':video_counts,'expert_calls':response['expert_calls'],
                        'expert_guards_enabled':response['expert_guards_enabled'],'ever_grasped':response['ever_grasped'],
                        'grasp_events':response['grasp_events'],'max_bound_particles':response['max_bound_particles'],
                        'joint_target_clips':response['joint_target_clips'],'inference_mean_ms':float(np.mean(inference_times)),
                        'inference_p95_ms':float(np.percentile(inference_times,95)),'wall_time_s':time.monotonic()-started,
                        'fold_checkpoint':str(a.fold_checkpoint) if a.fold_checkpoint else None,
                        'switch_step':a.switch_step,
                        'checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                        'physics_steps_per_action':2,'control_hz':30,'n_action_steps':a.action_steps,'ui':not a.headless,
                        'max_camera_position_error_m':visual.max_camera_error,'max_link_pose_component_error':visual.max_pose_error,
                        'limitations':['single fixed initial scene; kinematic robot and local assisted grasp',
                            'synchronized simulation time; rendering may be slower than wall-clock real time']}
                    if evaluation['expert_calls']!=0:raise AssertionError('Expert guard count nonzero')
                    (output/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
                    if labels:labels['outcome'].text='Task PASSED' if evaluation['task_success'] else 'Task FAILED - see evaluation.json'
                    completed=True
                    if not a.headless:
                        import omni.renderer_capture
                        omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(output/'ui_completed.png'))
                    print('ACT EVALUATION COMPLETE:',json.dumps(evaluation),flush=True)
                    if a.headless or a.exit_on_done:break
                if flags['playing'] and not completed and pending is None:
                    request_id+=1
                    pending=executor.submit(infer_and_advance,request_id,state,last_rgbs,refs)
                app.update();time.sleep(.002)
        except BaseException:
            error=traceback.format_exc();traceback.print_exc()
        finally:
            for writer in writers.values():writer.release()
            if executor:executor.shutdown(wait=True,cancel_futures=True)
            for conn in connections:
                try:
                    send_message(conn,{'command':'quit','request_id':request_id+1})
                    receive_message(conn)
                except (OSError,EOFError,ValueError):pass
                conn.close()
            for process in processes:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            final={'status':'error' if error else ('completed' if completed else 'aborted'),'error':error,
                'task_success':evaluation['task_success'] if evaluation else None,'policy_steps':len(actions),
                'all_workers_stopped':all(v.poll() is not None for v in processes),'wall_time_s':time.monotonic()-started}
            (output/'result.json').write_text(json.dumps(final,indent=2))
            print('ACT LIVE FINISHED:',json.dumps(final),flush=True)
            if app:app.close()
    return 1 if error else (0 if completed and evaluation['task_success'] else 2)

if __name__=='__main__':raise SystemExit(main())
