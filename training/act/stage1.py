"""Standard LeRobot ACT: local-data acceptance, short training and reload test."""
import argparse,copy,json,os,random,time,traceback,importlib.metadata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset', type=Path)
    source.add_argument('--collection-batch', type=Path, help='batch.json from run_collect.sh --episodes')
    parser.add_argument('--recovery-dataset', type=Path, action='append',
                        help='Add one recorded recovery episode while keeping the main dataset statistics')
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--balance-recovery', action='store_true')
    parser.add_argument('--finetune-z0', action='store_true',
                        help='Fine-tune the deployment path with a zero latent and no VAE encoder')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save-every', type=int, default=1000, help='Save deployable intermediate checkpoints; 0 disables')
    parser.add_argument('--learning-rate', type=float, help='Override optimizer learning rate')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Exclude earlier observation windows from training; keep original normalization statistics')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.steps<1 or args.batch_size<1:parser.error('steps和batch-size必须为正数')
    if args.save_every < 0 or (args.learning_rate is not None and not 0 < args.learning_rate < 1):
        parser.error('Invalid save interval or learning rate')
    if args.finetune_z0 and not args.init_checkpoint:
        parser.error('--finetune-z0 requires --init-checkpoint')
    args.batch_sources = []
    if args.collection_batch:
        manifest = args.collection_batch.resolve()
        entries = json.loads(manifest.read_text())['episodes']
        args.batch_sources = [(manifest.parent / row['dataset']).resolve()
                              for row in entries if row['passed']]
        if not args.batch_sources or len(set(args.batch_sources)) != len(args.batch_sources):
            parser.error('Batch must contain successful, distinct datasets')
        args.dataset = args.batch_sources[0]
    if args.dataset is None:
        choices=sorted(ROOT.glob('outputs/collect/*/dataset/SUCCESS.json'),key=lambda p:p.stat().st_mtime)
        # Default runs start at home; recovery-only recordings start at a near-miss.
        def is_home_collection(success_path):
            metadata_path = success_path.parents[1] / 'metadata.json'
            return not metadata_path.exists() or 'recovery_plan' not in json.loads(metadata_path.read_text())
        choices = [p for p in choices if is_home_collection(p)]
        if not choices:raise SystemExit('没有成功采集的数据，请先运行 ./run_collect.sh')
        args.dataset=choices[-1].parent
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output=(args.output or ROOT/f'outputs/act_stage1/{stamp}_{os.getpid()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    print('ACT阶段1输出：',output,flush=True)
    try:
        run(args,output)
    except BaseException:
        (output/'FAILED.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2))
        raise

def run(args,output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_IMAGES
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition,transition_to_policy_action
    from common import CHUNK,HEIGHT,WIDTH,load_dataset,make_config,observations_only,validate_dataset,training_processors
    from prepare_data import prepare
    assert importlib.metadata.version('lerobot')=='0.4.4'
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA不可用，请在宿主GPU环境运行；没有自动回退CPU')
        torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark=False
    versions={p:importlib.metadata.version(p) for p in ['lerobot','torch','torchvision','datasets','numpy','pyarrow','av','torchcodec']}
    versions.update(python=__import__('sys').version,device=args.device,cuda=torch.version.cuda)
    if args.device=='cuda':
        versions['gpu']=torch.cuda.get_device_name()
        versions['gpu_total_gb']=torch.cuda.get_device_properties(0).total_memory/1024**3
    (output/'environment.json').write_text(json.dumps(versions,indent=2))
    # Preserve the raw episode. Compatibility adjustments are made only to this copy.
    data_root=output/'dataset'
    prepare(args.dataset,data_root)
    dataset=load_dataset(data_root)
    summary=validate_dataset(dataset)
    main_datasets = [dataset]
    main_records = [{'source_dataset': str(args.dataset.resolve()), 'frames': len(dataset)}]
    if args.collection_batch:
        reference_meta = json.loads((args.dataset.parent / 'metadata.json').read_text())
        if 'recovery_plan' in reference_meta:
            raise ValueError('The batch must contain full demonstrations')
        for number, path in enumerate(args.batch_sources[1:], start=1):
            meta = json.loads((path.parent / 'metadata.json').read_text())
            config = copy.deepcopy(meta['config'])
            config['towel']['center'] = reference_meta['config']['towel']['center']
            if 'recovery_plan' in meta or (config, meta['urdf_sha256']) != (
                reference_meta['config'], reference_meta['urdf_sha256']
            ):
                raise ValueError('Full demonstrations must share the robot and scene, except towel centre')
            episode_root = output / f'episode_{number:03d}' / 'dataset'
            prepare(path, episode_root)
            episode = load_dataset(episode_root)
            if episode.features != dataset.features:
                raise ValueError('Batch episode feature schemas differ')
            check = validate_dataset(episode)
            main_datasets.append(episode)
            main_records.append({'source_dataset': str(path), 'frames': len(episode), 'check': check})
    windows = []
    for episode in main_datasets:
        if not 0 <= args.start_frame <= len(episode) - CHUNK:
            raise ValueError('Start frame must leave at least one complete action chunk in each episode')
        windows.append(torch.utils.data.Subset(episode, range(args.start_frame, len(episode)))
                       if args.start_frame else episode)
    training_dataset = torch.utils.data.ConcatDataset(windows) if len(windows) > 1 else windows[0]
    summary.update(main_episodes=len(main_datasets), main_datasets=main_records)
    summary.update(training_start_frame=args.start_frame, training_frames=len(training_dataset))
    recovery_runtime = {}
    main_frames = len(training_dataset)
    recovery_datasets = []
    recovery_records = []
    for number, recovery_source in enumerate(args.recovery_dataset or []):
        source_metadata = json.loads((args.dataset.parent / 'metadata.json').read_text())
        recovery_metadata = json.loads((recovery_source.parent / 'metadata.json').read_text())
        if 'recovery_plan' not in recovery_metadata or any(
            source_metadata[key] != recovery_metadata[key] for key in ('config', 'urdf_sha256')
        ):
            raise ValueError('Recovery must be recorded in the same robot and camera scene')
        recovery_root = output / f'recovery_{number}' / 'dataset'
        prepare(recovery_source, recovery_root)
        recovery = load_dataset(recovery_root)
        if dataset.features != recovery.features:
            raise ValueError('Recovery and main dataset feature schemas differ')
        recovery_datasets.append(recovery)
        recovery_records.append({
            'source_dataset': str(recovery_source.resolve()), 'training_copy': str(recovery_root),
            'frames': len(recovery), 'check': validate_dataset(recovery),
        })
    if recovery_datasets:
        training_dataset = torch.utils.data.ConcatDataset([training_dataset, *recovery_datasets])
        summary.update(recoveries=recovery_records, training_frames=len(training_dataset))
        first = recovery_records[0]
        recovery_runtime = {
            'recovery_source_dataset': first['source_dataset'],
            'recovery_training_copy': first['training_copy'], 'recovery_frames': first['frames'],
            'recoveries': recovery_records,
        }
    if args.balance_recovery and not recovery_datasets:
        raise ValueError('Balanced sampling requires recovery datasets')
    (output/'dataset_check.json').write_text(json.dumps(summary,indent=2))
    print('官方LeRobotDataset验收通过：',summary,flush=True)
    config=make_config(dataset,args.device)
    print('创建官方ACTPolicy：512维、4层编码器、1层解码器、4层VAE编码器。',flush=True)
    if args.init_checkpoint:
        policy = ACTPolicy.from_pretrained(args.init_checkpoint, local_files_only=True).to(args.device)
        if policy.config != config:
            raise ValueError('Fine-tuning requires an identical ACT configuration')
    else:
        policy=ACTPolicy(config).to(args.device)
    dataset_stats = (aggregate_stats([episode.meta.stats for episode in main_datasets])
                     if len(main_datasets) > 1 else dataset.meta.stats)
    pre,post=training_processors(config,dataset_stats,args.init_checkpoint)
    parameters=sum(p.numel() for p in policy.parameters())
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=args.learning_rate or config.optimizer_lr,weight_decay=config.optimizer_weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.balance_recovery:
        recovery_frames = len(training_dataset) - main_frames
        weights = [0.5 / main_frames] * main_frames + [0.5 / recovery_frames] * recovery_frames
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(training_dataset), generator=generator)
    loader=DataLoader(training_dataset,batch_size=args.batch_size,shuffle=sampler is None,
                      sampler=sampler,num_workers=0,drop_last=False,generator=generator)
    iterator=iter(loader)
    eval_indices=np.linspace(0,len(dataset)-CHUNK,8,dtype=int).tolist()
    # Standard processors are used for normalization, device placement and unnormalization.
    eval_raw=[next(iter(DataLoader(torch.utils.data.Subset(dataset,[i]),batch_size=1))) for i in eval_indices]
    def evaluate():
        policy.eval();errors=[];reference=None;unnormalized=None
        def forbidden(*unused):raise AssertionError('z=0推理不应调用VAE编码器')
        handle=policy.model.vae_encoder.register_forward_pre_hook(forbidden)
        try:
            with torch.no_grad():
                for raw in eval_raw:
                    batch=pre(copy.deepcopy(raw))
                    pred=policy.predict_action_chunk(observations_only(batch))
                    valid=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
                    errors.append(float((pred-batch['action']).abs()[valid].mean()))
                    if reference is None:
                        reference=pred.detach().cpu().clone()
                        unnormalized=post(pred).detach().cpu().clone()
        finally:handle.remove()
        return float(np.mean(errors)),reference,unnormalized
    runtime={'resize':{'height':HEIGHT,'width':WIDTH,'antialias':True,'center_crop':False},
             'camera_order':list(dataset.meta.camera_keys),'state_names':dataset.features['observation.state']['names'],
             'action_names':dataset.features['action']['names'],'fps':dataset.fps,'chunk_size':CHUNK,
             'n_action_steps':config.n_action_steps,'video_backend':'pyav','seed':args.seed,
             'source_dataset':str(args.dataset.resolve()),'training_copy':str(data_root),
             'training_start_frame':args.start_frame,'training_frames':len(training_dataset),
             'collection_batch':str(args.collection_batch.resolve()) if args.collection_batch else None,
             'main_datasets':main_records,
             **recovery_runtime,
             'init_checkpoint':str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
             'normalization_source':str(args.init_checkpoint.resolve()) if args.init_checkpoint else 'training_main_episodes',
             'balanced_recovery':args.balance_recovery,
             'training_objective':'z0_arm_and_gripper_l1' if args.finetune_z0 else 'act_vae',
             'limitations':['fixed-scene training episodes; no independent validation/test or closed-loop control',
                            'joint targets and binary closed flags need conversion before simulation execution']}
    def save_checkpoint(destination, step):
        policy.save_pretrained(destination)
        pre.save_pretrained(destination, config_filename='policy_preprocessor.json')
        post.save_pretrained(destination, config_filename='policy_postprocessor.json')
        (destination / 'runtime.json').write_text(json.dumps(dict(runtime, training_step=step), indent=2))
        torch.save({'step': step, 'optimizer': optimizer.state_dict()}, destination / 'optimizer_state.pt')

    initial_l1,initial_chunk,_=evaluate()
    history=[];started=time.monotonic()
    print(f'参数量={parameters:,}，初始固定训练片段z=0 L1={initial_l1:.6f}',flush=True)
    with (output/'train_metrics.jsonl').open('w') as log:
        for step in range(1,args.steps+1):
            try:raw=next(iterator)
            except StopIteration:iterator=iter(loader);raw=next(iterator)
            batch=pre(raw)
            assert batch['action'].shape[1:]==(CHUNK,16)
            if args.finetune_z0:
                # Evaluation mode selects z=0 and disables dropout. Keep autograd
                # enabled so that the same path used in control can learn.
                policy.eval()
                inputs = observations_only(batch)
                inputs[OBS_IMAGES] = [inputs[key] for key in config.image_features]
                predicted, _ = policy.model(inputs)
                valid = ~batch['action_is_pad']
                errors = (predicted - batch['action']).abs()[valid]
                arm_loss = errors[:, [*range(7), *range(8, 15)]].mean()
                gripper_loss = errors[:, [7, 15]].mean()
                loss = arm_loss + gripper_loss
                parts = {'z0_arm_l1': float(arm_loss.detach()),
                         'z0_gripper_l1': float(gripper_loss.detach())}
            else:
                policy.train()
                loss, parts = policy(batch)
            if not torch.isfinite(loss):raise RuntimeError(f'第{step}步损失非有限值')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            record={'step':step,'loss':float(loss.detach()),'grad_norm':float(grad_norm),
                    **{k:float(v) for k,v in parts.items()},'wall_time_s':time.monotonic()-started}
            if step==1 or step%25==0 or step==args.steps:
                eval_l1,_,_=evaluate();record['fixed_training_z0_l1']=eval_l1
                print(json.dumps(record),flush=True)
            history.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if args.save_every and step % args.save_every == 0 and step < args.steps:
                save_checkpoint(output / f'checkpoint_step_{step:06d}', step)
    final_l1,reference,unnormalized=evaluate()
    policy.eval()
    checkpoint=output/'checkpoint';policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint,config_filename='policy_preprocessor.json')
    post.save_pretrained(checkpoint,config_filename='policy_postprocessor.json')
    (checkpoint/'runtime.json').write_text(json.dumps(runtime,indent=2))
    print('已保存标准模型与处理器，开始重载验收。',flush=True)
    restored=ACTPolicy.from_pretrained(checkpoint,local_files_only=True).to(args.device);restored.eval()
    restored_pre=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_preprocessor.json',local_files_only=True)
    restored_post=PolicyProcessorPipeline.from_pretrained(checkpoint,config_filename='policy_postprocessor.json',local_files_only=True,
        to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
    raw=copy.deepcopy(eval_raw[0])
    batch=restored_pre(raw)
    with torch.no_grad():
        chunk=restored.predict_action_chunk(observations_only(batch))
        real_chunk=restored_post(chunk)
        restored.reset();first=restored.select_action(observations_only(batch));first_real=restored_post(first)
    assert chunk.shape==(1,CHUNK,16) and torch.isfinite(real_chunk).all()
    max_error=float((chunk.cpu()-reference).abs().max())
    torch.testing.assert_close(chunk.cpu(),reference,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(real_chunk.cpu(),unnormalized,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(first.cpu(),chunk[:,0].cpu(),atol=1e-5,rtol=1e-5)
    assert first_real.shape==(1,16)
    assert not torch.allclose(reference,initial_chunk,atol=1e-6),'训练后预测没有变化'
    np.savez_compressed(output/'inference_example.npz',normalized_chunk=chunk.cpu().numpy(),
                        unnormalized_chunk=real_chunk.cpu().numpy(),first_action=first_real.cpu().numpy())
    # Optimizer state is retained to make this run resumable by future tooling.
    torch.save({'step':args.steps,'optimizer':optimizer.state_dict()},output/'optimizer_state.pt')
    result={'status':'passed','phase':'standard ACT stage1, not robot rollout','steps':args.steps,'batch_size':args.batch_size,
            'parameters':parameters,'initial_fixed_training_z0_l1':initial_l1,'final_fixed_training_z0_l1':final_l1,
            'training_error_decreased':final_l1<initial_l1,'checkpoint':str(checkpoint),'reload_max_abs_error':max_error,
            'normalized_output_shape':list(chunk.shape),'single_action_shape':list(first.shape),
            'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3 if args.device=='cuda' else None,
            'training_wall_time_s':time.monotonic()-started,'evaluation':'fixed clips from the training episode only',
            'expert_not_used_in_inference':True,'source_data_unchanged':True}
    if not args.init_checkpoint and not result['training_error_decreased']:raise RuntimeError('短程训练未降低固定片段误差，需要排查；不标记成功')
    # Plotting is independent of GPU and never starts an application window.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    measured=[r for r in history if 'fixed_training_z0_l1' in r]
    ax.plot([0]+[r['step'] for r in measured],[initial_l1]+[r['fixed_training_z0_l1'] for r in measured],marker='o')
    ax.set(xlabel='Training updates',ylabel='Normalized action L1, z=0',title='Standard ACT: fixed training clips (not validation)')
    fig.savefig(output/'training.png',dpi=150);plt.close(fig)
    (output/'SUCCESS.json').write_text(json.dumps(result,indent=2))
    print('ACT STAGE1 PASSED:',json.dumps(result),flush=True)

if __name__=='__main__':main()
