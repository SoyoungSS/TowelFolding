"""Physics-only ACT worker: explicit commands, guarded against expert use."""
import argparse,json,socket,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
from bridge_transport import send_message,receive_message
from newton_env import NewtonTowelEnv,JOINT_ORDER
from act_io import internal_action,measure_fold,fold_verdict,grasp_verdict


def serve(connection,env,seed,duration,initial_plan=None,goal='fold'):
    env.reset(seed)
    initialization = None
    if initial_plan:
        from recovery_collection import reproduce_prefix
        with np.load(initial_plan, allow_pickle=False) as plan:
            _, joint_error, cloth_error = reproduce_prefix(
                env, plan['prefix'], plan['joint_state'], plan['cloth_points'])
        initialization = dict(plan=str(initial_plan.resolve()), joint_error=joint_error, cloth_error=cloth_error)
    origin_frame = env.frame_id
    origin_time = env.scene.sim_time
    expert_calls=0
    def forbidden(*unused,**kwargs):
        nonlocal expert_calls
        expert_calls+=1
        raise AssertionError('Expert/planner function invoked during ACT rollout')
    env.start_expert=forbidden;env.expert_action=forbidden
    for name in ('update','attach','move','start_fold','enter'):
        setattr(env.scene.fold,name,forbidden)
    bodies=[env.scene.model.body_label.index(f'dual_arm/arm_{s}_link_gripper') for s in ('l','r')]
    ever=[False,False];previous=[False,False];events=[0,0];max_bound=0
    calls=0;last_request=0;terminal=False;internal=None;clipped_joints=0
    held_frames = max_held_frames = 0
    def observation():
        state = env.get_state()
        state['frame_id'] -= origin_frame
        state['sim_time'] -= origin_time
        return state
    def reply(request_id):
        state=observation()
        state['phase']='act_finished' if terminal else 'act_running'
        owners=np.asarray(state['grasp_owner'])
        metrics=measure_fold(state['cloth_points'],env.scene.config)
        success,checks=fold_verdict(metrics,env.scene.config,ever,np.all(owners<0))
        if goal == 'grasp':
            success, checks = grasp_verdict(held_frames)
        return {'type':'state','request_id':request_id,'state':state,'terminal':terminal,
                'metrics':metrics,'task_success':bool(terminal and success),'checks':checks,
                'expert_calls':expert_calls,'expert_guards_enabled':True,'action':internal,
                'policy_steps':calls,'ever_grasped':ever,'grasp_events':events,'max_bound_particles':max_bound,
                'joint_target_clips':clipped_joints,'goal':goal,
                'final_both_held_s':held_frames / 60, 'max_both_held_s':max_held_frames / 60}
    ready=reply(0);ready['metadata']=env.metadata();ready['metadata']['controller']='ACT explicit commands; expert guarded'
    ready['metadata']['initialization'] = initialization
    send_message(connection,ready)
    while True:
        request=receive_message(connection);rid=request['request_id']
        if rid!=last_request+1:raise ValueError('Non-sequential request')
        if request['command']=='quit':
            send_message(connection,{'type':'bye','request_id':rid});return
        state=observation()
        if (request['episode_id'],request['frame_id'])!=(state['episode_id'],state['frame_id']):raise ValueError('Wrong physics frame acknowledgement')
        if request['command']!='advance' or terminal:raise ValueError('Only explicit advance is supported before terminal')
        count=request.get('steps',2)
        if count!=2:raise ValueError('ACT evaluation requires 30Hz targets and 60Hz physics')
        internal=internal_action(request['policy_action'])
        for i,name in enumerate(JOINT_ORDER[:14]):
            low,high=env.scene.limits[name]
            bounded=np.clip(internal[i],low,high)
            clipped_joints+=int(bounded!=internal[i]);internal[i]=bounded
        for _ in range(count):
            state=env.step(internal)
            owners=np.asarray(state['grasp_owner'])
            held_frames = held_frames + 1 if all(np.any(owners == body) for body in bodies) else 0
            max_held_frames = max(max_held_frames, held_frames)
            max_bound=max(max_bound,int((owners>=0).sum()))
            for i,body in enumerate(bodies):
                grasped=bool(np.any(owners==body))
                events[i]+=int(grasped and not previous[i]);previous[i]=grasped;ever[i]|=grasped
        calls+=1;terminal=state['sim_time']-origin_time>=duration-1e-8;last_request=rid
        send_message(connection,reply(rid))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--initial-plan', type=Path)
    p.add_argument('--goal', choices=['fold', 'grasp'], default='fold')
    a=p.parse_args();env=None
    if not np.isfinite(a.duration) or a.duration<=0:p.error('duration must be positive')
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(a.socket));listener.listen(1)
            with listener.accept()[0] as connection:
                env=NewtonTowelEnv();serve(connection,env,a.seed,a.duration,a.initial_plan,a.goal)
    except EOFError:print('ACT physics peer disconnected',flush=True)
    finally:
        if env:env.close()
        a.socket.unlink(missing_ok=True)

if __name__=='__main__':main()
