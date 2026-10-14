"""Execute selected interventions on restored recipients; cache lookups are never scored as calls."""
import argparse,json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from physical_action_primitives.construction.response import constrain,restore,command,snap,rotate,yaw,numpy
from physical_action_primitives.construction.train import CLASSES,identify,predictions
from physical_action_primitives.construction.interfaces import ResponseParameters,choose_intervention,RESPONSE_SCALE
from physical_action_primitives.paths import resolve

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(resolve('data','query_invoke')));ap.add_argument('--models',default=str(resolve('checkpoints','pap')),help='directory of <construction>/model.pt');ap.add_argument('--output',default=str(resolve('results','construction','calls')));ap.add_argument('--cohort',default='development');ap.add_argument('--model-names',default='');ap.add_argument('--extended-calls',action='store_true');ap.add_argument('--donor-environment',default='');ap.add_argument('--donor-cohort',default='training');a=ap.parse_args()
    torch.set_num_threads(4)
    root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
    model_paths=sorted(Path(a.models).glob('*/model.pt'))
    if a.model_names:model_paths=[p for p in model_paths if p.parent.name in a.model_names.split(',')]
    bound=2.5 if a.extended_calls else 1.25
    grid=np.array([(x,y) for x in np.linspace(-bound,bound,9) for y in np.linspace(-bound,bound,9)],np.float32)
    for task in ['PushT-v1','PokeCube-v1','RollBall-v1']:
        source_dir=root/a.cohort/task;meta=json.loads((source_dir/'COMPLETE.json').read_text());s=meta['spec'];d=dict(np.load(root/f'features_{a.cohort}'/f'{task}.npz'));d['src']=d['source_indices']
        donor_task=a.donor_environment or task
        training=dict(np.load(root/f'features_{a.donor_cohort}'/f'{donor_task}.npz'));training['src']=training['source_indices']
        donor_spec=json.loads((root/a.donor_cohort/donor_task/'COMPLETE.json').read_text())['spec']
        for z,zspec in [(d,s),(training,donor_spec)]:
            z['grid']=z['xi'].copy()
            z['xi']=np.broadcast_to((z['grid']*zspec['perturb']+[0.,zspec['nominal']])/.01,(len(z['h']),25,2)).astype(np.float32)
        n=len(d['h']);env=gym.make(task,num_envs=n,obs_mode='state_dict',reward_mode='none',control_mode='pd_ee_delta_pos',sim_backend='gpu',render_mode='rgb_array',robot_init_qpos_noise=0,max_episode_steps=10000)
        try:
            env.reset(seed=meta['seed'],options={'reconfigure':True});b=env.unwrapped;obj=getattr(b,s['actor']);constrain(b)
            parent=torch.load(source_dir/'source_parent_state.pt',map_location=b.device,weights_only=False)
            direction=torch.tensor(d['direction'],device=b.device)
            candidates=torch.tensor(grid,device=b.device).expand(n,-1,-1)
            physical_candidates=(candidates*s['perturb']+torch.tensor([0.,s['nominal']],device=b.device))/.01
            # Donor selection sees geometry/current state only, never recipient responses.
            distance=((d['h'][:,None]-training['h'][None,:2])**2).sum(-1)
            donor=distance.argmin(-1)
            np.save(out/f'{task}_donors.npy',donor)
            methods=[]
            for path in model_paths:
                ck=torch.load(path,map_location='cpu',weights_only=False);m=CLASSES[ck['method']](ck['context_dim']).to(b.device);m.load_state_dict(ck['state']);m.eval()
                for mode in (['fixed_identity_new_recipient'] if a.donor_environment else ['new_source','fixed_identity_new_recipient']):
                    z=d if mode=='new_source' else training;ix=np.arange(n) if mode=='new_source' else donor
                    zd={**z,'h':(z['h']-ck['h_mean'])/ck['h_std']}
                    with torch.no_grad():(u,_),_=identify(m,zd,ix,b.device,probes=ck.get('probes',7))
                    h=torch.as_tensor((d['h']-ck['h_mean'])/ck['h_std'],device=b.device)
                    methods.append((path.parent.name,mode,m,ck['method'],u,h))
            replay=np.load(root/a.donor_cohort/donor_task/'branches.npz')['action'][12][:,donor]
            if not a.donor_environment:
                methods.extend([('oracle_response','new_source',None,'oracle',None,None),('seven_probe_linear','new_source',None,'linear7',None,None),('fixed_source_replay','fixed_identity_new_recipient',None,'replay',None,None),('no_op','new_source',None,'zero',None,None)])
            for name,mode,m,kind,u,h in methods:
                # Shared target calls: magnitude and lateral direction varied independently of u.
                magnitudes=[.015,.018,.021] if a.extended_calls else [.006,.009,.012]
                for goal,(dx,dy) in enumerate([(x,y) for y in magnitudes for x in [-.003,0.,.003]]):
                    alpha=torch.tensor([dx,dy,0.,0.,0.,0.],device=b.device).expand(n,-1)/torch.tensor(RESPONSE_SCALE,device=b.device)
                    with torch.no_grad():
                        if kind=='policy':selected=m.invoke(u,h,alpha)
                        elif kind in ['oracle','linear7']:
                            ids=np.arange(25) if kind=='oracle' else d['src'];x=np.c_[np.ones(len(ids)),d['grid'][ids]]
                            coef=np.einsum('ip,npr->nir',np.linalg.pinv(x),d['y'][:,ids])
                            params=ResponseParameters(torch.tensor(coef[:,0],device=b.device,dtype=torch.float32),torch.tensor(coef[:,1:].transpose(0,2,1),device=b.device,dtype=torch.float32),torch.ones(n,device=b.device))
                            selected=choose_intervention(params,alpha,candidates)
                        elif kind in ['zero','replay']:selected=torch.zeros((n,2),device=b.device)
                        elif kind=='affine':
                            physical=choose_intervention(m.response_parameters(u,h),alpha,physical_candidates)
                            selected=(physical*.01-torch.tensor([0.,s['nominal']],device=b.device))/s['perturb']
                        else:
                            pred=predictions(m,kind,u,h,physical_candidates);index=(pred-alpha[:,None]).square().sum(-1).argmin(-1);selected=candidates[torch.arange(n),index]
                    restore(b,parent);p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone();tcp0=b.agent.tcp.pose.p.clone()
                    travel=rotate(selected*s['perturb']+torch.tensor([0.,s['nominal']],device=b.device),direction)
                    trace=[snap(b,obj)];acts=[];joints=[]
                    for t in range(meta['steps']):
                        desired=tcp0.clone();desired[:,:2]+=travel*min((t+1)/(meta['steps']-4),1.)
                        action=command(b,desired)
                        if kind=='zero':
                            action.zero_()
                            if action.shape[-1]>3:action[:,3]=-1
                        if kind=='replay':action=torch.as_tensor(replay[t],device=b.device)
                        env.step(action);trace.append(snap(b,obj));acts.append(numpy(action).copy());joints.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy())
                    delta=rotate(obj.pose.p[:,:2]-p0[:,:2],-direction);dyaw=torch.atan2(torch.sin(yaw(obj.pose.q)-y0),torch.cos(yaw(obj.pose.q)-y0));vel=rotate(obj.linear_velocity[:,:2],-direction)
                    actual=numpy(torch.cat([delta,dyaw[:,None],vel,obj.angular_velocity[:,2:3]],-1));target=numpy(alpha)*np.asarray(RESPONSE_SCALE)
                    error=np.linalg.norm(actual[:,:2]-target[:,:2],axis=-1);ye=np.abs(np.arctan2(np.sin(actual[:,2]),np.cos(actual[:,2])));ve=np.linalg.norm(actual[:,3:5],axis=-1)
                    jt=np.stack(joints);limits=numpy(b.agent.robot.get_qlimits())[:,:7];valid=((jt>=limits[None,:,:,0]-1e-5)&(jt<=limits[None,:,:,1]+1e-5)).all((0,2))
                    success=(error<=.005)&(ye<=np.deg2rad(5))&(ve<=.05)&valid
                    tag=f'{task}_{name}_{mode}_goal{goal}'
                    np.savez_compressed(out/f'{tag}.npz',actual=actual,target=target,success=success,error_m=error,yaw_error_rad=ye,terminal_speed_m_s=ve,selected_xi=numpy(selected),action=np.stack(acts),joint_target=jt,**{key:np.stack([r[key] for r in trace]) for key in trace[0]})
                    record=dict(environment=task,source_environment=donor_task if mode=='fixed_identity_new_recipient' else task,method=name,mode=mode,goal=goal,dx_m=dx,dy_m=dy,success=int(success.sum()),total=n,mean_error_m=float(error.mean()),valid_joint_targets=int(valid.sum()),artifact=tag+'.npz');records.append(record);print(json.dumps(record),flush=True)
            del methods
        finally:env.close()
    (out/'RESULTS.json').write_text(json.dumps(dict(records=records,cohort=a.cohort,call_family='amplitude_extrapolation' if a.extended_calls else 'local_calls',candidate_bound=bound,action_steps=16,candidates=81,source_probes=7,success='position<=5mm, wrapped yaw<=5deg, terminal speed<=0.05m/s, legal joints',invocation='simulator-executed world targets using the shared controller adapter, full state restored before each call'),indent=2))

if __name__=='__main__':main()
