"""Simulated tests of explicit action-interface operations: execution substitution, calibration-based repair, sequential invocation."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from physical_action_primitives.construction.response import constrain, restore, snap, command, rotate, yaw, numpy
from physical_action_primitives.construction.train import CLASSES, identify, predictions
from physical_action_primitives.construction.interfaces import RESPONSE_SCALE, choose_intervention
from physical_action_primitives.paths import resolve

TASKS=['PushT-v1','PokeCube-v1','RollBall-v1']

def context(b,obj,s,direction):
    p=snap(b,obj)['object'];tcp=numpy(b.agent.tcp.pose.p);rel=tcp-p[:,:3]
    xy=numpy(rotate(torch.as_tensor(rel[:,:2],device=b.device),-direction))
    return np.concatenate([xy/.1,rel[:,2:3]/.1,p[:,7:10]/.05,p[:,10:13],np.tile([s['back']/.1,s['tool']/.1,s['height']/.1],(len(p),1))],-1).astype(np.float32)

def execute(env,obj,direction,travel,profile='linear',scale=1.,replay=None):
    b=env.unwrapped;n=len(travel);tcp0=b.agent.tcp.pose.p.clone();tr=[snap(b,obj)];acts=[];joints=[];requested=[]
    world=rotate(travel,direction)
    gain=torch.as_tensor(scale,device=b.device,dtype=torch.float32)
    if gain.ndim:gain=gain[:,None]
    for t in range(16):
        z=min((t+1)/12.,1.)
        if profile=='smoothstep':z=z*z*(3-2*z)
        if profile=='front_loaded':z=np.sqrt(z)
        desired=tcp0.clone();desired[:,:2]+=world*z*gain
        action=command(b,desired) if replay is None else torch.as_tensor(replay[t],device=b.device)
        env.step(action);tr.append(snap(b,obj));acts.append(numpy(action).copy());requested.append(numpy(desired).copy());joints.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy())
    return dict(**{k:np.stack([x[k] for x in tr]) for k in tr[0]},action=np.stack(acts),joint_target=np.stack(joints),desired_tcp=np.stack(requested),requested_travel=numpy(travel))

def prepare(env,obj,s,direction,offset):
    b=env.unwrapped;n=len(direction);local=torch.stack([offset,torch.full_like(offset,-s['back']-s['tool']-.009)],-1)
    behind=obj.pose.p[:,:2].clone()+rotate(local,direction);stage=torch.zeros(n,device=b.device,dtype=torch.long);tr=[snap(b,obj)];acts=[]
    for t in range(180):
        tcp=b.agent.tcp.pose.p
        stage=torch.where((stage==0)&(tcp[:,2]>.13),1,stage)
        stage=torch.where((stage==1)&(torch.linalg.vector_norm(tcp[:,:2]-behind,dim=-1)<.003),2,stage)
        stage=torch.where((stage==2)&((tcp[:,2]-s['height']).abs()<.004),3,stage)
        desired=torch.cat([behind,torch.full((n,1),.15,device=b.device)],-1);desired[stage==0,:2]=tcp[stage==0,:2];desired[stage>=2,2]=s['height']
        action=command(b,desired);env.step(action);acts.append(numpy(action).copy());tr.append(snap(b,obj))
    return dict(**{k:np.stack([x[k] for x in tr]) for k in tr[0]},action=np.stack(acts),stage=numpy(stage))

def outcome(b,obj,tr,direction,target,p0,y0):
    delta=rotate(obj.pose.p[:,:2]-p0[:,:2],-direction);angle=torch.atan2(torch.sin(yaw(obj.pose.q)-y0),torch.cos(yaw(obj.pose.q)-y0));vel=rotate(obj.linear_velocity[:,:2],-direction)
    actual=numpy(torch.cat([delta,angle[:,None],vel,obj.angular_velocity[:,2:3]],-1));error=np.linalg.norm(actual[:,:2]-target[:,:2],axis=-1)
    ye=np.abs(np.arctan2(np.sin(actual[:,2]-target[:,2]),np.cos(actual[:,2]-target[:,2])));speed=np.linalg.norm(actual[:,3:5],axis=-1)
    limits=numpy(b.agent.robot.get_qlimits())[:,:7];jt=tr['joint_target'];valid=((jt>=limits[None,:,:,0]-1e-5)&(jt<=limits[None,:,:,1]+1e-5)).all((0,2))
    success=(error<=.005)&(ye<=np.deg2rad(5))&(speed<=.05)&valid
    return dict(actual=actual,target=target,success=success,error_m=error,yaw_error_rad=ye,terminal_speed_m_s=speed,valid_joint_targets=valid)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(resolve('data','query_invoke','confirmation')),help='directory of <task>/ recipient data');ap.add_argument('--donor-root',default=str(resolve('data','query_invoke')),help='directory holding development/ and features_development/ donor data');ap.add_argument('--output',default=str(resolve('results','execution')));ap.add_argument('--models',default=str(resolve('checkpoints','pap')),help='directory of <construction>/model.pt');a=ap.parse_args()
    torch.set_num_threads(4)
    root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[];started=time.time()
    for task in TASKS:
        source=root/task;meta=json.loads((source/'COMPLETE.json').read_text());s=meta['spec'];n=meta['source_states'];br=np.load(source/'branches.npz')
        env=gym.make(task,num_envs=n,obs_mode='state_dict',reward_mode='none',control_mode='pd_ee_delta_pos',sim_backend='gpu',render_mode='rgb_array',robot_init_qpos_noise=0,max_episode_steps=10000)
        try:
            env.reset(seed=meta['seed'],options={'reconfigure':True});b=env.unwrapped;obj=getattr(b,s['actor']);constrain(b);parent=torch.load(source/'source_parent_state.pt',map_location=b.device,weights_only=False);restore(b,parent)
            direction=torch.as_tensor(br['direction'],device=b.device);offset=torch.as_tensor(br['offset'],device=b.device);h0=context(b,obj,s,direction)
            donorroot=Path(a.donor_root);d=dict(np.load(donorroot/'features_development'/f'{task}.npz'));d['src']=d['source_indices'];d['xi']=np.broadcast_to((d['xi']*s['perturb']+[0.,s['nominal']])/.01,(len(d['h']),25,2)).astype(np.float32)
            donor=((h0[:,None]-d['h'][None,:2])**2).sum(-1).argmin(-1);models={}
            for name in ['affine','endpoint']:
                ck=torch.load(Path(a.models)/name/'model.pt',map_location='cpu',weights_only=False);m=CLASSES[ck['method']](ck['context_dim']).to(b.device);m.load_state_dict(ck['state']);m.eval();m.requires_grad_(False)
                with torch.no_grad():(u,_),_=identify(m,{**d,'h':(d['h']-ck['h_mean'])/ck['h_std']},donor,b.device)
                models[name]=(m,ck,u)
            np.savez(out/f'{task}_identities.npz',donor=donor,initial_context=h0,**{k:numpy(v[2]) for k,v in models.items()})
            grid=torch.tensor([(x,y) for x in np.linspace(-2.5,2.5,9) for y in np.linspace(-2.5,2.5,9)],device=b.device,dtype=torch.float32);candidates=(grid*s['perturb']+torch.tensor([0.,s['nominal']],device=b.device))/.01;candidates=candidates.expand(n,-1,-1)
            def select(name,target):
                m,ck,u=models[name];h=torch.as_tensor((context(b,obj,s,direction)-ck['h_mean'])/ck['h_std'],device=b.device);alpha=torch.as_tensor(target/np.asarray(RESPONSE_SCALE),device=b.device,dtype=torch.float32)
                with torch.no_grad():
                    if name=='affine':
                        params=m.response_parameters(u,h);xi=choose_intervention(params,alpha,candidates);pred=params.predict(xi[:,None])[:,0];extra={'response_bias':numpy(params.bias),'response_jacobian':numpy(params.jacobian)}
                    else:
                        allpred=predictions(m,'endpoint',u,h,candidates);ix=(allpred-alpha[:,None]).square().sum(-1).argmin(-1);xi=candidates[torch.arange(n),ix];pred=allpred[torch.arange(n),ix];extra={}
                return xi*.01,dict(predicted_response=numpy(pred)*np.asarray(RESPONSE_SCALE),public_context=context(b,obj,s,direction),**extra)
            def save(family,method,condition,goal,tr,result,**extra):
                tag=f'{task}_{family}_{method}_{condition}_{goal}';np.savez_compressed(out/(tag+'.npz'),**tr,**result,**extra)
                rec=dict(environment=task,family=family,method=method,condition=condition,goal=goal,success=int(result['success'].sum()),total=n,mean_error_mm=float(result['error_m'].mean()*1000),artifact=tag+'.npz');records.append(rec);print(json.dumps(rec),flush=True)
            replay=np.load(donorroot/'development'/task/'branches.npz')['action'][12][:,donor]
            goals=[(-.003,.012),(0.,.018),(.003,.021)]
            for g,(dx,dy) in enumerate(goals):
                target=np.tile([dx,dy,0.,0.,0.,0.],(n,1)).astype(np.float32)
                for name in models:
                    restore(b,parent);travel,extra=select(name,target)
                    for profile in ['linear','smoothstep','front_loaded']:
                        restore(b,parent);p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone();tr=execute(env,obj,direction,travel,profile);result=outcome(b,obj,tr,direction,target,p0,y0);save('substitution',name,profile,g,tr,result,**extra)
                restore(b,parent);p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone();tr=execute(env,obj,direction,torch.zeros((n,2),device=b.device),replay=replay);save('substitution','source_replay','nominal',g,tr,outcome(b,obj,tr,direction,target,p0,y0))
            # Two short strokes separate displacement scale from tracking offset.
            restore(b,parent);caltravel=torch.tensor([0.,.020],device=b.device).expand(n,-1);cal=execute(env,obj,direction,caltravel,scale=.55)
            tracked=rotate(torch.as_tensor(cal['tcp'][-1,:,:2]-cal['tcp'][0,:,:2],device=b.device),-direction)
            restore(b,parent);cal2=execute(env,obj,direction,torch.tensor([0.,.030],device=b.device).expand(n,-1),scale=.55)
            tracked2=rotate(torch.as_tensor(cal2['tcp'][-1,:,:2]-cal2['tcp'][0,:,:2],device=b.device),-direction)
            estimate=((tracked2[:,1]-tracked[:,1])/.010).clamp(.2,1.5)
            np.savez_compressed(out/f'{task}_calibration.npz',**cal,estimated_scale=numpy(estimate),injected_scale=np.asarray(.55),tracked_second=numpy(tracked2))
            np.savez_compressed(out/f'{task}_calibration_second.npz',**cal2)
            for g,(dx,dy) in enumerate(goals):
                target=np.tile([dx,dy,0.,0.,0.,0.],(n,1)).astype(np.float32)
                for name in models:
                    restore(b,parent);travel,extra=select(name,target)
                    for label,scale,budget in [('normal',1.,0),('mismatch',.55,0),('retry',.55,32),('interface_repair',.55/estimate,32),('fixed_gain_adjustment',.55*1.5,32)]:
                        restore(b,parent);p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone();tr=execute(env,obj,direction,travel,scale=scale);save('debugging',name,label,g,tr,outcome(b,obj,tr,direction,target,p0,y0),calibration_steps=np.asarray(budget),estimated_scale=numpy(estimate),**extra)
            for name in ['affine','endpoint','repeat_first_intervention']:
                restore(b,parent);origin=obj.pose.p.clone();origin_yaw=yaw(obj.pose.q).clone();chain=[];all_success=np.ones(n,bool);first=None
                for stage,(dx,dy) in enumerate([(0.,.009),(.003,.018),(0.,.027)]):
                    if stage:
                        prep=prepare(env,obj,s,direction,offset);np.savez_compressed(out/f'{task}_sequence_{name}_preparation{stage}.npz',**prep)
                    absolute=np.tile([dx,dy,0.,0.,0.,0.],(n,1)).astype(np.float32)
                    current=numpy(rotate(obj.pose.p[:,:2]-origin[:,:2],-direction));target=absolute.copy();target[:,:2]-=current;target[:,2]=-numpy(torch.atan2(torch.sin(yaw(obj.pose.q)-origin_yaw),torch.cos(yaw(obj.pose.q)-origin_yaw)))
                    if stage==0 or name!='repeat_first_intervention':
                        travel,extra=select('affine' if name=='repeat_first_intervention' else name,target)
                        if stage==0:first=travel.clone()
                    else:travel=first;extra={}
                    tr=execute(env,obj,direction,travel);result=outcome(b,obj,tr,direction,absolute,origin,origin_yaw);all_success&=result['success'];chain.append(result['error_m']);save('sequence',name,'stage',stage,tr,result,total_control_steps=np.asarray((stage+1)*196),**extra)
                np.savez_compressed(out/f'{task}_sequence_{name}_chain.npz',success=all_success,stage_error_m=np.stack(chain));records.append(dict(environment=task,family='sequence_complete',method=name,condition='three_waypoints',goal=0,success=int(all_success.sum()),total=n,mean_error_mm=float(chain[-1].mean()*1000),artifact=f'{task}_sequence_{name}_chain.npz'))
        finally:env.close()
    payload=dict(records=records,seconds=time.time()-started,backbone_updates=0,head_updates=0,source_identity='first two development packets, public initial-context nearest, fixed through every operation',success='position<=5mm; yaw<=5deg; terminal speed<=.05m/s; legal call joint targets',debugging='known-location, two-stroke scale calibration; paired reset; recovery includes32 calibration steps',sequence='no state reset between stages; shared180-step preparation per call;588 control steps including initial preparation')
    (out/'RESULTS.json').write_text(json.dumps(payload,indent=2));print(json.dumps({'status':'COMPLETE','seconds':payload['seconds']}),flush=True)

if __name__=='__main__':main()
