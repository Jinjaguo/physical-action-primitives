"""Exchange retained source conditions between paired recipients and compare against a context-only control."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from physical_action_primitives.construction.response import constrain,restore,command,snap,rotate,yaw,numpy
from physical_action_primitives.construction.train import identify
from physical_action_primitives.construction.interfaces import LawInterface,ResponseParameters,choose_intervention,RESPONSE_SCALE
from physical_action_primitives.source_exchange.context_only import ContextOnlyLaw
from physical_action_primitives.simulation import paired_damping_task
from physical_action_primitives.paths import resolve

def main(a):
    torch.set_num_threads(4)
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    source=root/'evaluation/RollBall-v1';meta=json.loads((source/'COMPLETE.json').read_text());s=meta['spec']
    d=dict(np.load(root/'features_evaluation/RollBall-v1.npz'));d['src']=d['source_indices'];grid=d['xi'].copy();n=len(d['h'])
    d['xi']=np.broadcast_to((grid*s['perturb']+[0.,s['nominal']])/.01,(n,25,2)).astype(np.float32)
    env=gym.make(meta['runtime_environment'],damping_low=meta['damping'][0],damping_high=meta['damping'][1],num_envs=n,obs_mode='state_dict',reward_mode='none',control_mode='pd_ee_delta_pos',sim_backend='gpu',render_mode='rgb_array',robot_init_qpos_noise=0,max_episode_steps=10000)
    records=[]
    try:
        env.reset(seed=meta['seed'],options={'reconfigure':True});b=env.unwrapped;obj=b.ball;constrain(b);device=b.device
        parent=torch.load(source/'source_parent_state.pt',map_location=device,weights_only=False)
        ck=torch.load(a.affine,map_location='cpu',weights_only=False);m=LawInterface(ck['context_dim']).to(device);m.load_state_dict(ck['state']);m.eval()
        zd={**d,'h':(d['h']-ck['h_mean'])/ck['h_std']}
        with torch.no_grad():(u,_),h=identify(m,zd,np.arange(n),device)
        no=torch.load(a.no_identity,map_location='cpu',weights_only=False);nm=ContextOnlyLaw(no['context_dim'],no['width']).to(device);nm.load_state_dict(no['state']);nm.eval()
        nh=torch.as_tensor((d['h']-no['h_mean'])/no['h_std'],device=device)
        x=np.c_[np.ones(7),d['xi'][0,d['src']]];coef=np.einsum('ip,npr->nir',np.linalg.pinv(x),d['y'][:,d['src']])
        lp=ResponseParameters(torch.tensor(coef[:,0],device=device,dtype=torch.float32),torch.tensor(coef[:,1:].transpose(0,2,1),device=device,dtype=torch.float32),torch.ones(n,device=device))
        with torch.no_grad():params={'correct_u':m.response_parameters(u,h),'swapped_u':m.response_parameters(u[torch.arange(n,device=device)^1],h),'no_identity':nm.response_parameters(nh),'source_linear':lp}
        candidates=torch.tensor([(x,y) for x in np.linspace(-1.25,1.25,9) for y in np.linspace(-1.25,1.25,9)],device=device,dtype=torch.float32).expand(n,-1,-1)
        physical=(candidates*s['perturb']+torch.tensor([0.,s['nominal']],device=device))/.01
        direction=torch.tensor(d['direction'],device=device);scale=torch.tensor(RESPONSE_SCALE,device=device)
        np.savez_compressed(out/'identities.npz',u=numpy(u),context=d['h'],source_linear_coefficients=coef,pair=np.arange(n)//2,condition=np.arange(n)%2)
        predictions={name:numpy(p.predict(torch.tensor(d['xi'],device=device))) for name,p in params.items()}
        np.savez_compressed(out/'held_response_predictions.npz',truth=d['y'],source_indices=d['src'],**predictions)
        for name,p in params.items():
            for amplitude in [.006,.009,.012]:
                alpha=torch.tensor([0.,amplitude,0.,0.,0.,0.],device=device).expand(n,-1)/scale
                with torch.no_grad():chosen=choose_intervention(p,alpha,physical)
                restore(b,parent);p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone();tcp=b.agent.tcp.pose.p.clone();travel=rotate(chosen*.01,direction)
                trace=[snap(b,obj)];actions=[];joints=[]
                for t in range(16):
                    desired=tcp.clone();desired[:,:2]+=travel*min((t+1)/meta['moving_steps'],1.)
                    action=command(b,desired);env.step(action);actions.append(numpy(action).copy());joints.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy());trace.append(snap(b,obj))
                delta=rotate(obj.pose.p[:,:2]-p0[:,:2],-direction);dyaw=torch.atan2(torch.sin(yaw(obj.pose.q)-y0),torch.cos(yaw(obj.pose.q)-y0));vel=rotate(obj.linear_velocity[:,:2],-direction)
                actual=numpy(torch.cat([delta,dyaw[:,None],vel,obj.angular_velocity[:,2:3]],-1));target=numpy(alpha*scale);error=np.linalg.norm(actual[:,:2]-target[:,:2],axis=-1)
                jt=np.stack(joints);limits=numpy(b.agent.robot.get_qlimits())[:,:7];valid=((jt>=limits[None,:,:,0]-1e-5)&(jt<=limits[None,:,:,1]+1e-5)).all((0,2))&np.isfinite(actual).all(-1)&np.isfinite(jt).all((0,2))
                success=(error<=.003)&valid&(d['stage']==3);tag=f'{name}_{amplitude:.3f}.npz'
                np.savez_compressed(out/tag,actual=actual,target=target,error_m=error,success=success,selected_physical_xi=numpy(chosen),action=np.stack(actions),joint_target=jt,**{key:np.stack([r[key] for r in trace]) for key in trace[0]})
                for i in range(n):records.append(dict(method=name,amplitude_m=amplitude,pair=i//2,condition=i%2,damping=meta['damping'][i%2],lane=i,error_m=float(error[i]),success=bool(success[i]),achieved_m=float(actual[i,1]),terminal_speed_m_s=float(np.linalg.norm(actual[i,3:5])),trajectory=tag))
                print(json.dumps(dict(method=name,amplitude=amplitude,success=int(success.sum()),mean_error_mm=float(error.mean()*1000))),flush=True)
        (out/'calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'RESULTS.json').write_text(json.dumps(dict(calls=len(records),records=records,context_pair_max_abs=float(np.abs(d['h'][::2]-d['h'][1::2]).max()),identity_pair_mean_l2=float(np.linalg.norm(numpy(u)[::2]-numpy(u)[1::2],axis=-1).mean()),source_probes=7,call_steps=16,candidates=81,success_definition='position error <=3mm; finite and legal arm targets; prepared contact',status='COMPLETE'),indent=2))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(resolve('data','source_exchange')),help='directory holding evaluation/RollBall-v1 and features_evaluation/RollBall-v1.npz');p.add_argument('--output',default=str(resolve('results','source_exchange')));p.add_argument('--affine',default=str(resolve('checkpoints','pap','affine','model.pt')));p.add_argument('--no-identity',default=str(resolve('checkpoints','source_exchange','context_only','model.pt')));main(p.parse_args())
