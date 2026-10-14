"""Physical source packets under two damping laws at matched visible states."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from physical_action_primitives.construction.response import SPECS,GRID,SOURCE,constrain,clone_tree,snap,restore,command,rotate,numpy,yaw
from mani_skill.utils.structs import Pose
from physical_action_primitives.simulation import paired_damping_task
from physical_action_primitives.paths import resolve

def main(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(4);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);n=a.parents*2
    s=SPECS['RollBall-v1'];started=time.time()
    env=gym.make('PairedDampingRollBall-v1',damping_low=a.low,damping_high=a.high,num_envs=n,obs_mode='state_dict',reward_mode='none',control_mode='pd_ee_delta_pos',sim_backend='gpu',render_mode='rgb_array',robot_init_qpos_noise=0,max_episode_steps=10000)
    try:
        env.reset(seed=a.seed,options={'reconfigure':True});b=env.unwrapped;obj=b.ball;constrain(b)
        rng=np.random.default_rng(a.seed);centers=np.tile(s['center'],(a.parents,1));centers[:,:2]+=rng.uniform(-.018,.018,(a.parents,2))
        pos=torch.tensor(np.repeat(centers,2,0),device=b.device,dtype=torch.float32)
        obj.set_pose(Pose.create_from_pq(pos));obj.set_linear_velocity(torch.zeros_like(pos));obj.set_angular_velocity(torch.zeros_like(pos))
        b.scene._gpu_apply_all();b.scene._gpu_fetch_all()
        direction=torch.full((n,),s['heading'],device=b.device);offset=torch.zeros(n,device=b.device)
        local=torch.stack([offset,torch.full_like(offset,-s['back']-s['tool']-.009)],-1)
        behind=obj.pose.p[:,:2].clone()+rotate(local,direction)
        stage=torch.zeros(n,device=b.device,dtype=torch.long);preparation=[]
        for t in range(180):
            tcp=b.agent.tcp.pose.p
            stage=torch.where((stage==0)&(tcp[:,2]>.13),1,stage)
            stage=torch.where((stage==1)&(torch.linalg.vector_norm(tcp[:,:2]-behind,dim=-1)<.003),2,stage)
            stage=torch.where((stage==2)&((tcp[:,2]-s['height']).abs()<.004),3,stage)
            desired=torch.cat([behind,torch.full((n,1),.15,device=b.device)],-1)
            desired[stage==0,:2]=tcp[stage==0,:2];desired[stage>=2,2]=s['height']
            env.step(command(b,desired));preparation.append(snap(b,obj))
        # Explicit quiescent source initialization; same operation in both conditions.
        obj.set_linear_velocity(torch.zeros_like(pos));obj.set_angular_velocity(torch.zeros_like(pos));b.scene._gpu_apply_all();b.scene._gpu_fetch_all()
        parent=clone_tree(b.get_state_dict());torch.save(parent,out/'source_parent_state.pt');reference=snap(b,obj)
        traces=[];responses=[];requests=[];restore_errors=[]
        for k in list(range(25))+[12,12]:
            restore(b,parent);actual=snap(b,obj)
            restore_errors.append(max(float(np.max(np.abs(actual[key]-reference[key]))) for key in actual))
            origin=b.agent.tcp.pose.p.clone();p0=obj.pose.p.clone();yaw0=yaw(obj.pose.q).clone()
            xi=torch.tensor(GRID[k]*s['perturb']+[0.,s['nominal']],device=b.device,dtype=torch.float32).expand(n,2)
            travel=rotate(xi,direction);rec=[snap(b,obj)];actions=[];targets=[]
            for t in range(16):
                desired=origin.clone();desired[:,:2]+=travel*min((t+1)/a.moving_steps,1.)
                action=command(b,desired);env.step(action);actions.append(numpy(action).copy());targets.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy());rec.append(snap(b,obj))
            delta=rotate(obj.pose.p[:,:2]-p0[:,:2],-direction);dyaw=torch.atan2(torch.sin(yaw(obj.pose.q)-yaw0),torch.cos(yaw(obj.pose.q)-yaw0));vel=rotate(obj.linear_velocity[:,:2],-direction)
            response=numpy(torch.cat([delta,dyaw[:,None],vel,obj.angular_velocity[:,2:3]],-1));responses.append(response)
            traces.append({**{key:np.stack([r[key] for r in rec]) for key in rec[0]},'action':np.stack(actions),'joint_target':np.stack(targets)})
            requests.append(dict(probe=k,xi_m=numpy(xi).tolist()))
            print(json.dumps(dict(probe=k,low_mean_dy_m=float(response[0::2,1].mean()),high_mean_dy_m=float(response[1::2,1].mean()))),flush=True)
        np.savez_compressed(out/'branches.npz',response=np.stack(responses),grid=GRID,source_indices=SOURCE,stage=numpy(stage),direction=numpy(direction),offset=numpy(offset),condition=np.arange(n)%2,pair=np.arange(n)//2,**{key:np.stack([tr[key] for tr in traces]) for key in traces[0]})
        np.savez_compressed(out/'preparation.npz',**{key:np.stack([r[key] for r in preparation]) for key in preparation[0]})
        matched={key:float(np.max(np.abs(value[0::2]-value[1::2]))) for key,value in reference.items()}
        meta=dict(environment='RollBall-v1',runtime_environment='PairedDampingRollBall-v1',parents=a.parents,source_states=n,preparations_per_parent=2,seed=a.seed,spec=s,steps=16,moving_steps=a.moving_steps,source_probe_indices=SOURCE,damping=[a.low,a.high],actual_damping=b.actual_damping,setup_ready=int((numpy(stage)==3).sum()),matched_parent_max_abs=matched,restore_max_abs=max(restore_errors),control_freq=b.control_freq,sim_freq=b.sim_freq,seconds=time.time()-started,backbone_updates=0,condition_labels_used_by_models=False,preparation_control_steps=n*180,source_acquisition_control_steps=n*7*16,extra_grid_and_repeat_control_steps=n*20*16,bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        (out/'probe_requests.json').write_text(json.dumps(requests))
        (out/'COMPLETE.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(resolve('data','source_exchange','evaluation','RollBall-v1')));p.add_argument('--parents',type=int,default=4);p.add_argument('--seed',type=int,default=120100);p.add_argument('--low',type=float,default=0.);p.add_argument('--high',type=float,default=8.);p.add_argument('--moving-steps',type=int,default=8);main(p.parse_args())
