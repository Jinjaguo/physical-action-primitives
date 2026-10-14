"""Local response construction: measure object responses to a grid of tool displacements from restorable branch parents."""
import argparse
import copy
import json
from pathlib import Path
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import mani_skill.envs
from mani_skill.utils.structs import Pose

from physical_action_primitives.simulation.controller_adapter import constrain
from physical_action_primitives.simulation.geometry_utils import rotate, yaw, numpy

SPECS = {
    'PushT-v1': dict(actor='tee', center=[-.25,.05,.021], back=.0625, tool=.008, height=.024, offset=.030, nominal=.026, perturb=.008, heading=0.),
    'PokeCube-v1': dict(actor='cube', center=[-.15,.05,.021], back=.020, tool=.018, height=.025, offset=.010, nominal=.015, perturb=.006, heading=0.),
    'RollBall-v1': dict(actor='ball', center=[-.10,.55,.036], back=.035, tool=.018, height=.035, offset=.012, nominal=.022, perturb=.008, heading=np.pi),
}
GRID = np.array([(x,y) for x in [-1.,-.5,0.,.5,1.] for y in [-1.,-.5,0.,.5,1.]],np.float32)
SOURCE = [12,2,22,10,14,0,24]

def clone_tree(x):
    if isinstance(x,dict): return {k:clone_tree(v) for k,v in x.items()}
    return x.clone() if torch.is_tensor(x) else copy.deepcopy(x)

def snap(b,obj):
    return {k:numpy(v).copy() for k,v in dict(object=torch.cat([obj.pose.raw_pose,obj.linear_velocity,obj.angular_velocity],-1),tcp=b.agent.tcp.pose.raw_pose,qpos=b.agent.robot.get_qpos(),qvel=b.agent.robot.get_qvel()).items()}

def command(b,desired):
    # The controller takes base-frame Cartesian deltas; convert the world-frame vector for rotated bases.
    delta=desired-b.agent.tcp.pose.p
    a=rotate(delta[:,:2],-yaw(b.agent.robot.pose.q))
    arm=torch.cat([a,delta[:,2:3]],-1).div(.1).clamp(-.2,.2)
    if b.agent.robot.get_qpos().shape[-1]>7:
        return torch.cat([arm,-torch.ones((len(arm),1),device=b.device)],-1)
    return arm

def restore(b,state):
    b.set_state_dict(clone_tree(state))
    b.agent.controller.reset()
    if 'controller' in state: b.agent.controller.set_state(clone_tree(state['controller']))
    b._elapsed_steps.zero_()

def frame(b):
    a=numpy(b.render_rgb_array())
    if a.ndim==4: a=a[0]
    return a.astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--env',choices=list(SPECS),required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--parents',type=int,default=12)
    ap.add_argument('--seed',type=int,default=101000)
    ap.add_argument('--steps',type=int,default=16)
    ap.add_argument('--single-preparation',action='store_true')
    ap.add_argument('--no-video',action='store_true')
    args=ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    s=SPECS[args.env];copies=1 if args.single_preparation else 2;n=args.parents*copies
    env=gym.make(args.env,num_envs=n,obs_mode='state_dict',reward_mode='none',control_mode='pd_ee_delta_pos',sim_backend='gpu',render_mode='rgb_array',robot_init_qpos_noise=0,max_episode_steps=10000)
    started=time.time()
    try:
        env.reset(seed=args.seed,options={'reconfigure':True})
        b=env.unwrapped;obj=getattr(b,s['actor']);constrain(b)
        rng=np.random.default_rng(args.seed)
        centers=np.tile(s['center'],(args.parents,1));centers[:,:2]+=rng.uniform(-.018,.018,(args.parents,2))
        angles=rng.uniform(-.15,.15,args.parents) if args.env!='RollBall-v1' else np.zeros(args.parents)
        p=torch.tensor(np.repeat(centers,copies,0),device=b.device,dtype=torch.float32)
        theta=torch.tensor(np.repeat(angles,copies),device=b.device,dtype=torch.float32)
        q=torch.zeros((n,4),device=b.device);q[:,0]=torch.cos(theta/2);q[:,3]=torch.sin(theta/2)
        obj.set_pose(Pose.create_from_pq(p,q));obj.set_linear_velocity(torch.zeros_like(p));obj.set_angular_velocity(torch.zeros_like(p))
        if args.env=='PokeCube-v1':
            # Alternate contact preparation: closed Panda fingers push the cube directly.
            # Park the unused peg during INITIALIZATION, and retain it in saved state.
            peg=b.peg.pose.p.clone();peg[:,0]=.40;peg[:,1]=-.35
            b.peg.set_pose(Pose.create_from_pq(peg,b.peg.pose.q))
        b.scene._gpu_apply_all();b.scene._gpu_fetch_all()
        initial=clone_tree(b.get_state_dict());torch.save(initial,out/'initial_state.pt')
        direction=theta+s['heading']
        offset=torch.tensor(np.resize([0.,s['offset']],n),device=b.device,dtype=torch.float32)
        local=torch.stack([offset,torch.full_like(offset,-s['back']-s['tool']-.009)],-1)
        behind=obj.pose.p[:,:2].clone()+rotate(local,direction)
        stage=torch.zeros(n,device=b.device,dtype=torch.long);frames=[];setup=[]
        for t in range(180):
            tcp=b.agent.tcp.pose.p
            stage=torch.where((stage==0)&(tcp[:,2]>.13),1,stage)
            stage=torch.where((stage==1)&(torch.linalg.vector_norm(tcp[:,:2]-behind,dim=-1)<.003),2,stage)
            stage=torch.where((stage==2)&((tcp[:,2]-s['height']).abs()<.004),3,stage)
            desired=torch.cat([behind,torch.full((n,1),.15,device=b.device)],-1)
            desired[stage==0,:2]=tcp[stage==0,:2]
            desired[stage>=2,2]=s['height']
            # Stop outside the surface. The nonzero nominal probe establishes contact.
            a=command(b,desired);env.step(a)
            setup.append(snap(b,obj))
            if t%4==0 and not args.no_video: frames.append(frame(b))
        parent=clone_tree(b.get_state_dict());torch.save(parent,out/'source_parent_state.pt')
        reference=snap(b,obj)
        rows=[];responses=[];traces=[];restoration_errors=[]
        for k in list(range(25))+[12,12]:
            restore(b,parent)
            actual=snap(b,obj)
            err=max(float(np.max(np.abs(actual[key]-reference[key]))) for key in actual)
            restoration_errors.append(err)
            if err>1e-5: raise RuntimeError(f'branch restore mismatch {err}')
            tcp0=b.agent.tcp.pose.p.clone();p0=obj.pose.p.clone();y0=yaw(obj.pose.q).clone()
            xi=torch.tensor(GRID[k]*s['perturb']+[0.,s['nominal']],device=b.device,dtype=torch.float32).expand(n,2)
            travel=rotate(xi,direction)
            rec=[snap(b,obj)];acts=[];targets=[]
            for t in range(args.steps):
                desired=tcp0.clone();desired[:,:2]+=travel*min((t+1)/(args.steps-4),1.)
                a=command(b,desired);env.step(a)
                acts.append(numpy(a).copy());targets.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy());rec.append(snap(b,obj))
                if k==12 and len(responses)==12 and not args.no_video: frames.append(frame(b))
            delta=rotate(obj.pose.p[:,:2]-p0[:,:2],-direction)
            dyaw=torch.atan2(torch.sin(yaw(obj.pose.q)-y0),torch.cos(yaw(obj.pose.q)-y0))
            vel=rotate(obj.linear_velocity[:,:2],-direction)
            response=numpy(torch.cat([delta,dyaw[:,None],vel,obj.angular_velocity[:,2:3]],-1))
            tracked=numpy(rotate(b.agent.tcp.pose.p[:,:2]-tcp0[:,:2],-direction))
            limits=numpy(b.agent.robot.get_qlimits())[:,:7]
            jt=np.stack(targets);valid=((jt>=limits[None,:,:,0]-1e-5)&(jt<=limits[None,:,:,1]+1e-5)).all((0,2))
            rows.append(dict(probe=k,xi_m=numpy(xi).tolist(),actual_tcp_delta_m=tracked.tolist(),joint_valid=valid.tolist()))
            responses.append(response)
            tr={key:np.stack([r[key] for r in rec]) for key in rec[0]};tr['action']=np.stack(acts);tr['joint_target']=jt;traces.append(tr)
            print(json.dumps(dict(env=args.env,probe=k,moving=int((np.linalg.norm(response[:,:2],axis=-1)>.002).sum()),total=n,median_displacement_m=float(np.median(np.linalg.norm(response[:,:2],axis=-1))))),flush=True)
        response=np.stack(responses)
        np.savez_compressed(out/'branches.npz',response=response,grid=GRID,source_indices=SOURCE,stage=numpy(stage),direction=numpy(direction),offset=numpy(offset),**{key:np.stack([r[key] for r in traces]) for key in traces[0]})
        np.savez_compressed(out/'preparation.npz',**{key:np.stack([r[key] for r in setup]) for key in setup[0]})
        if frames:imageio.mimsave(out/'preview.mp4',frames,fps=10)
        noise=np.max(np.abs(response[25:]-response[12]),axis=0)
        summary=dict(environment=args.env,status='CONSTRUCTION_MEASURED',parents=args.parents,preparations_per_parent=copies,branches=27*n,source_probe_indices=SOURCE,heldout_probe_indices=[i for i in range(25) if i not in SOURCE],setup_ready=int((numpy(stage)==3).sum()),nominal_moving=int((np.linalg.norm(response[12,:,:2],axis=-1)>.002).sum()),source_states=n,repeat_max_abs_by_response=noise.max(0).tolist(),restore_max_abs=max(restoration_errors),control_freq=b.control_freq,sim_freq=b.sim_freq,steps=args.steps,spec=s,seed=args.seed,seconds=time.time()-started,contact_mask=False,contact_observation='unavailable; no force channel inferred from commands',tool='PandaStick' if args.env=='PushT-v1' else 'closed Panda gripper direct contact',backbone_updates=0,bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        (out/'probe_requests.json').write_text(json.dumps(rows,indent=2))
        (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally: env.close()

if __name__=='__main__':main()
