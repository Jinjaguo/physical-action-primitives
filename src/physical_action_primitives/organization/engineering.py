"""Fresh-process, matched-preparation branch measurement for the organization audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def child(a):
    import gymnasium as gym
    import numpy as np
    import torch
    from physical_action_primitives.construction.response import command, constrain, rotate, snap, SPECS, yaw, clone_tree
    from physical_action_primitives.simulation import continuous_damping_task  # noqa: F401  (registers the environment)
    from mani_skill.utils.structs import Pose
    torch.set_num_threads(4)
    spec = SPECS['RollBall-v1']
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    env = gym.make('SequentialDampingRollBall-v1', preparation_damping=4.,
                   num_envs=a.parents, obs_mode='state_dict', reward_mode='none',
                   control_mode='pd_ee_delta_pos', sim_backend='gpu',
                   render_mode=None, robot_init_qpos_noise=0, max_episode_steps=10000)
    try:
        env.reset(seed=a.seed, options={'reconfigure': True})
        b = env.unwrapped
        obj = b.ball
        constrain(b)
        rng = np.random.default_rng(a.seed)
        centers = np.tile(spec['center'], (a.parents, 1))
        centers[:, :2] += rng.uniform(-a.jitter, a.jitter, (a.parents, 2))
        p = torch.tensor(centers, device=b.device, dtype=torch.float32)
        obj.set_pose(Pose.create_from_pq(p))
        obj.set_linear_velocity(torch.zeros_like(p)); obj.set_angular_velocity(torch.zeros_like(p))
        b.scene._gpu_apply_all(); b.scene._gpu_fetch_all()
        direction = torch.full((a.parents,), spec['heading'], device=b.device)
        local = torch.tensor([0., -spec['back']-spec['tool']-.009], device=b.device).expand(a.parents, 2)
        behind = obj.pose.p[:, :2].clone() + rotate(local, direction)
        stage = torch.zeros(a.parents, device=b.device, dtype=torch.long)
        for _ in range(180):
            tcp = b.agent.tcp.pose.p
            stage = torch.where((stage == 0) & (tcp[:, 2] > .13), 1, stage)
            stage = torch.where((stage == 1) & (torch.linalg.vector_norm(tcp[:, :2]-behind, dim=-1) < .003), 2, stage)
            stage = torch.where((stage == 2) & ((tcp[:, 2]-spec['height']).abs() < .004), 3, stage)
            desired = torch.cat([behind, torch.full((a.parents, 1), .15, device=b.device)], -1)
            desired[stage == 0, :2] = tcp[stage == 0, :2]
            desired[stage >= 2, 2] = spec['height']
            env.step(command(b, desired))
        if not bool((stage == 3).all()):
            raise RuntimeError('Preparation failed')
        obj.set_linear_velocity(torch.zeros_like(p)); obj.set_angular_velocity(torch.zeros_like(p))
        b.agent.robot.set_qvel(torch.zeros_like(b.agent.robot.get_qvel()))
        b.agent.controller.reset()
        b.scene._gpu_apply_all(); b.scene._gpu_fetch_all()
        parent = clone_tree(b.get_state_dict())
        torch.save(parent, out/'parent.pt')
        pre = snap(b, obj)
        b.set_lane_damping([a.damping]*a.parents)
        getter = np.asarray(b.get_lane_damping())
        if np.max(np.abs(getter-a.damping)) > 1e-7:
            raise RuntimeError('Damping getter mismatch')
        origin = b.agent.tcp.pose.p.clone()
        p0 = obj.pose.p.clone(); q0 = obj.pose.q.clone(); yaw0 = yaw(q0)
        requested = torch.tensor(a.xi, dtype=torch.float32, device=b.device).expand(a.parents, 2)
        travel = rotate(requested, direction)
        trace_steps = [pre]; actions = []; targets = []; desireds = []; raw_commands = []
        for step in range(a.steps):
            desired = origin.clone()
            desired[:, :2] += travel * min((step+1)/a.moving_steps, 1.)
            delta = desired-b.agent.tcp.pose.p
            raw = torch.cat([rotate(delta[:, :2], -yaw(b.agent.robot.pose.q)), delta[:, 2:3]], -1)/.1
            action = command(b, desired)
            raw_commands.append(raw.detach().cpu().numpy().copy())
            desireds.append(desired.detach().cpu().numpy().copy())
            env.step(action)
            actions.append(action.detach().cpu().numpy().copy())
            targets.append(b.agent.controller.controllers['arm']._target_qpos.detach().cpu().numpy().copy())
            trace_steps.append(snap(b, obj))
        trace = {k: np.stack([s[k] for s in trace_steps]) for k in pre}
        delta = rotate(obj.pose.p[:, :2]-p0[:, :2], -direction)
        dyaw = torch.atan2(torch.sin(yaw(obj.pose.q)-yaw0), torch.cos(yaw(obj.pose.q)-yaw0))
        vel = rotate(obj.linear_velocity[:, :2], -direction)
        response = torch.cat([delta, dyaw[:, None], vel, obj.angular_velocity[:, 2:3]], -1).detach().cpu().numpy()
        desireds = np.stack(desireds)
        tracking = np.linalg.norm(trace['tcp'][1:, :, :3]-desireds, axis=-1)
        realized = rotate(b.agent.tcp.pose.p[:, :2]-origin[:, :2], -direction).detach().cpu().numpy()
        limits=b.agent.robot.get_qlimits().detach().cpu().numpy()
        target_array=np.stack(targets)
        joint_valid=((target_array>=limits[None,:,:7,0]-1e-5)&(target_array<=limits[None,:,:7,1]+1e-5)).all(axis=(0,2))
        np.savez_compressed(out/'trace.npz', **trace, action=np.stack(actions), joint_target=np.stack(targets),
                            desired_tcp=desireds, raw_command=np.stack(raw_commands), response=response,
                            requested=np.asarray(a.xi), realized=realized, direction=direction.cpu().numpy(),
                            root_centers=centers, tracking_m=tracking,joint_valid=joint_valid,
                            state_time_s=np.arange(a.steps+1)/b.control_freq,command_time_s=np.arange(a.steps)/b.control_freq)
        arm = b.agent.controller.controllers['arm']
        result = dict(status='MEASURED', seed=a.seed, parents=a.parents,
                      damping=a.damping, xi=a.xi, steps=a.steps, moving_steps=a.moving_steps, jitter=a.jitter,
                      tracking_max_m=float(tracking.max()), endpoint_error_max_m=float(np.linalg.norm(realized-a.xi,axis=-1).max()),
                      control_freq=b.control_freq, sim_freq=b.sim_freq, controller_config=repr(arm.config),
                      controller_class=type(arm).__module__+'.'+type(arm).__name__,
                      physical_response=response.tolist(), versions=dict(torch=torch.__version__))
        (out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--child', action='store_true')
    p.add_argument('--parents', type=int, default=4); p.add_argument('--seed', type=int, default=24091401)
    p.add_argument('--damping',type=float, default=4.); p.add_argument('--xi', type=float,nargs=2,default=[0.,.022])
    p.add_argument('--steps',type=int,default=16); p.add_argument('--moving-steps',type=int,default=8)
    p.add_argument('--jitter',type=float,default=.018)
    p.add_argument('--calibration',action='store_true')
    a=p.parse_args()
    if a.child: return child(a)
    import numpy as np
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    schedules=[(d,[0.,.022],rep) for rep in range(2) for d in [0.,4.,8.]]
    schedules += [(d,[x,.030],0) for x in [-.008,.008] for d in [0.,8.]]
    if a.calibration:
        schedules=[(d,xi,rep) for rep in range(3) for xi in [[0.,.022],[-.008,.030],[.008,.030]] for d in [0.,2.,4.,6.,8.]]
    rows=[]; start=time.time()
    for index,(d,xi,rep) in enumerate(schedules):
        dest=out/f'trial_{index:02d}'
        cmd=[sys.executable,'-m','physical_action_primitives.organization.engineering','--child','--output',str(dest),'--parents',str(a.parents),
             '--seed',str(a.seed),'--damping',str(d),'--xi',*map(str,xi),
             '--steps',str(a.steps),'--moving-steps',str(a.moving_steps),'--jitter',str(a.jitter)]
        subprocess.run(cmd,check=True)
        rows.append(dict(index=index,damping=d,xi=xi,repeat=rep,path=str(dest)))
    data=[dict(np.load(Path(r['path'])/'trace.npz')) for r in rows]
    pre_error={key:max(float(np.max(np.abs(v[key][0]-data[0][key][0]))) for v in data) for key in ['object','tcp','qpos','qvel']}
    scale=np.array([.005,.005,np.deg2rad(5),.05,.05,1.])
    repeat=np.stack([np.abs(data[i]['response']-data[j]['response'])/scale
                     for i in range(len(rows)) for j in range(i)
                     if rows[i]['damping']==rows[j]['damping'] and rows[i]['xi']==rows[j]['xi']])
    signal=np.stack([np.abs(data[i]['response']-data[j]['response'])/scale
                     for i in range(len(rows)) for j in range(i)
                     if rows[i]['repeat']==rows[j]['repeat']==0 and rows[i]['xi']==rows[j]['xi'] and rows[i]['damping']!=rows[j]['damping']])
    profile_gap=max(float(np.linalg.norm(data[i]['tcp'][..., :3]-data[j]['tcp'][..., :3],axis=-1).max())
                    for i in range(len(data)) for j in range(i)
                    if np.array_equal(data[i]['requested'],data[j]['requested']))
    result=dict(status='MEASURED',topology='fresh Python process and identical full preparation for every branch',
                pre_state_max_error=pre_error,repeat_normalized_mean=float(repeat.mean()),repeat_normalized_max=float(repeat.max()),
                adjacent_signal_normalized_mean=float(signal.mean()),
                matched_request_tcp_profile_gap_max_m=profile_gap,
                tracking_max_m=max(float(d['tracking_m'].max()) for d in data),
                endpoint_error_max_m=max(float(np.linalg.norm(d['realized']-d['requested'],axis=-1).max()) for d in data),
                rows=rows,seconds=time.time()-start)
    if a.calibration:
        from physical_action_primitives.organization.core import response12,group_distances
        units=[]
        for i,row in enumerate(rows):
            if row['repeat']!=0: continue
            ids=[j for j,r in enumerate(rows) if r['damping']==row['damping'] and r['xi']==row['xi']]
            responses=[response12(data[j]['object'],data[j]['direction']) for j in ids]
            gaps=np.stack([group_distances(responses[k],responses[l]) for k in range(len(ids)) for l in range(k)])
            units.append(gaps.max(0))
        units=np.concatenate(units)
        result['repeat_q95_groups']=np.quantile(units,.95,axis=0).tolist()
        result['repeat_max_groups']=units.max(0).tolist()
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
