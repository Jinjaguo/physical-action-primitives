"""One physical intervention per lane, fresh simulator and deterministic preparation."""
import argparse
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from physical_action_primitives.construction.response import command, constrain, rotate, snap, SPECS, yaw, Pose, numpy


def recipe(seed, blocks):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-.018, .018, (blocks, 2))
    angle = rng.uniform(-.15, .15, blocks)
    receive_xy = np.repeat(xy, 3, axis=0)+rng.uniform(-.002, .002, (blocks*3, 2))
    receive_angle = np.repeat(angle, 3)+rng.uniform(-.02, .02, blocks*3)
    ids = np.r_[np.arange(blocks), np.repeat(np.arange(blocks), 3)]
    return np.r_[xy, receive_xy], np.r_[angle, receive_angle], ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--blocks', type=int, required=True)
    ap.add_argument('--request', required=True, help='n x 2 physical local displacements in metres; n multiple of 4*blocks')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    xi = np.load(a.request).astype(np.float32)
    base_n = 4*a.blocks
    assert xi.ndim==2 and xi.shape[1]==2 and len(xi)%base_n==0 and np.isfinite(xi).all()
    n = len(xi); copies = n//base_n; s = SPECS[a.task]
    xy, angle, block_ids = recipe(a.seed, a.blocks)
    centers = np.tile(s['center'], (base_n, 1)); centers[:, :2] += xy
    if a.task=='RollBall-v1': angle = np.zeros_like(angle)
    centers = np.tile(centers, (copies, 1)); angle = np.tile(angle, copies)
    offsets = np.tile((block_ids%2)*s['offset'], copies)
    env = gym.make(a.task, num_envs=n, obs_mode='state_dict', reward_mode='none', control_mode='pd_ee_delta_pos',
                   sim_backend='gpu', render_mode=None, robot_init_qpos_noise=0, max_episode_steps=10000)
    started = time.time()
    try:
        env.reset(seed=a.seed, options={'reconfigure': True})
        b = env.unwrapped; obj = getattr(b, s['actor']); constrain(b)
        p = torch.tensor(centers, device=b.device, dtype=torch.float32)
        theta = torch.tensor(angle, device=b.device, dtype=torch.float32)
        q = torch.zeros((n, 4), device=b.device); q[:, 0] = torch.cos(theta/2); q[:, 3] = torch.sin(theta/2)
        obj.set_pose(Pose.create_from_pq(p, q)); obj.set_linear_velocity(torch.zeros_like(p)); obj.set_angular_velocity(torch.zeros_like(p))
        if a.task=='PokeCube-v1':
            peg = b.peg.pose.p.clone(); peg[:, 0] = .40; peg[:, 1] = -.35
            b.peg.set_pose(Pose.create_from_pq(peg, b.peg.pose.q))
        b.scene._gpu_apply_all(); b.scene._gpu_fetch_all()
        direction = theta+s['heading']
        offset = torch.tensor(offsets, device=b.device, dtype=torch.float32)
        local = torch.stack([offset, torch.full_like(offset, -s['back']-s['tool']-.009)], -1)
        behind = obj.pose.p[:, :2].clone()+rotate(local, direction)
        stage = torch.zeros(n, device=b.device, dtype=torch.long)
        for _ in range(180):
            tcp = b.agent.tcp.pose.p
            stage = torch.where((stage==0)&(tcp[:, 2]>.13), 1, stage)
            stage = torch.where((stage==1)&(torch.linalg.vector_norm(tcp[:, :2]-behind, dim=-1)<.003), 2, stage)
            stage = torch.where((stage==2)&((tcp[:, 2]-s['height']).abs()<.004), 3, stage)
            desired = torch.cat([behind, torch.full((n, 1), .15, device=b.device)], -1)
            desired[stage==0, :2] = tcp[stage==0, :2]; desired[stage>=2, 2] = s['height']
            env.step(command(b, desired))
        pre = snap(b, obj)
        tcp0 = b.agent.tcp.pose.p.clone(); p0 = obj.pose.p.clone(); y0 = yaw(obj.pose.q).clone()
        travel = rotate(torch.tensor(xi, device=b.device), direction)
        trace = [pre]; actions = []; joints = []
        for t in range(16):
            desired = tcp0.clone(); desired[:, :2] += travel*min((t+1)/12, 1.)
            action = command(b, desired); env.step(action)
            actions.append(numpy(action).copy()); joints.append(numpy(b.agent.controller.controllers['arm']._target_qpos).copy())
            trace.append(snap(b, obj))
        delta = rotate(obj.pose.p[:, :2]-p0[:, :2], -direction)
        dyaw = torch.atan2(torch.sin(yaw(obj.pose.q)-y0), torch.cos(yaw(obj.pose.q)-y0))
        velocity = rotate(obj.linear_velocity[:, :2], -direction)
        response = numpy(torch.cat([delta, dyaw[:, None], velocity, obj.angular_velocity[:, 2:3]], -1))
        jt = np.stack(joints); limits = numpy(b.agent.robot.get_qlimits())[:, :7]
        valid = ((jt>=limits[None, :, :, 0]-1e-5)&(jt<=limits[None, :, :, 1]+1e-5)).all((0, 2))
        np.savez_compressed(out/'branch.npz', response=response, xi_m=xi, stage=numpy(stage), joint_valid=valid,
                            direction=numpy(direction), offset=numpy(offset), action=np.stack(actions), joint_target=jt,
                            **{k: np.stack([r[k] for r in trace]) for k in pre})
        meta = dict(task=a.task, seed=a.seed, blocks=a.blocks, lanes=n, copies=copies, steps=16, spec=s,
                    seconds=time.time()-started, all_setup_ready=bool((numpy(stage)==3).all()),
                    all_joint_valid=bool(valid.all()),
                    topology='fresh Python process and full preparation; one intervention per lane; no branch restore')
        (out/'COMPLETE.json').write_text(json.dumps(meta, indent=2)); print(json.dumps(meta), flush=True)
    finally:
        env.close()


if __name__=='__main__': main()
