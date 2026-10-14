"""Arm controller adapter: upright end-effector IK target with a hard per-step joint box."""

import torch

from physical_action_primitives.simulation import geometry_utils as base

def constrain(b):
    arm=b.agent.controller.controllers['arm']
    if getattr(arm,'joint_box',False):return
    def set_action(action):
        delta=arm._preprocess_action(action);arm._step=0;arm._start_qpos=arm.qpos
        current=arm.ee_pose_at_base
        upright=torch.zeros_like(current.q);upright[:,1]=1
        target=base.Pose.create_from_pq(current.p+delta,upright)
        arm._target_pose=target
        arm._target_qpos=arm.kinematics.compute_ik(pose=target,q0=arm.articulation.get_qpos(),is_delta_pose=False,current_pose=current,solver_config=arm.config.delta_solver_config)
        if arm._target_qpos is None:arm._target_qpos=arm._start_qpos
        limits=b.agent.robot.get_qlimits()[:,:7]
        # Hard joint box (+/-0.1 rad per step, inside joint limits) applied after the IK solve.
        lo=torch.maximum(limits[:,:,0]+.005,arm._start_qpos-.10)
        hi=torch.minimum(limits[:,:,1]-.005,arm._start_qpos+.10)
        arm._target_qpos=torch.maximum(torch.minimum(arm._target_qpos,hi),lo)
        if arm.config.interpolate:arm._step_size=(arm._target_qpos-arm._start_qpos)/arm._sim_steps
        else:arm.set_drive_targets(arm._target_qpos)
    arm.set_action=set_action;arm.joint_box=True
