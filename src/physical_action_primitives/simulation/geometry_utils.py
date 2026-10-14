"""Tensor geometry helpers: numpy conversion, yaw extraction, planar rotation."""

import torch

from mani_skill.utils.structs import Pose

def numpy(x):
    return x.detach().cpu().numpy()

def yaw(q):
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def rotate(x,theta):
    c,s=torch.cos(theta),torch.sin(theta)
    return torch.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1]),-1)
