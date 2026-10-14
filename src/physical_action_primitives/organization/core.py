"""Physical-unit geometry for the organization audit; no primitive labels define truth."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCALE=np.asarray([.005]*3+[np.deg2rad(5)]*3+[.05]*3+[1.]*3,dtype=np.float32)


def local(vector, heading):
    v=np.asarray(vector).copy()
    c,s=np.cos(heading),np.sin(heading)
    v[...,0]=c*vector[...,0]+s*vector[...,1]
    v[...,1]=-s*vector[...,0]+c*vector[...,1]
    return v


def response12(trace, heading):
    """[time,root,13] -> exact 3D displacement/rotvec/velocity response."""
    initial,final=trace[0],trace[-1]
    qi=Rotation.from_quat(initial[:,[4,5,6,3]])
    qf=Rotation.from_quat(final[:,[4,5,6,3]])
    rot=(qf*qi.inv()).as_rotvec()
    return np.concatenate([local(final[:,:3]-initial[:,:3],heading),local(rot,heading),
                            local(final[:,7:10],heading),local(final[:,10:13],heading)],axis=-1).astype(np.float32)


def state_sequence(trace):
    return np.concatenate([trace[k] for k in ['object','tcp','qpos','qvel']],axis=-1).astype(np.float32)


def group_distances(a,b):
    """Broadcast [...,12] arrays, returning [...,4] in m,rad,m/s,rad/s."""
    shape=np.broadcast_shapes(a.shape[:-1],b.shape[:-1])
    aa=np.broadcast_to(a,shape+(12,)); bb=np.broadcast_to(b,shape+(12,))
    qa=Rotation.from_rotvec(aa[...,3:6].reshape(-1,3).copy()).as_quat()
    qb=Rotation.from_rotvec(bb[...,3:6].reshape(-1,3).copy()).as_quat()
    dot=np.clip(np.abs(np.sum(qa*qb,axis=-1)),0,1)
    angle=(2*np.arccos(dot)).reshape(shape)
    return np.stack([np.linalg.norm(aa[...,:3]-bb[...,:3],axis=-1),angle,
                     np.linalg.norm(aa[...,6:9]-bb[...,6:9],axis=-1),
                     np.linalg.norm(aa[...,9:12]-bb[...,9:12],axis=-1)],axis=-1)


def physical_distance(a,b):
    return (group_distances(a,b)/np.array([.005,np.deg2rad(5),.05,1.])).mean(-1)


def witnessed(a,b,position_m=.005,rotation_rad=np.deg2rad(5)):
    gaps=group_distances(a,b)
    return ((gaps[...,0]>position_m)|(gaps[...,1]>rotation_rad)).sum(-1)>=2


def validate_config(config):
    s=np.asarray(config['source_xi_m']); a=np.asarray(config['audit_xi_m'])
    if s.ndim!=2 or a.ndim!=2 or s.shape[1]!=2 or a.shape[1]!=2:
        raise ValueError('Invalid intervention shapes')
    if (np.linalg.norm(s[:,None]-a[None,:],axis=-1)<1e-9).any():
        raise ValueError('Source/audit overlap')
    roots=[row['seed'] for row in config['batches']]
    if len(roots)!=len(set(roots)):
        raise ValueError('Root preparation seed reused across batches')
    if set(row['split'] for row in config['batches'])!={'train','development','query','gallery'}:
        raise ValueError('Missing split')


def load_config(path):
    config=json.loads(Path(path).read_text())
    validate_config(config)
    return config
