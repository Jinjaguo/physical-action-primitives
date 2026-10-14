"""Matched-source PAP-Affine operator and PAP-Nonlinear learned-dynamics model.

A raw-state source encoder is trained from scratch for both methods (no pretrained
backbone). The affine head is RigidLawHead; targets use the exact local rotation vector.
"""
import collections
import collections.abc
import sys
import torch
from torch import nn
from physical_action_primitives.simulation.rigid_common import RigidLawHead
from physical_action_primitives.paths import resolve


def mlp(i,o,width):
    return nn.Sequential(nn.Linear(i,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,o))


class ResponseOrganizer(nn.Module):
    def __init__(self,state_dim,kind,width=128):
        super().__init__();self.kind=kind
        self.temporal=mlp(state_dim,32,width)
        self.probe=mlp(2+12+32+state_dim,32,width)
        if kind=='affine': self.head=RigidLawHead(32,state_dim,width)
        elif kind=='nonlinear': self.head=mlp(32+state_dim+2,12,width)
        else: raise ValueError(kind)

    def identify(self,states,source_y,source_xi,h):
        temporal=self.temporal(states).mean(-2)
        packet=torch.cat([source_xi,source_y,temporal,h[:,None].expand(-1,states.shape[1],-1)],-1)
        return self.probe(packet).mean(1)

    def forward(self,states,source_y,source_xi,h,query):
        u=self.identify(states,source_y,source_xi,h)
        if self.kind=='affine': return self.head(u,h,query)
        return self.head(torch.cat([u[:,None].expand(-1,query.shape[1],-1),
                                    h[:,None].expand(-1,query.shape[1],-1),query],-1))


def spirl_model(state_dim,action_dim,steps,device,width=128):
    # Compatibility aliases for the SPiRL utility layer (collections ABCs); no algorithm changes.
    for name in ['Mapping','MutableMapping','Sequence']:
        if not hasattr(collections,name): setattr(collections,name,getattr(collections.abc,name))
    sys.path.insert(0,str(resolve('external','python_deps')))
    sys.path.insert(0,str(resolve('external','spirl')))
    from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
    from spirl.utils.general_utils import AttrDict
    config=AttrDict(state_dim=state_dim,action_dim=action_dim,n_rollout_steps=steps,cond_decode=True,
                   kl_div_weight=5e-4,nz_enc=width,nz_mid=width,n_processing_layers=5,
                   batch_size=64,device=device)
    return ClSPiRLMdl(config).to(device),AttrDict


def reset_spirl_batch(model,batch_size):
    """Official CustomLSTMCell stores a configured batch size; reset per batch.

    Only allocate the same zero recurrent state at the actual batch size.
    No change to architecture, posterior, decoder, prior or losses.
    """
    from spirl.modules.recurrent_modules import CustomLSTMCell
    model._hp.batch_size=batch_size
    for module in model.modules():
        if isinstance(module,CustomLSTMCell):
            module._hp.batch_size=batch_size
            module.reset()
