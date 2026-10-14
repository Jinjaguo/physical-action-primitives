"""Response constructions: source identity u, invocation target alpha, and public-parameter execution are separate APIs.

Class names map to the paper constructions: LawInterface = PAP-Affine, DynamicsInterface = PAP-Nonlinear,
EndpointInterface = PAP-Endpoint, ActionSequenceInterface = PAP-ActionSeq.
"""
from dataclasses import dataclass
import torch
from torch import nn

RESPONSE_SCALE = (.005,.005,.0872664626,.05,.05,1.)

def mlp(inputs,outputs,width=128):
    return nn.Sequential(nn.Linear(inputs,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,outputs))

@dataclass
class ResponseParameters:
    bias: torch.Tensor
    jacobian: torch.Tensor
    support_logits: torch.Tensor

    def predict(self,xi):
        return self.bias[:,None,:]+torch.einsum('bri,bki->bkr',self.jacobian,xi)

class LawInterface(nn.Module):
    """Shared temporal physical projection followed by a DeepSets source encoder.

    Hidden tensors are frozen encoder outputs, shape [batch,probes,time,2032].
    The seven probe responses are observations, not query-response targets.
    h is public recipient context. Environment IDs and private materials absent.
    """
    def __init__(self,context_dim,hidden_dim=2032,width=128,identity_dim=32):
        super().__init__()
        self.step_projection=mlp(hidden_dim,6,width)
        self.probe_encoder=mlp(2+6+6+context_dim,identity_dim,width)
        self.law_head=mlp(identity_dim+context_dim,6*3+1,width)

    def identify(self,hidden,source_xi,source_response,source_context,probe_mask=None):
        step=self.step_projection(hidden)
        temporal=torch.cat([step[...,:3].sum(-2),step[...,-1,3:]],-1)
        packet=torch.cat([source_xi,source_response,temporal,source_context],-1)
        encoded=self.probe_encoder(packet)
        if probe_mask is None: probe_mask=torch.ones_like(encoded[...,0])
        if (probe_mask.sum(-1)==0).any(): raise ValueError('Empty identification packet')
        u=(encoded*probe_mask[...,None]).sum(1)/probe_mask.sum(1,keepdim=True)
        return u,step

    def response_parameters(self,u,h):
        z=self.law_head(torch.cat([u,h],-1))
        return ResponseParameters(z[:,:6],z[:,6:18].reshape(-1,6,2),z[:,18])

class EndpointInterface(LawInterface):
    """Same packet, context, and capacity; bottleneck exposes endpoint effect.

    This is a goal-conditioned response model, retaining candidate dependence.
    It must not be implemented as a constant prediction for all interventions.
    """
    def __init__(self,context_dim,**kw):
        super().__init__(context_dim,**kw)
        width=kw.get('width',128);identity_dim=kw.get('identity_dim',32)
        self.endpoint=mlp(identity_dim,6,width)
        self.goal_dynamics=mlp(6+context_dim+2,6,width)

    def predict(self,u,h,xi):
        goal=self.endpoint(u)
        return self.goal_dynamics(torch.cat([goal[:,None].expand(-1,xi.shape[1],-1),h[:,None].expand(-1,xi.shape[1],-1),xi],-1))

class DynamicsInterface(LawInterface):
    def __init__(self,context_dim,**kw):
        super().__init__(context_dim,**kw)
        self.dynamics=mlp(kw.get('identity_dim',32)+context_dim+2,6,kw.get('width',128))

    def predict(self,u,h,xi):
        return self.dynamics(torch.cat([u[:,None].expand(-1,xi.shape[1],-1),h[:,None].expand(-1,xi.shape[1],-1),xi],-1))

class SourceConditionedPolicy(LawInterface):
    def __init__(self,context_dim,**kw):
        super().__init__(context_dim,**kw)
        self.policy=mlp(kw.get('identity_dim',32)+context_dim+6,2,kw.get('width',128))

    def invoke(self,u,h,alpha):
        return self.policy(torch.cat([u,h,alpha],-1)).tanh()

class TrajectoryInterface(DynamicsInterface):
    """Action reconstruction auxiliary replaces physical step supervision."""
    def __init__(self,context_dim,**kw):
        super().__init__(context_dim,**kw)
        self.action_reconstruction=mlp(kw.get('identity_dim',32)+2,8,kw.get('width',128))

class ActionSequenceInterface(DynamicsInterface):
    def __init__(self,context_dim,**kw):
        super().__init__(context_dim,**kw)
        self.action_reconstruction=mlp(kw.get('identity_dim',32)+2,16*8,kw.get('width',128))

def choose_intervention(parameters,alpha,candidates,weights=None):
    """E has no access to u: choose from public b,J and shared candidates.

    alpha contains the desired six physical responses in normalized units.
    Scoring requires executing the selected intervention over the same action horizon.
    """
    predicted=parameters.predict(candidates)
    error=(predicted-alpha[:,None]).square()
    if weights is not None:error=error*weights
    index=error.sum(-1).argmin(-1)
    return candidates[torch.arange(len(index),device=index.device),index]
