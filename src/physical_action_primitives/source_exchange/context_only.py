"""Capacity-comparable local response law with no source identity input (context-only control)."""
import torch
from torch import nn
from physical_action_primitives.construction.interfaces import ResponseParameters,mlp

class ContextOnlyLaw(nn.Module):
    def __init__(self,context_dim=12,width=552):
        super().__init__();self.net=mlp(context_dim,19,width)

    def response_parameters(self,h):
        z=self.net(h)
        return ResponseParameters(z[:,:6],z[:,6:18].reshape(-1,6,2),z[:,18])

    def forward(self,h,xi):
        return self.response_parameters(h).predict(xi)
