"""Contract checks: probe-permutation invariance of the source identity and explicit-parameter candidate execution."""
import torch
from physical_action_primitives.construction.interfaces import LawInterface,ResponseParameters,choose_intervention

torch.manual_seed(10)
m=LawInterface(12).eval()
h=torch.randn(2,12);hidden=torch.randn(2,7,8,2032);x=torch.randn(2,7,2);y=torch.randn(2,7,6);ctx=h[:,None].expand(-1,7,-1)
with torch.no_grad():
    u,_=m.identify(hidden,x,y,ctx)
    perm=torch.tensor([4,2,6,0,3,1,5])
    up,_=m.identify(hidden[:,perm],x[:,perm],y[:,perm],ctx[:,perm])
    torch.testing.assert_close(u,up,atol=1e-6,rtol=1e-5)
    pars=ResponseParameters(torch.zeros(2,6),torch.eye(6,2).expand(2,-1,-1),torch.ones(2))
    candidates=torch.tensor([[[0.,0.],[1.,0.],[0.,1.]]]).expand(2,-1,-1)
    alpha=torch.tensor([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.]])
    torch.testing.assert_close(choose_intervention(pars,alpha,candidates),torch.tensor([[1.,0.],[0.,1.]]))
print('PASS: probe permutation invariance; explicit physical-parameter candidate executor')
