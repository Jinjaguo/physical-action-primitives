"""Train the context-only (no source identity) control on the same data and development split as the PAP constructions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from physical_action_primitives.construction.train import load
from physical_action_primitives.paths import resolve
from physical_action_primitives.source_exchange.context_only import ContextOnlyLaw

def main(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.root);tr=load(root/'features_training');de=load(root/'features_development')
    mean=tr['h'].mean(0);std=np.maximum(tr['h'].std(0),.01)
    for d in [tr,de]:d['h']=(d['h']-mean)/std
    m=ContextOnlyLaw(len(mean)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5)
    held=np.array([i for i in range(25) if i not in tr['src']]);best=float('inf');history=[];start=time.time()
    F=torch.nn.functional
    for step in range(a.updates):
        idx=rng.integers(len(tr['h']),size=32)
        h=torch.tensor(tr['h'][idx],device='cuda');xi=torch.tensor(tr['xi'][idx],device='cuda');truth=torch.tensor(tr['y'][idx],device='cuda',dtype=torch.float32)
        params=m.response_parameters(h);pred=params.predict(xi)
        lp=F.smooth_l1_loss(pred[:,held],truth[:,held]);ld=F.smooth_l1_loss(pred-pred.mean(1,keepdim=True),truth-truth.mean(1,keepdim=True))
        distance=torch.cdist(h,h);distance.fill_diagonal_(float('inf'));other=distance.argmin(-1)
        near=distance[torch.arange(len(idx)),other]<.5
        equiv=(truth[:,tr['src']]-truth[other][:,tr['src']]).abs().amax((1,2))<1.
        pair=near&equiv;le=F.smooth_l1_loss(pred[pair],pred[other][pair]) if pair.any() else pred.sum()*0
        goals=rng.integers(25,size=len(idx));alpha=truth[torch.arange(len(idx)),goals]
        cost=(pred-alpha[:,None]).square().mean(-1);truecost=(truth-alpha[:,None]).square().mean(-1)
        lc=-(torch.softmax(-truecost.detach()*4,dim=-1)*torch.log_softmax(-cost*4,dim=-1)).sum(-1).mean()
        ls=F.binary_cross_entropy_with_logits(params.support_logits,torch.tensor(tr['stage'][idx]==3,device='cuda',dtype=torch.float32))
        loss=lp+.5*ld+.2*le+.2*ls+.5*lc
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if (step+1)%500==0:
            m.eval();scores=[]
            with torch.no_grad():
                for begin in range(0,len(de['h']),32):
                    ix=np.arange(begin,min(begin+32,len(de['h'])))
                    pp=m(torch.tensor(de['h'][ix],device='cuda'),torch.tensor(de['xi'][ix],device='cuda'))
                    tt=torch.tensor(de['y'][ix],device='cuda',dtype=torch.float32)
                    scores.append(float(F.smooth_l1_loss(pp[:,held],tt[:,held])))
            score=float(np.mean(scores));row=dict(step=step+1,development_score=score,train_loss=float(loss));history.append(row)
            if score<best:
                best=score;torch.save(dict(state={k:v.detach().cpu() for k,v in m.state_dict().items()},h_mean=mean,h_std=std,context_dim=len(mean),width=552,step=step+1,seed=a.seed,development_score=score,backbone_updates=0),out/'model.pt')
            print(json.dumps(row),flush=True);m.train()
    (out/'TRAINING.json').write_text(json.dumps(dict(seed=a.seed,updates=a.updates,training_parents=len(tr['h']),development_parents=len(de['h']),parameters=sum(p.numel() for p in m.parameters()),history=history,seconds=time.time()-start,method='context_only_law',input='public h only; no source packet, identity or condition label',loss='same response, difference, equivalence, support and invocation terms; no unused feature projection auxiliary'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(resolve('data','query_invoke')),help='directory holding features_training/ and features_development/');p.add_argument('--output',default=str(resolve('checkpoints','source_exchange','context_only')));p.add_argument('--seed',type=int,default=104002);p.add_argument('--updates',type=int,default=10000);main(p.parse_args())
