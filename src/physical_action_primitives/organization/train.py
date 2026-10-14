"""Common-root training with development-set checkpoint selection; never opens query/gallery audit data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from physical_action_primitives.organization.core import SCALE,load_config
from physical_action_primitives.paths import resolve
from physical_action_primitives.organization.models import ResponseOrganizer,spirl_model,reset_spirl_batch


def read_split(folder,split):
    s=dict(np.load(Path(folder)/f'{split}_source.npz'))
    a=dict(np.load(Path(folder)/f'{split}_audit.npz'))
    if not np.array_equal(s['id'],a['id']): raise ValueError('Training pair order mismatch')
    mask=s['valid']&a['valid']
    return {k:v[mask] for k,v in s.items()},{k:v[mask] for k,v in a.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json')));p.add_argument('--data',default=str(resolve('data','organization','assembled')))
    p.add_argument('--output',default=None,help='default: checkpoints/organization/<method>/seed_<seed>/width_<width>');p.add_argument('--method',choices=['spirl','affine','nonlinear'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--width',type=int,default=128)
    a=p.parse_args();c=load_config(a.protocol)
    torch.set_num_threads(4);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    tr,ta=read_split(a.data,'train');de,da=read_split(a.data,'development')
    if set(tr['root'])&set(de['root']): raise ValueError('Root leakage')
    mean=tr['states'].mean((0,1,2));std=tr['states'].std((0,1,2)).clip(.01)
    amean=np.concatenate([tr['actions'],ta['actions']],1).mean((0,1,2))
    astd=np.concatenate([tr['actions'],ta['actions']],1).std((0,1,2)).clip(.001)
    out=Path(a.output) if a.output else resolve('checkpoints','organization',a.method,f'seed_{a.seed}',f'width_{a.width}');out.mkdir(parents=True,exist_ok=False)
    device='cuda';history=[];start=time.time();best=float('inf')
    source_xi=(np.asarray(c['source_xi_m'])-np.array([0.,.022]))/.008
    audit_xi=(np.asarray(c['audit_xi_m'])-np.array([0.,.022]))/.008
    def tensor(x): return torch.as_tensor(x,dtype=torch.float32,device=device)
    if a.method=='spirl':
        model,AttrDict=spirl_model(len(mean),len(amean),c['steps'],device,a.width)
        train_states=np.concatenate([tr['states'],ta['states']],1).reshape(-1,c['steps']+1,len(mean))
        train_actions=np.concatenate([tr['actions'],ta['actions']],1).reshape(-1,c['steps'],len(amean))
        dev_states=np.concatenate([de['states'],da['states']],1).reshape(-1,c['steps']+1,len(mean))
        dev_actions=np.concatenate([de['actions'],da['actions']],1).reshape(-1,c['steps'],len(amean))
        def batch(states,actions,idx):
            reset_spirl_batch(model,len(idx))
            return AttrDict(states=tensor((states[idx]-mean)/std),actions=tensor((actions[idx]-amean)/astd))
        def loss_train():
            ix=rng.integers(len(train_states),size=64);data=batch(train_states,train_actions,ix)
            return model.loss(model(data),data).total.value
        def validate():
            scores=[]
            # Fixed posterior noise and complete held-in development sequence population.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(94001)
                for i in range(0,len(dev_states),128):
                    ix=np.arange(i,min(i+128,len(dev_states)));data=batch(dev_states,dev_actions,ix)
                    scores.append((float(model.loss(model(data),data).total.value),len(ix)))
            return sum(v*n for v,n in scores)/sum(n for _,n in scores)
    else:
        model=ResponseOrganizer(len(mean),a.method,a.width).to(device)
        def response_batch(d,idx,queries):
            n=len(idx)
            return (tensor((d['states'][idx]-mean)/std),tensor(d['y'][idx]/SCALE),
                    tensor(np.broadcast_to(source_xi,(n,*source_xi.shape))),tensor((d['h'][idx]-mean)/std),
                    tensor(np.broadcast_to(queries,(n,*queries.shape))))
        def loss_train():
            ix=rng.integers(len(tr['y']),size=64)
            pred=model(*response_batch(tr,ix,audit_xi))
            # Same physical target loss for both heads; no final response or damping labels.
            return torch.nn.functional.smooth_l1_loss(pred,tensor(ta['y'][ix]/SCALE))
        def validate():
            values=[]
            for i in range(0,len(de['y']),128):
                ix=np.arange(i,min(i+128,len(de['y'])))
                pred=model(*response_batch(de,ix,audit_xi))
                values.append((float(torch.nn.functional.smooth_l1_loss(pred,tensor(da['y'][ix]/SCALE))),len(ix)))
            return sum(v*n for v,n in values)/sum(n for _,n in values)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    for step in range(c['training_updates']):
        model.train();loss=loss_train()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%c['validation_interval']==0:
            model.eval()
            with torch.no_grad(): score=validate()
            row=dict(step=step+1,train_loss=float(loss.detach()),development_score=score)
            history.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=score
                torch.save(dict(state=model.state_dict(),method=a.method,width=a.width,seed=a.seed,
                    state_dim=len(mean),action_dim=len(amean),steps=c['steps'],mean=mean,std=std,
                    action_mean=amean,action_std=astd,development_score=score,step=step+1),out/'model.pt')
    result=dict(status='TRAINING_COMPLETE',method=a.method,width=a.width,seed=a.seed,
                 train_roots=len(set(tr['root'])),development_roots=len(set(de['root'])),
                 train_interactions=len(tr['root']),development_interactions=len(de['root']),
                 parameters=sum(x.numel() for x in model.parameters()),best_development_score=best,
                 history=history,seconds=time.time()-start)
    (out/'TRAINING.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)

if __name__=='__main__':main()
