"""Equal source packets and rollout budgets; development selects each candidate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from physical_action_primitives.paths import resolve
from physical_action_primitives.construction.interfaces import LawInterface,EndpointInterface,DynamicsInterface,TrajectoryInterface,SourceConditionedPolicy,ActionSequenceInterface

CLASSES={'affine':LawInterface,'endpoint':EndpointInterface,'nonlinear':DynamicsInterface,'trajectory':TrajectoryInterface,'policy':SourceConditionedPolicy,'action_seq':ActionSequenceInterface}

def load(root,sequence=False):
    rows=[];tasks=[]
    for p in sorted(Path(root).glob('*.npz')):
        d=dict(np.load(p));cohort=Path(root).name.removeprefix('features_')
        meta=json.loads((Path(root).parent/cohort/p.stem/'COMPLETE.json').read_text())
        d['physical_xi']=np.broadcast_to((d['xi']*meta['spec']['perturb']+[0.,meta['spec']['nominal']])/.01,(len(d['h']),25,2)).astype(np.float32)
        if sequence:
            br=np.load(Path(root).parent/cohort/p.stem/'branches.npz');ids=d['source_indices']
            arm=(br['joint_target'][ids]-br['qpos'][ids,:-1,:,:7])/.1
            actions=np.zeros((*arm.shape[:-1],8),np.float32);actions[...,:7]=arm
            if br['action'].shape[-1]>3:actions[...,7]=br['action'][ids,...,3]
            d['action_seq']=actions.transpose(2,0,1,3).reshape(len(d['h']),7,128)
        rows.append(d);tasks.extend([p.stem]*len(d['h']))
    result={k:np.concatenate([d[k] for d in rows]) for k in ['hidden','step','h','y','stage']}
    if all('action' in d for d in rows):result['action']=np.concatenate([d['action'] for d in rows])
    if sequence:result['action_seq']=np.concatenate([d['action_seq'] for d in rows])
    result.update(xi=np.concatenate([d['physical_xi'] for d in rows]),grid=rows[0]['xi'],src=rows[0]['source_indices'],tasks=tasks)
    return result

def identify(model,d,idx,device,probes=7):
    src=d['src'][:probes]
    hidden=torch.as_tensor(d['hidden'][idx,:probes],device=device,dtype=torch.float32)
    h=torch.as_tensor(d['h'][idx],device=device)
    xi=torch.as_tensor(d['xi'][idx][:,src],device=device)
    response=torch.as_tensor(d['y'][idx][:,src],device=device,dtype=torch.float32)
    return model.identify(hidden,xi,response,h[:,None].expand(-1,probes,-1)),h

def predictions(m,kind,u,h,xi):
    if kind=='affine':return m.response_parameters(u,h).predict(xi)
    return m.predict(u,h,xi)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(resolve('data','query_invoke')),help='directory holding features_training/ and features_development/ and the matching cohort data');ap.add_argument('--output',default=None,help='default: checkpoints/pap_candidates/<method>_<seed>');ap.add_argument('--method',choices=list(CLASSES),required=True);ap.add_argument('--seed',type=int,default=104000);ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--probes',type=int,choices=[1,3,7],default=7);ap.add_argument('--no-step',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    out=Path(a.output) if a.output else resolve('checkpoints','pap_candidates',f'{a.method}_{a.seed}');out.mkdir(parents=True,exist_ok=False);root=Path(a.root)
    tr=load(root/'features_training',sequence=a.method=='action_seq');de=load(root/'features_development',sequence=a.method=='action_seq')
    mean=tr['h'].mean(0);std=np.maximum(tr['h'].std(0),.01)
    for d in [tr,de]:d['h']=(d['h']-mean)/std
    m=CLASSES[a.method](len(mean)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5)
    grid=torch.tensor(tr['grid'],device='cuda');held=np.array([i for i in range(25) if i not in tr['src'][:a.probes]]);hist=[];best=float('inf');t0=time.time()
    for step in range(a.updates):
        idx=rng.integers(len(tr['h']),size=32)
        (u,step_pred),h=identify(m,tr,idx,'cuda',probes=a.probes)
        xi=torch.as_tensor(tr['xi'][idx],device='cuda')
        truth=torch.as_tensor(tr['y'][idx],device='cuda',dtype=torch.float32)
        step_truth=torch.as_tensor(tr['step'][idx,:a.probes],device='cuda')
        physical=torch.nn.functional.smooth_l1_loss(step_pred,step_truth)
        if a.no_step:physical=physical*0
        if a.method=='policy':
            # Direct inverse learning uses exactly the same 25 measured training branches.
            goals=rng.integers(25,size=len(idx));alpha=truth[torch.arange(len(idx)),goals]
            chosen=m.invoke(u,h,alpha)
            loss=torch.nn.functional.smooth_l1_loss(chosen,grid[goals])+.2*physical
        else:
            pred=predictions(m,a.method,u,h,xi)
            lp=torch.nn.functional.smooth_l1_loss(pred[:,held],truth[:,held])
            ld=torch.nn.functional.smooth_l1_loss(pred-pred.mean(1,keepdim=True),truth-truth.mean(1,keepdim=True))
            # Measured response difference defines an equivalence target at near-identical
            # public contexts; no contact labels or environment IDs enter this pairing.
            contextual=torch.cdist(h,h)
            contextual.fill_diagonal_(float('inf'));other=contextual.argmin(-1)
            near=contextual[torch.arange(len(idx)),other]<.5
            equiv=(truth[:,tr['src']]-truth[other][:,tr['src']]).abs().amax((1,2))<1.
            pair=near&equiv
            le=torch.nn.functional.smooth_l1_loss(pred[pair],pred[other][pair]) if pair.any() else pred.sum()*0
            goals=rng.integers(25,size=len(idx));alpha=truth[torch.arange(len(idx)),goals]
            cost=(pred-alpha[:,None]).square().mean(-1)
            truecost=(truth-alpha[:,None]).square().mean(-1)
            lc=-(torch.softmax(-truecost.detach()*4,dim=-1)*torch.log_softmax(-cost*4,dim=-1)).sum(-1).mean()
            support=m.response_parameters(u,h).support_logits
            ls=torch.nn.functional.binary_cross_entropy_with_logits(support,torch.as_tensor(tr['stage'][idx]==3,device='cuda',dtype=torch.float32))
            loss=lp+.5*ld+.2*le+.2*ls+.5*lc+physical
            if a.method=='endpoint':loss=loss+torch.nn.functional.smooth_l1_loss(m.endpoint(u),truth[:,12])
            if a.method in ['trajectory','action_seq']:
                ax=xi[:,tr['src']]
                rec=m.action_reconstruction(torch.cat([u[:,None].expand(-1,7,-1),ax],-1))
                la=torch.nn.functional.smooth_l1_loss(rec,torch.as_tensor(tr['action_seq' if a.method=='action_seq' else 'action'][idx],device='cuda'))
                loss=loss-physical+la
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if (step+1)%500==0:
            m.eval();scores=[]
            with torch.no_grad():
                for begin in range(0,len(de['h']),32):
                    ix=np.arange(begin,min(begin+32,len(de['h'])));(uu,_),hh=identify(m,de,ix,'cuda',probes=a.probes);truth=torch.as_tensor(de['y'][ix],device='cuda',dtype=torch.float32)
                    if a.method=='policy':
                        for k in held:
                            pp=m.invoke(uu,hh,truth[:,k]);scores.append(float((pp-grid[k]).square().mean()))
                    else:
                        pp=predictions(m,a.method,uu,hh,torch.as_tensor(de['xi'][ix],device='cuda'));scores.append(float(torch.nn.functional.smooth_l1_loss(pp[:,held],truth[:,held])))
            score=float(np.mean(scores));hist.append(dict(step=step+1,development_score=score,train_loss=float(loss)))
            if score<best:
                best=score;torch.save(dict(state={k:v.detach().cpu() for k,v in m.state_dict().items()},method=a.method,context_dim=len(mean),h_mean=mean,h_std=std,step=step+1,seed=a.seed,development_score=best,backbone_updates=0,probes=a.probes,step_supervision=not a.no_step,intervention_contract='object-local requested displacement / 0.01m'),out/'model.pt')
            print(json.dumps(hist[-1]),flush=True);m.train()
    (out/'TRAINING.json').write_text(json.dumps(dict(method=a.method,seed=a.seed,updates=a.updates,training_parents=len(tr['h']),development_parents=len(de['h']),parameters=sum(p.numel() for p in m.parameters()),history=hist,seconds=time.time()-t0,backbone_updates=0),indent=2))

if __name__=='__main__':main()
