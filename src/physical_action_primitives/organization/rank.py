"""Compute source-only distances and neighbor rankings for the query and gallery cohorts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.distance import cdist
from physical_action_primitives.organization.core import SCALE,load_config,physical_distance
from physical_action_primitives.paths import resolve
from physical_action_primitives.organization.models import ResponseOrganizer,spirl_model,reset_spirl_batch


def retain_complete_roots(d,conditions):
    roots=np.unique(d['root']);keep=[]
    for root in roots:
        ix=np.flatnonzero(d['root']==root)
        if len(ix)!=conditions: raise ValueError('Incomplete root bundle')
        if d['valid'][ix].all(): keep.extend(ix.tolist())
    if len(keep)<.95*len(d['root']):raise ValueError('Source invalidity exceeds the 5% root limit')
    keep=np.asarray(sorted(keep))
    return {k:v[keep] for k,v in d.items()}


def load_source(folder,split,c):
    return retain_complete_roots(dict(np.load(Path(folder)/f'{split}_source.npz')),len(c['damping_values']))


def operator_distances(q,g):
    return np.stack([physical_distance(row[None],g).mean(-1) for row in q])


def select_model(models,method,seed,widths):
    paths=[Path(models)/method/f'seed_{seed}'/f'width_{w}' for w in widths]
    rows=[json.loads((p/'TRAINING.json').read_text()) for p in paths]
    best=min(range(len(rows)),key=lambda i:(rows[i]['best_development_score'],widths[i]))
    return paths[best]/'model.pt',rows[best]


def learned(d,path,c,kind):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    mean,std=ck['mean'],ck['std']
    if kind=='spirl': model,AttrDict=spirl_model(ck['state_dim'],ck['action_dim'],ck['steps'],'cuda',ck['width'])
    else: model=ResponseOrganizer(ck['state_dim'],kind,ck['width']).cuda()
    model.load_state_dict(ck['state']);model.eval();result=[]
    def tensor(x): return torch.as_tensor(x,dtype=torch.float32,device='cuda')
    src=(np.asarray(c['source_xi_m'])-np.array([0.,.022]))/.008
    query=(np.asarray(c['virtual_query_xi_m'])-np.array([0.,.022]))/.008
    with torch.no_grad():
        for begin in range(0,len(d['id']),64):
            ix=np.arange(begin,min(begin+64,len(d['id'])));n=len(ix)
            states=(d['states'][ix]-mean)/std
            if kind=='spirl':
                ns=states.shape[1]
                reset_spirl_batch(model,n*ns)
                data=AttrDict(states=tensor(states.reshape(-1,ck['steps']+1,ck['state_dim'])),
                              actions=tensor(((d['actions'][ix]-ck['action_mean'])/ck['action_std']).reshape(-1,ck['steps'],ck['action_dim'])))
                value=model._run_inference(data).mu.reshape(n,ns*10)
            else:
                value=model(tensor(states),tensor(d['y'][ix]/SCALE),tensor(np.broadcast_to(src,(n,*src.shape))),
                            tensor((d['h'][ix]-mean)/std),tensor(np.broadcast_to(query,(n,*query.shape))))
                value=value*tensor(SCALE)
            result.append(value.cpu().numpy())
    del model;torch.cuda.empty_cache()
    return np.concatenate(result)


def dmp_features(d,c):
    sys.path.insert(0,str(resolve('external','python_deps')))
    from pydmps.dmp_discrete import DMPs_discrete
    result=[];errors=[]
    for motions in d['tcp']:
        packet=[]
        for motion in motions:
            model=DMPs_discrete(n_dmps=3,n_bfs=c['dmp_basis_count'],dt=1/c['steps'])
            model.imitate_path(y_des=motion.T)
            packet.extend(np.concatenate([model.w.ravel(),motion[0],motion[-1],[c['steps']/20]]))
            reproduced=model.rollout()[0]
            # Library fit diagnostic on its own time grid, without audit response access.
            target=np.stack([np.interp(np.linspace(0,1,len(reproduced)),np.linspace(0,1,len(motion)),motion[:,j]) for j in range(3)],-1)
            errors.append(float(np.sqrt(np.mean((target-reproduced)**2))))
        result.append(packet)
    return np.asarray(result),dict(mean_reconstruction_rmse_m=float(np.mean(errors)),p95_reconstruction_rmse_m=float(np.quantile(errors,.95)))


def context_mask(q,g,c):
    # state order: object13, tcp7, qpos9, qvel9.
    qh,gh=q['h'],g['h']
    relq=qh[:,:3]-qh[:,13:16];relg=gh[:,:3]-gh[:,13:16]
    mask=cdist(relq,relg)<=c['context_relative_position_tolerance_m']
    mask&=(np.abs(qh[:,None,20:27]-gh[None,:,20:27]).max(-1)<=c['context_qpos_tolerance_rad'])
    mask&=(np.abs(qh[:,None,27:29]-gh[None,:,27:29]).max(-1)<=c['context_gripper_tolerance_m'])
    mask&=(q['root'][:,None]!=g['root'][None,:])
    for i in range(len(qh)):
        mask[i]&=(np.linalg.norm(q['tcp'][i][None]-g['tcp'],axis=-1).max(axis=(1,2))<=c['cross_root_profile_tolerance_m'])
    if not (mask.sum(1)>=c['minimum_gallery_interactions']).all():
        raise ValueError('Insufficient context-matched gallery; do not change after audit')
    return mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json')));p.add_argument('--data',default=str(resolve('data','organization','assembled')))
    p.add_argument('--models',default=str(resolve('checkpoints','organization')),help='directory of <method>/seed_<seed>/width_<width>/');p.add_argument('--output',default=str(resolve('results','organization','rankings')));a=p.parse_args()
    c=load_config(a.protocol);torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    tr=load_source(a.data,'train',c);q=load_source(a.data,'query',c);g=load_source(a.data,'gallery',c)
    if set(q['root'])&set(g['root']):raise ValueError('Query-gallery roots overlap')
    if set(q['fingerprint'])&set(g['fingerprint']):raise ValueError('Query-gallery physical parent fingerprint overlap')
    if (set(q['fingerprint'])|set(g['fingerprint']))&set(tr['fingerprint']):raise ValueError('Training-final physical parent overlap')
    eligible=context_mask(q,g,c)
    # Full query-root condition pairs used by the independent split test.
    pi=[];pj=[]
    for root in np.unique(q['root']):
        ix=np.flatnonzero(q['root']==root)
        for u,i in enumerate(ix):
            for j in ix[:u]:pi.append(i);pj.append(j)
    pi=np.asarray(pi);pj=np.asarray(pj)
    meta=dict(methods={},query_roots=len(set(q['root'])),gallery_roots=len(set(g['root'])))
    np.savez_compressed(out/'cohort.npz',query_id=q['id'],gallery_id=g['id'],query_root=q['root'],gallery_root=g['root'],
                         eligible=eligible,pair_i=pi,pair_j=pj,query_fingerprint=q['fingerprint'],gallery_fingerprint=g['fingerprint'])
    def save(name,dist,pair_dist,extra=None):
        if not np.isfinite(dist).all() or not np.isfinite(pair_dist).all():raise ValueError('Nonfinite representation')
        # A seeded hash of the interaction ids defines the tie order for every method, never physical outcomes.
        import hashlib
        tie=np.asarray([[int.from_bytes(hashlib.sha256((c['tie_seed']+str(i)+'|'+str(j)).encode()).digest()[:8],'big')
                         for j in g['id']] for i in q['id']],dtype=np.uint64)
        masked=np.where(eligible,dist,np.inf)
        order=np.stack([np.lexsort((tie[i],masked[i])) for i in range(len(q['id']))])
        path=out/(name+'.npz');np.savez_compressed(path,distance=dist,order=order,pair_distance=pair_dist)
        meta['methods'][name]=dict(**(extra or {}))
        print(json.dumps(dict(method=name,status='RANKED')),flush=True)
    save('oracle_semantic',np.zeros(eligible.shape),np.zeros(len(pi)))
    save('endpoint',operator_distances(q['y'][:,:1],g['y'][:,:1]),physical_distance(q['y'][pi,0],q['y'][pj,0]))
    tf,tinfo=dmp_features(tr,c);qf,qinfo=dmp_features(q,c);gf,ginfo=dmp_features(g,c)
    norm=tf.std(0).clip(1e-6)
    save('dmp',cdist(qf/norm,gf/norm),np.linalg.norm((qf[pi]-qf[pj])/norm,axis=-1),
         dict(training_fit=tinfo,query_fit=qinfo,gallery_fit=ginfo))
    for method in ['spirl','nonlinear','affine']:
        for seed in c['training_seeds']:
            path,training=select_model(a.models,method,seed,c['model_widths'])
            qf=learned(q,path,c,method);gf=learned(g,path,c,method)
            if method=='spirl':
                tf=learned(tr,path,c,method);norm=tf.std(0).clip(1e-3)
                dist=cdist(qf/norm,gf/norm);pairs=np.linalg.norm((qf[pi]-qf[pj])/norm,axis=-1)
            else:
                dist=operator_distances(qf,gf);pairs=physical_distance(qf[pi],qf[pj]).mean(-1)
            save(f'{method}_seed_{seed}',dist,pairs,dict(development_score=training['best_development_score'],width=training['width']))
    meta['status']='RANKED'
    (out/'RANKINGS.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
