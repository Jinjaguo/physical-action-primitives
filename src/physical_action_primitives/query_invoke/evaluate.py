"""Prospective Query/Invoke evaluation of selected checkpoints on independent source blocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from physical_action_primitives.construction.response import SPECS, GRID, SOURCE
from physical_action_primitives.construction.train import CLASSES, identify, predictions
from physical_action_primitives.construction.interfaces import RESPONSE_SCALE
from physical_action_primitives.paths import resolve

SCALE=np.asarray(RESPONSE_SCALE)
TASKS=['PushT-v1','PokeCube-v1','RollBall-v1']


def child(task, seed, blocks, request, output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-m','physical_action_primitives.query_invoke.collect_branch','--task',task,'--seed',str(seed),
                        '--blocks',str(blocks),'--request',str(request),'--output',str(output)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def load_npz(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def run(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(a.protocol).read_text())
    task=a.task; seed=protocol['seeds'][task]; blocks=protocol['source_blocks']; n=4*blocks
    spec=SPECS[task]; held=np.array([k for k in range(25) if k not in SOURCE])
    receiving=np.arange(blocks,n); donor=np.repeat(np.arange(blocks),3)
    started=time.time(); checks=[]
    # Source and recipient audit panels are collected in isolated fresh processes.
    # No audit label enters any model; the checkpoints are fixed before collection.
    rows=[]
    def collect_panel(k):
        request=out/f'panel_request_{k}.npy'
        np.save(request,np.tile(GRID[k]*spec['perturb']+[0.,spec['nominal']],(n,1)))
        target=out/f'panel_{k}'
        child(task,seed,blocks,request,target)
        print(json.dumps(dict(stage='panel_collected',task=task,intervention=k)),flush=True)
        return load_npz(target/'branch.npz')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(collect_panel,range(25)))
    for k,d in enumerate(rows):
        if k:
            for key in ['object','tcp','qpos','qvel']:
                err=float(np.max(np.abs(d[key][0]-rows[0][key][0])))
                assert err<=1e-5, f'Panel initial-state mismatch {k} {key} {err}'
        print(json.dumps(dict(stage='panel',task=task,completed=k+1,total=25)),flush=True)
    pack=out/'data'/'confirmation'/task; pack.mkdir(parents=True)
    fields=['response','object','tcp','qpos','qvel','action','joint_target']
    np.savez_compressed(pack/'branches.npz',**{k:np.stack([r[k] for r in rows]) for k in fields},
                        grid=GRID,source_indices=SOURCE,stage=rows[12]['stage'],direction=rows[12]['direction'],offset=rows[12]['offset'])
    (pack/'COMPLETE.json').write_text(json.dumps(dict(environment=task,spec=spec,seed=seed,steps=16,
                                                       parents=n,source_blocks=blocks)))
    feature_command=[sys.executable,'-m','physical_action_primitives.construction.extract_features','--data',str(out/'data'/'confirmation'),
                     '--output',str(out/'features')]
    if 'backbone_plan' in protocol:
        plan_path=out/'BACKBONE_PLAN.json'
        plan_path.write_text(json.dumps(protocol['backbone_plan'],indent=2))
        feature_command.extend(['--backbone-plan',str(plan_path)])
    subprocess.run(feature_command,check=True)
    d=load_npz(out/'features'/f'{task}.npz')
    physical_xi=np.broadcast_to((GRID*spec['perturb']+[0.,spec['nominal']])/.01,(n,25,2)).astype(np.float32).copy()
    # Only seven responses per root are exposed to source identification.
    source_y=np.full_like(d['y'],np.nan); source_y[:,SOURCE]=d['y'][:,SOURCE]
    safe={**d,'y':source_y,'xi':physical_xi,'src':np.array(SOURCE)}
    del d
    selections=[]; query_outputs={}
    selection=json.loads(resolve(protocol['selection']).read_text())
    configs=[]
    for method in ['affine','endpoint','nonlinear','action_seq']:
        for c in sorted(selection['candidates'][method],key=lambda z:z['seed']): configs.append((method,c['seed'],resolve(c['path'])))
    configs += [('nearest_source_probe',-1,None),('direct_affine7',-1,None)]
    for method,model_seed,path in configs:
        m=None
        if path:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            m=CLASSES[ck['method']](ck['context_dim']).cuda();m.load_state_dict(ck['state']);m.eval()
            sd={**safe,'h':(safe['h']-ck['h_mean'])/ck['h_std']}
            with torch.no_grad():
                (all_u,_),_=identify(m,sd,np.arange(n),'cuda',probes=7)
            h=torch.as_tensor((safe['h'][receiving]-ck['h_mean'])/ck['h_std'],device='cuda')
        for mode, source_ids in [('fixed_source',donor),('new_source',receiving)]:
            sx=physical_xi[source_ids][:,SOURCE]; sy=source_y[source_ids][:,SOURCE]
            def predict(x):
                if m is not None:
                    with torch.no_grad():
                        return predictions(m,ck['method'],all_u[source_ids],h,torch.as_tensor(x,device='cuda')).cpu().numpy()
                if method=='nearest_source_probe':
                    ix=((x[:,:,None]-sx[:,None])**2).sum(-1).argmin(-1)
                    return sy[np.arange(len(sy))[:,None],ix]
                design=np.concatenate([np.ones((*sx.shape[:-1],1)),sx],-1)
                coef=np.linalg.pinv(design)@sy
                return np.concatenate([np.ones((*x.shape[:-1],1)),x],-1)@coef
            qpred=predict(physical_xi[receiving][:,held])
            query_outputs[f'{method}_{model_seed}_{mode}']=qpred
            for family,bound,mags in [('local',1.25,[.006,.009,.012]),('extrapolation',2.5,[.015,.018,.021])]:
                grid=np.array([(x,y) for x in np.linspace(-bound,bound,9) for y in np.linspace(-bound,bound,9)],np.float32)
                xi_m=(grid*spec['perturb']+[0.,spec['nominal']]).astype(np.float32)
                x=np.broadcast_to(xi_m/.01,(len(receiving),81,2)).copy()
                pred=predict(x)
                assert pred.shape==(3*blocks,81,6) and np.isfinite(pred).all()
                goals=np.array([[dx,dy,0.,0.,0.,0.] for dy in mags for dx in [-.003,0.,.003]])
                loss=((pred[None]-goals[:,None,None]/SCALE)**2).sum(-1)
                chosen=loss.argmin(-1)
                # One fresh process per request, always the same n recipe lanes.
                request=np.tile(np.array([0.,spec['nominal']],np.float32),(9,n,1))
                request[:,receiving]=xi_m[chosen]
                tag=f'{method}_{model_seed}_{mode}_{family}'
                predpath=out/f'predictions_{tag}.npz'
                np.savez_compressed(predpath,candidate_prediction=pred,candidate_xi_m=xi_m,goals=goals,
                                    selected_index=chosen,selected_predicted_loss=np.take_along_axis(loss,chosen[...,None],axis=-1)[...,0],
                                    source_ids=source_ids,recipient_ids=receiving)
                reqpath=out/f'request_{tag}.npy';np.save(reqpath,request)
                selections.append(dict(tag=tag,method=method,seed=model_seed,mode=mode,family=family,
                                       predictions=str(predpath),request=str(reqpath)))
        m=None
        if path: del all_u
        torch.cuda.empty_cache()
        print(json.dumps(dict(stage='predictions',task=task,method=method,seed=model_seed)),flush=True)
    np.savez_compressed(out/'QUERY_PREDICTIONS.npz',**query_outputs)
    # Audit outcomes are only used after every query/candidate prediction is written.
    truth=load_npz(pack/'branches.npz')['response'].transpose(1,0,2)
    result=dict(task=task,source_blocks=blocks,receiving_roots=3*blocks,query=[],invoke=[],
                topology='fresh process per request; constant 80 lanes')
    scores={}
    for tag,pred in query_outputs.items():
        physical=pred*SCALE-truth[receiving][:,held]
        err=((physical/SCALE)**2).sum(-1)
        scores['query__'+tag]=err.reshape(blocks,3,18).mean((1,2))
        result['query'].append(dict(tag=tag,mean_error=float(err.mean()),position_mm=float(np.linalg.norm(physical[...,:2],axis=-1).mean()*1000),
                                   yaw_degrees=float(np.rad2deg(np.abs(physical[...,2])).mean())))
    reference=rows[0]
    for index,c in enumerate(selections):
        target=out/'calls'/c['tag']
        requests=np.load(c['request']);p=load_npz(c['predictions'])
        target.mkdir(parents=True,exist_ok=False)
        def execute_goal(g):
            req=target/f'request_{g}.npy';np.save(req,requests[g])
            dest=target/f'goal_{g}'
            child(task,seed,blocks,req,dest)
            observed=load_npz(dest/'branch.npz')
            for key in ['object','tcp','qpos','qvel']:
                err=float(np.max(np.abs(observed[key][0]-reference[key][0])))
                assert err<=1e-5, f'Call context mismatch {c["tag"]} {g} {key} {err}'
            return observed
        with ThreadPoolExecutor(max_workers=4) as pool:
            observed=list(pool.map(execute_goal,range(9)))
        actual=np.stack([v['response'][receiving] for v in observed])
        physical=actual-p['goals'][:,None]
        loss=((physical/SCALE)**2).sum(-1)
        assert np.isfinite(loss).all()
        pos=np.linalg.norm(physical[...,:2],axis=-1)
        yaw=np.abs(np.arctan2(np.sin(physical[...,2]),np.cos(physical[...,2])))
        speed=np.linalg.norm(actual[...,3:5],axis=-1)
        valid=np.stack([v['joint_valid'][receiving] for v in observed])
        success=(pos<=.005)&(yaw<=np.deg2rad(5))&(speed<=.05)&valid
        scores['invoke__'+c['tag']]=loss.reshape(9,blocks,3).mean((0,2))
        scores['success__'+c['tag']]=success.reshape(9,blocks,3).mean((0,2))
        np.savez_compressed(target/'scores.npz',actual=actual,target=p['goals'],loss=loss,success=success,joint_valid=valid)
        result['invoke'].append(dict(**c,mean_error=float(loss.mean()),median_error=float(np.median(loss)),p95_error=float(np.quantile(loss,.95)),
                                     success=int(success.sum()),total=success.size,position_mm=float(pos.mean()*1000),
                                     yaw_degrees=float(np.rad2deg(yaw).mean()),terminal_speed=float(speed.mean()),invalid_joints=int((~valid).sum())))
        (out/'PROGRESS.json').write_text(json.dumps(dict(stage='invoke',completed=index+1,total=len(selections),latest=result['invoke'][-1])))
        print(json.dumps(dict(stage='invoke',task=task,completed=index+1,total=len(selections),method=c['tag'])),flush=True)
    np.savez_compressed(out/'BLOCK_SCORES.npz',**scores)
    result['seconds']=time.time()-started
    result['data_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',task=task,seconds=result['seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--protocol',default=str(resolve('configs','query_invoke','protocol.json')));p.add_argument('--output',default=None,help='default: results/query_invoke/prospective/<task>')
    a=p.parse_args()
    if a.output is None:a.output=str(resolve('results','query_invoke','prospective',a.task))
    run(a)
