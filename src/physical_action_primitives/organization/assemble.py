"""Assemble source-only representations separately from audit outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from physical_action_primitives.paths import resolve
from physical_action_primitives.organization.core import load_config,response12,state_sequence,local


def assemble(protocol,root,phase,splits,output):
    c=load_config(protocol); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    buckets={s:[] for s in splits}; summaries=[]
    for batch_index,batch in enumerate(c['batches']):
        split=batch['split']
        if split not in splits: continue
        base=Path(root)/f'batch_{batch_index:03d}'
        trace_rows=[]; responses=[]; valid=[]; fingerprints=[]; reference_state=None; reference_tcp={}
        for condition in range(len(c['damping_values'])):
            branches=[]; ys=[]; masks=[]
            for branch in range(len(c[phase+'_xi_m'])):
                path=base/phase/f'condition_{condition:02d}'/f'branch_{branch:03d}'/'trace.npz'
                with np.load(path) as f: d=dict(f)
                if not np.allclose(d['requested'],c[phase+'_xi_m'][branch],atol=1e-9,rtol=0):
                    raise ValueError('Query mismatch')
                if d['object'].shape[0]!=c['steps']+1 or d['action'].shape[0]!=c['steps']:
                    raise ValueError('Time alignment mismatch')
                ys.append(response12(d['object'],d['direction']))
                states=state_sequence(d).transpose(1,0,2)
                if reference_state is None: reference_state=states[:,0].copy()
                if np.max(np.abs(states[:,0]-reference_state))>5e-6:
                    raise ValueError('Matched-parent state restoration failed')
                tcp=(d['tcp'][...,:3]-d['tcp'][0:1,:,:3])
                tcp=local(tcp,d['direction']).transpose(1,0,2)
                if condition==0: reference_tcp[branch]=tcp.copy()
                profile_match=np.linalg.norm(tcp-reference_tcp[branch],axis=-1).max(-1)<=c['within_root_profile_tolerance_m']
                branches.append(dict(states=states,actions=d['action'].transpose(1,0,2),tcp=tcp))
                endpoint=np.linalg.norm(d['realized']-d['requested'],axis=-1)
                quat=d['tcp'][...,3:7].astype(float);quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
                orientation=(2*np.arccos(np.clip(np.abs((quat*quat[0:1]).sum(-1)),0,1))).max(0)
                masks.append((d['tracking_m'].max(0)<=c['tracking_profile_tolerance_m']) &
                              (endpoint<=c['tracking_endpoint_tolerance_m']) & d['joint_valid'] & profile_match &
                              (orientation<=c['tracking_orientation_tolerance_rad']))
                if condition==0 and branch==0:
                    import hashlib
                    fingerprints=[hashlib.sha256(states[i,0].tobytes()).hexdigest() for i in range(batch['parents'])]
            trace_rows.append(branches);responses.append(np.stack(ys,axis=1));valid.append(np.stack(masks,axis=1).all(1))
        # Rows are condition-major, with root ancestry retained separately.
        data=dict(y=np.concatenate(responses),valid=np.concatenate(valid),
                  root=np.tile([f'{batch["seed"]}:{i}' for i in range(batch['parents'])],len(trace_rows)),
                  id=np.asarray([f'{batch["seed"]}:{i}:v{cond}' for cond in range(len(trace_rows)) for i in range(batch['parents'])]),
                  fingerprint=np.tile(fingerprints,len(trace_rows)))
        data['tcp']=np.concatenate([np.stack([b['tcp'] for b in row],axis=1) for row in trace_rows])
        if phase=='source' or split in ['train','development']:
            for key in ['states','actions','tcp']:
                data[key]=np.concatenate([np.stack([b[key] for b in row],axis=1) for row in trace_rows])
            data['h']=data['states'][:,0,0]
        buckets[split].append(data)
        summaries.append(dict(batch=batch_index,split=split,interactions=len(data['root']),invalid=int((~data['valid']).sum())))
    for split,rows in buckets.items():
        if not rows: raise ValueError('Empty split '+split)
        data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(out/f'{split}_{phase}.npz',**data)
    result=dict(status='ASSEMBLED',phase=phase,splits=splits,summaries=summaries)
    (out/f'{phase}_ASSEMBLY_{"_".join(splits)}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json')));p.add_argument('--root',default=str(resolve('data','organization','raw')),help='directory of batch_<index>/ collections')
    p.add_argument('--phase',choices=['source','audit'],required=True);p.add_argument('--splits',nargs='+',required=True)
    p.add_argument('--output',default=str(resolve('data','organization','assembled')));a=p.parse_args()
    assemble(a.protocol,a.root,a.phase,a.splits,a.output)
