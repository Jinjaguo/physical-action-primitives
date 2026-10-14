"""Collection driver; every branch runs in a fresh process with full preparation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from physical_action_primitives.organization.core import load_config
from physical_action_primitives.paths import resolve


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json'))); p.add_argument('--batch',type=int,required=True)
    p.add_argument('--phase',choices=['source','audit','both'],required=True)
    p.add_argument('--output',default=None,help='default: data/organization/raw/batch_<index>')
    a=p.parse_args()
    c=load_config(a.protocol); batch=c['batches'][a.batch]
    if a.phase=='both' and batch['split'] in ['query','gallery']:
        raise ValueError('Final source and audit collection must be separate')
    root=Path(a.output) if a.output else resolve('data','organization','raw',f'batch_{a.batch:03d}'); root.mkdir(parents=True,exist_ok=True)
    manifest=dict(batch=a.batch,split=batch['split'],seed=batch['seed'],parents=batch['parents'],phase=a.phase)
    manifest_path=root/(a.phase+'_MANIFEST.json')
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Refusing incompatible collection resume')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    phases=['source','audit'] if a.phase=='both' else [a.phase]
    start=time.time(); completed=0
    # Source order is fixed in the config; scheduling is independent of outcomes.
    for phase in phases:
        for condition,damping in enumerate(c['damping_values']):
            for index,xi in enumerate(c[phase+'_xi_m']):
                dest=root/phase/f'condition_{condition:02d}'/f'branch_{index:03d}'
                if (dest/'RESULT.json').exists():
                    completed+=1; continue
                if dest.exists():
                    raise RuntimeError(f'Incomplete attempt requires explicit preserved retry: {dest}')
                cmd=[sys.executable,'-m','physical_action_primitives.organization.engineering','--child','--output',str(dest),
                     '--parents',str(batch['parents']),'--seed',str(batch['seed']),
                     '--damping',str(damping),'--xi',*map(str,xi),'--steps',str(c['steps']),
                     '--moving-steps',str(c['moving_steps']),'--jitter',str(c['position_jitter_m'])]
                logfile=root/f'{phase}_condition_{condition:02d}_branch_{index:03d}.log'
                with logfile.open('w') as f:
                    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                completed+=1
                print(json.dumps(dict(batch=a.batch,phase=phase,condition=condition,branch=index,
                                      completed=completed,seconds=time.time()-start)),flush=True)
    (root/(a.phase.upper()+'_COMPLETE.json')).write_text(json.dumps(dict(**manifest,
            status='COLLECTION_COMPLETE',branches=completed,seconds=time.time()-start),indent=2)+'\n')

if __name__=='__main__': main()
