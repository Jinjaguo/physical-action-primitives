"""Presentation-only outputs: retrieval denominators and a fixed-rule illustrative pair."""
import argparse
import json
from pathlib import Path
import numpy as np
from physical_action_primitives.paths import resolve
from physical_action_primitives.organization.core import load_config,group_distances

p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json')));p.add_argument('--data',default=str(resolve('data','organization','assembled')))
p.add_argument('--rankings',default=str(resolve('results','organization','rankings')));p.add_argument('--results',default=str(resolve('results','organization','audit')),help='audit output directory (physical_audit.npz, REPORT.md)');a=p.parse_args()
c=load_config(a.protocol);out=Path(a.results);rank=Path(a.rankings)
co=dict(np.load(rank/'cohort.npz'));phys=dict(np.load(out/'physical_audit.npz'));dmp=dict(np.load(rank/'dmp.npz'))
cut=float(np.quantile(dmp['pair_distance'],.25))
qualified=np.flatnonzero(phys['pair_witness']&(dmp['pair_distance']<=cut))
eligible=co['eligible'];n=eligible.sum(1)
denominators={str(cov):dict(minimum_k=int(np.ceil(n*cov).min()),maximum_k=int(np.ceil(n*cov).max()),
                           retrieved_directed_pairs=int(np.ceil(n*cov).sum())) for cov in c['coverage']}
record=dict(rule=c['examples'],source_dmp_quartile_cut=cut,qualifying_pairs=len(qualified),retrieval_denominators=denominators)
if len(qualified):
    index=min(qualified,key=lambda v:(co['query_id'][co['pair_i'][v]],co['query_id'][co['pair_j'][v]]))
    i,j=co['pair_i'][index],co['pair_j'][index]
    def read(phase):
        d=dict(np.load(Path(a.data)/f'query_{phase}.npz'));lookup={v:k for k,v in enumerate(d['id'])}
        ix=[lookup[co['query_id'][k]] for k in [i,j]]
        return {key:val[ix] for key,val in d.items()}
    source=read('source');audit=read('audit');gaps=group_distances(audit['y'][0],audit['y'][1])
    witnesses=(gaps[:,0]>c['witness_position_m'])|(gaps[:,1]>c['witness_rotation_rad'])
    record.update(status='EXAMPLE_FOUND',interaction_ids=co['query_id'][[i,j]].tolist(),
                  source_dmp_distance=float(dmp['pair_distance'][index]),
                  source_nominal_gap=group_distances(source['y'][0,0],source['y'][1,0]).tolist(),
                  source_tcp_max_gap_m=float(np.linalg.norm(source['tcp'][0]-source['tcp'][1],axis=-1).max()),
                  audit_witness_count=int(witnesses.sum()),audit_witness_indices=np.flatnonzero(witnesses).tolist(),
                  audit_group_gaps=gaps.tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12.8,3.8))
    for k,label in enumerate(['Interaction A','Interaction B']):
        tcp=source['tcp'][k,0]*1000
        axs[0].plot(np.arange(len(tcp))/c['control_freq'],tcp[:,1],label=label,linestyle='-' if k==0 else '--')
        axs[1].plot(np.arange(len(c['audit_xi_m'])),audit['y'][k,:,1]*1000,marker='o',ms=3,label=label,linestyle='-' if k==0 else '--')
    axs[0].set(xlabel='Source time (s)',ylabel='Measured TCP forward travel (mm)',title='Executed source motion')
    axs[1].set(xlabel='Audit intervention index',ylabel='Object forward displacement (mm)',title='Independent object responses')
    axs[2].plot(np.arange(len(gaps)),gaps[:,0]*1000,label='Translation gap (mm)',marker='o',ms=3)
    axs[2].plot(np.arange(len(gaps)),np.rad2deg(gaps[:,1]),label='Rotation gap (degrees)',marker='s',ms=3)
    axs[2].axhline(5,color='black',linestyle=':',label='5 mm / 5 degree physical tolerance')
    axs[2].set(xlabel='Audit intervention index',ylabel='Physical response gap',title='Witnessed separation')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'boundary_example.png',dpi=220);fig.savefig(out/'boundary_example.pdf');plt.close(fig)
else:
    record['status']='NO_QUALIFYING_PAIR'
(out/'EXAMPLES_AND_DENOMINATORS.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'REPORT.md').open('a') as f:
    f.write('\n## Retrieval denominators and fixed-rule example\n\n')
    f.write('| Coverage | K range | Retrieved directed pairs per method |\n|---|---:|---:|\n')
    for cov,r in denominators.items():f.write(f'| {float(cov)*100:g}% | {r["minimum_k"]}–{r["maximum_k"]} | {r["retrieved_directed_pairs"]} |\n')
    if len(qualified):
        f.write(f'\nThe first metadata-ordered qualifying pair has {record["audit_witness_count"]} witnessed audit branches. It was selected using the bottom-quartile source DMP distance rule, not maximal response discrepancy.\n\n![Fixed-rule example](boundary_example.png)\n')
    else:f.write('\nNo pair qualified under the example rule; no illustrative pair was substituted.\n')
print(json.dumps(record))
