"""Source-block paired comparison across four model seeds and three tasks."""
import argparse
import json
from pathlib import Path
import numpy as np

from physical_action_primitives.paths import resolve

METHODS=['nearest_source_probe','endpoint','nonlinear','affine','action_seq','direct_affine7']
LABELS={'nearest_source_probe':'Nearest Source','endpoint':'PAP-Endpoint','nonlinear':'PAP-Nonlinear','affine':'PAP-Affine','action_seq':'PAP-ActionSeq','direct_affine7':'Direct affine (7 probes)'}
TASKS=['PushT-v1','PokeCube-v1','RollBall-v1']


def interval(delta):
    # One row per environment, one column per independent source block.
    delta=np.atleast_2d(delta)
    rng=np.random.default_rng(150025)
    draws=np.zeros(10000)
    for values in delta:
        draws+=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)/len(delta)
    return dict(mean_delta=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                simultaneous_family6_interval=np.quantile(draws,[.05/12,1-.05/12]).tolist())


def vectors(scores,method,endpoint):
    if endpoint=='query': prefix='query__'+method+'_';suffix='_fixed_source'
    else: prefix='invoke__'+method+'_';suffix='_fixed_source_extrapolation'
    values=[v for k,v in scores.items() if k.startswith(prefix) and k.endswith(suffix)]
    assert len(values)==(1 if method in ['nearest_source_probe','direct_affine7'] else 4),(method,endpoint,len(values))
    return np.stack(values).mean(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(resolve('results','query_invoke','prospective')),help='directory holding one <task>/ output of query_invoke.evaluate per task');p.add_argument('--output',default=str(resolve('results','query_invoke','summary')));a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data=[];scores=[]
    for i,task in enumerate(TASKS):
        root=Path(a.results)/task
        assert json.loads((root/'COMPLETE.json').read_text())['status']=='COMPLETE'
        d=json.loads((root/'RESULTS.json').read_text());assert d['task']==task
        data.append(d)
        with np.load(root/'BLOCK_SCORES.npz') as z:scores.append({k:z[k] for k in z.files})
    rows=[];contrasts=[]
    for i,task in enumerate(TASKS):
        for method in METHODS:
            calls=[r for r in data[i]['invoke'] if r['method']==method and r['mode']=='fixed_source' and r['family']=='extrapolation']
            rows.append(dict(task=task,method=method,query_error=float(vectors(scores[i],method,'query').mean()),
                             invoke_error=float(vectors(scores[i],method,'invoke').mean()),
                             success_rate=float(np.mean([r['success']/r['total'] for r in calls])),
                             successful_executions=sum(r['success'] for r in calls),executions=sum(r['total'] for r in calls),
                             position_mm=float(np.mean([r['position_mm'] for r in calls])),
                             yaw_degrees=float(np.mean([r['yaw_degrees'] for r in calls])),
                             model_seeds=len(calls)))
        for endpoint in ['query','invoke']:
            for baseline in [m for m in METHODS if m!='affine']:
                delta=vectors(scores[i],'affine',endpoint)-vectors(scores[i],baseline,endpoint)
                contrasts.append(dict(task=task,endpoint=endpoint,baseline=baseline,**interval(delta)))
    primary=[]
    for endpoint in ['query','invoke']:
        for baseline in ['nearest_source_probe','endpoint','nonlinear']:
            delta=np.stack([vectors(z,'affine',endpoint)-vectors(z,baseline,endpoint) for z in scores])
            row=dict(task='equal_environment_mean',endpoint=endpoint,baseline=baseline,**interval(delta))
            row['supports_lower_error']=row['simultaneous_family6_interval'][1]<0
            primary.append(row)
    result=dict(status='COMPLETE',
                rows=rows,primary=primary,per_task_descriptive_contrasts=contrasts,
                joint_superiority_supported=all(r['supports_lower_error'] for r in primary),
                uncertainty='Whole source-block paired bootstrap; all descendants, requests, methods and seeds retained; conditional on four training seeds.',
                total_data_bytes=sum(d['data_bytes'] for d in data))
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    lines=['# Matched-source Query and Invoke', '',
           'Prospective fixed-checkpoint replication: 20 independent source blocks per environment, three receiving states per block, seven source probes, 18 withheld Query interventions, 81 Invoke candidates, nine goals. Four training seeds for each learned method.', '',
           'The primary comparison uses fixed-source receiving-state Query and fixed-source extrapolation Invoke. All methods use identical measured source information and execution rules. Fresh processes preserve the same 80-lane preparation for every intervention.', '',
           '| Environment | Method | Query error ↓ | Realized goal error ↓ | Success ↑ |', '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {LABELS[r['method']]} | {r['query_error']:.4f} | {r['invoke_error']:.4f} | {r['success_rate']:.1%} |")
    lines+=['','Both continuous errors are sums of squared six-field residuals divided by the existing physical scales. Learned-method rates average four seeds; their total executed denominator is 2,160 per environment/condition. Nonlearned methods execute 540 calls.', '',
            '| Primary endpoint | Baseline | PAP-Affine minus baseline | Simultaneous interval | Lower error supported |', '|---|---|---:|---|---|']
    for r in primary:lines.append(f"| {r['endpoint']} | {LABELS[r['baseline']]} | {r['mean_delta']:.4f} | {r['simultaneous_family6_interval']} | {r['supports_lower_error']} |")
    lines+=['','## Interpretation','',
            'Joint superiority over all three specified baselines on both primary endpoints: '+('SUPPORTED.' if result['joint_superiority_supported'] else 'NOT SUPPORTED.'), '',
            'These results compare explicit affine response construction with source-conditioned nonlinear predictors, not PAP-Affine versus all dynamics models. Both constructions retain source information. No source-memory mechanism claim follows from the main comparison alone.', '',
            'Checkpoints are fixed before collection and no width tuning is performed, so the study does not establish superiority over an optimally tuned dynamics family. Confidence intervals condition on the four training seeds and the declared physical regime.', '',
            'Complete local/new-source conditions, every seed, physical component errors and raw execution traces are retained in the per-environment outputs. Per-task intervals are descriptive, not an additional corrected hypothesis family.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    palette=['#9aa3ad','#db9842','#7967b0','#237b91','#7e9c66','#b4a4a0']
    for col,task in enumerate(TASKS):
        rr=[r for r in rows if r['task']==task]
        for row,key in enumerate(['query_error','invoke_error']):
            ax=axes[row,col];ax.bar(range(len(rr)),[r[key] for r in rr],color=palette)
            ax.set_title(task);ax.set_xticks(range(len(rr)),[LABELS[r['method']] for r in rr],rotation=40,ha='right')
            ax.set_ylabel('Query normalized error' if row==0 else 'Realized goal error');ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE')))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
