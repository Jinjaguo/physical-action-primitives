"""Simulator-only physical boundary evaluator for the neighbor rankings produced by rank.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from physical_action_primitives.paths import resolve
from physical_action_primitives.organization.core import load_config,physical_distance,witnessed,group_distances


def root_average(values,roots):
    return np.stack([values[roots==r].mean(0) for r in np.unique(roots)])


def interval(samples,alpha=.05):
    return np.quantile(samples,[alpha/2,1-alpha/2],axis=0).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(resolve('configs','organization','protocol.json')));p.add_argument('--rankings',default=str(resolve('results','organization','rankings')))
    p.add_argument('--data',default=str(resolve('data','organization','assembled')));p.add_argument('--output',default=str(resolve('results','organization','audit')));a=p.parse_args()
    c=load_config(a.protocol);rankdir=Path(a.rankings)
    cohort=dict(np.load(rankdir/'cohort.npz'));coverage=np.asarray(c['coverage'])
    def audit(split):
        d=dict(np.load(Path(a.data)/f'{split}_audit.npz'));lookup={v:i for i,v in enumerate(d['id'])}
        ix=np.asarray([lookup[v] for v in cohort[split+'_id']])
        if not d['valid'][ix].all():raise ValueError('Incomplete final audit panel: no primary result permitted')
        if not np.array_equal(d['fingerprint'][ix],cohort[split+'_fingerprint']):raise ValueError('Source/audit parent mismatch')
        return d['y'][ix],d['tcp'][ix]
    (q,qtcp),(g,gtcp)=audit('query'),audit('gallery')
    for i in range(len(q)):
        aligned=np.linalg.norm(qtcp[i][None]-gtcp,axis=-1).max(axis=(1,2))<=c['cross_root_profile_tolerance_m']
        if not aligned[cohort['eligible'][i]].all():raise ValueError('Final actual intervention profiles not aligned')
    truth=np.stack([physical_distance(row[None],g).mean(-1) for row in q])
    separated=np.stack([witnessed(row[None],g,c['witness_position_m'],c['witness_rotation_rad']) for row in q])
    components=np.stack([group_distances(row[None],g).mean(-2) for row in q])
    pi,pj=cohort['pair_i'],cohort['pair_j']
    pair_truth=physical_distance(q[pi],q[pj]).mean(-1)
    pair_witness=witnessed(q[pi],q[pj],c['witness_position_m'],c['witness_rotation_rad'])
    families=['oracle_semantic','dmp','spirl','endpoint','nonlinear','affine']
    nq=len(q);nr=len(np.unique(cohort['query_root']));rng=np.random.default_rng(c['bootstrap_seed'])
    resamples=rng.integers(nr,size=(c['bootstrap_resamples'],nr))
    root_lookup={r:i for i,r in enumerate(np.unique(cohort['query_root']))}
    pair_root=np.asarray([root_lookup[r] for r in cohort['query_root'][pi]])
    root_counts=np.stack([np.bincount(sample,minlength=nr) for sample in resamples])
    all_method_roots={};results={};all_seeds={};split_results={}
    for family in families:
        names=[family] if family in ['oracle_semantic','dmp','endpoint'] else [f'{family}_seed_{seed}' for seed in c['training_seeds']]
        seed_curves=[];seed_fmr=[];seed_parts=[];seed_split=[];seed_split_fmr=[];split_boot=[];split_fmr_boot=[];ranks=[]
        for name in names:
            path=rankdir/(name+'.npz')
            d=dict(np.load(path));curves=[];fmr=[];parts=[]
            for row in range(nq):
                n=int(cohort['eligible'][row].sum());rowcurve=[];rowfmr=[];rowpart=[]
                for cov in coverage:
                    k=int(np.ceil(cov*n));ix=d['order'][row,:k]
                    if not cohort['eligible'][row,ix].all():raise ValueError('Invalid retrieved neighbor')
                    rowcurve.append(truth[row,ix].mean());rowfmr.append(separated[row,ix].mean());rowpart.append(components[row,ix].mean(0))
                curves.append(rowcurve);fmr.append(rowfmr);parts.append(rowpart)
            seed_curves.append(root_average(np.asarray(curves),cohort['query_root']))
            seed_fmr.append(root_average(np.asarray(fmr),cohort['query_root']))
            seed_parts.append(root_average(np.asarray(parts),cohort['query_root']))
            # Split test uses seeded tie hashes on pair identifiers.
            import hashlib
            tie=np.asarray([int.from_bytes(hashlib.sha256((c['tie_seed']+cohort['query_id'][i]+cohort['query_id'][j]).encode()).digest()[:8],'big') for i,j in zip(pi,pj)],dtype=np.uint64)
            order=np.lexsort((tie,d['pair_distance']));sc=[];sf=[]
            for cov in c['split_coverage']:
                ix=order[:max(1,int(np.ceil(cov*len(order))))]
                sc.append(float(pair_truth[ix].mean()));sf.append(float(pair_witness[ix].mean()))
            seed_split.append(sc);seed_split_fmr.append(sf)
            bs=[];bf=[]
            for counts in root_counts:
                boot_order=np.repeat(order,counts[pair_root[order]])
                bs.append([pair_truth[boot_order[:max(1,int(np.ceil(cov*len(boot_order))))]].mean() for cov in c['split_coverage']])
                bf.append([pair_witness[boot_order[:max(1,int(np.ceil(cov*len(boot_order))))]].mean() for cov in c['split_coverage']])
            split_boot.append(bs);split_fmr_boot.append(bf)
            # Root-balanced full eligible bank: each root has same condition multiplicity.
            x=d['distance'][cohort['eligible']];y=truth[cohort['eligible']]
            ranks.append(None if np.ptp(x)==0 else float(spearmanr(x,y).statistic))
        roots=np.mean(seed_curves,axis=0);fmr_roots=np.mean(seed_fmr,axis=0)
        boot=roots[resamples].mean(1);boot_fmr=fmr_roots[resamples].mean(1)
        area=np.trapezoid(roots,coverage,axis=-1)/(coverage[-1]-coverage[0]);area_boot=area[resamples].mean(1)
        all_method_roots[family]=area;all_seeds[family]=np.asarray(seed_curves)
        results[family]=dict(discrepancy=roots.mean(0).tolist(),discrepancy_95_ci=interval(boot),
            witnessed_fmr=fmr_roots.mean(0).tolist(),witnessed_fmr_95_ci=interval(boot_fmr),
            normalized_area=float(area.mean()),normalized_area_95_ci=interval(area_boot),
            seed_mean_curves=np.asarray(seed_curves).mean(1).tolist(),
            physical_components_m_rad_mps_radps=np.mean(seed_parts,axis=(0,1)).tolist(),
            spearman_by_seed=ranks,rank_note='NA for constant distance; descriptive pair geometry, not iid pair inference')
        split_results[family]=dict(coverage=c['split_coverage'],discrepancy=np.mean(seed_split,axis=0).tolist(),
                                   discrepancy_95_ci=interval(np.mean(split_boot,axis=0)),
                                   witnessed_fmr=np.mean(seed_split_fmr,axis=0).tolist(),
                                   witnessed_fmr_95_ci=interval(np.mean(split_fmr_boot,axis=0)),
                                   uncertainty='Whole-root resampling; pair-retention ranking recomputed in each sample')
    contrasts={}
    for family in families[:-1]:
        difference=all_method_roots['affine']-all_method_roots[family]
        boot=difference[resamples].mean(1)
        contrasts[family]=dict(affine_minus_comparator=float(difference.mean()),paired_99_ci=interval(boot,.01))
    supported=all(contrasts[m]['paired_99_ci'][1]<0 for m in ['oracle_semantic','dmp','spirl','endpoint'])
    result=dict(status='COMPLETE',coverage=coverage.tolist(),methods=results,
        boundary_split=split_results,contrasts=contrasts,affine_superiority_decision='SUPPORT' if supported else 'UNSUPPORTED',
        query_roots=nr,gallery_roots=len(set(cohort['gallery_root'])),query_interactions=len(q),gallery_interactions=len(g),
        audit_branches=len(c['audit_xi_m']),valid_pair_count=int(cohort['eligible'].sum()),
        random_eligible_discrepancy=float(np.mean([truth[i,cohort['eligible'][i]].mean() for i in range(nq)])),
        same_verb_witnessed_rate=float(separated[cohort['eligible']].mean()),
        uncertainty='Query-root bootstrap conditional on fixed gallery and trained models; training seed mean is primary.',
        scope='RollBall matched-context planar tool travel; hidden linear+angular damping; exact 3D rigid-body response audit.')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'physical_audit.npz',truth=truth,separated=separated,components=components,
                         pair_truth=pair_truth,pair_witness=pair_witness,**{f'roots_{m}':v for m,v in all_method_roots.items()})
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(oracle_semantic='Oracle Semantic',dmp='DMP',spirl='SPiRL',endpoint='PAP-Endpoint',nonlinear='PAP-Nonlinear',affine='PAP-Affine')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for method in families:
        row=results[method]
        for ax,key,ci in [(axes[0],'discrepancy','discrepancy_95_ci'),(axes[1],'witnessed_fmr','witnessed_fmr_95_ci')]:
            line=ax.plot(coverage*100,row[key],marker='o',label=labels[method])[0]
            ax.fill_between(coverage*100,*row[ci],color=line.get_color(),alpha=.10)
    axes[0].set_ylabel('Audit response discrepancy (physical-scale normalized)')
    axes[1].set_ylabel('Witnessed physical false-merge rate')
    for ax in axes:ax.set_xlabel('Retrieval coverage (%)');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'boundary_audit.png',dpi=220);fig.savefig(out/'boundary_audit.pdf');plt.close(fig)
    lines=['# Physical Organization Audit of Action Primitive Representations','',
           f'Completed audit. PAP-Affine superiority over the four comparator organizers (oracle semantic, DMP, SPiRL, endpoint): **{result["affine_superiority_decision"]}**.',
           '',f'{nr} query roots; {result["gallery_roots"]} fixed gallery roots; {len(c["audit_xi_m"])} independent audit interventions.',
           '', '| Organizer | Normalized discrepancy area | 95% conditional root interval |', '|---|---:|---|']
    for method in families:
        row=results[method];lines.append(f'| {labels[method]} | {row["normalized_area"]:.4f} | {row["normalized_area_95_ci"]} |')
    lines+=['','The simulator alone defines the physical response audit. No-separation observations do not establish equivalence.',
            'Intervals are conditional on the fixed gallery; they preserve all condition variants within query roots.',
            'The study is scoped to matched RollBall contexts and the declared damping intervention. It does not establish universal cross-task boundaries.',
            'PAP-Affine uses the explicit affine rigid response head with a raw-state source encoder trained from scratch; no pretrained backbone is used.',
            'SPiRL uses the official closed-loop skill model. DMP fits measured TCP trajectories. Per-method model selection details are in RANKINGS.json.',
            '', '![Boundary audit](boundary_audit.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
