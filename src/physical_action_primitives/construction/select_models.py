"""Select one model per construction (lowest development score across four equal-budget seeds)."""
import argparse,json,shutil
from pathlib import Path
import torch
from physical_action_primitives.paths import project_root,resolve

FAMILIES=['affine','endpoint','nonlinear','policy','action_seq']

def rel(p):
    try:return str(Path(p).resolve().relative_to(project_root()))
    except ValueError:return str(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--roots',nargs='+',default=[str(resolve('checkpoints','pap_candidates'))],help='directories holding <construction>_<seed>/model.pt candidates');ap.add_argument('--output',default=str(resolve('checkpoints','pap')),help='selected models are written to <output>/<construction>/model.pt');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);choices={};candidates={}
    for family in FAMILIES:
        rows=[]
        for root in a.roots:
            for p in Path(root).glob(f'{family}_*/model.pt'):
                training=p.parent/'TRAINING.json'
                if not training.exists():continue
                ck=torch.load(p,map_location='cpu',weights_only=False)
                if ck['method']!=family:continue
                log=json.loads(training.read_text())
                assert log['updates']==10000
                assert ck['intervention_contract']=='object-local requested displacement / 0.01m'
                rows.append(dict(path=rel(p),seed=ck['seed'],development_score=ck['development_score'],step=ck['step'],parameters=log['parameters']))
        assert sorted(r['seed'] for r in rows)==[104000,104001,104002,104003],(family,rows)
        winner=min(rows,key=lambda r:(r['development_score'],r['seed']));dst=out/family;dst.mkdir(parents=True)
        src=resolve(winner['path']);shutil.copy2(src,dst/'model.pt');shutil.copy2(src.parent/'TRAINING.json',dst/'TRAINING.json')
        choices[family]=winner;candidates[family]=rows
    selection=dict(rule='minimum development prediction loss; policy uses development inverse-intervention error; same rule across seeds within each construction',choices=choices,candidates=candidates,training_budget_per_construction='four seeds x 10000 head updates; frozen 800M backbone; identical seven-probe source packets',evaluation=dict(seed=105000,parents_per_environment=60,paired_damping_seed=105900,paired_damping_parents_per_environment=12,source_identity_cohort='development first two metadata-ordered parents',local_target_dy_m=[.006,.009,.012],extrapolation_target_dy_m=[.015,.018,.021],target_dx_m=[-.003,0,.003],position_tolerance_m=.005,yaw_tolerance_rad=0.0872664626,terminal_speed_tolerance_m_s=.05,action_steps=16,source_probe_indices=[12,2,22,10,14,0,24]))
    (out/'SELECTION.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection),flush=True)

if __name__=='__main__':main()
