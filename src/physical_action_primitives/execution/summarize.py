"""Tables, figures and a report for the execution-substitution, calibration and sequential-invocation study."""
import argparse,json
from pathlib import Path
import numpy as np
from physical_action_primitives.paths import resolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS=['PushT-v1','PokeCube-v1','RollBall-v1']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(resolve('results','execution')),help='output directory of execution.evaluate; figures and tables are written here');a=ap.parse_args();root=Path(a.results);data=root;payload=json.loads((data/'RESULTS.json').read_text());records=payload['records'];fig=root/'figures';fig.mkdir(exist_ok=True);groups=[];examples={};rng=np.random.default_rng(111900)
    for key in sorted(set((r['family'],r['environment'],r['method'],r['condition']) for r in records)):
        rows=[r for r in records if (r['family'],r['environment'],r['method'],r['condition'])==key];arrays=[dict(np.load(data/r['artifact'])) for r in rows];success=np.stack([z['success'] for z in arrays]);n=success.shape[1];samples=success[:,rng.integers(n,size=(2000,n))].mean((0,2));lo,hi=np.quantile(samples,[.025,.975]);groups.append(dict(family=key[0],environment=key[1],method=key[2],condition=key[3],success=int(success.sum()),total=int(success.size),rate=float(success.mean()),ci95=[float(lo),float(hi)],mean_error_mm=float(np.mean([r['mean_error_mm'] for r in rows]))))
    def group(f,t,m,c):return next(r for r in groups if (r['family'],r['environment'],r['method'],r['condition'])==(f,t,m,c))
    def get(f,t,m,c,g):
        r=next(r for r in records if (r['family'],r['environment'],r['method'],r['condition'],r['goal'])==(f,t,m,c,g));return r,dict(np.load(data/r['artifact']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#176ca4','#e67c30','#36936a']
    def savefig(name):
        plt.savefig(fig/(name+'.png'),dpi=180,bbox_inches='tight');plt.savefig(fig/(name+'.pdf'),bbox_inches='tight');plt.close()
    f,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,t in zip(axes,TASKS):
        for j,method in enumerate(['affine','endpoint']):
            vals=[group('substitution',t,method,c) for c in ['linear','smoothstep','front_loaded']];rates=np.array([x['rate'] for x in vals])*100
            ax.bar(np.arange(3)+(j-.5)*.32,rates,.32,label=method,color=colors[j]);ax.errorbar(np.arange(3)+(j-.5)*.32,rates,yerr=np.array([[v['rate']-v['ci95'][0] for v in vals],[v['ci95'][1]-v['rate'] for v in vals]])*100,fmt='none',ecolor='black',capsize=2)
        ax.set_xticks(range(3),['Linear','Smooth','Front-loaded']);ax.set_ylim(0,108);ax.set_title(t);ax.set_ylabel('Successful calls (%)')
    axes[0].legend();f.suptitle('Fixed identity and invocation, changed execution timing');f.tight_layout();savefig('figure1_execution_substitution')
    f,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,t in zip(axes,TASKS):
        cs=['normal','mismatch','retry','interface_repair','fixed_gain_adjustment'];vals=[group('debugging',t,'affine',c) for c in cs]
        ax.bar(range(5),[x['rate']*100 for x in vals],color=['#176ca4','#aaa','#aaa','#36936a','#e67c30']);ax.set_xticks(range(5),['Normal','Scale\nmismatch','Retry','Measured\nrepair','Fixed\nadjustment']);ax.set_ylim(0,108);ax.set_title(t);ax.set_ylabel('Successful calls (%)')
    f.suptitle('A visible execution-scale discrepancy supports targeted repair');f.tight_layout();savefig('figure2_interface_debugging')
    f,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,t in zip(axes,TASKS):
        for j,m in enumerate(['affine','endpoint','repeat_first_intervention']):
            vals=[]
            for stage in range(3):
                _,z=get('sequence',t,m,'stage',stage);vals.append(z['success'].mean()*100)
            ax.plot([1,2,3],vals,'o-',label=m,color=colors[j])
        ax.set_xticks([1,2,3]);ax.set_xlabel('Waypoint (no reset between calls)');ax.set_ylabel('Waypoint success (%)');ax.set_ylim(0,108);ax.set_title(t)
    axes[0].legend(fontsize=8);f.suptitle('One source identity, three sequential invocations');f.tight_layout();savefig('figure3_sequential_reuse')
    # Illustration selection is descriptive and does not change any denominator.
    best=None
    for t in TASKS:
        for g in range(3):
            r,normal=get('debugging',t,'affine','normal',g);_,bad=get('debugging',t,'affine','mismatch',g);rr,repair=get('debugging',t,'affine','interface_repair',g)
            improvement=bad['error_m']-repair['error_m'];lane=int(improvement.argmax());score=float(improvement[lane])
            if best is None or score>best[0]:best=(score,t,g,lane,r,rr,normal,bad,repair)
    score,t,g,lane,r,rr,normal,bad,repair=best;examples['debugging']=dict(environment=t,goal=g,lane=lane,selection='largest measured position-error reduction among all affine mismatch/repair pairs; full aggregate tables retained',repair_artifact=rr['artifact'],normal_artifact=r['artifact'],improvement_mm=score*1000)
    f,axes=plt.subplots(1,2,figsize=(9,3.5))
    for z,label,col in [(normal,'Normal',colors[0]),(bad,'Mismatch','#999'),(repair,'Measured repair',colors[2])]:
        tcp=z['tcp'][:,lane,:2];distance=np.linalg.norm(tcp-tcp[0],axis=-1)*1000;axes[0].plot(np.arange(len(tcp))/20,distance,label=label,color=col)
        axes[1].scatter(z['actual'][lane,0]*1000,z['actual'][lane,1]*1000,label=label,color=col,s=65)
    axes[0].set(xlabel='Time (s)',ylabel='TCP displacement (mm)');axes[0].legend();axes[1].scatter(*repair['target'][lane,:2]*1000,marker='*',s=150,color='black',label='Target');axes[1].set(xlabel='Local x (mm)',ylabel='Local y (mm)');axes[1].legend();f.suptitle(f'Illustrated debugging case: {t}, parent {lane}, goal {g}');f.tight_layout();savefig('debugging_example')
    # Record actual timed-trajectory differences with paired calls.
    differences=[]
    for t in TASKS:
        for c in ['smoothstep','front_loaded']:
            values=[]
            for g in range(3):
                _,ref=get('substitution',t,'affine','linear',g);_,other=get('substitution',t,'affine',c,g);values.extend(np.sqrt(np.square(other['tcp'][:,:,:3]-ref['tcp'][:,:,:3]).sum(-1).mean(0)).tolist())
            differences.append(dict(environment=t,profile=c,tcp_timed_rms_mm=float(np.mean(values)*1000)))
    summary=dict(groups=groups,examples=examples,trajectory_differences=differences)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(root/'EXAMPLES.json').write_text(json.dumps(examples,indent=2))
    def count(v):return f"{v['success']}/{v['total']} ({100*v['rate']:.1f}%)"
    text=['# Explicit Action Interfaces in Use','','Three-environment execution study using the selected PAP models; no additional training.','','The study covers execution substitution, interface-guided repair, and sequential invocation without state reset. It uses explicit contact preparation and simulator state; all selected parent states, targets and methods appear in the tables below.','','## 1. Fixed identity, changed execution timing','','![Execution substitution](figures/figure1_execution_substitution.png)','','The source identity and the invocation target are fixed while the progress function of the 12-step motion is changed: linear, smoothstep, or square-root front-loaded progress, followed by 4 holding steps. These change the realized time course; the spatial motion adapter, arm and control frequency are shared. The three targets are (-3, 12), (0, 18) and (3, 21) mm.','','| Method / implementation | PushT | PokeCube | RollBall |','|---|---:|---:|---:|']
    for m in ['affine','endpoint']:
        for c in ['linear','smoothstep','front_loaded']:text.append('| '+m+' / '+c+' | '+' | '.join(count(group('substitution',t,m,c)) for t in TASKS)+' |')
    text.append('| nominal source replay | '+' | '.join(count(group('substitution',t,'source_replay','nominal')) for t in TASKS)+' |')
    text+=['','| Environment | Smoothstep vs. linear TCP timed RMS | Front-loaded vs. linear TCP timed RMS |','|---|---:|---:|']
    for t in TASKS:text.append('| '+t+' | '+' | '.join(f"{next(v['tcp_timed_rms_mm'] for v in differences if v['environment']==t and v['profile']==c):.3f} mm" for c in ['smoothstep','front_loaded'])+' |')
    text+=['','## 2. Interface-guided targeted repair','','![Debugging](figures/figure2_interface_debugging.png)','','An execution displacement-scale mismatch of 0.55 is injected. Two calibration strokes of 20 mm and 30 mm (16 steps each) provide the realized TCP displacement; the scale estimate is the difference between the two measured forward displacements divided by 10 mm, clipped to 0.2..1.5, and the correction is its inverse. The estimate does not read the injected scale. The fault location is given by construction, so this is a sensor-guided repair of a known fault. Calibration includes the realized tracking dynamics, so the estimate reflects the implementation response in this setting.','','Recovery conditions all count a 32-step calibration budget and then execute the 16-step target call from the same parent state; before/after comparisons are paired resets. Retry keeps the original implementation; fixed adjustment multiplies by 1.5; measured repair uses the measured inverse. The affine and endpoint constructions receive identical corrections.','','| Method / condition | PushT | PokeCube | RollBall |','|---|---:|---:|---:|']
    for m in ['affine','endpoint']:
        for c in ['normal','mismatch','retry','interface_repair','fixed_gain_adjustment']:text.append('| '+m+' / '+c+' | '+' | '.join(count(group('debugging',t,m,c)) for t in TASKS)+' |')
    text+=['','![Debugging example](figures/debugging_example.png)','','Example selection: the affine pair with the largest position-error reduction is shown. The selection is illustrative only; the complete method/environment denominators are in the main table, and goal-level records and continuous errors are stored in RESULTS.json and the npz files.','','## 3. Three sequential invocations','','![Sequential reuse](figures/figure3_sequential_reuse.png)','','The targets are three waypoints (0, 9), (3, 18) and (0, 27) mm relative to the initial object position. State and object are not reset between stages. Each call shares a 180-step physical preparation and a 16-step execution (588 control steps in total); later calls are planned from the actual current context, with the source identity unchanged. The control repeats the first physical intervention with the same contact preparation. A full sequence succeeds only if all three waypoints meet the outcome tolerance.','','| Method / all three waypoints succeed | PushT | PokeCube | RollBall |','|---|---:|---:|---:|']
    for m in ['affine','endpoint','repeat_first_intervention']:text.append('| '+m+' | '+' | '.join(count(group('sequence_complete',t,m,'three_waypoints')) for t in TASKS)+' |')
    text+=['','## Implementation','','Parent states per environment: '+str(len(np.load(data/records[0]['artifact'])['success']))+'. Collection seeds are in the COMPLETE.json file of each cohort. Preparation alternates centered and offset contact; the tool is PandaStick or a closed Panda gripper; for PokeCube the idle peg is moved out of the workspace; control runs at 20 Hz and simulation at 100 Hz. The source identity is the first two development source packets, matched to each recipient by public initial context, and is fixed across all operations. The current 12-dimensional context is recomputed at every call.','','Models: checkpoints/pap/{affine,endpoint}/model.pt. Frozen 800M backbone features come from the development cohort; there are no backbone or head updates. There are 81 candidates in metric units, over the same extrapolation range as the Invoke study.','','Outcome tolerances: position 5 mm, yaw 5 degrees, terminal speed 0.05 m/s, and legal joint targets. They describe each outcome and do not exclude parent states or block later execution. 95% intervals resample parent states; SUMMARY.json stores the intervals for each group and the continuous errors.','']
    (root/'REPORT.md').write_text('\n'.join(text));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
