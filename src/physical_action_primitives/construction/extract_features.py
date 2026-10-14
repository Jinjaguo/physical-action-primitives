"""Extract frozen-backbone features for each branch cohort, using realized joint targets as the action channel."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from physical_action_primitives.backbone.model import EmergentActionUnitLearner,UnitLearnerConfig
from physical_action_primitives.backbone.training_data import normalization_from_manifest
from physical_action_primitives.backbone.data import fresh_unit_arrays,normalize_unit_arrays,extract_lazy_window_batch
from physical_action_primitives.construction.interfaces import RESPONSE_SCALE
from physical_action_primitives.paths import resolve

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--backbone-plan',default=str(resolve('configs','backbone','analysis_plan_800m.json')));a=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(a.backbone_plan).read_text())
    ck=torch.load(resolve(plan['checkpoint']),map_location='cpu',weights_only=False)
    model=EmergentActionUnitLearner(UnitLearnerConfig(**ck['model_config'])).cuda();model.load_state_dict(ck['model_state']);del ck
    model.eval();model.requires_grad_(False);norm=normalization_from_manifest(resolve(plan['normalization']))
    sources=[]
    for entry in sorted(Path(a.data).glob('*/COMPLETE.json')):
        meta=json.loads(entry.read_text());d=np.load(entry.parent/'branches.npz');src=d['source_indices'];n=d['object'].shape[2];steps=d['action'].shape[1]
        hidden=np.empty((n,7,steps,2032),np.float16);physical=np.empty((n,7,steps,6),np.float32);actions=np.zeros((n,7,8),np.float32)
        for j,k in enumerate(src):
            primary=d['object'][k];tcp=d['tcp'][k];qpos=d['qpos'][k];qvel=d['qvel'][k]
            # pd_joint_delta_pos uses +/-0.1 rad. Encode the realized IK joint target
            # relative to the pre-step qpos, not the Cartesian command.
            action=(d['joint_target'][k]-qpos[:-1,:,:7])/.1
            if qpos.shape[-1]>7:action=np.concatenate([action,d['action'][k,:,:,3:4]],-1)
            actions[:,j,:action.shape[-1]]=action.mean(0)
            payload={'physical/primary':primary,'physical/context':np.zeros_like(primary),'physical/context_mask':np.zeros((steps+1,n,1),bool),'physical/tcp_pose':tcp,'physical/qpos':qpos,'physical/qvel':qvel,'physical/typed_contact_force':np.zeros((steps+1,n,21),np.float32),'physical/contact_channel_mask':np.zeros((steps+1,n,7),bool),'action/processed_controller_space':action,'validity/finite':np.ones((steps,n),bool)}
            arrays=normalize_unit_arrays(fresh_unit_arrays(payload),norm)
            for begin in range(0,n,32):
                ix=np.arange(begin,min(begin+32,n))
                # Eight-step context windows, as in backbone training, applied to each block.
                for start in range(0,steps,8):
                    batch=extract_lazy_window_batch(arrays,ix,np.full(len(ix),start),length=8)
                    values=np.concatenate([batch[z] for z in ['state','action','state_delta','effect','contact_active']],-1).astype(np.float32)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        h=model.input_projection(model.input_norm(torch.tensor(values,device='cuda')))
                        for block in model.encoder:h=block(h)
                        h=model.encoder_norm(h)
                    hidden[ix,j,start:start+8]=h.float().cpu().numpy().astype(np.float16)
            q=primary[:,:,3:7];w,x,y,z=np.moveaxis(q,-1,0);angle=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));da=np.arctan2(np.sin(np.diff(angle,axis=0)),np.cos(np.diff(angle,axis=0)))
            direction=d['direction'];c=np.cos(direction);s=np.sin(direction)
            delta=np.diff(primary[:,:,:2],axis=0);local=np.stack([delta[:,:,0]*c+delta[:,:,1]*s,-delta[:,:,0]*s+delta[:,:,1]*c],-1)
            v=primary[1:,:,7:9];lv=np.stack([v[:,:,0]*c+v[:,:,1]*s,-v[:,:,0]*s+v[:,:,1]*c],-1)
            physical[:,j]=np.concatenate([local,da[:,:,None],lv,primary[1:,:,12:13]],-1).transpose(1,0,2)/np.asarray(RESPONSE_SCALE)
        # Public h: relative tool geometry, observed velocity and object scale.
        primary=d['object'][12,0];tcp=d['tcp'][12,0];rel=tcp[:,:3]-primary[:,:3];direction=d['direction'];c=np.cos(direction);s=np.sin(direction)
        rxy=np.stack([rel[:,0]*c+rel[:,1]*s,-rel[:,0]*s+rel[:,1]*c],-1)
        h=np.concatenate([rxy/.1,rel[:,2:3]/.1,primary[:,7:10]/.05,primary[:,10:13],np.tile([meta['spec']['back']/.1,meta['spec']['tool']/.1,meta['spec']['height']/.1],(n,1))],-1).astype(np.float32)
        np.savez(out/f'{meta["environment"]}.npz',hidden=hidden,step=physical,action=actions,h=h,y=d['response'][:25].transpose(1,0,2)/np.asarray(RESPONSE_SCALE),xi=d['grid'],source_indices=src,stage=d['stage'],direction=d['direction'],offset=d['offset'])
        sources.append(dict(environment=meta['environment'],parents=n,seed=meta['seed']))
        print(json.dumps(sources[-1]),flush=True)
    (out/'FEATURES.json').write_text(json.dumps(dict(checkpoint=plan['checkpoint'],normalization=plan['normalization'],representation='frozen encoder_norm; two consecutive 8-step blocks; FP16 cache',backbone_updates=0,action_encoding='realized joint target delta / 0.1rad plus original gripper command',sources=sources),indent=2))

if __name__=='__main__':main()
