"""Train clean CNN/Context/RL, then refine CNN on configured domains and warm-start Context/RL.
No heldout scoring is performed in training. Use the official evaluator afterwards.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import amcal_burst_official_protocol as official
core=official.core


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--stage1-rl-epochs',type=int,default=12)
    p.add_argument('--cnn-multishift-epochs',type=int,default=30)
    p.add_argument('--cnn-multishift-patience',type=int,default=10)
    p.add_argument('--stage2-pretrain-epochs',type=int,default=0)
    p.add_argument('--reuse-multishift-run',type=Path)
    p.add_argument('--refine-routing-only',action='store_true',help='Reuse CNN/Context unchanged; allow new route auxiliary weight and refine selector only.')
    p.add_argument('--multishift-levels',type=int,nargs='+',default=[0,3,12])
    extra,remaining=p.parse_known_args()
    levels=tuple(sorted(set(extra.multishift_levels)))
    if len(levels)<2 or levels[0]!=0 or any(v<0 for v in levels):p.error('Specify zero and at least one distinct positive Burst level')
    reuse=extra.reuse_multishift_run.resolve() if extra.reuse_multishift_run else None
    if extra.refine_routing_only and (reuse is None or extra.stage2_pretrain_epochs != 0):p.error('Routing refinement requires --reuse-multishift-run and --stage2-pretrain-epochs 0')
    if extra.stage2_pretrain_epochs < 0:p.error('Pretraining epochs must be nonnegative')
    if min(extra.stage1_rl_epochs,extra.cnn_multishift_epochs,extra.cnn_multishift_patience)<1:p.error('Stage epochs/patience must be positive')
    if 'train' not in remaining:p.error('This entry point is training only; use amcal_burst_official_protocol.py evaluate for testing')
    if '--official-split-dir' not in remaining:p.error('--official-split-dir is required')
    root=Path(remaining[remaining.index('--burst-dir')+1]).resolve()
    if '--train-perturbations' in remaining:p.error('Training stages use clean 0, then --multishift-levels; omit --train-perturbations')
    def train_stages(dataset_path,base_epoch=100,base_patience=15,adver_epoch=100,adver_patience=15,pretrain_epochs=20):
        train,val,_unused_test,full,nclasses,width,counts=core.load_data(dataset_path=dataset_path)
        if set(core.TRAIN_PERTURBATIONS)!=set(levels):raise ValueError('Final domain configuration differs')
        digest=core.file_sha256('Models/protocol_preprocessing.joblib')
        def save_cnn(cnn):
            torch.save(dict(preprocessing_sha256=digest,protocol_version=3,model_state_dict=cnn.state_dict(),input_length=width,num_classes=nclasses),'Models/initial_cnn_model.pth')
        state_width=width+2*nclasses+2+(4 if core.ROUTE_HISTORY_ENABLED else 0)
        if reuse is None:
            print('\nSTAGE 1A: CNN on clean training rows only; clean validation checkpoint selection.',flush=True)
            cnn=core.CNN1D(width,nclasses).to(core.device)
            cnn=core.train_initial_cnn(cnn,train,val,counts,base_epoch,base_patience)
            save_cnn(cnn)
            state_width=width+2*nclasses+2+(4 if core.ROUTE_HISTORY_ENABLED else 0)
            agent=core.DQNSelectorAgent(state_width,core.device)
            context=core.ContextAwareNetwork(width).to(core.device)
            core.TRAIN_PERTURBATIONS=(0.0,)
            print('\nSTAGE 1B: Context and RL on clean rows only.',flush=True)
            cnn,agent,context=core.train_adversarial_system(cnn,agent,context,full,val,counts,width,nclasses,extra.stage1_rl_epochs,min(adver_patience,extra.stage1_rl_epochs),pretrain_epochs)
            archive=Path('Stage1/Models');archive.mkdir(parents=True,exist_ok=True)
            for path in Path('Models').iterdir():
                if path.is_file():shutil.copy2(path,archive/path.name)
            shutil.copy2('burst_training_domains.json','Stage1/burst_training_domains.json')
            prior_policy=deepcopy(agent.policy_net.state_dict());prior_context=deepcopy(context.state_dict())
        else:
            print('Reusing multishift CNN, Context and policy; no CNN retraining.',flush=True)
            old=joblib.load(reuse/'Models/protocol_preprocessing.joblib')
            new=joblib.load('Models/protocol_preprocessing.joblib')
            if old['source_sha256']!=new['source_sha256'] or old['feature_names']!=new['feature_names']:
                raise ValueError('Reuse source/features differ')
            for key in ('train_ids','val_ids','test_ids'):
                if not np.array_equal(old[key],new[key]):raise ValueError('Reuse split differs: '+key)
            for key in ('mean_','scale_'):
                if not np.allclose(getattr(old['scaler'],key),getattr(new['scaler'],key)):raise ValueError('Reuse scaling differs')
            if not np.array_equal(old['label_encoder'].classes_,new['label_encoder'].classes_):raise ValueError('Reuse classes differ')
            def saved(name):return torch.load(reuse/'Models'/name,map_location='cpu',weights_only=False)
            cnn=core.CNN1D(width,nclasses).to(core.device)
            cnn.load_state_dict(saved('initial_cnn_model.pth')['model_state_dict'])
            prior_context=saved('adversarial_best_context.pth')['context_model_state_dict']
            policy=saved('adversarial_best_dqn.pth')
            if policy.get('routing_reward_mode','absolute')!=core.ROUTING_REWARD_MODE or policy.get('route_history',False)!=core.ROUTE_HISTORY_ENABLED or (not extra.refine_routing_only and policy.get('route_aux_weight',0)!=core.ROUTE_AUX_WEIGHT):
                raise ValueError('Reuse selector configuration differs')
            prior_policy=policy['dqn_policy_net_state_dict']
            context=core.ContextAwareNetwork(width).to(core.device)
        if reuse is not None:
            old_config=json.loads((reuse/'two_stage_training.json').read_text())
            if set(old_config['cnn_refinement_domains'])!=set(levels):raise ValueError('Reused CNN saw different intensities; use a fresh run.')
        art=joblib.load('Models/protocol_preprocessing.joblib')
        def combined_loader(ids,shuffle):
            frames=[]
            for level in levels:
                path=root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
                frame,source_ids,_=official.mapped_domain(path,art,root,False)
                lookup={int(v):i for i,v in enumerate(source_ids)}
                selected=[lookup[int(i)] for i in ids if int(i) in lookup]
                if len(selected)!=len(ids) and not art.get('allow_partial_domains',False):raise ValueError('Supplied domain does not cover partition')
                if not selected:raise ValueError('No supplied-domain rows in this partition')
                frames.append(frame.iloc[selected])
            frame=pd.concat(frames,ignore_index=True)
            x=art['scaler'].transform(frame.iloc[:,:-1].to_numpy()).astype(np.float32)
            if not np.isfinite(x).all():raise ValueError('Non-finite domain features')
            y=art['label_encoder'].transform(frame.iloc[:,-1])
            ds=TensorDataset(torch.tensor(x,device=core.device).unsqueeze(1),torch.tensor(y,dtype=torch.long,device=core.device))
            return DataLoader(ds,batch_size=64,shuffle=shuffle)
        multitrain=combined_loader(art['train_ids'],True)
        multival=combined_loader(art['val_ids'],False)
        if reuse is None:
            print(f'\nSTAGE 2A: Continue CNN on training domains {levels}; validation-only selection.',flush=True)
            for parameter in cnn.parameters():parameter.requires_grad_(True)
            cnn=core.train_initial_cnn(cnn,multitrain,multival,counts,extra.cnn_multishift_epochs,extra.cnn_multishift_patience)
        cnn.eval()
        for parameter in cnn.parameters():parameter.requires_grad_(False)
        save_cnn(cnn)
        print(f'\nSTAGE 2B: Warm-start Context and RL; fresh replay, optimizer and target sync; domains {levels}.',flush=True)
        agent=core.DQNSelectorAgent(state_width,core.device)
        agent.policy_net.load_state_dict(prior_policy);agent.update_target_network()
        assert len(agent.replay_buffer)==0
        context.load_state_dict(prior_context)
        core.TRAIN_PERTURBATIONS=tuple(float(v) for v in levels)
        # Pretrain against the new CNN on configured TRAIN domains, then keep per-domain RL episodes.
        pretrain_loader=multitrain if extra.stage2_pretrain_epochs else full
        core.train_adversarial_system(cnn,agent,context,pretrain_loader,val,counts,width,nclasses,adver_epoch,adver_patience,extra.stage2_pretrain_epochs,episode_loaders=(full,val))
        if extra.refine_routing_only:
            saved_context=torch.load('Models/adversarial_best_context.pth',map_location='cpu',weights_only=False)['context_model_state_dict']
            if any(not torch.equal(saved_context[k].cpu(),v.cpu()) for k,v in prior_context.items()):
                raise RuntimeError('Routing refinement unexpectedly changed initial Context')
            print('Verified: initial Context weights unchanged; CNN reused unchanged. Only selector checkpoint refined.')
        ckpath=Path('Models/adversarial_best_dqn.pth');ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        ck['two_stage_protocol']='clean CNN/Context/RL -> CNN multishift refinement -> warm-start Context/RL with empty replay'
        torch.save(ck,ckpath)
        manifest=dict(refine_routing_only=extra.refine_routing_only,reused_run=str(reuse) if reuse else None,stage2_pretrain_epochs=extra.stage2_pretrain_epochs,stage1_domains=[0],stage1_rl_epochs=extra.stage1_rl_epochs,cnn_refinement_domains=list(levels),cnn_refinement_max_epochs=extra.cnn_multishift_epochs,stage2_rl_domains=list(levels),stage2_rl_max_epochs=adver_epoch,context_warm_start=True,policy_warm_start=True,selector_replay_reset=True,heldout_scoring_during_training=False,preprocessing_sha256=digest)
        Path('two_stage_training.json').write_text(json.dumps(manifest,indent=2))
        print('\nTwo-stage training complete. Final models saved in Models; stage-1 snapshots in Stage1/Models. No heldout test was scored.',flush=True)
    core.train_main=train_stages
    sys.argv=[sys.argv[0]]+remaining+['--train-perturbations']+[str(v) for v in levels]
    official.main()

if __name__=='__main__':main()
