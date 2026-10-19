"""Warm-started K-Context bank: 1+2K DDQN actions, no class/flow routing.

Default uniform feedback isolates prediction/update routing from query scarcity.
Each episode/domain starts from the same healthy source Context weights.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import shutil

import joblib
import numpy as np
import torch
from torch import nn
import amcal_burst_official_protocol as official

core = official.core


def route_index(action, slots):
    if not 0 <= action < 1+2*slots:
        raise ValueError('Invalid bank action.')
    return 0 if action == 0 else (action-1) % slots + 1


def legal_actions(slots, remaining, scheduled):
    if remaining <= 0 or scheduled == 0:
        return list(range(slots+1))
    if scheduled == 1:
        return list(range(slots+1, 2*slots+1))
    return list(range(2*slots+1))


class BankQ(nn.Module):
    def __init__(self, width, slots):
        super().__init__()
        hidden = min(512, max(128, width*2))
        self.network = nn.Sequential(nn.Linear(width,hidden), nn.ReLU(), nn.Dropout(.3),
                                     nn.Linear(hidden,hidden//2), nn.ReLU(), nn.Dropout(.3),
                                     nn.Linear(hidden//2,1+2*slots))

    def forward(self, state):
        return self.network(state)


def configure(a):
    slots = a.slots
    core.DQNSelector = lambda width: BankQ(width,slots)
    def mask(values, states):
        q = values.clone()
        # Last state values: scheduled query (-1=policy), budget, elapsed.
        deny_query = (states[:,-2] <= 0) | (states[:,-3] == 0)
        force_query = (states[:,-2] > 0) & (states[:,-3] == 1)
        q[deny_query,slots+1:] = -torch.inf
        q[force_query,:slots+1] = -torch.inf
        return q
    core.mask_query_action = mask
    core.SELECTION_COST, core.GAIN_WEIGHT, core.RETENTION_WEIGHT = a.selection_cost,a.gain_weight,a.retention_weight
    core.REFERENCE_BATCH_SIZE, core.REFERENCE_HISTORY_SIZE = 32,256
    core.ROUTE_AUX_WEIGHT = 0


def read_domain(root, artifact, level, split):
    path = root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
    frame, ids, _ = official.mapped_domain(path,artifact,root,False)
    lookup = {int(v):i for i,v in enumerate(ids)}
    selected = artifact[split]
    frame = frame.iloc[[lookup[int(i)] for i in selected]]
    x = artifact['scaler'].transform(frame.iloc[:,:-1]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite domain features.')
    y = artifact['label_encoder'].transform(frame.iloc[:,-1])
    return torch.tensor(x,device=core.device).unsqueeze(1), y


def make_models(artifact, models, slots):
    width, classes = len(artifact['feature_names']),len(artifact['label_encoder'].classes_)
    cnn = core.CNN1D(width,classes).to(core.device)
    cnn.load_state_dict(torch.load(models/'initial_cnn_model.pth',map_location='cpu',weights_only=False)['model_state_dict'])
    cnn.eval()
    for p in cnn.parameters():p.requires_grad_(False)
    initial = torch.load(models/'adversarial_best_context.pth',map_location='cpu',weights_only=False)['context_model_state_dict']
    bank = [core.ContextAwareNetwork(width).to(core.device) for _ in range(slots)]
    for context in bank:
        context.load_state_dict(initial)
        context.eval()
    state_width = width+(slots+1)*classes+slots+3
    agent = core.DQNSelectorAgent(state_width,core.device,batch_size=32,epsilon_start=.2,epsilon_end=.05)
    return cnn,bank,agent,initial


def state_at(cnn,bank,x,used,cap,position,count,slot_queries,scheduled):
    with torch.no_grad():
        logits = [cnn(x)] + [cnn(context(x)) for context in bank]
        extra = x.new_tensor([[*(v/max(cap,1) for v in slot_queries),scheduled,
                               (cap-used)/max(cap,1),position/max(count,1)]])
        state = torch.cat([x.flatten(1),*[z.softmax(1) for z in logits],extra],dim=1)
    return state,logits


def observer(cnn,bank,agent,x,y,seed,mode):
    """Retrospective no-feedback scores; never used by training rewards."""
    all_logits = []
    with torch.no_grad():
        for start in range(0,len(y),64):
            bx = x[start:start+64]
            all_logits.append(torch.stack([cnn(bx),*[cnn(c(bx)) for c in bank]],dim=1))
        logits = torch.cat(all_logits)
        extras = x.new_zeros((len(y),len(bank)+3))
        extras[:,-3] = 0  # Observer permits prediction routes only, no query.
        extras[:,-1] = torch.arange(len(y),device=core.device)/max(len(y),1)
        states = torch.cat([x.flatten(1),logits.softmax(2).flatten(1),extras],dim=1)
        if mode == 'random':
            routes = torch.tensor(np.random.default_rng(seed+30003).integers(0,len(bank)+1,len(y)),device=core.device)
        else:
            routes = core.mask_query_action(agent.policy_net(states),states).argmax(1)
        predictions = logits[torch.arange(len(y),device=core.device),routes].argmax(1).cpu().numpy()
        per_slot = (logits.argmax(2).cpu().numpy()==np.asarray(y)[:,None]).mean(0)
    return predictions,per_slot.tolist()


def episode(a,cnn,bank,agent,initial,x,y,mode,epsilon,learning,training=False,diagnostic=None):
    slots,n = len(bank),len(y)
    for context in bank:
        context.load_state_dict(initial)
        context.eval()
    optimizers = [torch.optim.Adam(c.parameters(),lr=a.context_lr,weight_decay=1e-4) for c in bank]
    order = np.random.default_rng(a.seed).permutation(n)
    x,y = x[order],np.asarray(y)[order]
    cap = int(np.floor(a.budget_fraction*n))
    schedule = set(np.random.default_rng(a.seed+819003).choice(n,cap,replace=False).tolist()) if a.query_schedule=='uniform' else None
    def gate(pos):return int(pos in schedule) if schedule is not None else -1
    rng = np.random.default_rng(a.seed+719003)
    counts,slot_queries,actions = [0]*slots,[0]*slots,Counter()
    used,updates,oracle_correct = 0,0,0
    predictions,frozen_predictions,trace,losses,rewards = [],[],[],[],[]
    history = core.TrainingProbe(x,torch.tensor(y,device=core.device),a.seed+910033) if training else core.PaidHistory(a.seed+910033)
    agent.policy_net.eval()
    initial_observation = observer(cnn,bank,agent,*diagnostic,a.seed,mode) if diagnostic is not None else None
    for pos in range(n):
        inputs = x[pos:pos+1]
        state,logits = state_at(cnn,bank,inputs,used,cap,pos,n,slot_queries,gate(pos))
        allowed = legal_actions(slots,cap-used,gate(pos))
        if mode=='random' or rng.random()<epsilon:
            action = int(rng.choice(allowed))
        else:
            with torch.no_grad():action = int(core.mask_query_action(agent.policy_net(state),state).argmax(1).item())
        route = route_index(action,slots)
        route_predictions = [int(v.argmax(1).item()) for v in logits]
        prediction = route_predictions[route]
        # Record before accessing this sample's feedback or changing any weight.
        predictions.append(prediction)
        frozen_predictions.append(route_predictions[0])
        actions[action] += 1
        if route:counts[route-1] += 1
        truth = int(y[pos])  # observer; only the paid branch exposes it to learners
        oracle_correct += any(p==truth for p in route_predictions)
        reward = None
        if action > slots:
            assert used < cap
            selected = route-1
            label = torch.tensor([truth],device=core.device)
            # Counterfactual no-update successor; no additional labels acquired.
            skip_next = None
            if learning and pos+1<n:
                skip_next,_ = state_at(cnn,bank,x[pos+1:pos+2],used,cap,pos+1,n,slot_queries,gate(pos+1))
            batch = history.sample(inputs)
            before = core.probe_losses(cnn,bank[selected],batch)
            optimizers[selected].zero_grad()
            loss = core.context_objective(nn.functional.cross_entropy(cnn(bank[selected](inputs)),label),
                                          nn.functional.cross_entropy(logits[0],label))
            loss.backward()
            optimizers[selected].step()
            after = core.probe_losses(cnn,bank[selected],batch)
            utility,gain,harm = core.transfer_utility(before,after)
            reward = (1.0 if prediction==truth else -1.0)+utility
            used += 1
            slot_queries[selected] += 1
            rewards.append(reward)
            if not training:history.add(inputs,label)
            if learning:
                done = pos+1==n or used==cap
                if done:next_state = torch.zeros_like(state)
                else:next_state,_ = state_at(cnn,bank,x[pos+1:pos+2],used,cap,pos+1,n,slot_queries,gate(pos+1))
                agent.replay_buffer.push(state,action,reward,next_state,float(done))
                # The same paid label gives pre-feedback correctness for all skip routes.
                skip_state = state.clone()
                if schedule is not None:skip_state[:,-3] = 0
                skip_terminal = pos+1==n
                if skip_next is None:skip_next = torch.zeros_like(state)
                for other in range(slots+1):
                    agent.replay_buffer.push(skip_state,other,1.0 if route_predictions[other]==truth else -1.0,skip_next,float(skip_terminal))
                agent.policy_net.train()
                result = agent.optimize()
                agent.policy_net.eval()
                if result is not None:
                    losses.append(result);updates += 1
                    if updates%15==0:agent.update_target_network()
        trace.append(dict(step=pos+1,prediction=prediction,truth=truth,action=action,route=route,
                          queried=action>slots,reward=reward,labels_used=used))
    if schedule is not None:assert used==cap
    metrics = dict(accuracy=float(core.accuracy_score(y,predictions)),macro_f1=float(core.f1_score(y,predictions,average='macro',zero_division=0)),
                   frozen_accuracy=float(core.accuracy_score(y,frozen_predictions)),oracle_accuracy=oracle_correct/n,
                   labels=used,label_cap=cap,slot_queries=slot_queries,slot_routes=counts,actions=dict(actions),
                   selector_updates=updates,mean_dqn_loss=float(np.mean(losses)) if losses else None,
                   mean_paid_reward=float(np.mean(rewards)) if rewards else None,
                   protocol='pre-feedback score; 1+2K actions; shared label cap; no oracle Context routing',
                   query_schedule=a.query_schedule,context_lr=a.context_lr,mode=mode,seed=a.seed)
    if diagnostic is not None:
        final,slot_final = observer(cnn,bank,agent,*diagnostic,a.seed,mode)
        original,slot_original = initial_observation
        truth_ref = np.asarray(diagnostic[1])
        good0,good1 = original==truth_ref,final==truth_ref
        # Routed reference is affected by BOTH the selector and Context bank.
        metrics['reference'] = dict(initial=float(good0.mean()),final=float(good1.mean()),
                                   drop_pp=float(100*(good0.mean()-good1.mean())),broken=int((good0&~good1).sum()),
                                   per_route_initial=slot_original,per_route_final=slot_final,
                                   protocol='observer-only routed clean validation; per-route Context scores also saved')
        _,slot_final_target = observer(cnn,bank,agent,x,y,a.seed,mode)
        metrics['final_current_per_route'] = slot_final_target
    return metrics,trace


def write_trace(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--source-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--burst-dir',type=Path,required=True)
    p.add_argument('--slots',type=int,default=3)
    p.add_argument('--levels',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--context-lr',type=float,default=0.001)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--selection-cost',type=float,default=.0001)
    p.add_argument('--gain-weight',type=float,default=1)
    p.add_argument('--retention-weight',type=float,default=1)
    p.add_argument('--query-schedule',choices=['uniform','policy'],default='uniform')
    p.add_argument('--train-epsilon',type=float,default=.2)
    p.add_argument('--eval-epsilon',type=float,default=0)
    p.add_argument('--modes',nargs='+',choices=['learned','random'],default=['learned','random'])
    a = p.parse_args()
    if a.slots<1 or min(a.epochs,a.patience)<1 or not 0<=a.budget_fraction<=1:
        p.error('Invalid slots, epochs, patience or budget.')
    if not np.isfinite(a.context_lr) or a.context_lr<=0 or any(not np.isfinite(v) or v<0 for v in (a.selection_cost,a.gain_weight,a.retention_weight)):
        p.error('Invalid LR or reward configuration.')
    if not 0<=a.eval_epsilon<=1 or not 0<=a.train_epsilon<=1:p.error('Epsilon must be in [0,1].')
    a.output,a.burst_dir = a.output.resolve(),a.burst_dir.resolve()
    core.set_seed(a.seed);configure(a)
    models = a.output/'Models'
    if a.command=='train':
        if a.source_run is None:p.error('Training requires --source-run.')
        if a.output.exists():p.error('Use a fresh output directory.')
        source = a.source_run.resolve()/'Models'
        artifact = joblib.load(source/'protocol_preprocessing.joblib')
        old = torch.load(source/'adversarial_best_dqn.pth',map_location='cpu',weights_only=False)
        levels = a.levels or [0,1,3]
        if set(levels)!=set(old.get('training_perturbations',[])):
            p.error('Bank training domains must match source CNN/Context development domains.')
        a.output.mkdir(parents=True);shutil.copytree(source,models)
        cnn,bank,agent,initial = make_models(artifact,models,a.slots)
        domains = {(v,s):read_domain(a.burst_dir,artifact,v,s) for v in levels for s in ('train_ids','val_ids')}
        best,stale = -1,0
        hashes = {name:core.file_sha256(models/name) for name in ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib')}
        for epoch in range(a.epochs):
            level = levels[epoch%len(levels)]
            m,_ = episode(a,cnn,bank,agent,initial,*domains[level,'train_ids'],'learned',a.train_epsilon,True,True)
            # Validation gradients adapt temporary Contexts only; policy/RNG unchanged.
            py_rng = random.getstate()
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                results = [episode(a,cnn,bank,agent,initial,*domains[v,'val_ids'],'learned',0,False)[0]['accuracy'] for v in levels]
            random.setstate(py_rng)
            score = float(np.mean(results))
            print(f'Epoch {epoch+1}/{a.epochs} | domain {level} | train {100*m["accuracy"]:.2f}% | validation {100*score:.2f}% | labels {m["labels"]}/{m["label_cap"]} | slot queries {m["slot_queries"]}',flush=True)
            if score>best:
                best,stale = score,0
                torch.save(dict(version=1,slots=a.slots,state_dict=deepcopy(agent.policy_net.state_dict()),
                                target_state_dict=deepcopy(agent.target_net.state_dict()),model_hashes=hashes,
                                training_levels=levels,validation_accuracy=best,initialization='all slots clone the healthy source Context; independent weights and optimizers',state_protocol='features; all route probabilities; per-slot paid counts; schedule gate; remaining budget; elapsed',reward_protocol='pre-feedback routed correctness plus selected-slot transfer gain minus damage minus cost; paid labels only',config=vars(a)|{'output':str(a.output),'burst_dir':str(a.burst_dir),'source_run':str(a.source_run)}),models/'context_bank_ddqn.pth')
            else:
                stale += 1
                if stale>=a.patience:break
        print(f'Bank training complete. {1+2*a.slots} actions; {a.slots} healthy identical initial checkpoints; best validation {100*best:.2f}%. No heldout test scored.')
        return
    artifact = joblib.load(models/'protocol_preprocessing.joblib')
    saved = torch.load(models/'context_bank_ddqn.pth',map_location='cpu',weights_only=False)
    if a.slots!=saved['slots']:p.error('Slot count differs from bank checkpoint.')
    for name,digest in saved['model_hashes'].items():
        if core.file_sha256(models/name)!=digest:raise ValueError('Bank source checkpoint mismatch: '+name)
    for name in ('selection_cost','gain_weight','retention_weight'):
        if getattr(a,name)!=saved['config'][name]:p.error('Reward differs from training: '+name)
    levels = a.levels or [5,7,10,12,15,17,20]
    if set(levels)&set(saved['training_levels']):p.error('Heldout test levels must exclude training levels.')
    cnn,bank,agent,initial = make_models(artifact,models,a.slots)
    root = a.output/'Results';root.mkdir(exist_ok=True)
    clean_ref = read_domain(a.burst_dir,artifact,0,'val_ids')
    rows=[]
    for level in levels:
        target = read_domain(a.burst_dir,artifact,level,'test_ids')
        pair={}
        for mode in a.modes:
            # Fresh DDQN replay/optimizer and bank optimizers per independent level.
            width = len(artifact['feature_names'])+(a.slots+1)*len(artifact['label_encoder'].classes_)+a.slots+3
            core.set_seed(a.seed)
            agent = core.DQNSelectorAgent(width,core.device,batch_size=32)
            agent.policy_net.load_state_dict(saved['state_dict'])
            agent.update_target_network()
            m,trace = episode(a,cnn,bank,agent,initial,*target,mode,a.eval_epsilon,mode=='learned',diagnostic=clean_ref)
            m['level']=level
            tag=f'bank{a.slots}_{mode}_per{level}_seed{a.seed}_lr{a.context_lr:g}_eps{a.eval_epsilon:g}_{a.query_schedule}'
            (root/(tag+'_metrics.json')).write_text(json.dumps(m,indent=2))
            write_trace(root/(tag+'_trace.csv'),trace)
            print(f'Level {level} | {mode} | online {100*m["accuracy"]:.2f}% | oracle {100*m["oracle_accuracy"]:.2f}% | labels {m["labels"]}/{m["label_cap"]} | slot queries {m["slot_queries"]} | routed reference drop {m["reference"]["drop_pp"]:+.2f} pp',flush=True)
            for slot,(v0,v1) in enumerate(zip(m['reference']['per_route_initial'][1:],m['reference']['per_route_final'][1:]),start=1):
                print(f'  Clean reference Context {slot}: {100*v0:.2f}% -> {100*v1:.2f}% | drop {100*(v0-v1):+.2f} pp')
            pair[mode]=m
        row={'PER':level,'Frozen':100*next(iter(pair.values()))['frozen_accuracy']}
        for mode,prefix in [('learned','RL'),('random','Random')]:
            if mode not in pair:continue
            m=pair[mode]
            row.update({prefix:100*m['accuracy'],'Oracle'+prefix:100*m['oracle_accuracy'],
                        'Labels'+prefix:m['labels'],'Ref'+prefix:100*m['reference']['final'],
                        'Drop'+prefix:m['reference']['drop_pp']})
        if 'RL' in row and 'Random' in row:row['Gap']=row['RL']-row['Random']
        rows.append(row)
    columns=list(rows[0]);mean={'PER':'Mean'}
    for key in columns[1:]:mean[key]=float(np.mean([r[key] for r in rows]))
    rows.append(mean)
    print('\nCombined bank results (%; Gap/Drop in pp; reference is observer-only routed clean validation):')
    from amcal_burst_results_summary import display
    display(rows,columns)
    filename=root/f'combined_bank{a.slots}_seed{a.seed}_lr{a.context_lr:g}_eps{a.eval_epsilon:g}_{a.query_schedule}.csv'
    write_trace(filename,rows)
    print('Saved:',filename)


if __name__=='__main__':main()
