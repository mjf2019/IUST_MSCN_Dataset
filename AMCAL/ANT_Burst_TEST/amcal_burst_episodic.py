"""Budgeted episodic Monte Carlo selector diagnostic; not legacy DDQN training."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
if '--deterministic' in sys.argv:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
import amcal_burst_legacy as core


def discounted_returns(rewards, gamma, terminal_bonus):
    value = float(terminal_bonus)
    result = [0.0] * len(rewards)
    for i in range(len(rewards)-1, -1, -1):
        value = float(rewards[i]) + gamma * value
        result[i] = value
    return result


def partition(dataset, seed):
    x, y = dataset.tensors
    _, rows, inverse = np.unique(x.detach().cpu().reshape(len(x), -1).numpy(),
                                 axis=0, return_index=True, return_inverse=True)
    labels = y.detach().cpu().numpy()
    group_labels = labels[rows]
    if np.any(group_labels[inverse] != labels):
        raise ValueError('Conflicting labels in identical feature groups.')
    rng = np.random.default_rng(seed)
    support, reference = [], []
    for c in np.unique(group_labels):
        pool = rng.permutation(rows[group_labels == c])
        if len(pool) < 2:
            raise ValueError('Each class needs at least two distinct groups.')
        count = min(len(pool)-1, max(1, int(round(0.1 * len(pool)))))
        reference.extend(pool[:count].tolist())
        support.extend(pool[count:].tolist())
    return np.asarray(support), np.asarray(reference)


def episode_indices(dataset, support, reference, length, seed):
    rng = np.random.default_rng(seed)
    labels = dataset.tensors[1].detach().cpu().numpy()
    counts = np.bincount(labels[support])
    probabilities = 1.0 / counts[labels[support]]
    probabilities /= probabilities.sum()
    # Sampled row order is a synthetic training stream, not capture chronology.
    stream = rng.choice(support, min(length, len(support)), replace=False, p=probabilities)
    refs = []
    for c in np.unique(labels[reference]):
        candidates = reference[labels[reference] == c]
        refs.extend(rng.choice(candidates, min(8, len(candidates)), replace=False).tolist())
    return stream, np.asarray(refs, dtype=int)


def observe(cnn, context, x, used, budget, step, rows):
    with torch.no_grad():
        base = cnn(x)
        modulated = cnn(context(x))
        joint = core.joint_selector_state(x, base, modulated)
        extra = torch.tensor([[(budget-used)/max(budget,1), step/max(rows,1)]],
                             dtype=joint.dtype, device=joint.device)
        return torch.cat([joint, extra], dim=1).detach(), base, modulated


def rollout(cnn, context_seed, agent, dataset, indices, refs, args, mode, epsilon, seed, budget=None):
    context = deepcopy(context_seed).requires_grad_(True).eval()
    optimizer = torch.optim.Adam(context.parameters(), lr=args.online_lr, weight_decay=1e-4)
    x_all, y_all = dataset.tensors
    n = len(indices)
    budget = int(np.floor(args.budget_fraction*n)) if budget is None else min(budget,n)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    ref_x = ref_y = None
    if refs is not None:
        ref_x, ref_y = x_all[refs].to(core.device), y_all[refs].to(core.device)
        before_reference = core.reference_loss(cnn, context, ref_x, ref_y)
    used = 0
    trajectory, trace, truth, predictions, base_predictions = [], [], [], [], []
    for step, row in enumerate(indices, 1):
        x = x_all[int(row):int(row)+1].to(core.device)
        state, base, logits = observe(cnn, context, x, used, budget, step, n)
        released = core.released_label_budget(step, n, budget)
        action = 0
        if used < released:
            if mode == 'random' or rng.random() < epsilon:
                action = int(rng.integers(0,2))
            else:
                with core.evaluation_mode(agent.policy_net), torch.no_grad():
                    action = int(agent.policy_net(state).argmax(dim=1).item())
        # Scoring label never enters state/action. Only selected labels update.
        y = y_all[int(row):int(row)+1].to(core.device)
        before_pred, base_pred = int(logits.argmax(1).item()), int(base.argmax(1).item())
        truth.append(int(y.item())); predictions.append(before_pred); base_predictions.append(base_pred)
        reward = 0.0
        if action == 1:
            used += 1
            optimizer.zero_grad()
            loss = criterion(cnn(context(x)), y)
            base_loss = criterion(base.detach(), y)
            core.context_objective(loss, base_loss, online=True).backward()
            optimizer.step()
            with torch.no_grad():
                after_pred = int(cnn(context(x)).argmax(1).item())
            # Normalize total competitive utility by the episode label cap.
            if mode != 'random':
                reward = core.compute_reward(1, base_pred, after_pred, int(y.item())) / max(budget,1)
        trajectory.append((state.detach().cpu(), action, reward))
        trace.append(dict(step=step, source_row=int(row), action=action, labels_used=used,
                          released_budget=released, remaining_budget_fraction=(budget-used)/max(budget,1),
                          progress=step/max(n,1), prediction_before_update=before_pred,
                          base_prediction=base_pred, true_label=int(y.item()), immediate_reward=reward))
    terminal_bonus = 0.0
    relative_gain = None
    if ref_x is not None:
        after_reference = core.reference_loss(cnn, context, ref_x, ref_y)
        relative_gain = float(np.clip((before_reference-after_reference)/max(before_reference,1e-8),-1,1))
        terminal_bonus = args.terminal_gain_weight * relative_gain
    returns = discounted_returns([r for _,_,r in trajectory], args.gamma, terminal_bonus)
    result = dict(accuracy_before_update=float(accuracy_score(truth,predictions)),
                  precision=float(core.precision_score(truth,predictions,average='weighted',zero_division=0)),
                  recall=float(core.recall_score(truth,predictions,average='weighted',zero_division=0)),
                  macro_f1=float(f1_score(truth,predictions,average='macro',zero_division=0)),
                  weighted_f1=float(f1_score(truth,predictions,average='weighted',zero_division=0)),
                  base_accuracy=float(accuracy_score(truth,base_predictions)),
                  rows=n, labels_used=used, label_budget=budget,
                  reference_relative_ce_gain=relative_gain, terminal_bonus=terminal_bonus)
    return trajectory, returns, trace, result


class MonteCarloAgent(core.DQNSelectorAgent):
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, returns, _, _ = self.replay_buffer.sample(self.batch_size)
        values = self.policy_net(states.to(self.device)).gather(1,actions.to(self.device).unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(values,returns.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),1.0)
        self.optimizer.step()
        return float(loss.item())


def source_models(folder):
    folder = folder.resolve() / 'Models'
    c = torch.load(folder/'initial_cnn_model.pth',map_location=core.device,weights_only=False)
    k = torch.load(folder/'adversarial_best_context.pth',map_location=core.device,weights_only=False)
    cnn = core.CNN1D(c['input_length'],c['num_classes']).to(core.device)
    cnn.load_state_dict(c['model_state_dict']); cnn.eval().requires_grad_(False)
    context = core.ContextAwareNetwork(c['input_length']).to(core.device)
    context.load_state_dict(k['context_model_state_dict']); context.eval().requires_grad_(False)
    return cnn,context,c,k


def train(args):
    output = args.output.resolve()
    if (output/'Models').exists():
        raise ValueError('Use a fresh output folder; Models already exist.')
    cnn,context,c,k = source_models(args.source)
    loaders = core.load_training_data(str(args.data_dir.resolve()))
    train_loader,val_loader = loaders[:2]
    if train_loader is None or loaders[5] != c['input_length'] or loaders[4] != c['num_classes']:
        raise ValueError('Training schema/model mismatch.')
    train_pool,train_refs = partition(train_loader.dataset,args.seed)
    val_pool,val_refs = partition(val_loader.dataset,args.seed+1)
    width = c['input_length']+2*c['num_classes']+2
    agent = MonteCarloAgent(width,core.device,gamma=args.gamma,epsilon_start=1.0,
                           epsilon_end=0.2,epsilon_decay=0.96)
    if 4 * min(args.episode_length, len(train_pool)) < agent.batch_size:
        raise ValueError('Four training episodes must supply at least 256 transitions.')
    initial_cnn = deepcopy(cnn.state_dict()); initial_context = deepcopy(context.state_dict())
    history,best,stale = [],None,0
    for round_number in range(1,args.rounds+1):
        # Fresh on-policy trajectory batch; no returns from older policy rounds.
        agent.replay_buffer.buffer.clear()
        train_results = []
        for j in range(4):
            seed = args.seed+round_number*100+j
            indices,refs = episode_indices(train_loader.dataset,train_pool,train_refs,args.episode_length,seed)
            trajectory,returns,_,result = rollout(cnn,context,agent,train_loader.dataset,indices,refs,args,
                                                  'learned',agent.epsilon,seed)
            train_results.append(result)
            for (state,action,_),value in zip(trajectory,returns):
                # Reward field holds the full discounted return, not immediate r.
                agent.replay_buffer.push(state,action,value,state,1.0)
        agent.policy_net.train()
        losses = [agent.optimize() for _ in range(10)]
        agent.update_epsilon()
        print(f"Episodic round {round_number}/{args.rounds}: 4 episodes; epsilon={agent.epsilon:.3f}; "
              f"MC loss={np.mean([v for v in losses if v is not None]) if any(v is not None for v in losses) else 0:.4f}",flush=True)
        if round_number == 1 or round_number % 5 == 0:
            validation = []
            with core.preserve_random_state():
                for j in range(3):
                    seed = args.seed+100000+j
                    indices,refs = episode_indices(val_loader.dataset,val_pool,val_refs,args.episode_length,seed)
                    validation.append(rollout(cnn,context,agent,val_loader.dataset,indices,refs,args,
                                              'learned',0.0,seed)[3])
            accuracy = float(np.mean([v['accuracy_before_update'] for v in validation]))
            labels = float(np.mean([v['labels_used'] for v in validation]))
            key = (accuracy,-labels)
            history.append(dict(round=round_number,train=train_results,validation=validation))
            print(f"Validation episode pre-update accuracy={accuracy:.4f}; mean labels={labels:.1f}",flush=True)
            if best is None or key > best['key']:
                best = dict(key=key,round=round_number,weights=deepcopy(agent.policy_net.state_dict()))
                stale=0
            else:
                stale+=1
            if round_number >= 25 and stale >= args.patience:
                print('Episodic early stopping at round',round_number);break
    if any(not torch.equal(v,initial_cnn[n]) for n,v in cnn.state_dict().items()) or any(
            not torch.equal(v,initial_context[n]) for n,v in context.state_dict().items()):
        raise RuntimeError('Source model changed; refusing to save.')
    models=output/'Models';models.mkdir(parents=True)
    torch.save(c,models/'initial_cnn_model.pth');torch.save(k,models/'adversarial_best_context.pth')
    torch.save(dict(weights=best['weights'],input_width=width,best_round=best['round'],
                    selection_cost=args.selection_cost,gamma=args.gamma,terminal_gain_weight=args.terminal_gain_weight),
               models/'episodic_selector.pth')
    manifest=dict(method='episodic Monte Carlo Q control, not DDQN',source=str(args.source.resolve()),
                  seed=args.seed,episode_length=args.episode_length,budget_fraction=args.budget_fraction,
                  budget_schedule=args.budget_schedule,gamma=args.gamma,terminal_gain_weight=args.terminal_gain_weight,
                  train_support_indices=train_pool.tolist(),train_reference_indices=train_refs.tolist(),
                  validation_support_indices=val_pool.tolist(),validation_reference_indices=val_refs.tolist(),
                  source_cnn_context_unchanged=True,best_round=best['round'],history=history,
                  source_hashes={n:core.sha256_file(args.source/'Models'/n) for n in
                                 ['initial_cnn_model.pth','adversarial_best_context.pth']},
                  csv_hashes={n:core.sha256_file(args.data_dir/n) for n in
                              ['oversampled_train_dataset.csv','oversampled_validation_dataset.csv']})
    (output/'episodic_training.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Saved episodic selector; best validation round:',best['round'])


def evaluate(args):
    output=args.output.resolve()
    cnn,context,c,k=source_models(output)
    saved=torch.load(output/'Models'/'episodic_selector.pth',map_location=core.device,weights_only=False)
    if saved['selection_cost'] != args.selection_cost:
        raise ValueError('Selection cost must match episodic training.')
    if saved['input_width'] != c['input_length'] + 2*c['num_classes'] + 2:
        raise ValueError('Episodic selector input width/model mismatch.')
    agent=MonteCarloAgent(saved['input_width'],core.device)
    agent.policy_net.load_state_dict(saved['weights'])
    metadata=core.load_metadata(str(args.data_dir/'AdvBurst_FTSC-IAT'),0)
    if len(metadata)!=5 or metadata[0] is None:
        raise ValueError('Could not load level-0 attack metadata.')
    classes,encoder,scaler,length,_=metadata
    if classes!=c['num_classes'] or length!=c['input_length']:
        raise ValueError('Attack schema mismatch.')
    summary=[]
    for level in args.levels:
        path=args.data_dir/'AdvBurst_FTSC-IAT'/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
        frame=pd.read_csv(path)
        x=torch.tensor(scaler.transform(frame.iloc[:,:-1].to_numpy()),dtype=torch.float32,device=core.device).unsqueeze(1)
        y=torch.tensor(encoder.transform(frame.iloc[:,-1].to_numpy()),dtype=torch.long,device=core.device)
        dataset=torch.utils.data.TensorDataset(x,y)
        for mode in ['learned','random']:
            core.seed_run(args.seed)
            _,_,trace,result=rollout(cnn,context,agent if mode=='learned' else None,dataset,np.arange(len(y)),None,args,mode,
                                    0.05 if mode=='learned' else 1.0,args.seed,budget=args.budget)
            result.update(level=level,mode=mode,selector_updates=0,policy_scope='frozen weights, state-conditioned actions',
                          protocol='predict-score-query-update',alignment='file-local rows; source overlap unverified',
                          attack_sha256=core.sha256_file(path))
            folder=output/'Results'/f'episodic_{mode}_seed{args.seed}_{args.budget_schedule}'
            folder.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(trace).to_csv(folder/f'PER_{level}_trace.csv',index=False)
            (folder/f'PER_{level}_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(f"\nResults for Perturbation Level {level} ({mode}):\n"
                  f"  Frozen CNN Accuracy: {result['base_accuracy']:.4f}\n"
                  f"  Pre-update Accuracy: {result['accuracy_before_update']:.4f}\n"
                  f"  Precision (weighted): {result['precision']:.4f}\n"
                  f"  Recall (weighted): {result['recall']:.4f}\n"
                  f"  F1 (weighted): {result['weighted_f1']:.4f}\n"
                  f"  Macro-F1: {result['macro_f1']:.4f}\n"
                  f"  Context Updates Used: {result['labels_used']}/{result['label_budget']}\n"
                  f"  Selector Updates: 0 (trained policy weights fixed)",flush=True)
            summary.append(result)
    pd.DataFrame(summary).to_csv(output/'Results'/'episodic_comparison.csv',index=False)
    print('\nCombined Results Table:')
    print(pd.DataFrame(summary)[['level','mode','accuracy_before_update','macro_f1','weighted_f1','labels_used','label_budget']].to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate','run'])
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=core.BURST_DATA_DIR)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--rounds',type=int,default=75)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--episode-length',type=int,default=100)
    p.add_argument('--budget-fraction',type=float,default=0.2)
    p.add_argument('--budget-schedule',choices=['full','linear'],default='linear')
    p.add_argument('--gamma',type=float,default=0.99)
    p.add_argument('--terminal-gain-weight',type=float,default=1.0)
    p.add_argument('--selection-cost',type=float,default=0.42)
    p.add_argument('--online-lr',type=float,default=0.25)
    p.add_argument('--budget',type=int,default=214)
    p.add_argument('--levels',type=int,nargs='+',choices=[0,1,3,5,7,10,12,15,17,20],default=[0,7,20])
    args=p.parse_args()
    if args.command!='evaluate' and args.source is None:p.error('train/run needs --source.')
    if min(args.rounds,args.patience,args.episode_length)<1:p.error('Positive duration/length required.')
    if not 0<args.budget_fraction<=1 or not 0<=args.gamma<1:p.error('Invalid fraction/gamma.')
    if not np.isfinite(args.online_lr) or args.online_lr<=0 or not np.isfinite(args.selection_cost) or args.selection_cost<0:
        p.error('Invalid learning rate/cost.')
    if not np.isfinite(args.terminal_gain_weight) or args.terminal_gain_weight<0 or args.budget<0:p.error('Invalid gain/budget.')
    args.data_dir=args.data_dir.resolve()
    if args.deterministic:core.configure_deterministic(args.seed)
    else:core.seed_run(args.seed)
    core.SELECTOR_REWARD_MODE='paper';core.SELECTOR_REWARD_PROFILE='improvement'
    core.CONTEXT_LOSS_MODE='legacy';core.SELECTION_COST=args.selection_cost
    core.ONLINE_BUDGET_SCHEDULE=args.budget_schedule
    if args.command in ['train','run']:train(args)
    if args.command in ['evaluate','run']:evaluate(args)


if __name__=='__main__':main()
