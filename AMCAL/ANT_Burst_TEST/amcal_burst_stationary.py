"""Stationary, counterfactual one-step selector-training diagnostic.

Run locally; the saved CNN/Context never learn during selector training.
Each action-1 target uses its own disposable Context copy and fresh Adam.
Terminal mode isolates immediate reward fitting. Optional local DDQN mode
bootstraps from the same sample after a disposable update, not future traffic.
"""
import argparse
import json
from pathlib import Path
from copy import deepcopy
import subprocess
import sys
import os
if '--deterministic' in sys.argv:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import amcal_burst_legacy as core


def balanced_indices(labels, maximum, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if maximum < len(classes):
        raise ValueError('Sample limit must cover every class.')
    selected = []
    pools = {int(c): rng.permutation(np.flatnonzero(labels == c)).tolist() for c in classes}
    while len(selected) < maximum and any(pools.values()):
        for c in classes:
            if pools[int(c)] and len(selected) < maximum:
                selected.append(pools[int(c)].pop())
    return np.asarray(sorted(selected), dtype=int)


def build_table(cnn, context_seed, dataset, counts, limit, seed, lr, label, gain_weight=0.0):
    inputs, labels = dataset.tensors
    indices = balanced_indices(labels.detach().cpu().numpy(), limit, seed)
    inputs, labels = inputs[indices], labels[indices]
    weights = torch.as_tensor(counts, dtype=torch.float32, device=core.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    states, next_states, rewards, base_predictions, before_predictions, after_predictions = [], [], [], [], [], []
    cache = {}
    for i in range(len(indices)):
        x, y = inputs[i:i+1].to(core.device), labels[i:i+1].to(core.device)
        # Cache duplicates within this source only. All source Context copies
        # start from identical weights/BN buffers, optimizer, and eval mode.
        key = (x.detach().cpu().numpy().tobytes(), int(y.item()))
        if key not in cache:
            with torch.no_grad():
                base_logits = cnn(x)
                before_logits = cnn(context_seed(x))
                uncertainty = 1 - before_logits.softmax(dim=1).max(dim=1, keepdim=True).values
                state = torch.cat([x.squeeze(1), uncertainty], dim=1).detach().cpu()
                base_pred = int(base_logits.argmax(dim=1).item())
                before_pred = int(before_logits.argmax(dim=1).item())
                before_ce = float(core.reward_ce(before_logits, y).item())
            # A separate copy per candidate, never shared across candidates.
            probe = deepcopy(context_seed).requires_grad_(True).eval()
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
            optimizer.zero_grad()
            loss = criterion(cnn(probe(x)), y)
            base_loss = criterion(base_logits.detach(), y)
            core.context_objective(loss, base_loss, online=True).backward()
            optimizer.step()
            with torch.no_grad():
                after_logits = cnn(probe(x))
                after_pred = int(after_logits.argmax(dim=1).item())
                after_uncertainty = 1 - after_logits.softmax(dim=1).max(dim=1, keepdim=True).values
                next_state = torch.cat([x.squeeze(1), after_uncertainty], dim=1).detach().cpu()
                after_ce = float(core.reward_ce(after_logits, y).item())
            reward = core.compute_reward(1, base_pred, after_pred, int(y.item()))
            reward += gain_weight * (before_ce - after_ce)
            cache[key] = (state, next_state, float(reward), base_pred, before_pred, after_pred)
            del optimizer, probe
        state, next_state, reward, base_pred, before_pred, after_pred = cache[key]
        states.append(state)
        next_states.append(next_state)
        rewards.append(reward)
        base_predictions.append(base_pred)
        before_predictions.append(before_pred)
        after_predictions.append(after_pred)
        if (i+1) % 50 == 0 or i+1 == len(indices):
            print(f'{label} counterfactual rewards: {i+1}/{len(indices)}; unique probes={len(cache)}', flush=True)
    return dict(states=torch.cat(states), apply_next_states=torch.cat(next_states),
                apply_rewards=torch.tensor(rewards, dtype=torch.float32),
                labels=labels.detach().cpu(), source_indices=indices.tolist(),
                base_predictions=base_predictions, before_predictions=before_predictions,
                after_predictions=after_predictions, unique_probes=len(cache))


def policy_quality(agent, table):
    with core.evaluation_mode(agent.policy_net), torch.no_grad():
        q = agent.policy_net(table['states'].to(core.device)).cpu()
    rewards, labels = table['apply_rewards'], table['labels']
    actions = q.argmax(dim=1)
    utilities = rewards * (actions == 1).float()
    errors = (q - torch.stack([torch.zeros_like(rewards), rewards], dim=1)).abs().mean(dim=1)
    if core.SELECTOR_TRANSITION_MODE == 'local':
        with core.evaluation_mode(agent.policy_net, agent.target_net), torch.no_grad():
            action1_next = table['apply_next_states'].to(core.device)
            next_actions = agent.policy_net(action1_next).argmax(dim=1)
            next_values = agent.target_net(action1_next).gather(1, next_actions[:,None]).squeeze(1).cpu()
            skip_next = table['states'].to(core.device)
            skip_actions = agent.policy_net(skip_next).argmax(dim=1)
            skip_values = agent.target_net(skip_next).gather(1, skip_actions[:,None]).squeeze(1).cpu()
        targets = torch.stack([agent.gamma*skip_values, rewards+agent.gamma*next_values], dim=1)
        errors = (q-targets).abs().mean(dim=1)
    classes = torch.unique(labels)
    balanced = lambda values: float(torch.stack([values[labels == c].mean() for c in classes]).mean())
    return dict(greedy_utility=balanced(utilities), value_mae=balanced(errors),
                value_error_scope='DDQN target residual' if core.SELECTOR_TRANSITION_MODE=='local' else 'immediate reward MAE',
                select_fraction=float((actions == 1).float().mean()),
                random_50_50_expected_utility=0.5 * balanced(rewards),
                immediate_oracle_utility=balanced(rewards.clamp(min=0)))


def train(args):
    source = args.source.resolve() / 'Models'
    output = args.output.resolve()
    if (output / 'Models').exists():
        raise ValueError('Choose a fresh output folder; existing Models are preserved.')
    if args.deterministic:
        core.configure_deterministic(args.seed)
    else:
        core.seed_run(args.seed)
    core.SELECTOR_REWARD_MODE = 'paper'
    core.SELECTOR_REWARD_PROFILE = args.reward_profile
    core.SELECTOR_STATE_MODE = 'context'
    core.CONTEXT_LOSS_MODE = 'legacy'
    core.SELECTION_COST = args.selection_cost
    core.SELECTOR_TRANSITION_MODE = args.transition_mode
    core.SELECTOR_GAMMA = args.gamma
    core.TRAIN_GREEDY_MODE = 'legacy'
    names = ['initial_cnn_model.pth', 'adversarial_best_context.pth', 'adversarial_best_dqn.pth']
    checkpoints = {n: torch.load(source / n, map_location=core.device, weights_only=False) for n in names}
    required = dict(selector_reward_mode='paper', selector_reward_profile=args.reward_profile,
                    selector_state_mode='context', context_loss_mode='legacy',
                    reward_gain_weight=args.reward_gain_weight, reward_generalization_weight=0.0,
                    selection_cost=args.selection_cost, selector_transition_mode='terminal', selector_gamma=0.2)
    # These optional fields were added after the older reference was trained.
    # Absence means the historical disabled/default setting, never a mismatch.
    historical_defaults = dict(selector_reward_profile='original', reward_generalization_weight=0.0)
    for name, checkpoint in checkpoints.items():
        for key, expected in required.items():
            actual = checkpoint.get(key, historical_defaults.get(key))
            if actual != expected:
                raise ValueError(f'{name}: source {key} must match {expected!r}.')
    cnn_checkpoint = checkpoints[names[0]]
    context_checkpoint = checkpoints[names[1]]
    dqn_checkpoint = checkpoints[names[2]]
    length, classes = cnn_checkpoint['input_length'], cnn_checkpoint['num_classes']
    cnn = core.CNN1D(length, classes).to(core.device)
    cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn.eval().requires_grad_(False)
    context = core.ContextAwareNetwork(length).to(core.device)
    context.load_state_dict(context_checkpoint['context_model_state_dict'])
    context.eval().requires_grad_(False)
    # Keep exact source tensors for an end-of-training immutability check.
    frozen_cnn = {k: v.detach().cpu().clone() for k, v in cnn.state_dict().items()}
    frozen_context = {k: v.detach().cpu().clone() for k, v in context.state_dict().items()}
    train_loader, val_loader, _, _, num_classes, input_length, counts = core.load_training_data(str(args.data_dir.resolve()))
    if train_loader is None or input_length != length or num_classes != classes:
        raise ValueError('Training files and source model dimensions do not match.')
    train_table = build_table(cnn, context, train_loader.dataset, counts, args.train_samples, args.seed, args.probe_lr, 'TRAIN', args.reward_gain_weight)
    val_table = build_table(cnn, context, val_loader.dataset, counts, args.val_samples, args.seed+1, args.probe_lr, 'VALIDATION', args.reward_gain_weight)
    # Start from the source policy, but no old replay/Adam or counter states.
    agent = core.DQNSelectorAgent(length+1, core.device, epsilon_start=1.0,
                                 epsilon_end=0.2, epsilon_decay=0.9)
    agent.policy_net.load_state_dict(dqn_checkpoint['dqn_policy_net_state_dict'])
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    initial_quality = policy_quality(agent, val_table)
    print('Source-policy validation quality before selector training:', initial_quality, flush=True)
    history, best, stale = [], None, 0
    target_counter = 0
    for epoch in range(1, args.epochs+1):
        agent.policy_net.train()
        order = torch.randperm(len(train_table['states']))
        losses = []
        action_counts = [0, 0]
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            states = train_table['states'][ix].to(core.device)
            actions = agent.select_action(states)
            for j, row in enumerate(ix.tolist()):
                action = int(actions[j].item())
                action_counts[action] += 1
                reward = float(train_table['apply_rewards'][row]) if action else 0.0
                next_state = train_table['apply_next_states'][row:row+1] if action else states[j:j+1]
                agent.replay_buffer.push(states[j:j+1], action, reward, next_state,
                                         0.0 if args.transition_mode=='local' else 1.0)
                target_counter += 1
            if args.transition_mode=='terminal' and target_counter >= 5:
                agent.update_target_network()
                target_counter = 0
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)
        agent.update_epsilon()
        if args.transition_mode=='local' and epoch % 15 == 0:
            agent.update_target_network()
        quality = policy_quality(agent, val_table)
        key = (quality['greedy_utility'], -quality['value_mae'])
        history.append(dict(epoch=epoch, **quality, dqn_loss=float(np.mean(losses)) if losses else None,
                            actions=action_counts, epsilon=agent.epsilon))
        print(f"Selector epoch {epoch}/{args.epochs}: val utility={quality['greedy_utility']:.4f}; "
              f"random expected={quality['random_50_50_expected_utility']:.4f}; value MAE={quality['value_mae']:.4f}; actions={action_counts}", flush=True)
        if best is None or key > best['key']:
            best = dict(key=key, epoch=epoch, quality=quality,
                        policy=deepcopy(agent.policy_net.state_dict()), target=deepcopy(agent.target_net.state_dict()),
                        replay=deepcopy(list(agent.replay_buffer.buffer)),
                        counters=dict(epsilon=agent.epsilon, steps_done=agent.steps_done,
                                      bootstrap_optimizer_steps=agent.bootstrap_optimizer_steps))
            stale = 0
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f'Selector early stopping at epoch {epoch}.')
            break
    for model, snapshot in [(cnn, frozen_cnn), (context, frozen_context)]:
        if any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in model.state_dict().items()):
            raise RuntimeError('Source CNN/Context changed; refusing to save.')
    model_dir = output / 'Models'
    model_dir.mkdir(parents=True)
    output_settings = dict(required, selector_transition_mode=args.transition_mode, selector_gamma=args.gamma)
    tags = dict(output_settings, training_state_version=core.TRAINING_STATE_VERSION,
                training_selector_mode='original', agreement_penalty=None,
                training_greedy_mode='legacy', training_replay_reset_every=0,
                selector_only_training=True, stationary_probe_lr=args.probe_lr,
                stationary_source=str(args.source.resolve()), stationary_best_epoch=best['epoch'])
    for name in names[:2]:
        saved = dict(checkpoints[name])
        saved.update(tags)
        torch.save(saved, model_dir / name)
    torch.save(dict(tags, dqn_policy_net_state_dict=best['policy'], dqn_target_net_state_dict=best['target'],
                    replay_buffer=best['replay'], selector_training_counters=best['counters'],
                    best_epoch=best['epoch'], input_length=length, num_classes=classes,
                    stationary_validation=best['quality']), model_dir / names[2])
    manifest = dict(experiment='stationary counterfactual selector fine-tuning',
                    transition_mode=args.transition_mode, gamma=args.gamma,
                    reward_profile=args.reward_profile, reward_gain_weight=args.reward_gain_weight,
                    target_sync_rule='every 15 selector epochs' if args.transition_mode=='local' else 'legacy sample counter',
                    source=str(args.source.resolve()), output=str(output), seed=args.seed,
                    source_checkpoint_sha256={n: core.sha256_file(source/n) for n in names},
                    source_csv_sha256={n: core.sha256_file(args.data_dir/n) for n in
                                       ['oversampled_train_dataset.csv','oversampled_validation_dataset.csv']},
                    train_indices=train_table['source_indices'], validation_indices=val_table['source_indices'],
                    probe_lr=args.probe_lr, source_cnn_unchanged=True, source_context_unchanged=True,
                    counterfactual_context_update='separate eval-mode clone, fresh Adam, one online Context-objective step per feature/label',
                    policy_initialization='source policy weights; empty replay and new Adam',
                    validation_scope='class-balanced immediate greedy reward; no stream budget; not traffic accuracy',
                    initial_validation=initial_quality,
                    best_epoch=best['epoch'], validation=best['quality'], history=history,
                    reproducibility=core.reproducibility_settings())
    (output/'stationary_training.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for label, table in [('train',train_table),('validation',val_table)]:
        np.savez_compressed(output/f'stationary_{label}_targets.npz', states=table['states'].numpy(),
                            apply_next_states=table['apply_next_states'].numpy(),
                            labels=table['labels'].numpy(), action1_rewards=table['apply_rewards'].numpy(),
                            source_indices=np.asarray(table['source_indices']),
                            base_predictions=table['base_predictions'], before_predictions=table['before_predictions'],
                            after_predictions=table['after_predictions'])
    print(f"Saved stationary selector snapshot epoch {best['epoch']}. CNN/Context weights unchanged.")


def evaluate(args):
    script = Path(__file__).with_name('amcal_burst_legacy.py')
    for level in args.levels:
        common = [sys.executable, str(script), 'evaluate', '--output', str(args.output.resolve()),
                  '--data-dir', str(args.data_dir.resolve()), '--seed', str(args.seed),
                  '--query-gate','none','--budget',str(args.budget),'--budget-schedule','linear',
                  '--online-lr',str(args.online_lr),'--per',str(level)]
        if args.deterministic:
            common.append('--deterministic')
        learned = common + ['--reward-mode','paper','--reward-profile',args.reward_profile,
                            '--transition-mode',args.transition_mode,'--gamma',str(args.gamma),
                            '--reward-gain-weight',str(args.reward_gain_weight),
                            '--selection-cost',str(args.selection_cost),'--selector-state','context',
                            '--selector-mode','learned','--online-reward-timing','post',
                            '--online-skip-replay','--skip-optimizer-mode','queries',
                            '--epsilon-start','0.05','--epsilon-end','0.05']
        subprocess.run(learned, check=True)
        subprocess.run(common + ['--selector-mode','random'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train','evaluate','run'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=core.BURST_DATA_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--val-samples', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--probe-lr', type=float, default=0.25)
    parser.add_argument('--selection-cost', type=float, default=0.42)
    parser.add_argument('--reward-profile', choices=['original','improvement'], default='improvement')
    parser.add_argument('--reward-gain-weight', type=float, default=0.0)
    parser.add_argument('--transition-mode', choices=['terminal','local'], default='terminal')
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--online-lr', type=float, default=0.25)
    parser.add_argument('--budget', type=int, default=214)
    parser.add_argument('--levels', type=int, nargs='+', choices=[0,1,3,5,7,10,12,15,17,20], default=[0,7,20])
    args = parser.parse_args()
    if args.command != 'evaluate' and args.source is None:
        parser.error('train/run requires --source with matching paper/context checkpoints trained in terminal mode.')
    if min(args.train_samples,args.val_samples,args.epochs,args.patience) < 1 or not 0 <= args.min_epochs <= args.epochs:
        parser.error('Positive sample/epoch/patience limits and valid minimum epochs are required.')
    if not all(np.isfinite(v) and v > 0 for v in [args.probe_lr,args.online_lr]) or not np.isfinite(args.selection_cost) or args.selection_cost < 0:
        parser.error('Learning rates must be finite/positive and cost finite/nonnegative.')
    if not 0 <= args.budget <= 1073:
        parser.error('Budget must be between 0 and 1073.')
    if not 0 <= args.gamma < 1 or not np.isfinite(args.reward_gain_weight) or args.reward_gain_weight < 0:
        parser.error('Invalid gamma/gain weight.')
    if args.command in ['train','run']:
        train(args)
    if args.command in ['evaluate','run']:
        evaluate(args)


if __name__ == '__main__':
    main()
