"""Isolated PairOnly shared-loss/DDQN control; no CNN or Context pretraining.

Only the existing future/update-value heads are fitted offline. Context copies
adapt with the original batch5 objective; all test levels reset independently.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

import amcal_final_enhancement as base
from enhancements.config import EnhancementConfig

LEVELS = [0, 1, 3]
TEST_LEVELS = [5, 7, 10, 12, 15, 17, 20]


def digest(path): return base.legacy.core.file_sha256(path)


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def subset(data, keep):
    at = torch.as_tensor(np.flatnonzero(keep), device=data[0].device)
    return data[0][at], np.asarray(data[1])[keep], np.asarray(data[2])[keep]


def keys(x): return [row.tobytes() for row in x.detach().cpu().contiguous().numpy()]


def domain_hash(data):
    result = hashlib.sha256()
    result.update(data[0].detach().cpu().contiguous().numpy().tobytes())
    for values in data[1:]: result.update(np.asarray(values).tobytes())
    return result.hexdigest()


def sample(data, count, seed):
    if len(data[1]) <= count: return data
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(data[1]):
        at = np.flatnonzero(np.asarray(data[1]) == cls)
        chosen.extend(rng.permutation(at)[:min(3, len(at))])
    if len(chosen) > count: raise ValueError('PilotRows cannot cover the classes.')
    available = np.setdiff1d(np.arange(len(data[1])), chosen)
    chosen.extend(rng.permutation(available)[:count-len(chosen)])
    keep = np.zeros(len(data[1]), dtype=bool); keep[chosen] = True
    return subset(data, keep)


def snapshot(agent):
    names = ('policy', 'target', 'td_optimizer')
    result = {name: deepcopy(getattr(agent, name).state_dict()) for name in names}
    result.update(rng=deepcopy(agent.rng.bit_generator.state), clock=agent.clock,
        td_updates=agent.td_updates, versions=agent.versions.copy(), td_memory=list(agent.td_memory))
    return result


def restore(agent, saved):
    for name in ('policy', 'target', 'td_optimizer'): getattr(agent, name).load_state_dict(saved[name])
    agent.rng.bit_generator.state = deepcopy(saved['rng'])
    agent.clock, agent.td_updates = saved['clock'], saved['td_updates']
    agent.versions = saved['versions'].copy()
    agent.td_memory = deque(saved['td_memory'], maxlen=agent.td_memory.maxlen)


def result_row(method, level, result, trace, classes):
    truth = [r['truth'] for r in trace]; predicted = [r['prediction'] for r in trace]
    pr, rc, f1, _ = precision_recall_fscore_support(truth, predicted,
        labels=list(range(classes)), average='macro', zero_division=0)
    reference = result['reference']
    return dict(method=method, level=level, AC=100*np.mean(np.asarray(truth) == predicted),
        PR=100*pr, RC=100*rc, F1=100*f1, samples=len(trace), labels=result['labels'],
        labels_percent=100*result['actual_label_fraction'],
        forgetting_AC=reference['drop_pp'],
        source_initial_AC=100*reference['initial'], source_final_AC=100*reference['final'],
        target_final_AC=100*result['same_level']['all_rows']['final'],
        context_forgetting_AC=100*(reference['per_route_initial'][1]-reference['per_route_final'][1]),
        td_updates=result['td_updates'], route_updates=result['route_updates'])


def show(row):
    print(f"{row['method']} level {row['level']}: AC {row['AC']:.2f}% | PR {row['PR']:.2f}% | "
        f"RC {row['RC']:.2f}% | F1 {row['F1']:.2f}% | labels {row['labels_percent']:.2f}% | "
        f"forgetting {row['forgetting_AC']:+.2f} pp | TD updates {row['td_updates']}", flush=True)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.models = args.source/'Models'
        checkpoint = self.models/'final_enhancement.pth'
        if not checkpoint.is_file():
            raise ValueError('PairOnly checkpoint not found: '+str(checkpoint)+'; specify --source if needed.')
        self.source_sha = digest(checkpoint)
        self.saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if self.saved.get('format_version') != base.FORMAT_VERSION:
            raise ValueError('Unsupported source checkpoint format.')
        if not set(self.saved['training_levels']).issubset(set(LEVELS)):
            raise ValueError('Source selector was trained on held-out intensities.')
        for name, expected in self.saved['model_hashes'].items():
            if digest(self.models/name) != expected: raise ValueError('Source model changed: '+name)
        for level, expected in self.saved['training_domain_sha256'].items():
            if digest(args.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != expected:
                raise ValueError('Source TRAIN data changed.')
        self.artifact = base.load_artifact(self.models, args.burst_dir)
        self.classes = len(self.artifact['label_encoder'].classes_)
        # All existing parser defaults remain available to the original episode.
        a = base.parser().parse_args(['train', '--output', str(args.output), '--burst-dir', str(args.burst_dir)])
        self.a = a
        a.models = self.models; a.seed = args.seed; a.classes = list(self.artifact['label_encoder'].classes_)
        a.context_count = 1; a.gamma = .95; a.context_lr = .00025; a.context_alpha = .8
        a.budget_fraction = .2; a.selection_cost = args.cost
        a.train_epsilon = .3; a.eval_epsilon = 0.; a.query_schedule = 'policy'; a.update_policy = 'learned'
        a.paid_replay_mode = 'batch5'; a.bank_utility = False; a.balanced_reward_reference = True
        a.query_budget_bonus = 0.; a.reference_size = self.saved['config'].get('reference_size', 32)
        a.gain_weight = 1.; a.retention_weight = 1.; a.continual = False
        a.config = replace(EnhancementConfig(**self.saved['config']), paired_weight=0.,
            shared_loss_rl=True, memory_ce_weight=.1).validated()
        a.device = args.device
        base.legacy.core.device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
            else 'cpu' if args.device == 'auto' else args.device)
        if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable.')
        base.legacy.core.set_seed(args.seed)
        cnn_saved = torch.load(self.models/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
        expected = 'legacy' if args.dataset == 'IAT' else 'light_ps_cnn_v1'
        if cnn_saved.get('architecture', 'legacy') != expected:
            raise ValueError('Expected original IAT CNN and Light PS CNN.')
        self.cnn = base.load_cnn(self.models, self.artifact)
        self.initial = torch.load(self.models/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
        self.domains = {}; self.audits = {}; self.audit_info = {}
        source = base.legacy.read_domain(args.burst_dir, self.artifact, 0, 'train_ids')
        self.anchors = source[:2]
        self.split('train_ids', LEVELS)
        train_audit_keys = set(keys(self.audits['train_ids'][0]))
        keep = np.asarray([key not in train_audit_keys for key in keys(source[0])])
        self.anchors = subset(source, keep)[:2]
        self.split('val_ids', LEVELS)
        self.threshold = self.calibrate()

    def split(self, split, levels):
        if split in self.domains: return self.domains[split], self.audits[split]
        source = base.legacy.read_domain(self.args.burst_dir, self.artifact, 0, split)
        anchor_keys = set(keys(self.anchors[0])) if split != 'train_ids' else set()
        if anchor_keys:
            source = subset(source, np.asarray([key not in anchor_keys for key in keys(source[0])]))
        rng = np.random.default_rng(self.args.seed+501)
        selected = []
        for cls in np.unique(source[1]):
            at = np.flatnonzero(np.asarray(source[1]) == cls)
            if len(at) < 3: raise ValueError('Source audit needs at least three eligible rows per class.')
            selected.extend(rng.permutation(at)[:max(1, int(.1*len(at)))])
        mask = np.zeros(len(source[1]), dtype=bool); mask[selected] = True
        audit = subset(source, mask)
        ids, blocked = set(map(int, audit[2])), set(keys(audit[0]))
        domains = {}
        for level in levels:
            data = base.legacy.read_domain(self.args.burst_dir, self.artifact, level, split)
            keep = np.asarray([int(i) not in ids and key not in blocked for i, key in zip(data[2], keys(data[0]))])
            domains[level] = sample(subset(data, keep), self.args.pilot_rows, self.args.seed+702)
        self.domains[split], self.audits[split] = domains, audit[:2]
        self.audit_info[split] = dict(ids=list(map(int, audit[2])),
            domain_hashes={str(k):domain_hash(v) for k, v in domains.items()}, audit_hash=domain_hash(audit),
            rule='source audit IDs/features excluded across all intensities before pilot sampling')
        return domains, audit[:2]

    def new_agent(self, saved=None, reset_values=False, parent=False):
        a = self.a
        if parent:
            a = deepcopy(a)
            a.config = replace(a.config, shared_loss_rl=False,
                paired_weight=self.saved['config'].get('paired_weight', .5)).validated()
        agent = base.new_agent(a, self.cnn, self.artifact, offline=False, saved=self.saved if saved is None else saved)
        if reset_values:
            # Old accuracy/utility returns are incompatible with the new payoff.
            for head in (agent.policy.future, agent.policy.update_value):
                output = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
                torch.nn.init.zeros_(output.weight); torch.nn.init.zeros_(output.bias)
            agent.target.load_state_dict(agent.policy.state_dict())
        return agent

    def calibrate(self):
        values = []
        context = base.build_bank(self.models, self.artifact, self.a.config, self.initial, slots=1)[0]
        # Source VAL features only. Labels never determine this cutoff.
        source = self.domains['val_ids'][0][0]
        with torch.no_grad():
            for start in range(0, len(source), 64):
                inputs = source[start:start+64]
                confidence = torch.stack([self.cnn(inputs), self.cnn(context(inputs))], 1).softmax(2).amax(2).max(1).values
                values.extend((1-confidence).cpu().tolist())
        return float(np.quantile(values, .75))

    def run(self, agent, data, reference, training, seed, method='SharedLossRL'):
        a = deepcopy(self.a); a.seed = seed
        mode = 'learned'
        if method == 'PairOnlyBonus1':
            a.config = replace(a.config, shared_loss_rl=False,
                paired_weight=self.saved['config'].get('paired_weight', .5)).validated()
            a.query_budget_bonus = 1.
            a.selection_cost = self.saved.get('training_args', {}).get('selection_cost', .0001)
        if method == 'Random': mode = 'random_al'
        if method == 'Uncertainty':
            a.uncertainty_routing = True; a.uncertainty_threshold = self.threshold
        if method in ('FrontLoadedContext', 'NoQueryContext'):
            a.architecture_ablation = 'no_rl'
            if method == 'FrontLoadedContext': a.context_only_frontloaded = True
            else: a.budget_fraction = 0.
        bank = base.build_bank(self.models, self.artifact, a.config, self.initial, slots=1)
        agent.uncertainty_routing = method == 'Uncertainty'
        agent.architecture_ablation = a.architecture_ablation
        agent.freeze_routing = training
        for p in agent.policy.routing_parameters(): p.requires_grad_(not training)
        return base.episode(a, self.cnn, bank, self.initial, agent, data, self.anchors,
            mode=mode, training=training, rollout_seed=seed, reference=reference,
            paid_supervision=True, same_level=not training)


def evaluate(experiment, saved, split, levels, root, controls=True):
    data, reference = experiment.split(split, levels)
    rows = []
    methods = ['SharedLossRL', 'PairOnlyBonus1', 'Uncertainty', 'Random', 'FrontLoadedContext', 'NoQueryContext'] if controls else ['SharedLossRL']
    for method in methods:
        for level in levels:
            base.legacy.core.set_seed(experiment.args.seed)
            agent = experiment.new_agent(experiment.saved if method == 'PairOnlyBonus1' else saved,
                parent=method == 'PairOnlyBonus1')
            result, trace = experiment.run(agent, data[level], reference, False, experiment.args.seed, method)
            item = result_row(method, level, result, trace, experiment.classes)
            rows.append(item); show(item)
            directory = root/method/f'level_{level}'; directory.mkdir(parents=True, exist_ok=True)
            base.write_json(directory/'metrics.json', result); base.write_csv(directory/'trace.csv', trace)
    base.write_csv(root/'summary.csv', rows)
    print('\nMEAN ONLINE METRICS | percent; forgetting in percentage points', flush=True)
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        print(method+': '+ ' | '.join(f'{k} {np.mean([r[k] for r in group]):.2f}'
            for k in ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting_AC')), flush=True)
    return rows


def train(experiment, protocol):
    args = experiment.args; root = args.output; model = root/'Models'/'shared_loss_rl.pth'
    complete = root/'training_complete.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['protocol'] != protocol or old['checkpoint_sha256'] != digest(model):
            raise ValueError('Completed checkpoint/protocol mismatch.')
        print('Using verified complete shared-loss RL training.', flush=True); return
    trainer = root/'Models'/'trainer_state.pth'
    agent = experiment.new_agent(reset_values=True)
    domains, _ = experiment.split('train_ids', LEVELS)
    if trainer.exists():
        state = torch.load(trainer, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol: raise ValueError('Resume settings changed; choose a new output.')
        restore(agent, state['agent']); history = state['history']; start = state['epoch']+1; best = state['best']
    else:
        if root.exists() and any(root.iterdir()): raise ValueError('Partial output preserved; choose a new --output.')
        root.mkdir(parents=True, exist_ok=True)
        history = []; start = 1; best = -float('inf')
        base.write_json(root/'protocol.json', protocol)
    for epoch in range(start, args.epochs+1):
        training_rows = []
        for level in LEVELS:
            result, trace = experiment.run(agent, domains[level], None, True, args.seed+epoch*1009+level*101)
            directory = root/'TRAIN'/f'epoch_{epoch:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            base.write_csv(directory/f'level_{level}_trace.csv', trace)
            rewards = [r['td_reward'] for r in trace]
            paid_rewards = [r['td_reward'] for r in trace if r['queried']]
            training_rows.append(dict(level=level, labels=result['labels'], cap=result['label_cap'],
                online_AC=100*result['accuracy'], td_updates=result['td_updates'],
                actual_transitions=len(trace), paid_loss_mean=float(np.mean([r['shared_loss_before'] for r in trace if r['queried']])) if result['labels'] else None,
                discounted_return=float(sum(experiment.a.gamma**t*r for t, r in enumerate(rewards))),
                positive_paid_rewards=sum(r > 0 for r in paid_rewards),
                mean_paid_reward=float(np.mean(paid_rewards)) if paid_rewards else None,
                TD_loss=agent.last_losses.get('td'), skipped_pair_buffer=len(agent.pair_memory)))
            print(f'TRAIN epoch {epoch}/{args.epochs} level {level}: AC {100*result["accuracy"]:.2f}% | '
                  f'labels {result["labels"]}/{result["label_cap"]} | TD updates {result["td_updates"]} | '
                  f'positive paid rewards {sum(r > 0 for r in paid_rewards)}/{len(paid_rewards)} | '
                  f'mean paid reward {np.mean(paid_rewards) if paid_rewards else 0.:+.4f}', flush=True)
        saved = dict(policy_state_dict=deepcopy(agent.policy.state_dict()), prerouter_state_dict=deepcopy(agent.pre_router.state_dict()))
        rows = evaluate(experiment, saved, 'val_ids', LEVELS, root/f'VAL_epoch_{epoch:02d}', controls=False)
        score = float(np.mean([r['AC']-max(0., r['forgetting_AC']) for r in rows]))
        history.append(dict(epoch=epoch, training=training_rows, validation=rows, selection_score=score))
        if score > best:
            best = score
            save_torch(model, dict(protocol=protocol, epoch=epoch, config=experiment.a.config.as_dict(), **saved))
        save_torch(trainer, dict(protocol=protocol, epoch=epoch, best=best, history=history, agent=snapshot(agent)))
        base.write_json(root/'training.json', dict(protocol=protocol, epochs=history))
    base.write_json(complete, dict(protocol=protocol, checkpoint_sha256=digest(model)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    parser.add_argument('--action', choices=['Train', 'Evaluate', 'All'], default='All')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cost', type=float, default=.05)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pilot_rows < 32 or args.epochs < 1 or not np.isfinite(args.cost) or args.cost < 0:
        parser.error('At least 32 pilot rows, positive epochs and finite nonnegative cost are required.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{args.dataset}_TEST'
    prefix = 'paironly_batch5' if args.dataset == 'IAT' else 'light_ps_paironly_batch5'
    args.source = args.source or root/'protocol_runs'/f'{prefix}_trainseed{args.seed}'/'paironly_batch5'
    args.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{args.dataset}'
    args.output = args.output or root/'protocol_runs'/f'shared_loss_rl_seed{args.seed}_p{args.pilot_rows}_e{args.epochs}_c{str(args.cost).replace(".","p")}_v1'
    experiment = Experiment(args)
    code_files = ['amcal_shared_loss_rl.py', *['enhancements/'+name+'.py' for name in
        ('agent', 'config', 'stream', 'paid_replay', 'counterfactual')]]
    protocol = dict(version=1, dataset=args.dataset, seed=args.seed, pilot_rows=args.pilot_rows, epochs=args.epochs,
        implementation_sha256={name:digest(Path(__file__).parent/name) for name in code_files},
        cost=args.cost, source_checkpoint_sha256=experiment.source_sha,
        source_models=experiment.saved['model_hashes'], config=experiment.a.config.as_dict(),
        source_training_levels=experiment.saved['training_levels'], training_levels=LEVELS,
        gamma=.95, context_lr=.00025, context_alpha=.8, budget_fraction=.2,
        replay_CE_weight=.1, prediction='a0 Frozen; a1/a2 Context; prediction precedes own feedback',
        payoff='query: actual pre-update batch5 Context objective including replay and modulation minus cost; skip: zero',
        optimization='alternating local shared-loss min-max approximation; ordinary sequential Double-DQN; no global convergence guarantee',
        heads='existing future/update_value heads trained; initial critic output layers reset to zero; routing frozen offline and paid-supervised online',
        transitions='one actual transition per row; no synthetic skip, paired target, correctness reward, bonus or future labels',
        validation='mean online AC minus positive original-domain forgetting; no TEST model selection',
        data_audits=deepcopy(experiment.audit_info))
    if args.action in ('Train', 'All'): train(experiment, protocol)
    if args.action in ('Evaluate', 'All'):
        path = args.output/'Models'/'shared_loss_rl.pth'
        complete = json.loads((args.output/'training_complete.json').read_text())
        if complete['protocol'] != protocol or complete['checkpoint_sha256'] != digest(path):
            raise ValueError('Selected checkpoint provenance mismatch.')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        evaluate(experiment, saved, 'test_ids', TEST_LEVELS, args.output/'Results')
        base.write_json(args.output/'evaluation_protocol.json', dict(protocol=protocol,
            audits=experiment.audit_info, controls='matched deployment replacements, not independently retrained no-RL ablations'))
    if digest(experiment.models/'final_enhancement.pth') != experiment.source_sha:
        raise RuntimeError('Source checkpoint changed.')


if __name__ == '__main__': main()
