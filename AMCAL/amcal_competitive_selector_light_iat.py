"""Opt-in compact shared-gap query policy; healthy CNN/Context retained.

Frozen-round contextual-bandit max steps alternate with real paid Context
updates. No sequential-DDQN or global saddle-point convergence is claimed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

import amcal_budget_actor as old
from enhancements.competitive_selector import CompetitiveSelector
from enhancements.budget_actor_math import fit_signal_scales
from enhancements.minimax_query import maximize_step

pilot = old.pilot
LEVELS = (0, 1, 3)


def save_model(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class Experiment:
    def __init__(self, a):
        self.a = a
        self.healthy, self.artifact, self.anchors, self.reference, self.teacher_hash = old.setup(a)
        # setup loads the healthy model's settings; the query intervention is explicit.
        a.competitive_selector = True
        a.competitive_spec = dict(mean=[0.]*(2*len(a.classes)+18), scale=[1.]*(2*len(a.classes)+18))
        a.collect_query_targets_only = True
        a.actor_experiment = None
        a.budget_actor_mode = True  # Fixed Context prediction; router never trained.
        a.eval_routing, a.eval_query_schedule = 'policy', 'policy'
        a.train_query_schedule = 'uniform'
        a.cost = 0. if getattr(a, 'original_paper_reward', False) else .05
        if a.context_lr != .00025 or a.replay_weight != .1:
            raise ValueError('This intervention requires the healthy LR .00025 and replay weight .1.')
        classifier = torch.load(a.models/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
        from enhancements.cnn_variant import validate_cnn_variant
        validate_cnn_variant(a.dataset, classifier.get('architecture', 'legacy'), getattr(a, 'light_cnn', False))
        held = set(map(int, self.reference[2]))
        held_keys = set(pilot.feature_keys(self.reference[0]))
        self.anchors = pilot.subset(self.anchors, np.asarray([
            k not in held_keys for k in pilot.feature_keys(self.anchors[0])]))
        self.train = {}
        for level in LEVELS:
            data = pilot.base.legacy.read_domain(a.burst_dir, self.artifact, level, 'train_ids')
            mask = np.asarray([int(i) not in held and k not in held_keys
                for i, k in zip(data[2], pilot.feature_keys(data[0]))])
            self.train[level] = old.pilot_domain(a, pilot.subset(data, mask), 702)
        self.evaluation_splits = {}

    def split(self, split, levels):
        if split not in self.evaluation_splits:
            source = pilot.base.legacy.read_domain(self.a.burst_dir, self.artifact, 0, split)
            anchor_keys = set(pilot.feature_keys(self.anchors[0]))
            source = pilot.subset(source, np.asarray([k not in anchor_keys for k in pilot.feature_keys(source[0])]))
            _, audit, _ = pilot.reference_partition(source, self.a.seed+501)
            self.evaluation_splits[split] = audit
        audit = self.evaluation_splits[split]
        ids, keys = set(map(int, audit[2])), set(pilot.feature_keys(audit[0]))
        domains = {}
        for level in levels:
            data = pilot.base.legacy.read_domain(self.a.burst_dir, self.artifact, level, split)
            mask = np.asarray([int(i) not in ids and k not in keys
                for i, k in zip(data[2], pilot.feature_keys(data[0]))])
            domains[level] = old.pilot_domain(self.a, pilot.subset(data, mask), 703 if split == 'val_ids' else 704)
        return domains, audit

    def collect(self, state, schedule, directory, round_index):
        states, scores, records = [], [], []
        self.a.train_query_schedule = schedule
        for level in LEVELS:
            pilot.base.legacy.core.set_seed(self.a.seed)
            live = old.runtime(self.a, self.healthy, self.artifact, state)
            recorder = old.Recorder()
            _, result, _ = pilot.stream(self.a, live, self.train[level], self.anchors,
                self.reference, self.reference, self.a.seed+round_index*1009+level*101,
                training=True, diagnostic_sink=recorder, target_endpoints=False)
            states.extend(recorder.states)
            scores.extend(recorder.scores)
            records.extend(dict(row, level=level) for row in recorder.rows)
            print(f'TRAIN round {round_index} level {level}: AC {result["online"]["AC"]:.2f}% | '
                  f'labels {result["labels"]}/{result["cap"]}', flush=True)
        if len(states) < 16:
            raise ValueError('Fewer than 16 measured TRAIN targets; output preserved.')
        directory.mkdir(parents=True, exist_ok=True)
        pilot.write_csv(directory/'targets.csv', records)
        return torch.stack(states), torch.tensor(scores, dtype=torch.float32), records

    def evaluate(self, state, split, levels, root):
        domains, audit = self.split(split, levels)
        rows = []
        for name, schedule in [('Actor', 'policy'), ('Uncertainty', 'uncertainty'), ('Random', 'policy'),
                               ('FrontLoaded', 'frontloaded'), ('NoQuery', 'policy')]:
            self.a.eval_query_schedule = schedule
            self.a.competitive_random_query = name == 'Random'
            for level in levels:
                pilot.base.legacy.core.set_seed(self.a.seed)
                live = old.runtime(self.a, self.healthy, self.artifact, state)
                budget = self.a.budget_fraction
                if name == 'NoQuery': self.a.budget_fraction = 0.
                try:
                    _, result, trace = pilot.stream(self.a, live, domains[level], self.anchors,
                        None, audit, self.a.seed, target_endpoints=True)
                finally:
                    self.a.budget_fraction = budget
                item = dict(method=name, level=level, **result['online'], labels=result['labels'],
                    labels_percent=result['labels_percent'], forgetting=result['original_domain_forgetting']['AC'],
                    source_initial_AC=result['source_before']['AC'], source_final_AC=result['source_after']['AC'],
                    target_final_AC=result['target_after']['AC'])
                rows.append(item)
                print(f'{name} level {level}: AC {item["AC"]:.2f}% | PR {item["PR"]:.2f}% | '
                      f'RC {item["RC"]:.2f}% | F1 {item["F1"]:.2f}% | labels {item["labels_percent"]:.2f}% | '
                      f'forgetting {item["forgetting"]:+.2f} pp', flush=True)
                directory = root/name/f'level_{level}'
                directory.mkdir(parents=True, exist_ok=True)
                pilot.write_json(directory/'metrics.json', result)
                pilot.write_csv(directory/'trace.csv', trace)
        self.a.eval_query_schedule = 'policy'
        self.a.competitive_random_query = False
        pilot.write_csv(root/'summary.csv', rows)
        return rows


def train(a, experiment, protocol):
    complete = a.output/'training_complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['protocol'] != protocol or saved['sha256'] != pilot.digest(a.output/'Models'/'selector.pth'):
            raise ValueError('Completed protocol/checkpoint mismatch; use a fresh output.')
        print('Using verified completed selector training.', flush=True)
        return
    resume = a.output/'Models'/'trainer_state.pth'
    if resume.exists():
        current = torch.load(resume, map_location='cpu', weights_only=False)
        if current['protocol'] != protocol: raise ValueError('Resume protocol mismatch.')
        a.competitive_spec, a.competitive_scales = current['spec'], current['scales']
        a.uncertainty_threshold = current['uncertainty_threshold']
    else:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError('Partial initialization preserved; use a fresh --output.')
        a.output.mkdir(parents=True, exist_ok=True)
        pilot.write_json(a.output/'protocol.json', protocol)
        states, _, records = experiment.collect(None, 'uniform', a.output/'initialization', 0)
        a.competitive_spec = old.normalizer(states)
        measured = [r for r in records if r.get('score') is not None]
        a.competitive_scales = (None if getattr(a, 'original_paper_reward', False)
                                else fit_signal_scales(measured))
        # Only TRAIN pre-feedback Context entropy determines the matched control.
        a.uncertainty_threshold = float(np.percentile(states[:, 2*len(a.classes)+5].numpy(), 75))
        current = dict(protocol=protocol, spec=a.competitive_spec, scales=a.competitive_scales,
            uncertainty_threshold=a.uncertainty_threshold, round=0, best=-float('inf'), history=[])
        pilot.base.legacy.core.set_seed(a.seed)
        head = CompetitiveSelector(a.competitive_spec)
        current['state'] = deepcopy(head.state_dict())
        current['optimizer'] = None
        save_model(resume, current)
    device = pilot.base.legacy.core.device
    head = CompetitiveSelector(a.competitive_spec).to(device)
    head.load_state_dict(current['state'])
    optimizer = torch.optim.Adam(head.parameters(), lr=a.actor_lr)
    if current['optimizer'] is not None: optimizer.load_state_dict(current['optimizer'])
    for round_index in range(current['round']+1, a.rounds+1):
        states, scores, _ = experiment.collect(deepcopy(head.state_dict()), 'policy',
            a.output/f'round_{round_index:02d}', round_index)
        states, scores = states.to(device), scores.to(device)
        rng = torch.Generator().manual_seed(a.seed+round_index)
        objectives = []
        for _ in range(a.steps):
            at = torch.randperm(len(states), generator=rng)[:min(64, len(states))].to(device)
            fit = maximize_step(head, optimizer, states[at], scores[at])
            objectives.append(-fit['actor_loss'])
        rows = experiment.evaluate(deepcopy(head.state_dict()), 'val_ids', LEVELS,
            a.output/f'VAL_round_{round_index:02d}')
        actor = [r for r in rows if r['method'] == 'Actor']
        # Accuracy/retention first; smaller label use only breaks equal scores.
        selection = float(np.mean([r['AC']-max(0., r['forgetting']) for r in actor]))
        labels = float(np.mean([r['labels_percent'] for r in actor]))
        key = (selection, -labels)
        old_key = tuple(current.get('best_key', (-float('inf'), 0.)))
        if key > old_key:
            current.update(best_key=list(key), selected_round=round_index)
            save_model(a.output/'Models'/'selector.pth', dict(protocol=protocol,
                spec=a.competitive_spec, scales=a.competitive_scales,
                uncertainty_threshold=a.uncertainty_threshold, state=deepcopy(head.state_dict()),
                selected_round=round_index))
        current['history'].append(dict(round=round_index, targets=len(scores),
            positive_targets=int((scores > 0).sum()), mean_target=float(scores.mean()),
            TRAIN_objective=float(np.mean(objectives)), VAL_selection=selection, VAL_labels_percent=labels))
        current.update(round=round_index, state=deepcopy(head.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
        save_model(resume, current)
        pilot.write_json(a.output/'training.json', dict(protocol=protocol, history=current['history']))
    pilot.write_json(complete, dict(protocol=protocol, selected_round=current['selected_round'],
        sha256=pilot.digest(a.output/'Models'/'selector.pth')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT'], required=True)
    parser.add_argument('--action', choices=['Train', 'Evaluate', 'All'], default='Train')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--actor-lr', type=float, default=.0001)
    parser.add_argument('--original-paper-reward', action='store_true',
        help='Refit only the compact query Actor using the original +3/+2/-2/0 reward; no future probes.')
    parser.add_argument('--light-cnn', action='store_true', default=True, help='Light IAT only; matching teacher required.')
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 32 or min(a.rounds, a.steps) < 1 or not np.isfinite(a.actor_lr) or a.actor_lr <= 0:
        parser.error('Positive rounds/steps/LR and at least 32 pilot rows are required.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    variant = 'light_iat_' if a.light_cnn and a.dataset == 'IAT' else ''
    a.teacher = a.teacher or root/'protocol_runs'/f'{variant}minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    suffix = '_paper_reward' if a.original_paper_reward else ''
    a.output = a.output or root/'protocol_runs'/f'{variant}competitive_selector_seed{a.seed}_p{a.pilot_rows}_r{a.rounds}_v1{suffix}'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.budget_fraction, a.progress_every = .2, 250
    experiment = Experiment(a)
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, pilot_rows=a.pilot_rows,
        rounds=a.rounds, steps=a.steps, actor_lr=a.actor_lr, budget_fraction=.2,
        teacher_sha256=experiment.teacher_hash, architecture='Light IAT CNN; Light PS CNN' if variant else 'original IAT CNN; Light PS CNN',
        context='healthy initial checkpoint; unchanged batch5/replay/LR; copies per domain',
        objective='max mean(p*(d/sd_d + weighted_G/sd_G - weighted_F/sd_F - .05))',
        theory='frozen-round shared-gap contextual-bandit min-max; not sequential DDQN or global saddle-point convergence',
        features='2*K+18; predictive uncertainty, paid-feedback coverage/novelty and causal budget/time',
        replay_probe='copy sampler RNG; sample the real update batch first, exclude its features from future probe',
        normalization='positive component SD and feature moments fitted once on TRAIN only; no score centering',
        policy='bounded logits; stochastic TRAIN requests; deterministic p>=.5 deployment; no forced spending',
        split='TRAIN/VAL 0/1/3; held source audit IDs/features excluded across levels; TEST only explicit Evaluate/All',
        selection='VAL online AC minus positive forgetting; fewer labels breaks ties',
        cost=.05, horizon=a.horizon, context_lr=a.context_lr, replay_weight=a.replay_weight,
        gain_weight=a.gain_weight, forgetting_weight=a.forgetting_weight,
        source_hashes=experiment.healthy['model_hashes'], train_hashes=experiment.healthy['training_domain_sha256'])
    if a.original_paper_reward:
        protocol.update(reward_mode='original paper Eq. reward',
            reward_table={'both_wrong':3., 'only_Frozen_wrong':2., 'only_Context_wrong':-2., 'both_correct':0.},
            objective='maximize mean(p*original_paper_reward); no gap/gain/harm/cost terms or target centering',
            theory='original-paper reward control with compact contextual-bandit policy; not original DDQN or the continuous shared-gap zero-sum objective',
            normalization='feature moments fitted on TRAIN; discrete reward remains unscaled',
            replay_probe='disabled; no copied Context/Adam, future-label probes or source-loss probes',
            initialization='same healthy CNN/Context checkpoint; new compact query head trained on original reward',
            horizon=None, cost=0., gain_weight=None, forgetting_weight=None)
    if a.action in ('Train', 'All'): train(a, experiment, protocol)
    if a.action in ('Evaluate', 'All'):
        completion = json.loads((a.output/'training_complete.json').read_text())
        path = a.output/'Models'/'selector.pth'
        if completion['protocol'] != protocol or completion['sha256'] != pilot.digest(path):
            raise ValueError('Selected checkpoint provenance mismatch.')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        a.competitive_spec, a.competitive_scales = saved['spec'], saved['scales']
        a.uncertainty_threshold = saved['uncertainty_threshold']
        experiment.evaluate(saved['state'], 'test_ids', pilot.TEST_LEVELS, a.output/'Results')
    if pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash:
        raise RuntimeError('Healthy teacher checkpoint changed.')


if __name__ == '__main__': main()
