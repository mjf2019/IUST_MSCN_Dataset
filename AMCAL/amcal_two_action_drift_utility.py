"""Opt-in utility DDQN on synthetic TRAIN/VAL feature drift; original TEST untouched."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from itertools import product
from dataclasses import asdict

import numpy as np
import torch

import amcal_competitive_selector as parent
from enhancements.competitive_selector import CompetitiveSelector, query_features, original_paper_reward
from enhancements.models import BankExecutor
from enhancements.two_action_ddqn import QueryDDQN
from enhancements.query_utility_reward import FixedProbeReward, prepare_train_probes
from enhancements.query_drift import fit_train_scale, transform_domain, episode_spec, choose_validation_candidate

pilot = parent.pilot
TRAIN = [0, 1, 3]
TEST = [5, 7, 10, 12, 15, 17, 20]
CONTROLS = ['ActorPaper', 'Uncertainty', 'Random', 'FrontLoaded', 'NoQuery']


def domain_hash(data):
    value = hashlib.sha256(data[0].detach().cpu().contiguous().numpy().tobytes())
    for part in data[1:]: value.update(np.asarray(part).tobytes())
    return value.hexdigest()


def weight_hash(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


@torch.no_grad()
def endpoint(experiment, context, data):
    x, y = data[:2]; predictions = []
    for start in range(0, len(y), 64):
        predictions.extend(experiment.cnn(context(x[start:start+64])).argmax(1).cpu().tolist())
    return pilot.metrics(y, predictions, len(experiment.a.classes))


class Experiment(parent.Experiment):
    def __init__(self, a):
        actor_file = a.actor_run/'Models'/'selector.pth'
        completion_file = a.actor_run/'training_complete.json'
        if not actor_file.is_file() or not completion_file.is_file():
            raise ValueError('Completed two-action paper-reward Actor missing: '+str(a.actor_run))
        saved = torch.load(actor_file, map_location='cpu', weights_only=False)
        completion = json.loads(completion_file.read_text(encoding='utf-8'))
        if completion['protocol'] != saved['protocol'] or completion['sha256'] != pilot.digest(actor_file):
            raise ValueError('Parent Actor checkpoint provenance mismatch.')
        p = saved['protocol']
        if p['dataset'] != a.dataset or p['seed'] != a.seed or p.get('reward_mode') != 'original paper Eq. reward':
            raise ValueError('Expected the completed paper-reward Actor for the same dataset and seed.')
        a.original_paper_reward = True
        super().__init__(a)
        if p['teacher_sha256'] != self.teacher_hash:
            raise ValueError('Actor and healthy CNN/Context teachers differ; supply the matching --teacher.')
        self.actor_sha = pilot.digest(actor_file)
        self.actor_saved = saved
        self.spec = saved['spec']
        if len(self.spec['mean']) != 2*len(a.classes)+18:
            raise ValueError('Parent compact state width differs.')
        self.actor = CompetitiveSelector(self.spec).to(pilot.base.legacy.core.device).eval().requires_grad_(False)
        self.actor.load_state_dict(saved['state'])
        for key in ('mean', 'scale'):
            if not torch.equal(torch.tensor(self.spec[key], dtype=torch.float32), saved['state'][key].float().cpu()):
                raise ValueError('Parent Actor normalization is inconsistent.')
        a.competitive_spec = self.spec
        self.threshold = float(saved['uncertainty_threshold'])
        self.cnn = pilot.base.load_cnn(a.models, self.artifact)
        self.template = pilot.base.build_bank(a.models, self.artifact, a.config,
            self.healthy['context_state_dict'], slots=1)[0].eval()
        self.split('val_ids', TRAIN)

    def context(self): return deepcopy(self.template).eval()

    def memory(self, seed):
        return pilot.Replay(self.anchors, seed+61, self.a.config, len(self.a.classes))

    def data_manifest(self, split, levels):
        domains, audit = (self.train, self.reference) if split == 'train_ids' else self.split(split, levels)
        return dict(streams={str(level):domain_hash(domains[level]) for level in levels},
                    source_audit=domain_hash(audit))



class DriftExperiment(Experiment):
    def enable_synthetic(self, probes, retention, train_anchors):
        self.clean_train = dict(self.train)
        self.base_probes = dict(zip(TRAIN, probes))
        self.retention_probe = retention
        self.train_anchors = train_anchors
        self.feature_scale = fit_train_scale(list(self.clean_train.values()))
        self.reward_parameters = None
        self.synthetic_enabled = True
        self.set_train_epoch(1)

    def set_train_epoch(self, epoch):
        self.train, self.shifted_probes, self.utility_by_domain = {}, {}, {}
        for level in TRAIN:
            spec = episode_spec(self.a.seed, 'TRAIN', level, epoch)
            self.train[level] = transform_domain(self.clean_train[level], self.feature_scale, spec)
            probe = transform_domain(self.base_probes[level], self.feature_scale, spec)
            self.shifted_probes[level] = probe
            self.utility_by_domain[id(self.train[level])] = (None if self.reward_parameters is None else
                FixedProbeReward([probe], self.retention_probe, **self.reward_parameters))
        self.reward_probe_keys = set().union(*(set(pilot.feature_keys(d[0])) for d in self.shifted_probes.values()))

    def reward_for_domain(self, domain):
        if id(domain) not in self.utility_by_domain:
            raise ValueError('TRAIN reward must use the probe from the same synthetic episode.')
        return self.utility_by_domain[id(domain)]

    def split(self, split, levels):
        domains, audit = super().split(split, levels)
        if getattr(self, 'synthetic_enabled', False) and split == 'val_ids':
            domains = {level:transform_domain(data, self.feature_scale,
                episode_spec(self.a.seed, 'VAL', level)) for level, data in domains.items()}
        return domains, audit

    def recipe(self):
        return dict(transform='label-blind feature-wise affine drift and row-ID seeded Gaussian noise',
            train_scale_sha256=hashlib.sha256(self.feature_scale.numpy().tobytes()).hexdigest(),
            scale_fitted_on='adaptive TRAIN feature values only; no labels, VAL or TEST',
            TRAIN={str(epoch):{str(level):asdict(episode_spec(self.a.seed, 'TRAIN', level, epoch))
                for level in TRAIN} for epoch in range(1, self.a.epochs+1)},
            VAL={str(level):asdict(episode_spec(self.a.seed, 'VAL', level)) for level in TRAIN},
            TEST='official unmodified inputs; no synthetic transforms; no selection',
            reward='per-episode matched shifted TRAIN probe; original-domain retention probe remains unshifted')

def run_episode(experiment, learner, domain, reference, seed, training=False, mode='DDQN', epsilon=0.):
    if mode not in ['DDQN', *CONTROLS] or (training and mode != 'DDQN'):
        raise ValueError('Unknown query policy or non-DDQN training override.')
    a = experiment.a; cnn = experiment.cnn; context = experiment.context()
    optimizer = torch.optim.Adam(context.parameters(), lr=a.context_lr, weight_decay=1e-4)
    memory = (pilot.Replay(experiment.train_anchors, seed+61, a.config, len(a.classes))
        if training and hasattr(experiment, 'train_anchors') else experiment.memory(seed))
    utility = (experiment.reward_for_domain(domain) if hasattr(experiment, 'reward_for_domain') else
        getattr(experiment, 'utility_reward', None)) if training else None
    order = np.random.default_rng(seed).permutation(len(domain[1]))
    x = domain[0][torch.as_tensor(order, device=domain[0].device)]
    y, ids = np.asarray(domain[1])[order], np.asarray(domain[2])[order]
    n = len(y); cap = int(a.budget_fraction*n)
    if not n: raise ValueError('Empty adaptation stream.')
    before = endpoint(experiment, context, reference) if reference is not None else {}
    target_before = endpoint(experiment, context, (x, y)) if not training else {}
    source_keys = set(pilot.feature_keys(experiment.reference[0])) if training else set()
    if training and hasattr(experiment, 'reward_probe_keys'): source_keys.update(experiment.reward_probe_keys)
    feature_keys = pilot.feature_keys(x)
    rng = np.random.default_rng(seed+71)
    paid_embeddings, paid_labels = [], []
    used = last_update = 0; pending = None
    traces, predictions, td_losses = [], [], []
    updates_before = learner.updates
    if x.is_cuda: torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    frozen, embeddings = pilot.base.legacy.base_cache(cnn, x)
    executor = BankExecutor(cnn, [context])
    for position in range(n):
        inputs = x[position:position+1]
        with torch.no_grad():
            outputs = executor.outputs(inputs, frozen[position:position+1])
            state = query_features(outputs, embeddings[position:position+1], paid_embeddings,
                paid_labels, used, cap, position, n, last_update).detach().cpu().reshape(-1)
            context_entropy = float(pilot.normalized_entropy(outputs[0][0, 1].cpu().numpy()))
        can_query = used < cap
        # Finish the previous transition only when this input actually arrives.
        # Its label is not accessed to construct the successor or TD target.
        if training and pending is not None:
            loss = learner.remember(*pending, state, False, can_query)
            if loss is not None: td_losses.append(loss['loss'])
        q_values = None; probability = None
        if mode == 'DDQN':
            action, q_values = learner.choose(state, can_query, epsilon if training else 0., rng)
        elif mode == 'ActorPaper':
            with torch.no_grad():
                value = experiment.actor(state.to(next(experiment.actor.parameters()).device)[None])[0]
                probability = float(value.sigmoid()[0])
            action = int(can_query and probability >= .5)
        elif mode == 'Uncertainty': action = int(can_query and context_entropy >= experiment.threshold)
        elif mode == 'Random': action = int(can_query and rng.random() < a.budget_fraction)
        elif mode == 'FrontLoaded': action = int(can_query and position < cap)
        else: action = 0
        if action == 1 and not can_query: raise RuntimeError('Feedback cap exceeded.')
        # All policies predict Context -> Frozen before acquiring the current label.
        frozen_prediction = int(outputs[0][0, 0].argmax())
        prediction = int(outputs[0][0, 1].argmax())
        truth = int(y[position]); reward = 0.; reward_components = {}
        if action == 1:
            reward = original_paper_reward(frozen_prediction, prediction, truth)
            label = torch.tensor([truth], device=x.device)
            paid, replay = memory.draw(ids[position], inputs, reference_keys=source_keys,
                current_key=feature_keys[position])
            if utility is not None:
                utility.assert_disjoint_update((inputs, label), paid, replay)
                probe_before = utility.measure(cnn, context)
            pilot.context_min_step(context, optimizer, cnn, (inputs, label), paid, replay, a.replay_weight)
            if utility is not None:
                reward, reward_components = utility.finish(cnn, context, probe_before)
            memory.add(ids[position], inputs, truth)
            paid_embeddings.append(embeddings[position:position+1].detach().clone())
            paid_labels.append(truth); used += 1; last_update = position+1
        predictions.append(prediction)
        traces.append(dict(step=position+1, source_row=int(ids[position]), truth=truth,
            prediction=prediction, frozen_prediction=frozen_prediction, context_prediction=prediction,
            action=action, route=1, queried=bool(action), labels_used=used, cap=cap,
            reward=reward, query_legal_before=can_query, context_entropy=context_entropy,
            probe_gain=reward_components.get('probe_gain'), probe_harm=reward_components.get('probe_harm'),
            weighted_probe_harm=reward_components.get('weighted_probe_harm'),
            request_cost=reward_components.get('request_cost'),
            generalization_loss_before=reward_components.get('generalization_loss_before'),
            generalization_loss_after=reward_components.get('generalization_loss_after'),
            retention_loss_before=reward_components.get('retention_loss_before'),
            retention_loss_after=reward_components.get('retention_loss_after'),
            novelty=float(state[-8]), paid_class_coverage=float(state[-6]),
            q_skip=None if q_values is None else float(q_values[0]),
            q_query=None if q_values is None else float(q_values[1]), actor_probability=probability,
            epsilon=epsilon if training else 0., terminal=position+1 == n))
        pending = (state, action, reward)
    if training:
        loss = learner.remember(*pending, torch.zeros_like(state), True, False)
        if loss is not None: td_losses.append(loss['loss'])
    if x.is_cuda: torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter()-started
    after = endpoint(experiment, context, reference) if reference is not None else {}
    target_after = endpoint(experiment, context, (x, y)) if not training else {}
    result = dict(online=pilot.metrics(y, predictions, len(a.classes)), samples=n, labels=used,
        cap=cap, labels_percent=100*used/n, source_before=before, source_after=after,
        original_domain_forgetting={key:before[key]-after[key] for key in before},
        target_before=target_before, target_after=target_after,
        td_updates=learner.updates-updates_before, mean_TD_loss=float(np.mean(td_losses)) if td_losses else None,
        positive_paid_rewards=sum(r['reward'] > 0 for r in traces if r['queried']),
        milliseconds_per_row=1000*elapsed/n,
        label_protocol='prediction before feedback; zero skip reward uses no label; fixed Context path',
        Q_learning='TRAIN only; actual successor at next input arrival; frozen DDQN during VAL/TEST')
    if training:
        paid_rewards = [r for r in traces if r['queried']]
        result['reward_diagnostics'] = dict(mode='independent TRAIN probe gain - weighted harm - cost' if utility is not None else 'original paper reward',
            positive_fraction=sum(r['reward'] > 0 for r in paid_rewards)/max(len(paid_rewards), 1),
            mean_reward=float(np.mean([r['reward'] for r in paid_rewards])) if paid_rewards else None,
            components={key:float(np.mean([r[key] for r in paid_rewards])) if paid_rewards else None
                for key in ('probe_gain', 'probe_harm', 'weighted_probe_harm', 'request_cost')} if utility is not None else {})
        print('TRAIN reward: '+json.dumps(result['reward_diagnostics'], sort_keys=True), flush=True)
    if used > cap or used != sum(r['queried'] for r in traces): raise RuntimeError('Invalid label accounting.')
    return result, traces


def result_row(mode, level, result):
    return dict(method=mode, level=level, **result['online'], samples=result['samples'],
        labels=result['labels'], labels_percent=result['labels_percent'],
        forgetting=result['original_domain_forgetting']['AC'],
        source_initial_AC=result['source_before']['AC'], source_final_AC=result['source_after']['AC'],
        target_final_AC=result['target_after']['AC'], td_updates=result['td_updates'])


def print_table(rows):
    print('\nMEAN ONLINE AC/PR/RC/F1 AND LABELS IN PERCENT; FORGETTING IN POINTS', flush=True)
    for mode in dict.fromkeys(row['method'] for row in rows):
        group = [row for row in rows if row['method'] == mode]
        print(mode+': '+ ' | '.join(f'{key} {np.mean([r[key] for r in group]):.2f}' for key in
            ('AC', 'PR', 'RC', 'F1', 'target_final_AC', 'labels_percent', 'forgetting')), flush=True)


def evaluate(experiment, learner, split, levels, directory, controls):
    domains, reference = experiment.split(split, levels)
    start_weights = weight_hash(learner.policy.state_dict()); start_updates = learner.updates
    rows = []
    for mode in ['DDQN', *CONTROLS] if controls else ['DDQN']:
        for level in levels:
            result, trace = run_episode(experiment, learner, domains[level], reference,
                experiment.a.seed, mode=mode)
            row = result_row(mode, level, result); rows.append(row)
            print(f'{mode} level {level}: AC {row["AC"]:.2f}% | F1 {row["F1"]:.2f}% | '
                f'labels {row["labels_percent"]:.2f}% | forgetting {row["forgetting"]:+.2f} pp', flush=True)
            root = directory/mode/f'level_{level}'; root.mkdir(parents=True, exist_ok=True)
            pilot.write_json(root/'metrics.json', result); pilot.write_csv(root/'trace.csv', trace)
    if start_weights != weight_hash(learner.policy.state_dict()) or start_updates != learner.updates:
        raise RuntimeError('Validation/test changed the DDQN weights.')
    pilot.write_csv(directory/'summary.csv', rows); print_table(rows)
    return rows


def train(experiment, learner, protocol):
    a = experiment.a; root = a.output
    model = root/'Models'/'two_action_ddqn.pth'; complete = root/'training_complete.json'
    if complete.exists():
        finished = json.loads(complete.read_text(encoding='utf-8'))
        if finished['protocol'] != protocol or finished['checkpoint_sha256'] != pilot.digest(model):
            raise ValueError('Completed run changed; use another output.')
        print('Using verified complete DDQN training.', flush=True); return
    trainer = root/'Models'/'trainer_state.pth'
    if trainer.exists():
        state = torch.load(trainer, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol: raise ValueError('Resume protocol differs; use another output.')
        learner.restore(state['learner']); history = state['history']; best = tuple(state['best']); start = state['epoch']+1
    else:
        if root.exists() and any(root.iterdir()): raise ValueError('Unmatched partial output preserved; choose a new --output.')
        root.mkdir(parents=True, exist_ok=True)
        pilot.write_json(root/'protocol.json', protocol)
        history = []; best = (-float('inf'), -float('inf')); start = 1
    for epoch in range(start, a.epochs+1):
        if hasattr(experiment, 'set_train_epoch'): experiment.set_train_epoch(epoch)
        epsilon = .3-.25*(epoch-1)/max(a.epochs-1, 1)
        training = []
        for level in TRAIN:
            result, trace = run_episode(experiment, learner, experiment.train[level], None,
                a.seed+epoch*1009+level*101, training=True, epsilon=epsilon)
            path = root/'TRAIN'/f'epoch_{epoch:02d}'; path.mkdir(parents=True, exist_ok=True)
            pilot.write_csv(path/f'level_{level}_trace.csv', trace)
            training.append(dict(level=level, epsilon=epsilon, **result))
            print(f'TRAIN epoch {epoch}/{a.epochs} level {level}: AC {result["online"]["AC"]:.2f}% | '
                f'labels {result["labels"]}/{result["cap"]} | TD updates {result["td_updates"]} | '
                f'positive paid rewards {result["positive_paid_rewards"]}', flush=True)
        rows = evaluate(experiment, learner, 'val_ids', TRAIN, root/f'VAL_epoch_{epoch:02d}', controls=False)
        score = float(np.mean([r['AC']-max(0., r['forgetting']) for r in rows]))
        labels = float(np.mean([r['labels_percent'] for r in rows]))
        key = (score, -labels)
        history.append(dict(epoch=epoch, training=training, validation=rows, selection_score=score,
            validation_labels_percent=labels))
        if key > best:
            best = key
            parent.save_model(model, dict(protocol=protocol, selected_epoch=epoch,
                policy_state=deepcopy(learner.policy.state_dict())))
        parent.save_model(trainer, dict(protocol=protocol, epoch=epoch, learner=learner.snapshot(),
            history=history, best=list(best)))
        pilot.write_json(root/'training.json', dict(protocol=protocol, epochs=history))
    selected = torch.load(model, map_location='cpu', weights_only=False)
    pilot.write_json(complete, dict(protocol=protocol, selected_epoch=selected['selected_epoch'],
        checkpoint_sha256=pilot.digest(model)))


def average_result(rows, name):
    return dict(candidate=name,
        AC=float(np.mean([r['AC'] for r in rows])),
        F1=float(np.mean([r['F1'] for r in rows])),
        labels_percent=float(np.mean([r['labels_percent'] for r in rows])),
        forgetting=float(np.mean([r['forgetting'] for r in rows])),
        positive_forgetting=float(np.mean([max(0., r['forgetting']) for r in rows])))


def select_on_validation(candidates, reference, accuracy_tolerance=.5, forgetting_tolerance=.5,
                         no_query=None, min_adaptation_gain=.5):
    if no_query is None: raise ValueError('A synthetic VAL NoQuery reference is required.')
    return choose_validation_candidate(candidates, reference, no_query, accuracy_tolerance,
        forgetting_tolerance, min_adaptation_gain)


def candidate_name(weight, cost):
    def token(value): return format(value, '.8g').replace('.', 'p').replace('-', 'm')
    return 'utility_f'+token(weight)+'_c'+token(cost)


def load_selected(experiment, name, protocol):
    model = experiment.a.output/'Models'/'two_action_ddqn.pth'
    completion = json.loads((experiment.a.output/'training_complete.json').read_text(encoding='utf-8'))
    if completion['protocol'] != protocol or completion['checkpoint_sha256'] != pilot.digest(model):
        raise ValueError('Selected candidate provenance mismatch: '+name)
    saved = torch.load(model, map_location='cpu', weights_only=False)
    if saved['protocol'] != protocol: raise ValueError('Candidate model protocol mismatch.')
    pilot.base.legacy.core.set_seed(experiment.a.seed)
    learner = QueryDDQN(experiment.spec, experiment.actor_saved['state'], pilot.base.legacy.core.device,
        experiment.a.seed)
    learner.policy.load_state_dict(saved['policy_state'])
    return learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    parser.add_argument('--action', choices=['Train', 'Evaluate', 'All'], default='All')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--probe-rows', type=int, default=24)
    parser.add_argument('--retention-rows', type=int, default=32)
    parser.add_argument('--costs', type=float, nargs='+', default=[0., .001])
    parser.add_argument('--forgetting-weights', type=float, nargs='+', default=[.5, 1.])
    parser.add_argument('--accuracy-tolerance', type=float, default=.5)
    parser.add_argument('--forgetting-tolerance', type=float, default=.5)
    parser.add_argument('--min-adaptation-gain', type=float, default=.5)
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 32 or a.epochs < 1: parser.error('Use at least 32 pilot rows and positive epochs.')
    if not all(np.isfinite(v) and v >= 0 for v in [*a.costs, *a.forgetting_weights,
            a.accuracy_tolerance, a.forgetting_tolerance, a.min_adaptation_gain]):
        parser.error('Costs, weights and validation tolerances must be finite and nonnegative.')
    a.costs, a.forgetting_weights = sorted(set(a.costs)), sorted(set(a.forgetting_weights))
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = a.output or root/'protocol_runs'/f'two_action_drift_utility_seed{a.seed}_p{a.pilot_rows}_e{a.epochs}_v1'
    output = a.output
    a.budget_fraction = .2
    experiment = DriftExperiment(a)
    probes, retention, train_anchors, probe_manifest = prepare_train_probes(experiment, pilot,
        a.probe_rows, a.retention_rows)
    experiment.enable_synthetic(probes, retention, train_anchors)
    probe_manifest.update(generalization_sha256=[domain_hash(d) for d in probes],
        retention_sha256=domain_hash(retention), TRAIN_anchors_sha256=domain_hash(train_anchors))
    files = ['amcal_two_action_drift_utility.py', 'enhancements/query_drift.py', 'enhancements/query_utility_reward.py',
        'enhancements/two_action_ddqn.py', 'enhancements/competitive_selector.py',
        'amcal_competitive_selector.py', 'amcal_budget_actor.py', 'amcal_minimax_paironly.py',
        'enhancements/minimax_query.py']
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, pilot_rows=a.pilot_rows, epochs=a.epochs,
        actor_checkpoint_sha256=experiment.actor_sha, healthy_teacher_sha256=experiment.teacher_hash,
        source_models=experiment.healthy['model_hashes'], train_files=experiment.healthy['training_domain_sha256'],
        implementation_sha256={name:pilot.digest(Path(__file__).parent/name) for name in files},
        train_data=experiment.data_manifest('train_ids', TRAIN), validation_data=experiment.data_manifest('val_ids', TRAIN),
        probes=probe_manifest, synthetic=experiment.recipe(), costs=a.costs, forgetting_weights=a.forgetting_weights,
        min_adaptation_gain_pp=a.min_adaptation_gain,
        accuracy_tolerance_pp=a.accuracy_tolerance, forgetting_tolerance_pp=a.forgetting_tolerance,
        architecture='unchanged healthy Context; original IAT CNN; Light PS CNN; fixed Context -> Frozen path',
        reward='query: TRAIN macro-CE decrease minus weighted positive TRAIN retention-CE increase minus cost; skip zero',
        measurement='before/after one actual batch5 update on an independent TRAIN probe with the same episode drift; no rollout',
        loss_units='raw macro cross entropy; no clipping, sign-only targets or TEST-based normalization',
        DDQN=dict(gamma=.95, lr=.0001, batch_size=32, learn_every=4, target_every=15,
            capacity=4096, epsilon='0.30 to 0.05; greedy VAL/TEST; same compact state and parent encoder'),
        context_lr=a.context_lr, replay_weight=a.replay_weight, budget_fraction=.2,
        evaluation='original anchors and same sampled IDs for all controls; synthetic VAL features, unchanged official TEST; reset adaptive state per level',
        query_learning='TRAIN only; DDQN/Actor frozen on VAL/TEST; observer TEST labels never form reward targets',
        comparison='paper-reward DDQN retrained on identical synthetic TRAIN episodes and probe-excluded anchors',
        theory='closed-loop sequential DDQN; Context competitive-gap descent unchanged; no formal shared min-max objective claimed')
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Utility experiment protocol differs; preserve this run and use a new --output.')
    elif any(output.iterdir()):
        raise ValueError('Unmatched output preserved; use a fresh --output.')
    else:
        if a.action == 'Evaluate': raise ValueError('Complete TRAIN/VAL selection before evaluation.')
        pilot.write_json(protocol_file, protocol)
        pilot.write_json(output/'probe_manifest.json', probe_manifest)
        pilot.write_json(output/'synthetic_recipe.json', experiment.recipe())
    base_args = deepcopy(a)
    configurations = {'paper':None}
    for weight, cost in product(a.forgetting_weights, a.costs):
        name = candidate_name(weight, cost)
        if name in configurations: raise ValueError('Candidate name collision; use more distinct grid values.')
        configurations[name] = dict(forgetting_weight=weight, cost=cost)

    def configure(name):
        experiment.a = deepcopy(base_args)
        experiment.a.output = output/'Candidates'/name
        parameters = configurations[name]
        experiment.reward_parameters = parameters
        experiment.set_train_epoch(1)
        return dict(protocol=protocol, candidate=name, utility=parameters)

    selection_file = output/'selection.json'
    if a.action in ('Train', 'All'):
        validations, controls, no_query = [], None, None
        for index, name in enumerate(configurations, 1):
            print(f'\nCANDIDATE {index}/{len(configurations)}: {name}; same healthy checkpoint; TRAIN/VAL only', flush=True)
            candidate_protocol = configure(name)
            pilot.base.legacy.core.set_seed(a.seed)
            learner = QueryDDQN(experiment.spec, experiment.actor_saved['state'], pilot.base.legacy.core.device, a.seed)
            train(experiment, learner, candidate_protocol)
            learner = load_selected(experiment, name, candidate_protocol)
            rows = evaluate(experiment, learner, 'val_ids', TRAIN,
                experiment.a.output/'VAL_selected', controls=name == 'paper')
            report = average_result([r for r in rows if r['method'] == 'DDQN'], name)
            validations.append(report)
            if name == 'paper': controls = average_result([r for r in rows if r['method'] == 'FrontLoaded'], 'FrontLoaded')
            if name == 'paper': no_query = average_result([r for r in rows if r['method'] == 'NoQuery'], 'NoQuery')
        selection = select_on_validation([r for r in validations if r['candidate'] != 'paper'], controls,
            a.accuracy_tolerance, a.forgetting_tolerance, no_query, a.min_adaptation_gain)
        selection.update(protocol=protocol, candidates=validations,
            model_sha256={name:pilot.digest(output/'Candidates'/name/'Models'/'two_action_ddqn.pth') for name in configurations})
        pilot.write_csv(output/'validation_candidates.csv', validations)
        pilot.write_json(selection_file, selection)
        print('VAL selection: '+json.dumps({k:selection[k] for k in ('selected', 'development_scenario_valid', 'validation_goal_met', 'eligible')}), flush=True)
        selected_report = next(r for r in validations if r['candidate'] == selection['selected'])
        print(f'VAL AC: NoQuery {no_query["AC"]:.2f}% | FrontLoaded {controls["AC"]:.2f}% | '
            f'selected {selected_report["AC"]:.2f}% | selected labels {selected_report["labels_percent"]:.2f}% | '
            f'gain over NoQuery {selection["adaptation_gain_vs_no_query_pp"]:+.2f} pp', flush=True)
    if a.action in ('Evaluate', 'All'):
        selection = json.loads(selection_file.read_text(encoding='utf-8'))
        if selection['protocol'] != protocol: raise ValueError('VAL selection protocol mismatch.')
        selected_name = selection['selected']
        combined = []
        for name, controls in [(selected_name, True), ('paper', False)]:
            candidate_protocol = configure(name)
            model = experiment.a.output/'Models'/'two_action_ddqn.pth'
            if pilot.digest(model) != selection['model_sha256'][name]:
                raise ValueError('Candidate changed after VAL selection.')
            learner = load_selected(experiment, name, candidate_protocol)
            rows = evaluate(experiment, learner, 'test_ids', TEST,
                output/'Results'/('SelectedUtility' if controls else 'PaperDDQN'), controls=controls)
            for row in rows:
                row = dict(row)
                if row['method'] == 'DDQN': row['method'] = 'UtilityDDQN' if controls else 'PaperDDQN'
                combined.append(row)
        pilot.write_csv(output/'Results'/'comparison_summary.csv', combined)
        pilot.write_json(output/'Results'/'execution.json', dict(protocol=protocol,
            selection=selection, test_data=experiment.data_manifest('test_ids', TEST)))
        print_table(combined)
        print('VAL consumption/accuracy goal met: '+str(selection['validation_goal_met'])+
            '; TEST is independent evidence, not a selection input.', flush=True)
    if pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha:
        raise RuntimeError('Parent Actor changed.')
    if pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash:
        raise RuntimeError('Healthy Context checkpoint changed.')


if __name__ == '__main__': main()
