"""DDQN on the strong two-action checkpoint; fixed Context prediction path.

The parent Actor, DDQN and controls use identical data, initial Context, replay,
learning rate and source audit. Only the query decision changes.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import amcal_competitive_selector_light_iat as parent
from enhancements.competitive_selector import CompetitiveSelector, query_features, original_paper_reward
from enhancements.models import BankExecutor
from enhancements.two_action_ddqn import QueryDDQN

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


def run_episode(experiment, learner, domain, reference, seed, training=False, mode='DDQN', epsilon=0.):
    if mode not in ['DDQN', *CONTROLS] or (training and mode != 'DDQN'):
        raise ValueError('Unknown query policy or non-DDQN training override.')
    a = experiment.a; cnn = experiment.cnn; context = experiment.context()
    optimizer = torch.optim.Adam(context.parameters(), lr=a.context_lr, weight_decay=1e-4)
    memory = experiment.memory(seed)
    order = np.random.default_rng(seed).permutation(len(domain[1]))
    x = domain[0][torch.as_tensor(order, device=domain[0].device)]
    y, ids = np.asarray(domain[1])[order], np.asarray(domain[2])[order]
    n = len(y); cap = int(a.budget_fraction*n)
    if not n: raise ValueError('Empty adaptation stream.')
    before = endpoint(experiment, context, reference) if reference is not None else {}
    target_before = endpoint(experiment, context, (x, y)) if not training else {}
    source_keys = set(pilot.feature_keys(experiment.reference[0])) if training else set()
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
        truth = int(y[position]); reward = 0.
        if action == 1:
            reward = original_paper_reward(frozen_prediction, prediction, truth)
            label = torch.tensor([truth], device=x.device)
            paid, replay = memory.draw(ids[position], inputs, reference_keys=source_keys,
                current_key=feature_keys[position])
            pilot.context_min_step(context, optimizer, cnn, (inputs, label), paid, replay, a.replay_weight)
            memory.add(ids[position], inputs, truth)
            paid_embeddings.append(embeddings[position:position+1].detach().clone())
            paid_labels.append(truth); used += 1; last_update = position+1
        predictions.append(prediction)
        traces.append(dict(step=position+1, source_row=int(ids[position]), truth=truth,
            prediction=prediction, frozen_prediction=frozen_prediction, context_prediction=prediction,
            action=action, route=1, queried=bool(action), labels_used=used, cap=cap,
            reward=reward, query_legal_before=can_query, context_entropy=context_entropy,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT'], required=True)
    parser.add_argument('--action', choices=['Train', 'Evaluate', 'All'], default='All')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--light-cnn', action='store_true', default=True, help='Light IAT only; matching teacher and Actor required.')
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 32 or a.epochs < 1: parser.error('Use at least 32 pilot rows and positive epochs.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    variant = 'light_iat_' if a.light_cnn and a.dataset == 'IAT' else ''
    a.teacher = a.teacher or root/'protocol_runs'/f'{variant}minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'{variant}competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = a.output or root/'protocol_runs'/f'two_action_ddqn_{variant}seed{a.seed}_p{a.pilot_rows}_e{a.epochs}_v1'
    a.budget_fraction = .2
    experiment = Experiment(a)
    device = pilot.base.legacy.core.device
    learner = QueryDDQN(experiment.spec, experiment.actor_saved['state'], device, a.seed)
    files = ['amcal_two_action_ddqn_light_iat.py', 'enhancements/two_action_ddqn.py', 'enhancements/competitive_selector.py',
        'amcal_competitive_selector_light_iat.py', 'amcal_minimax_paironly_light_iat.py',
        'enhancements/cnn_variant.py', 'amcal_budget_actor.py', 'amcal_minimax_paironly.py', 'enhancements/minimax_query.py']
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, pilot_rows=a.pilot_rows, epochs=a.epochs,
        actor_checkpoint_sha256=experiment.actor_sha, healthy_teacher_sha256=experiment.teacher_hash,
        source_models=experiment.healthy['model_hashes'], train_files=experiment.healthy['training_domain_sha256'],
        implementation_sha256={name:pilot.digest(Path(__file__).parent/name) for name in files},
        train_data=experiment.data_manifest('train_ids', TRAIN), validation_data=experiment.data_manifest('val_ids', TRAIN),
        architecture=('same healthy CNN/Context as two-action Actor; Light IAT CNN and Light PS CNN' if variant else
            'same healthy CNN/Context as two-action Actor; original IAT CNN and Light PS CNN'),
        actions={'0':'Context prediction; no feedback', '1':'Context prediction; acquire feedback and batch5 update'},
        reward={'both_wrong':3., 'only_Frozen_wrong':2., 'only_Context_wrong':-2., 'both_correct':0., 'skip':0.},
        gamma=.95, lr=.0001, batch_size=32, learn_every=4, target_every=15, replay_capacity=4096,
        feature_width=len(experiment.spec['mean']), normalization='unchanged frozen TRAIN normalizer from parent Actor',
        encoder='parent Actor 16-unit hidden layer; new two-output return layer initialized to zero',
        TRAIN_epsilon='0.30 to 0.05 across epochs', eval_epsilon=0.,
        context_lr=a.context_lr, replay_weight=a.replay_weight, budget_fraction=.2,
        online='Context adapts; DDQN and Actor weights frozen; every level resets Context/Adam/memory/budget',
        selection='VAL mean online AC minus positive original-domain forgetting; fewer labels only breaks ties',
        theory='closed-loop DDQN with competitive original-paper reward; no shared-objective min-max claim',
        scope='matched small-data pilot; no method superiority assumed; deployment controls, not retrained no-RL ablations')
    if a.action in ('Train', 'All'): train(experiment, learner, protocol)
    if a.action in ('Evaluate', 'All'):
        model = a.output/'Models'/'two_action_ddqn.pth'
        completion = json.loads((a.output/'training_complete.json').read_text(encoding='utf-8'))
        if completion['protocol'] != protocol or completion['checkpoint_sha256'] != pilot.digest(model):
            raise ValueError('Selected DDQN checkpoint provenance mismatch.')
        saved = torch.load(model, map_location='cpu', weights_only=False)
        learner.policy.load_state_dict(saved['policy_state'])
        evaluate(experiment, learner, 'test_ids', TEST, a.output/'Results', controls=True)
        pilot.write_json(a.output/'evaluation_protocol.json', dict(protocol=protocol,
            selected_epoch=completion['selected_epoch'], test_data=experiment.data_manifest('test_ids', TEST)))
    if pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha:
        raise RuntimeError('Parent Actor changed.')
    if pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash:
        raise RuntimeError('Healthy Context checkpoint changed.')


if __name__ == '__main__': main()
