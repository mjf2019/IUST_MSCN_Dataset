"""Opt-in DDQN: original paper reward + beta * same-domain TRAIN gain."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import amcal_competitive_selector as parent
from enhancements.competitive_selector import CompetitiveSelector, query_features, original_paper_reward
from enhancements.models import BankExecutor
from enhancements.two_action_ddqn import QueryDDQN
from enhancements.query_utility_reward import prepare_train_probes
from enhancements.paper_gain_reward import SameDomainGain

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



class PaperGainExperiment(Experiment):
    def set_gain_reward(self, probes, excluded_reference, beta):
        if len(probes) != len(TRAIN): raise ValueError('One independent gain probe per TRAIN level is required.')
        self.a.beta = beta
        self.gain_by_domain = {id(self.train[level]):(None if beta == 0 else
            SameDomainGain(probe, excluded_reference, beta)) for level, probe in zip(TRAIN, probes)}

    def reward_for_domain(self, domain):
        if id(domain) not in self.gain_by_domain:
            raise ValueError('Query gain must use the independent probe from the same TRAIN domain.')
        return self.gain_by_domain[id(domain)]


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
            reward_components = dict(paper_reward_component=reward, probe_gain=None,
                weighted_probe_gain=0. if training or getattr(a, 'beta', 0.) == 0 else None)
            label = torch.tensor([truth], device=x.device)
            paid, replay = memory.draw(ids[position], inputs, reference_keys=source_keys,
                current_key=feature_keys[position])
            if utility is not None:
                utility.assert_disjoint_update((inputs, label), paid, replay)
                probe_before = utility.measure(cnn, context)
            pilot.context_min_step(context, optimizer, cnn, (inputs, label), paid, replay, a.replay_weight)
            if utility is not None:
                reward, reward_components = utility.finish(cnn, context, probe_before, reward)
            memory.add(ids[position], inputs, truth)
            paid_embeddings.append(embeddings[position:position+1].detach().clone())
            paid_labels.append(truth); used += 1; last_update = position+1
        predictions.append(prediction)
        traces.append(dict(step=position+1, source_row=int(ids[position]), truth=truth,
            prediction=prediction, frozen_prediction=frozen_prediction, context_prediction=prediction,
            action=action, route=1, queried=bool(action), labels_used=used, cap=cap,
            reward=reward, query_legal_before=can_query, context_entropy=context_entropy,
            reward_beta=getattr(a, 'beta', 0.), reward_role='TRAIN target' if training else 'observer only; query policy frozen',
            probe_gain=reward_components.get('probe_gain'),
            paper_reward_component=reward_components.get('paper_reward_component'),
            weighted_probe_gain=reward_components.get('weighted_probe_gain'),
            generalization_loss_before=reward_components.get('generalization_loss_before'),
            generalization_loss_after=reward_components.get('generalization_loss_after'),
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
        result['reward_diagnostics'] = dict(mode='paper reward + beta * independent same-domain TRAIN gain' if utility is not None else 'original paper reward',
            positive_fraction=sum(r['reward'] > 0 for r in paid_rewards)/max(len(paid_rewards), 1),
            mean_reward=float(np.mean([r['reward'] for r in paid_rewards])) if paid_rewards else None,
            components={key:float(np.mean([r[key] for r in paid_rewards])) if paid_rewards else None
                for key in ('paper_reward_component', 'probe_gain', 'weighted_probe_gain')} if utility is not None else {})
        print(f'DATASET={getattr(a, "dataset", "fixture")} BETA={getattr(a, "beta", 0)} TRAIN reward: '+json.dumps(result['reward_diagnostics'], sort_keys=True), flush=True)
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
            print(f'DATASET={experiment.a.dataset} BETA={experiment.a.beta} SPLIT={split} {mode} level {level}: AC {row["AC"]:.2f}% | F1 {row["F1"]:.2f}% | '
                f'labels {row["labels_percent"]:.2f}% | forgetting {row["forgetting"]:+.2f} pp', flush=True)
            root = directory/mode/f'level_{level}'; root.mkdir(parents=True, exist_ok=True)
            pilot.write_json(root/'metrics.json', result); pilot.write_csv(root/'trace.csv', trace)
    if start_weights != weight_hash(learner.policy.state_dict()) or start_updates != learner.updates:
        raise RuntimeError('Validation/test changed the DDQN weights.')
    pilot.write_csv(directory/'summary.csv', rows)
    print(f'SUMMARY DATASET={experiment.a.dataset} BETA={experiment.a.beta} SPLIT={split}', flush=True)
    print_table(rows)
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
            print(f'DATASET={getattr(a, "dataset", "fixture")} BETA={getattr(a, "beta", 0)} TRAIN epoch {epoch}/{a.epochs} level {level}: AC {result["online"]["AC"]:.2f}% | '
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


def aggregate_validation(rows, beta):
    return dict(beta=float(beta), AC=float(np.mean([r['AC'] for r in rows])),
        F1=float(np.mean([r['F1'] for r in rows])), labels_percent=float(np.mean([r['labels_percent'] for r in rows])),
        positive_forgetting=float(np.mean([max(0., r['forgetting']) for r in rows])),
        forgetting=float(np.mean([r['forgetting'] for r in rows])))


def select_beta(reports):
    baseline = next((r for r in reports if r['beta'] == 0.), None)
    if baseline is None: raise ValueError('A matched beta=0 paper-reward control is required.')
    def rank(row): return row['AC']-row['positive_forgetting'], -row['labels_percent'], -row['beta']
    eligible = [r for r in reports if r['labels_percent'] > 0]
    selected = max(eligible, key=rank) if eligible else baseline
    improvement = selected['AC']-selected['positive_forgetting']-(baseline['AC']-baseline['positive_forgetting'])
    return dict(selected_beta=selected['beta'], baseline=baseline, selected=selected,
        validation_score_improved=selected['beta'] > 0 and improvement > 1e-8,
        score_improvement=improvement, labels_saved_pp=baseline['labels_percent']-selected['labels_percent'],
        excluded_zero_request_betas=[r['beta'] for r in reports if r['labels_percent'] == 0],
        criterion='max VAL mean online AC minus positive original-domain forgetting; fewer labels and smaller beta break ties; '
            'zero-request policies excluded; retain beta=0 if it wins or all candidates make no requests')


def beta_name(beta): return 'beta_'+format(beta, '.8g').replace('.', 'p').replace('-', 'm')


def load_selected(experiment, protocol):
    model = experiment.a.output/'Models'/'two_action_ddqn.pth'
    complete = json.loads((experiment.a.output/'training_complete.json').read_text(encoding='utf-8'))
    saved = torch.load(model, map_location='cpu', weights_only=False)
    if (complete['protocol'] != protocol or saved['protocol'] != protocol or
            complete['checkpoint_sha256'] != pilot.digest(model)):
        raise ValueError('Candidate checkpoint/completion provenance mismatch.')
    pilot.base.legacy.core.set_seed(experiment.a.seed)
    learner = QueryDDQN(experiment.spec, experiment.actor_saved['state'], pilot.base.legacy.core.device, experiment.a.seed)
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
    parser.add_argument('--betas', type=float, nargs='+', default=[0., 10., 100.])
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 32 or a.epochs < 1: parser.error('Use at least 32 pilot rows and positive epochs.')
    if not all(np.isfinite(beta) and beta >= 0 for beta in a.betas):
        parser.error('Beta values must be finite and nonnegative.')
    a.betas = sorted(set([0., *a.betas]))
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = a.output or root/'protocol_runs'/f'two_action_paper_gain_seed{a.seed}_p{a.pilot_rows}_e{a.epochs}_v1'
    output = a.output; a.budget_fraction = .2
    experiment = PaperGainExperiment(a)
    probes, excluded_reference, anchors, probe_manifest = prepare_train_probes(experiment, pilot, a.probe_rows, 32)
    experiment.train_anchors = anchors
    experiment.reward_probe_keys = set().union(*(set(pilot.feature_keys(d[0])) for d in probes))
    probe_manifest.update(generalization_sha256=[domain_hash(d) for d in probes],
        TRAIN_anchors_sha256=domain_hash(anchors),
        excluded_reference_use='reserved/excluded from adaptive updates; no reference loss is measured for query reward',
        loss='ordinary mean cross entropy on the single matching TRAIN level, not a mixed-domain or macro-class loss')
    files = ['amcal_two_action_paper_gain.py', 'enhancements/paper_gain_reward.py',
        'enhancements/query_utility_reward.py', 'enhancements/two_action_ddqn.py',
        'enhancements/competitive_selector.py', 'amcal_competitive_selector.py',
        'amcal_budget_actor.py', 'amcal_minimax_paironly.py', 'enhancements/minimax_query.py']
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, pilot_rows=a.pilot_rows, epochs=a.epochs,
        betas=a.betas, actor_checkpoint_sha256=experiment.actor_sha, healthy_teacher_sha256=experiment.teacher_hash,
        source_models=experiment.healthy['model_hashes'], train_files=experiment.healthy['training_domain_sha256'],
        implementation_sha256={name:pilot.digest(Path(__file__).parent/name) for name in files},
        train_data=experiment.data_manifest('train_ids', TRAIN), validation_data=experiment.data_manifest('val_ids', TRAIN),
        probes=probe_manifest, budget_fraction=.2, context_lr=a.context_lr, replay_weight=a.replay_weight,
        architecture='unchanged strong original IAT CNN / Light PS CNN / initial Context; fixed Context -> Frozen',
        reward='query: original paper reward + beta * (same-domain independent TRAIN CE before minus after one actual update); skip: 0',
        paper_reward={'both_wrong':3., 'only_Frozen_wrong':2., 'only_Context_wrong':-2., 'both_correct':0.},
        measurement='TRAIN only; signed unscaled mean CE on a fixed same-level probe; no cost, new forgetting penalty, '
            'sign target, clipping, synthetic drift, future rollout or copied update branch',
        DDQN=dict(gamma=.95, lr=.0001, batch_size=32, learn_every=4, target_every=15, capacity=4096,
            epsilon='0.30 to 0.05; same parent hidden encoder/normalizer; new return layer starts at zero'),
        matched_control='beta=0 retrained on the same streams, initial Q and probe-excluded TRAIN anchors',
        evaluation='VAL/TEST query policies frozen; only Context adapts; reset Context/Adam/memory/cap per intensity; '
            'gain probes are not accessed at evaluation; TEST does not select models',
        scope='single-seed small-data reward ablation, not full benchmark or independently retrained no-RL ablation',
        theory='closed-loop DDQN; Context competitive-gap descent preserved; no formal shared min-max equivalence asserted')
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Paper-plus-gain protocol differs; preserve outputs and use a new --output.')
    elif any(output.iterdir()): raise ValueError('Unmatched partial output preserved; use a fresh output.')
    else:
        if a.action == 'Evaluate': raise ValueError('Complete TRAIN/VAL selection first.')
        pilot.write_json(protocol_file, protocol); pilot.write_json(output/'probe_manifest.json', probe_manifest)
    base_args = deepcopy(a)
    if len({beta_name(beta) for beta in a.betas}) != len(a.betas):
        raise ValueError('Beta folder names collide; use more distinct beta values.')

    def configure(beta):
        experiment.a = deepcopy(base_args)
        experiment.a.output = output/'Candidates'/beta_name(beta)
        experiment.set_gain_reward(probes, excluded_reference, beta)
        return dict(protocol=protocol, beta=beta)

    selection_file = output/'selection.json'
    if a.action in ('Train', 'All'):
        reports, no_query = [], None
        for index, beta in enumerate(a.betas, 1):
            print(f'\nCANDIDATE {index}/{len(a.betas)} | DATASET={a.dataset} | BETA={beta:g} | '
                'REWARD=paper + beta*same-domain gain | TRAIN/VAL ONLY', flush=True)
            candidate_protocol = configure(beta)
            pilot.base.legacy.core.set_seed(a.seed)
            learner = QueryDDQN(experiment.spec, experiment.actor_saved['state'], pilot.base.legacy.core.device, a.seed)
            train(experiment, learner, candidate_protocol)
            learner = load_selected(experiment, candidate_protocol)
            rows = evaluate(experiment, learner, 'val_ids', TRAIN,
                experiment.a.output/'VAL_selected', controls=beta == 0)
            reports.append(aggregate_validation([r for r in rows if r['method'] == 'DDQN'], beta))
            if beta == 0: no_query = aggregate_validation([r for r in rows if r['method'] == 'NoQuery'], 0.)
        selection = select_beta(reports)
        selection.update(protocol=protocol, candidates=reports, no_query_reference=no_query,
            model_sha256={beta_name(beta):pilot.digest(output/'Candidates'/beta_name(beta)/'Models'/'two_action_ddqn.pth')
                for beta in a.betas})
        pilot.write_csv(output/'validation_candidates.csv', reports)
        pilot.write_json(selection_file, selection)
        print(f'DATASET={a.dataset} VAL SELECTION: BETA={selection["selected_beta"]:g} | '
            f'SCORE_IMPROVED={selection["validation_score_improved"]} | '
            f'LABEL_SAVING={selection["labels_saved_pp"]:+.2f} pp | '
            f'EXCLUDED_ZERO_REQUEST_BETAS={selection["excluded_zero_request_betas"]}', flush=True)
    if a.action in ('Evaluate', 'All'):
        selection = json.loads(selection_file.read_text(encoding='utf-8'))
        if selection['protocol'] != protocol: raise ValueError('VAL selection protocol changed.')
        selected_beta = selection['selected_beta']
        combined = []; selected_rows = None
        for beta, controls in [(selected_beta, True), (0., False)]:
            candidate_protocol = configure(beta)
            model_file = experiment.a.output/'Models'/'two_action_ddqn.pth'
            if pilot.digest(model_file) != selection['model_sha256'][beta_name(beta)]:
                raise ValueError('Selected model changed after VAL selection.')
            if not controls and selected_beta == 0:
                rows = [dict(row) for row in selected_rows if row['method'] == 'DDQN']
                print('BETA=0 was selected; matched paper TEST rows reused without rerunning inference.', flush=True)
            else:
                learner = load_selected(experiment, candidate_protocol)
                print(f'\nTEST | DATASET={a.dataset} | BETA={beta:g} | QUERY POLICY FROZEN', flush=True)
                rows = evaluate(experiment, learner, 'test_ids', TEST,
                    output/'Results'/('Selected' if controls else 'PaperControl'), controls=controls)
                if controls: selected_rows = rows
            for row in rows:
                row = dict(row)
                if row['method'] == 'DDQN':
                    row['method'] = f'SelectedDDQN_beta{selected_beta:g}' if controls else 'PaperDDQN_beta0'
                combined.append(row)
        pilot.write_csv(output/'Results'/'comparison_summary.csv', combined)
        pilot.write_json(output/'Results'/'execution.json', dict(protocol=protocol, selection=selection,
            test_data=experiment.data_manifest('test_ids', TEST)))
        print_table(combined)
    if (pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha or
            pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash):
        raise RuntimeError('A source checkpoint file was changed.')


if __name__ == '__main__': main()
