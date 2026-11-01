"""Clean-only learned-beta DDQN on the four supplied-attack datasets.

Reuse verified clean CNN/Context checkpoints. Train a dataset-local two-action
DDQN and beta using clean TRAIN only; choose its epoch using clean VAL only.
No attack, synthetic drift, cross-dataset weights or TEST row cap is used.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import torch

import amcal_two_action_paper_gain as base
from amcal_clean_supplied import DATASETS, ATTACKS
from amcal_external_data_audit import excluded
from enhancements.config import EnhancementConfig
from enhancements.meta_beta_ddqn import MetaBetaDDQN, LearnedGainReward, ComponentReward
from enhancements.independent_target_holdout import split_targets, purge_replay, run_holdout_case

api = base.pilot
ROOT = Path(__file__).resolve().parent
METRICS = ('AC', 'PR', 'RC', 'F1')
DEPENDENCIES = (
    'amcal_clean_meta_beta.py', 'amcal_clean_supplied.py',
    'amcal_two_action_paper_gain.py', 'amcal_competitive_selector.py',
    'amcal_minimax_paironly.py', 'amcal_batch5_pipeline.py',
    'amcal_final_enhancement.py', 'amcal_external_data_audit.py', 'enhancements/config.py',
    'enhancements/meta_beta_ddqn.py', 'enhancements/two_action_ddqn.py',
    'enhancements/paper_gain_reward.py', 'enhancements/query_utility_reward.py',
    'enhancements/competitive_selector.py', 'enhancements/minimax_query.py',
    'enhancements/independent_target_holdout.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    api.write_json(path, value)


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    api.write_csv(path, rows)


def check_classes(data, classes, name):
    if not len(data[1]) or set(map(int, np.unique(data[1]))) != set(range(classes)):
        raise ValueError(name + ': missing classes; no sampling/split fallback.')


def without(domain, held):
    ids = {int(i) for d in held for i in d[2]}
    keys = set().union(*(set(api.feature_keys(d[0])) for d in held)) if held else set()
    keep = np.asarray([int(i) not in ids and k not in keys
        for i, k in zip(domain[2], api.feature_keys(domain[0]))])
    return api.subset(domain, keep)


def reserve_clean(domain, seed, classes, meta_rows=300):
    """Reserve disjoint clean TRAIN roles, then keep ALL remaining inner rows."""
    check_classes(domain, classes, 'clean TRAIN')
    pool = domain
    parts = {}
    requests = [('reference', max(32, 3*classes)), ('gain', max(24, 3*classes)),
                ('meta', meta_rows), ('meta_gain', max(24, 3*classes))]
    for offset, (name, size) in enumerate(requests):
        # Never split an exact feature group between supervision roles.
        seen = set(); unique = np.zeros(len(pool[1]), dtype=bool)
        for index, key in enumerate(api.feature_keys(pool[0])):
            if key not in seen:
                unique[index] = True; seen.add(key)
        candidates = api.subset(pool, unique)
        check_classes(candidates, classes, name + ' candidates')
        if len(candidates[1]) < size or size < 3*classes:
            raise ValueError(name + ': insufficient unique clean TRAIN data.')
        chosen = api.pilot_indices(candidates[1], size, seed+7901+offset)
        mask = np.zeros(len(candidates[1]), dtype=bool); mask[chosen] = True
        parts[name] = api.subset(candidates, mask)
        check_classes(parts[name], classes, name)
        pool = without(pool, [parts[name]])
    check_classes(pool, classes, 'remaining full inner TRAIN')
    parts['inner'] = pool
    return parts


def part_manifest(parts):
    return {name: dict(rows=len(d[1]), sha256=base.domain_hash(d),
        ids=list(map(int, d[2]))) for name, d in parts.items()}


class UniformCollector:
    """Causal, exactly uniform cap collection; no Q or beta optimization."""
    def __init__(self, rows, fraction=.2, components=False):
        self.total = rows; self.cap = int(fraction*rows); self.index = 0
        self.steps = self.updates = 0
        self.states = []; self.rows = []; self.components = components

    def choose(self, state, can_query, epsilon, rng):
        self.states.append(torch.as_tensor(state).detach().cpu().clone())
        due = ((self.index+1)*self.cap//self.total > self.index*self.cap//self.total)
        self.index += 1
        return int(can_query and due), None

    def remember(self, state, action, reward, successor, terminal, next_can_query):
        if self.components and action:
            if not isinstance(reward, ComponentReward):
                raise ValueError('Independent TRAIN collection lacks raw reward components.')
            self.rows.append((torch.as_tensor(state).detach().cpu().clone(), reward.paper, reward.gain))


def normalization(states):
    matrix = torch.stack(states).double()
    if not torch.isfinite(matrix).all():
        raise ValueError('Nonfinite clean TRAIN features.')
    scale = matrix.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return dict(mean=matrix.mean(0).float().tolist(), scale=scale.float().tolist(),
        fitting_scope='all remaining clean TRAIN states; uniform temporary adaptation; no VAL/TEST')


def validate_recipe(recipe, dataset, seed):
    if (recipe.get('dataset') != dataset or recipe.get('seed') != seed
            or recipe.get('training_levels') != [0] or recipe.get('validation_levels') != [0]
            or recipe.get('architecture') != 'original CNN'):
        raise ValueError('Expected the matching original-CNN clean-only source recipe.')


class CleanExperiment:
    def __init__(self, a):
        self.a = a
        self.source_root = a.source_root
        self.manifest = read(a.source_root/'clean_supplied_protocol.json')
        recipe = self.manifest['recipe']
        validate_recipe(recipe, a.dataset, a.seed)
        self.training_dir = a.source_root/'TrainingDomains'
        models = a.source_root/'source'/'Models'
        marker = read(a.source_root/'source'/'clean_source_complete.json')
        if marker['recipe'] != recipe:
            raise ValueError('Clean source completion recipe mismatch.')
        for name, digest in marker['outputs'].items():
            if api.digest(a.source_root/'source'/name) != digest:
                raise ValueError('Clean source model changed: '+name)
        provenance = read(a.source_root/'source'/'source_training.json')
        if provenance.get('training_levels') != [0] or provenance.get('architecture') != 'legacy':
            raise ValueError('Source model was not trained solely on clean data.')
        # Validate prepared clean files without opening any target attack CSV.
        clean_hashes = {k: v for k, v in self.manifest['prepared_hashes'].items()
                       if k.startswith('TrainingDomains/')}
        if not clean_hashes:
            raise ValueError('Missing prepared clean TRAIN provenance.')
        for name, digest in clean_hashes.items():
            if api.digest(a.source_root/name) != digest:
                raise ValueError('Prepared clean data changed: '+name)
        self.artifact = api.base.load_artifact(models, self.training_dir)
        for key, name in [('train_ids','train'), ('val_ids','validation'), ('test_ids','test')]:
            if not np.array_equal(self.artifact[key], recipe['partitions'][name]):
                raise ValueError('Source partition mismatch: '+key)
        if any(excluded(name) for name in self.artifact['feature_names']):
            raise ValueError('Excluded identifiers found in source features.')
        a.classes = list(self.artifact['label_encoder'].classes_)
        a.config = EnhancementConfig.preset('accuracy').validated()
        preprocessing_hash = api.digest(models/'protocol_preprocessing.joblib')
        classifier = torch.load(models/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
        if classifier.get('architecture') != 'legacy':
            raise ValueError('These datasets must retain their original CNN architecture.')
        saved = torch.load(models/'adversarial_best_context.pth', map_location='cpu', weights_only=False)
        if saved.get('preprocessing_sha256') != preprocessing_hash:
            raise ValueError('Context preprocessing mismatch.')
        self.cnn = api.base.load_cnn(models, self.artifact)
        self.template = api.base.build_bank(models, self.artifact, a.config,
            saved['context_model_state_dict'], slots=1)[0].eval()
        self.original_weights = (base.weight_hash(self.cnn.state_dict()),
                                 base.weight_hash(self.template.state_dict()))
        self.model_files = {name: api.digest(models/name) for name in
            ('protocol_preprocessing.joblib','initial_cnn_model.pth','adversarial_best_context.pth')}
        self.full_train = self.domain(self.training_dir, 0, 'train_ids')
        self.validation = self.domain(self.training_dir, 0, 'val_ids')
        check_classes(self.full_train, len(a.classes), 'source TRAIN')
        check_classes(self.validation, len(a.classes), 'source VAL')
        self.parts = reserve_clean(self.full_train, a.seed, len(a.classes), a.meta_rows)
        self.train = self.parts['inner']; self.reference = self.parts['reference']
        self.anchors = self.full_train
        self.train_anchors = self.train
        self.reward_probe_keys = set().union(*(set(api.feature_keys(self.parts[n][0]))
            for n in ('gain','meta_gain')))
        self.gain_by_domain = {}
        self.val_stream, self.val_reference, _ = api.reference_partition(self.validation, a.seed, .2)
        self.val_stream = without(self.val_stream, [self.val_reference])
        check_classes(self.val_stream, len(a.classes), 'clean VAL stream')
        check_classes(self.val_reference, len(a.classes), 'clean VAL reference')

    def domain(self, directory, level, split):
        return api.base.legacy.read_domain(directory, self.artifact, level, split)

    def context(self):
        return deepcopy(self.template).eval()

    def memory(self, seed):
        return api.Replay(self.anchors, seed+61, self.a.config, len(self.a.classes))

    def reward_for_domain(self, domain):
        if id(domain) not in self.gain_by_domain:
            raise ValueError('Unknown TRAIN gain domain; target data cannot define training rewards.')
        return self.gain_by_domain[id(domain)]

    def bind(self, learner=None):
        self.gain_by_domain = {} if learner is None else {
            id(self.train): LearnedGainReward(self.parts['gain'], self.reference, learner, self.a),
            id(self.parts['meta']): LearnedGainReward(self.parts['meta_gain'], self.reference, learner, self.a)}

    def assert_source_unchanged(self):
        hashes = (base.weight_hash(self.cnn.state_dict()), base.weight_hash(self.template.state_dict()))
        if hashes != self.original_weights:
            raise RuntimeError('Source CNN or initial Context was modified.')
        for name, digest in self.model_files.items():
            if api.digest(self.source_root/'source'/'Models'/name) != digest:
                raise RuntimeError('Source file was modified: '+name)


def protocol_for(exp):
    a = exp.a
    return dict(version=1, dataset=a.dataset, seed=a.seed, training_levels=[0],
        validation_levels=[0], source_recipe=exp.manifest['recipe'],
        source_models_sha256=exp.model_files, train_partitions=part_manifest(exp.parts),
        validation_sha256=base.domain_hash(exp.validation), warmup_epochs=a.warmup_epochs,
        meta_epochs=a.epochs, budget_fraction=.2, context_lr=a.context_lr, replay_weight=.1,
        Q='fresh dataset-local 2K+18 -> 16/GELU -> 2; original-paper DDQN warmup; gamma .95',
        beta='TRAIN-only RMS calibration and unchanged MetaBetaDDQN virtual-step meta objective',
        meta_collection='independent clean TRAIN uniform paid trajectories; minimum 32 observed meta targets',
        selection='clean VAL online AC minus positive source forgetting; fewer labels break ties',
        initialization='verified original clean CNN/Context reused; no Burst weights or beta transferred',
        training_rows='all inner TRAIN after disjoint supervision reserves; no implicit cap',
        labeling='TRAIN probes are supervised; 20 percent is a trajectory/test cap, not total offline supervision',
        source_reference='TRAIN reserve during training; independent clean VAL reserve for selection; full clean VAL in TEST',
        theory='competitive Context update and closed-loop DDQN; no exact common saddle-point equivalence claimed',
        implementation_sha256={name: api.digest(ROOT/name) for name in DEPENDENCIES})


def ensure_protocol(path, protocol, training):
    if path.is_file():
        if read(path) != protocol:
            raise ValueError('Run recipe changed; use a fresh --output. Existing output is preserved.')
    elif path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError('Unmatched existing output preserved; use a fresh --output.')
    elif not training:
        raise ValueError('Train the dataset-local learned-beta checkpoint first.')
    else:
        write(path, protocol)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.parent.save_model(path, value)


def train(exp, protocol):
    a = exp.a; root = a.output; pf = root/'Models'/'learned_beta.pth'
    done = root/'training_complete.json'; state_file = root/'Models'/'trainer.pth'
    if done.is_file():
        load_selected(exp, protocol)
        print('Skipping verified completed clean-only DDQN/beta training.', flush=True)
        return
    initial = root/'Models'/'initialization.pth'
    if initial.exists():
        saved = torch.load(initial, map_location='cpu', weights_only=False)
        if saved['protocol'] != protocol:
            raise ValueError('Initialization protocol changed.')
        spec = saved['spec']
    else:
        print(f'NORMALIZATION DATASET={a.dataset} CLEAN_LEVEL=0 ROWS={len(exp.train[1])}', flush=True)
        # No reward oracle is required for normalization or original-paper warmup.
        exp.bind()
        collector = UniformCollector(len(exp.train[1]))
        # run_episode supports reward-free training through utility_reward=None.
        observer = PaperOnlyExperiment(exp)
        base.run_episode(observer, collector, exp.train, None, a.seed+601, training=True)
        spec = normalization(collector.states)
        save(initial, dict(protocol=protocol, spec=spec))
    learner = MetaBetaDDQN(spec, None, api.base.legacy.core.device, a.seed)
    start = 1; history = []; best = (-float('inf'), -float('inf'))
    if state_file.exists():
        saved = torch.load(state_file, map_location='cpu', weights_only=False)
        if saved['protocol'] != protocol or saved['spec'] != spec:
            raise ValueError('Resume checkpoint recipe changed.')
        learner.restore(saved['learner']); start = saved['epoch']+1
        history = saved['history']; best = tuple(saved['best'])
    else:
        warm_file = root/'Models'/'warmup.pth'
        api.base.legacy.core.set_seed(a.seed+4999)
        warm = base.QueryDDQN(spec, None, api.base.legacy.core.device, a.seed)
        first = 1
        if warm_file.exists():
            ws = torch.load(warm_file, map_location='cpu', weights_only=False)
            if ws['protocol'] != protocol:
                raise ValueError('Warmup recipe changed.')
            warm.restore(ws['learner']); first = ws['epoch']+1
        for epoch in range(first, a.warmup_epochs+1):
            api.base.legacy.core.set_seed(a.seed+5000+epoch)
            result, trace = base.run_episode(PaperOnlyExperiment(exp), warm, exp.train, None,
                a.seed+5000+epoch, training=True, epsilon=.3)
            csv(root/'Warmup'/f'epoch_{epoch}_trace.csv', trace)
            save(warm_file, dict(protocol=protocol, epoch=epoch, learner=warm.snapshot()))
            report(a, 'WARMUP', epoch, result, beta=0.)
        learner.policy.load_state_dict(warm.policy.state_dict())
        learner.target.load_state_dict(learner.policy.state_dict())
    exp.bind(learner)
    for epoch in range(start, a.epochs+1):
        api.base.legacy.core.set_seed(a.seed+epoch)
        collector = UniformCollector(len(exp.parts['meta'][1]), components=True)
        result, trace = base.run_episode(exp, collector, exp.parts['meta'], None,
            a.seed+epoch*1009+70000, training=True)
        learner.set_meta_rows(collector.rows)
        csv(root/'META'/f'epoch_{epoch}_trace.csv', trace)
        print(f'META DATASET={a.dataset} EPOCH={epoch} CLEAN_LEVEL=0 '
            f'PAID_TARGETS={len(collector.rows)} BETA={learner.beta_value():.6f} '
            f'GAIN_RMS={learner.gain_scale:.6g}', flush=True)
        result, trace = base.run_episode(exp, learner, exp.train, None,
            a.seed+epoch*1009, training=True,
            epsilon=.3-.25*(epoch-1)/max(a.epochs-1, 1))
        csv(root/'TRAIN'/f'epoch_{epoch}_trace.csv', trace)
        report(a, 'TRAIN', epoch, result, learner.beta_value())
        val, val_trace = base.run_episode(exp, learner, exp.val_stream, exp.val_reference,
            a.seed, training=False)
        csv(root/'VAL'/f'epoch_{epoch}_trace.csv', val_trace)
        score = val['online']['AC']-max(0., val['original_domain_forgetting']['AC'])
        rank = (score, -val['labels_percent'])
        history.append(dict(epoch=epoch, training=result, validation=val, score=score,
                            beta=learner.beta_value(), meta_updates=learner.meta_updates))
        if rank > best:
            best = rank
            save(pf, dict(protocol=protocol, spec=spec, epoch=epoch,
                policy_state=deepcopy(learner.policy.state_dict()), beta=learner.beta_value(),
                gain_scale=learner.gain_scale))
        save(state_file, dict(protocol=protocol, spec=spec, epoch=epoch,
            learner=learner.snapshot(), history=history, best=list(best)))
        write(root/'training.json', dict(protocol=protocol, epochs=history))
        csv(root/'beta_learning.csv', learner.meta_history)
        report(a, 'CLEAN_VAL', epoch, val, learner.beta_value())
    selected = torch.load(pf, map_location='cpu', weights_only=False)
    write(done, dict(protocol=protocol, selected_epoch=selected['epoch'],
        selected_beta=selected['beta'], sha256=api.digest(pf)))


class PaperOnlyExperiment:
    """Delegate all mechanics, with no independent-gain reward for warmup."""
    def __init__(self, exp):
        self.experiment = exp

    def __getattr__(self, name):
        return getattr(self.experiment, name)

    def reward_for_domain(self, domain):
        return None


def report(a, stage, epoch, result, beta):
    values = ' '.join(f'{k}={result["online"][k]:.2f}%' for k in METRICS)
    forget = result['original_domain_forgetting'].get('AC')
    suffix = '' if forget is None else f' ORIGINAL_FORGETTING={forget:+.2f}'
    print(f'{stage} DATASET={a.dataset} EPOCH={epoch} {values} '
        f'LABELS={result["labels"]}/{result["cap"]} ({result["labels_percent"]:.2f}%) '
        f'BETA={beta:.6f} TD_UPDATES={result["td_updates"]}{suffix}', flush=True)


def load_selected(exp, protocol):
    path = exp.a.output/'Models'/'learned_beta.pth'
    done = read(exp.a.output/'training_complete.json')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (done['protocol'] != protocol or saved['protocol'] != protocol
            or done['sha256'] != api.digest(path)):
        raise ValueError('Selected clean-only checkpoint verification failed.')
    learner = MetaBetaDDQN(saved['spec'], None, api.base.legacy.core.device, exp.a.seed)
    learner.policy.load_state_dict(saved['policy_state'])
    with torch.no_grad():
        learner.log_beta.copy_(learner.log_beta.new_tensor(np.log(saved['beta'])))
    learner.policy.eval().requires_grad_(False)
    learner.log_beta.requires_grad_(False)
    exp.a.beta = learner.beta_value()
    return learner, done


def attack_cases(exp):
    """Target files are opened only after a completed clean-only checkpoint."""
    domains = {}; descriptions = {}
    hashes = exp.manifest['prepared_hashes']
    for attack, strengths in ATTACKS.items():
        prefix = 'EvaluationDomains/'+attack+'/'
        selected = {k: v for k, v in hashes.items() if k.startswith(prefix)}
        if not selected:
            raise ValueError('Missing prepared supplied-attack provenance: '+attack)
        for name, digest in selected.items():
            if api.digest(exp.source_root/name) != digest:
                raise ValueError('Prepared target changed: '+name)
        for level, strength in enumerate(strengths, 1):
            case = len(domains)+1
            domains[case] = exp.domain(exp.source_root/'EvaluationDomains'/attack, level, 'test_ids')
            if (len(domains[case][1]) != len(exp.artifact['test_ids'])
                    or set(map(int, domains[case][2])) != set(map(int, exp.artifact['test_ids']))):
                raise ValueError('Target coverage differs from existing full TEST: '+attack)
            descriptions[case] = dict(attack=attack, strength=strength, level=level)
    return domains, descriptions


def summary_row(dataset, scope, description, result, beta):
    row = dict(dataset=dataset, protocol=scope, method='AMCAL-LearnedBeta-DDQN',
        **description, beta=beta, **result['online'], samples=result['samples'],
        labels=result['labels'], cap=result['cap'], labels_percent=result['labels_percent'],
        budget_used_percent=100*result['labels']/max(1, result['cap']),
        milliseconds_per_row=result['milliseconds_per_row'])
    for metric in METRICS:
        row['original_forgetting_'+metric] = result['original_domain_forgetting'][metric]
        row['source_final_'+metric] = result['source_after'][metric]
        if scope == 'Independent':
            row['target_final_'+metric] = result['target_after'][metric]
            row['target_gain_'+metric] = result['target_holdout_gain'][metric]
    if scope == 'Independent':
        row.update(target_holdout_rows=result['target_holdout_rows'],
            target_total_rows=result['target_total_rows'],
            labels_percent_of_all_target_rows=result['labels_percent_of_all_target_rows'])
    return row


def evaluate(exp, protocol):
    a = exp.a; learner, done = load_selected(exp, protocol)
    domains, descriptions = attack_cases(exp)
    scopes = ['Online', 'Independent'] if a.protocol == 'Both' else [a.protocol]
    full_anchors = exp.anchors
    policy_before = base.weight_hash(learner.policy.state_dict())
    for scope in scopes:
        exp.anchors = full_anchors
        root = a.output/'Results'/scope
        if scope == 'Independent':
            streams, audits, split = split_targets(domains, a.seed)
            exp.anchors, purge = purge_replay(full_anchors, audits)
        else:
            streams, audits, split, purge = domains, {}, None, None
        execution = dict(protocol=protocol, checkpoint_sha256=done['sha256'],
            test_scope=scope, budget_fraction=.2, model_reset_per_intensity=True,
            source_reference=base.domain_hash(exp.validation),
            targets={str(k): base.domain_hash(d) for k, d in domains.items()},
            split=split, replay_purge=purge,
            metric_order=list(METRICS), beta=learner.beta_value())
        ef = root/'execution.json'
        if ef.exists() and read(ef) != execution:
            raise ValueError('Existing TEST recipe differs; results preserved.')
        write(ef, execution)
        rows = []
        for case, description in descriptions.items():
            location = root/description['attack']/('level_'+str(description['level']))
            mf, tf, cf = (location/name for name in ('metrics.json','trace.csv','completed.json'))
            if cf.is_file():
                record = read(cf)
                if record['execution_sha256'] != api.digest(ef) or any(
                        api.digest(p) != record['sha256'][p.name] for p in (mf, tf)):
                    raise ValueError('Completed target case changed.')
                result = read(mf)
            else:
                if scope == 'Independent':
                    result, trace = run_holdout_case(exp, learner, streams[case], audits[case],
                        exp.validation, a.seed, 'LearnedBetaDDQN', 1)
                else:
                    result, trace = base.run_episode(exp, learner, streams[case], exp.validation,
                        a.seed, training=False)
                    result['adaptation_stream_seen_before'] = result.pop('target_before')
                    result['adaptation_stream_seen_after'] = result.pop('target_after')
                    result['target_final_definition'] = 'not reported; Online uses all TEST rows for pre-feedback scoring'
                result['case'] = description
                write(mf, result); csv(tf, trace)
                write(cf, dict(execution_sha256=api.digest(ef),
                    sha256={p.name: api.digest(p) for p in (mf, tf)}))
            if (result['samples'] != len(streams[case][1]) or result['td_updates'] != 0
                    or result['cap'] != int(.2*len(streams[case][1]))
                    or not 0 <= result['labels'] <= result['cap']):
                raise RuntimeError('TEST sample, frozen-Q or budget contract violated.')
            row = summary_row(a.dataset, scope, description, result, learner.beta_value())
            rows.append(row)
            values = ' '.join(f'{k}={row[k]:.2f}%' for k in METRICS)
            final = f' FINAL_HOLDOUT_AC={row["target_final_AC"]:.2f}%' if scope == 'Independent' else ''
            print(f'TEST DATASET={a.dataset} SCOPE={scope} ATTACK={description["attack"]} '
                f'STRENGTH={description["strength"]} {values} N={row["samples"]} '
                f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
                f'CAP_USED={row["budget_used_percent"]:.2f}% '
                f'ORIGINAL_FORGETTING={row["original_forgetting_AC"]:+.2f}{final}', flush=True)
        csv(root/'summary_by_intensity.csv', rows)
        means = []
        for attack in [*ATTACKS, 'ALL']:
            selected = [r for r in rows if attack == 'ALL' or r['attack'] == attack]
            mean = dict(dataset=a.dataset, protocol=scope, attack=attack, intensities=len(selected))
            numeric = [k for k in rows[0] if k not in ('dataset','protocol','method','attack','strength','level')]
            mean.update({k: float(np.mean([r[k] for r in selected])) for k in numeric})
            means.append(mean)
            print('MEAN '+json.dumps(mean, sort_keys=True), flush=True)
        csv(root/'summary_means.csv', means)
        write(root/'completed.json', dict(execution_sha256=api.digest(ef),
            cases=len(rows), summary_sha256=api.digest(root/'summary_by_intensity.csv')))
    exp.anchors = full_anchors
    if policy_before != base.weight_hash(learner.policy.state_dict()) or learner.steps or learner.updates:
        raise RuntimeError('TEST updated the frozen DDQN.')
    if not math.isclose(learner.beta_value(), done['selected_beta'], rel_tol=1e-6):
        raise RuntimeError('TEST updated the frozen beta.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--action', choices=['All','Train','Evaluate'], default='All')
    p.add_argument('--protocol', choices=['Both','Online','Independent'], default='Both')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='cuda')
    p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--meta-rows', type=int, default=300,
                   help='Independent clean TRAIN meta stream, not a TEST/main-TRAIN cap.')
    a = p.parse_args()
    if min(a.warmup_epochs, a.epochs) < 1 or int(.2*a.meta_rows) < 32:
        p.error('Positive epochs and at least 32 paid clean meta rows are required.')
    folder = ROOT/DATASETS[a.dataset][0]/'protocol_runs'
    a.source_root = (a.source_root or folder/f'clean0_AMCAL_attacks_seed{a.seed}').resolve()
    a.output = (a.output or folder/f'clean0_meta_beta_ddqn_seed{a.seed}_v1').resolve()
    if a.source_root == a.output or a.source_root in a.output.parents or a.output in a.source_root.parents:
        p.error('Output must be separate from the existing source run.')
    a.budget_fraction = .2; a.context_lr = .00025; a.replay_weight = .1; a.beta = 0.
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; choose --device cpu.')
    api.base.legacy.core.device = torch.device(device)
    api.base.legacy.core.set_seed(a.seed)
    exp = CleanExperiment(a); protocol = protocol_for(exp)
    ensure_protocol(a.output/'protocol.json', protocol, a.action in ('Train','All'))
    print(f'CLEAN-ONLY DATASET={a.dataset} SOURCE_TRAIN={len(exp.full_train[1])} '
        f'INNER_TRAIN={len(exp.train[1])} VAL={len(exp.validation[1])} '
        f'FULL_TEST_PER_INTENSITY={len(exp.artifact["test_ids"])} '
        f'WARMUP_EPOCHS={a.warmup_epochs} META_EPOCHS={a.epochs} '
        'BUDGET=20% CONTEXT_LR=0.00025 ORIGINAL_CNN | no TEST cap', flush=True)
    if a.action in ('Train','All'):
        train(exp, protocol)
    if a.action in ('Evaluate','All'):
        evaluate(exp, protocol)
    exp.assert_source_unchanged()
    print('OUTPUT: '+str(a.output), flush=True)


if __name__ == '__main__':
    main()
