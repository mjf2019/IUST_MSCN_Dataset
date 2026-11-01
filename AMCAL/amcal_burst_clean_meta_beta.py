"""Fresh clean-only light CNN/Context/DDQN/beta; full Burst IAT or PS TEST."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import re

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import amcal_clean_meta_beta as shared
import amcal_batch5_pipeline as source_fit
from amcal_light_cnn import LightCNN1D, ARCHITECTURE
from enhancements.config import EnhancementConfig
from enhancements.independent_target_holdout import split_targets, purge_replay, run_holdout_case

api = shared.api
base = shared.base
ROOT = Path(__file__).resolve().parent
EXPECTED_LEVELS = [0, 1, 3, 5, 7, 10, 12, 15, 17, 20]
MODEL_FILES = ('protocol_preprocessing.joblib', 'initial_cnn_model.pth', 'adversarial_best_context.pth')


def levels_on_disk(directory):
    pattern = re.compile(r'^AdvBurstTrainedBurst_size_(\d+)_Step_2000\.csv$')
    levels = sorted({int(match.group(1)) for p in directory.iterdir()
        if p.is_file() and (match := pattern.fullmatch(p.name))})
    if not set(EXPECTED_LEVELS).issubset(levels):
        raise ValueError('Expected complete official Burst levels 0,1,3,5,7,10,12,15,17,20.')
    return levels


def fit_clean_context(a, context, cnn, train, validation):
    """Supervised initial Context fit on clean TRAIN; clean VAL early stopping."""
    x, y = train; vx, vy = validation
    count = torch.bincount(y, minlength=len(a.classes)).float()
    if (count == 0).any():
        raise ValueError('Clean TRAIN class missing.')
    weights = 1/count; weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(context.parameters(), lr=a.context_lr, weight_decay=.0001)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    best = float('inf'); state = None; stale = 0; logs = []
    for epoch in range(a.context_epochs):
        context.eval(); total = 0.; hits = 0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            output = cnn(context(bx)); loss = criterion(output, by)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite clean Context training loss.')
            loss.backward(); nn.utils.clip_grad_norm_(context.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(by)
            hits += int((output.argmax(1) == by).sum())
        loss_sum = 0.; correct = 0
        with torch.no_grad():
            for start in range(0, len(vy), 32):
                bx, by = vx[start:start+32], vy[start:start+32]
                output = cnn(context(bx)); loss_sum += float(criterion(output, by))*len(by)
                correct += int((output.argmax(1) == by).sum())
        value = loss_sum/len(vy)
        if not math.isfinite(value):
            raise ValueError('Nonfinite clean Context validation loss.')
        logs.append(dict(epoch=epoch+1, train_loss=total/len(y), train_accuracy=hits/len(y),
            validation_loss=value, validation_accuracy=correct/len(vy)))
        print(f'CLEAN_CONTEXT DATASET={a.dataset} EPOCH={epoch+1}/{a.context_epochs} '
            f'TRAIN_AC={100*hits/len(y):.2f}% VAL_AC={100*correct/len(vy):.2f}% '
            f'LR={a.context_lr}', flush=True)
        if value < best:
            best = value; state = deepcopy(context.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= a.patience:
                break
    if state is None:
        raise ValueError('No valid clean Context checkpoint.')
    context.load_state_dict(state); context.eval()
    return logs


def source_stage(a):
    """Import partitions only; refit scaling and ALL weights using level 0."""
    source = a.output/'source'; models = source/'Models'
    completion = source/'clean_source_complete.json'; recipe_file = source/'recipe.json'
    template_models = a.source_template/'Models'
    template = api.base.load_artifact(template_models, a.burst_dir)
    if any(shared.excluded(name) for name in template['feature_names']):
        raise ValueError('Excluded identifiers found in Burst source features.')
    recipe = dict(version=1, dataset=a.dataset, seed=a.seed, architecture=ARCHITECTURE,
        training_levels=[0], validation_levels=[0], context_lr=a.context_lr,
        cnn_epochs=a.cnn_epochs, context_epochs=a.context_epochs, patience=a.patience,
        input_preprocessing_sha256=api.digest(template_models/'protocol_preprocessing.joblib'),
        clean_file_sha256=api.digest(a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'),
        initialization='partition metadata only; new clean TRAIN scaler and fresh CNN/Context weights',
        implementation_sha256={name: api.digest(ROOT/name) for name in
            ('amcal_burst_clean_meta_beta.py','amcal_light_cnn.py','amcal_batch5_pipeline.py')})
    if recipe_file.exists():
        if shared.read(recipe_file) != recipe:
            raise ValueError('Clean source recipe changed; use a fresh --output.')
    elif source.exists() and any(source.iterdir()):
        raise ValueError('Unmatched source output preserved; use a fresh --output.')
    elif a.action == 'Evaluate':
        raise ValueError('Train the fresh clean-only source first.')
    else:
        shared.write(recipe_file, recipe)
    if completion.is_file():
        done = shared.read(completion)
        if done['recipe'] != recipe or any(api.digest(models/name) != digest
                for name, digest in done['sha256'].items()):
            raise ValueError('Completed clean source changed.')
        print('Skipping verified clean-only LIGHT CNN and Context.', flush=True)
        return recipe
    if a.action == 'Evaluate':
        raise ValueError('Clean-only source training is incomplete.')
    models.mkdir(parents=True, exist_ok=True)
    # mapped_domain reads only the clean level-0 CSV here, never an attack CSV.
    frame, ids, _ = api.base.legacy.base.official.mapped_domain(
        a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv', template, a.burst_dir, False)
    lookup = {int(value): index for index, value in enumerate(ids)}
    at = [lookup[int(value)] for value in template['train_ids']]
    artifact = deepcopy(template)
    artifact['scaler'] = StandardScaler().fit(frame.iloc[at][artifact['feature_names']])
    preprocessor = models/'protocol_preprocessing.joblib'
    if preprocessor.exists():
        old = joblib.load(preprocessor)
        if any(not np.array_equal(getattr(old['scaler'], k), getattr(artifact['scaler'], k))
                for k in ('mean_','scale_','var_')):
            raise ValueError('Partial clean scaler differs.')
    else:
        temporary = preprocessor.with_suffix('.tmp')
        joblib.dump(artifact, temporary); temporary.replace(preprocessor)
    artifact = api.base.load_artifact(models, a.burst_dir)
    a.classes = list(artifact['label_encoder'].classes_)
    domains = [api.base.legacy.read_domain(a.burst_dir, artifact, 0, split)
               for split in ('train_ids','val_ids')]
    for data in domains:
        shared.check_classes(data, len(a.classes), 'clean supervised source')
    tensors = [(d[0], torch.as_tensor(d[1], dtype=torch.long, device=d[0].device)) for d in domains]
    width = len(artifact['feature_names']); classes = len(a.classes)
    checkpoint = models/'initial_cnn_model.pth'; cnn_log = source/'cnn_training.json'
    if checkpoint.exists() and cnn_log.exists():
        cnn = api.base.load_cnn(models, artifact)
    else:
        api.base.legacy.core.set_seed(a.seed+101)
        cnn = LightCNN1D(width, classes).to(api.base.legacy.core.device)
        print(f'CLEAN_CNN DATASET={a.dataset} TRAIN_LEVELS=[0] ARCHITECTURE={ARCHITECTURE}', flush=True)
        logs = source_fit.fit(a, cnn, None, *tensors)
        shared.save(checkpoint, dict(architecture=ARCHITECTURE, model_state_dict=cnn.state_dict(),
            preprocessing_sha256=api.digest(preprocessor), input_length=width, num_classes=classes,
            training_levels=[0]))
        shared.write(cnn_log, logs)
    cnn.eval().requires_grad_(False)
    api.base.legacy.core.set_seed(a.seed+102)
    context = api.base.legacy.core.ContextAwareNetwork(width).to(api.base.legacy.core.device).eval()
    logs = fit_clean_context(a, context, cnn, *tensors)
    shared.save(models/'adversarial_best_context.pth', dict(context_model_state_dict=context.state_dict(),
        preprocessing_sha256=api.digest(preprocessor), training_levels=[0]))
    shared.write(source/'source_training.json', dict(recipe=recipe, training_levels=[0],
        cnn=shared.read(cnn_log), context=logs, selection='clean VAL only; no perturbed domain loaded'))
    shared.write(completion, dict(recipe=recipe,
        sha256={name: api.digest(models/name) for name in MODEL_FILES}))
    return recipe


class BurstCleanExperiment(shared.CleanExperiment):
    def __init__(self, a, recipe):
        self.a = a; self.source_root = a.output; self.training_dir = a.burst_dir
        self.manifest = dict(recipe=recipe)
        models = a.output/'source'/'Models'
        self.artifact = api.base.load_artifact(models, a.burst_dir)
        a.classes = list(self.artifact['label_encoder'].classes_)
        a.config = EnhancementConfig.preset('accuracy').validated()
        self.cnn = api.base.load_cnn(models, self.artifact)
        saved = torch.load(models/'adversarial_best_context.pth', map_location='cpu', weights_only=False)
        self.template = api.base.build_bank(models, self.artifact, a.config,
            saved['context_model_state_dict'], slots=1)[0].eval()
        self.original_weights = (base.weight_hash(self.cnn.state_dict()), base.weight_hash(self.template.state_dict()))
        self.model_files = {name: api.digest(models/name) for name in MODEL_FILES}
        self.full_train = self.domain(a.burst_dir, 0, 'train_ids')
        self.validation = self.domain(a.burst_dir, 0, 'val_ids')
        self.parts = shared.reserve_clean(self.full_train, a.seed, len(a.classes), a.meta_rows)
        self.train = self.parts['inner']; self.reference = self.parts['reference']
        self.anchors = self.full_train; self.train_anchors = self.train
        self.reward_probe_keys = set().union(*(set(api.feature_keys(self.parts[name][0]))
            for name in ('gain','meta_gain')))
        self.gain_by_domain = {}
        self.val_stream, self.val_reference, _ = api.reference_partition(self.validation, a.seed, .2)
        self.val_stream = shared.without(self.val_stream, [self.val_reference])
        for data in (self.val_stream, self.val_reference):
            shared.check_classes(data, len(a.classes), 'clean VAL')


def evaluate(exp, protocol):
    a = exp.a; learner, done = shared.load_selected(exp, protocol)
    levels = levels_on_disk(a.burst_dir)
    domains = {level: exp.domain(a.burst_dir, level, 'test_ids') for level in levels}
    original_ids = set(map(int, exp.artifact['test_ids']))
    for level, data in domains.items():
        if len(data[1]) != len(original_ids) or set(map(int, data[2])) != original_ids:
            raise ValueError(f'Full TEST coverage mismatch at level {level}.')
    frozen_policy = base.weight_hash(learner.policy.state_dict()); beta = learner.beta_value()
    anchors = exp.anchors
    scopes = ['Online','Independent'] if a.protocol == 'Both' else [a.protocol]
    for scope in scopes:
        exp.anchors = anchors; split = purge = None
        streams = domains; audits = {}
        if scope == 'Independent':
            streams, audits, split = split_targets(domains, a.seed)
            exp.anchors, purge = purge_replay(anchors, audits)
        root = a.output/'Results'/scope
        execution = dict(protocol=protocol, checkpoint_sha256=done['sha256'], scope=scope,
            budget_fraction=.2, reset_per_level=True, levels=levels, beta=beta,
            domains={str(k): base.domain_hash(d) for k,d in domains.items()},
            source_reference=base.domain_hash(exp.validation), split=split, replay_purge=purge)
        ef = root/'execution.json'
        if ef.exists() and shared.read(ef) != execution:
            raise ValueError('Existing TEST execution differs; results preserved.')
        shared.write(ef, execution)
        rows = []
        for level in levels:
            target = root/f'level_{level}'; mf = target/'metrics.json'
            tf = target/'trace.csv'; cf = target/'completed.json'
            if cf.exists():
                record = shared.read(cf)
                if record['execution_sha256'] != api.digest(ef) or any(
                        api.digest(path) != record['sha256'][path.name] for path in (mf,tf)):
                    raise ValueError('Completed TEST level changed.')
                result = shared.read(mf)
            else:
                if scope == 'Independent':
                    result, trace = run_holdout_case(exp, learner, streams[level], audits[level],
                        exp.validation, a.seed, 'LearnedBetaDDQN', 1)
                else:
                    result, trace = base.run_episode(exp, learner, streams[level], exp.validation,
                        a.seed, training=False)
                    result['adaptation_stream_seen_before'] = result.pop('target_before')
                    result['adaptation_stream_seen_after'] = result.pop('target_after')
                shared.write(mf, result); shared.csv(tf, trace)
                shared.write(cf, dict(execution_sha256=api.digest(ef),
                    sha256={p.name: api.digest(p) for p in (mf,tf)}))
            if (result['samples'] != len(streams[level][1]) or result['td_updates'] != 0
                    or result['cap'] != int(.2*len(streams[level][1]))
                    or not 0 <= result['labels'] <= result['cap']):
                raise RuntimeError('TEST cap, row coverage or frozen-Q contract violated.')
            row = shared.summary_row(a.dataset, scope,
                dict(attack='Burst-'+a.dataset, strength=level, level=level), result, beta)
            rows.append(row)
            metrics = ' '.join(f'{k}={row[k]:.2f}%' for k in shared.METRICS)
            final = f' FINAL_HOLDOUT_AC={row["target_final_AC"]:.2f}%' if scope == 'Independent' else ''
            print(f'TEST DATASET={a.dataset} SCOPE={scope} LEVEL={level} {metrics} '
                f'N={row["samples"]} LABELS={row["labels"]}/{row["cap"]} '
                f'({row["labels_percent"]:.2f}%) CAP_USED={row["budget_used_percent"]:.2f}% '
                f'ORIGINAL_FORGETTING={row["original_forgetting_AC"]:+.2f}{final}', flush=True)
        shared.csv(root/'summary_by_level.csv', rows)
        means = []
        for name, selected in [('ALL_LEVELS', rows), ('PERTURBED_LEVELS', [r for r in rows if r['level'] != 0])]:
            mean = dict(dataset=a.dataset, protocol=scope, group=name, levels=len(selected))
            for key in rows[0]:
                if key not in ('dataset','protocol','method','attack','strength','level'):
                    mean[key] = float(np.mean([r[key] for r in selected]))
            means.append(mean); print('MEAN '+str(mean), flush=True)
        shared.csv(root/'summary_means.csv', means)
        shared.write(root/'completed.json', dict(execution_sha256=api.digest(ef), levels=levels,
            summary_sha256=api.digest(root/'summary_by_level.csv')))
    exp.anchors = anchors
    if (base.weight_hash(learner.policy.state_dict()) != frozen_policy or learner.steps or learner.updates
            or learner.beta_value() != beta):
        raise RuntimeError('Q or beta changed in TEST.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT','PS'], required=True)
    p.add_argument('--action', choices=['All','Train','Evaluate'], default='All')
    p.add_argument('--protocol', choices=['Both','Online','Independent'], default='Both')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='cuda')
    p.add_argument('--source-template', type=Path)
    p.add_argument('--burst-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--cnn-epochs', type=int, default=60)
    p.add_argument('--context-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--meta-rows', type=int, default=300)
    a = p.parse_args()
    if min(a.cnn_epochs,a.context_epochs,a.patience,a.warmup_epochs,a.epochs) < 1 or int(.2*a.meta_rows) < 32:
        p.error('Positive epochs/patience and at least 32 paid meta observations required.')
    folder = ROOT/f'ANT_Burst_{a.dataset}_TEST'; runs = folder/'protocol_runs'
    a.source_template = (a.source_template or runs/f'light_{a.dataset.lower()}_paironly_batch5_trainseed{a.seed}'/'source').resolve()
    a.output = (a.output or runs/f'clean0_light_meta_beta_ddqn_seed{a.seed}_v1').resolve()
    a.burst_dir = (a.burst_dir or folder/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}').resolve()
    if a.output == a.source_template or a.output in a.source_template.parents or a.source_template in a.output.parents:
        p.error('Use a separate output, preserving the old source.')
    a.budget_fraction = .2; a.context_lr = .00025; a.replay_weight = .1; a.beta = 0.
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; choose --device cpu.')
    api.base.legacy.core.device = torch.device(device); api.base.legacy.core.set_seed(a.seed)
    # Only filenames are inspected now. No perturbed CSV is read until Evaluate.
    levels = levels_on_disk(a.burst_dir)
    pf = a.output/'protocol.json'
    if a.output.exists() and not pf.exists() and any(p.name != 'source' for p in a.output.iterdir()):
        p.error('Unmatched output preserved; use a fresh --output.')
    recipe = source_stage(a)
    exp = BurstCleanExperiment(a, recipe)
    protocol = shared.protocol_for(exp)
    protocol.update(initialization='fresh LIGHT CNN/Context solely clean0; partition metadata only imported',
        cnn_architecture=ARCHITECTURE, test_levels=levels,
        implementation_extra_sha256=api.digest(Path(__file__)))
    if pf.exists():
        if shared.read(pf) != protocol:
            p.error('Training recipe changed; use a fresh --output.')
    elif a.action == 'Evaluate':
        p.error('Train the clean-only DDQN/beta first.')
    else:
        shared.write(pf, protocol)
    print(f'RUN DATASET={a.dataset} TRAIN=[0] VAL=[0] TEST={levels} LIGHT_CNN '
        f'FULL_TRAIN={len(exp.full_train[1])} INNER_TRAIN={len(exp.train[1])} '
        f'FULL_TEST_PER_LEVEL={len(exp.artifact["test_ids"])} BUDGET=20% | no TEST cap', flush=True)
    if a.action in ('All','Train'):
        shared.train(exp, protocol)
    if a.action in ('All','Evaluate'):
        evaluate(exp, protocol)
    exp.assert_source_unchanged()
    print('OUTPUT: '+str(a.output), flush=True)


if __name__ == '__main__':
    main()
