"""Isolated AMCAL enhancement experiments: train, evaluate and persistent audit.

The original structured-bank entry point, checkpoints and results are not edited.
See FINAL_ENHANCEMENT.md for matched controls, profiles and Windows commands.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EnhancementAgent
from enhancements.config import EnhancementConfig
from enhancements.models import ResidualContext, residual_bank
from enhancements.stream import episode, observer

FORMAT_VERSION = 1
SOURCE_FILES = ('initial_cnn_model.pth', 'adversarial_best_context.pth', 'protocol_preprocessing.joblib')
CONFIG_OPTIONS = {
    'rank_weight': float, 'paired_weight': float, 'route_prior_weight': float,
    'route_lr_online': float, 'version_half_life': float, 'memory_ce_weight': float,
    'memory_replay_size': int, 'source_capacity': int, 'historical_capacity': int,
    'recent_capacity': int, 'reference_size': int, 'damage_limit': float,
    'damage_weight': float, 'token_width': int, 'transformer_layers': int,
    'feedforward_width': int, 'residual_bound': float, 'residual_weight': float,
    'cascade_confidence': float, 'cascade_margin': float,
    'cascade_audit_fraction': float, 'cascade_max_gap': int, 'compute_cost': float,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def load_artifact(models, burst_dir):
    artifact = joblib.load(models / 'protocol_preprocessing.joblib')
    if legacy.core.file_sha256(burst_dir / 'AdvBurstTrainedBurst_size_0_Step_2000.csv') != artifact['source_sha256']:
        raise ValueError('Source data differ from the saved preprocessing provenance.')
    pools = [set(map(int, artifact[key])) for key in ('train_ids', 'val_ids', 'test_ids')]
    if any(pools[i] & pools[j] for i in range(3) for j in range(i)):
        raise ValueError('TRAIN, validation and TEST source IDs must be disjoint.')
    if any(not pool for pool in pools): raise ValueError('All source partitions must be nonempty.')
    return artifact


def load_cnn(models, artifact):
    width = len(artifact['feature_names']); classes = len(artifact['label_encoder'].classes_)
    cnn = legacy.core.CNN1D(width, classes).to(legacy.core.device)
    saved = torch.load(models / 'initial_cnn_model.pth', map_location='cpu', weights_only=False)
    expected = legacy.core.file_sha256(models / 'protocol_preprocessing.joblib')
    if saved.get('preprocessing_sha256', expected) != expected:
        raise ValueError('Frozen CNN and preprocessing do not match.')
    cnn.load_state_dict(saved['model_state_dict']); cnn.eval(); cnn.requires_grad_(False)
    return cnn


def build_bank(models, artifact, config, initial):
    width = len(artifact['feature_names'])
    if config.architecture != 'legacy': return residual_bank(width, config, initial, legacy.core.device)
    bank = [legacy.core.ContextAwareNetwork(width).to(legacy.core.device).eval() for _ in range(3)]
    for context in bank: context.load_state_dict(initial)
    return bank


def pretrain_context(a, models, artifact, cnn, domains):
    source = torch.load(models / 'adversarial_best_context.pth', map_location='cpu', weights_only=False)
    expected = legacy.core.file_sha256(models / 'protocol_preprocessing.joblib')
    if source.get('preprocessing_sha256', expected) != expected:
        raise ValueError('Source Context and preprocessing do not match.')
    if a.config.architecture == 'legacy': return source['context_model_state_dict']
    width = len(artifact['feature_names'])
    teacher = legacy.core.ContextAwareNetwork(width).to(legacy.core.device).eval()
    teacher.load_state_dict(source['context_model_state_dict']); teacher.requires_grad_(False)
    context = ResidualContext(width, a.config).to(legacy.core.device)
    train_x = torch.cat([domains[level, 'train_ids'][0] for level in a.levels])
    train_y = torch.tensor(np.concatenate([domains[level, 'train_ids'][1] for level in a.levels]),
                           dtype=torch.long, device=legacy.core.device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(context.parameters(), lr=.0003, weight_decay=1e-4)

    def validation_loss(model):
        model.eval(); values = []
        with torch.no_grad():
            for level in a.levels:
                vx, vy, _ = domains[level, 'val_ids']; total = 0.
                for start in range(0, len(vy), 64):
                    y = torch.tensor(vy[start:start+64], device=vx.device)
                    total += float(F.cross_entropy(cnn(model(vx[start:start+64])), y, reduction='sum'))
                values.append(total / len(vy))
        return float(np.mean(values))

    best = validation_loss(context); teacher_ce = validation_loss(teacher)
    best_state = deepcopy(context.state_dict()); stale = 0; log = []
    for epoch in range(a.context_epochs):
        context.train()
        for x, y in loader:
            with torch.no_grad(): target = teacher(x)
            optimizer.zero_grad(set_to_none=True)
            adapted = context(x)
            loss = .8 * F.cross_entropy(cnn(adapted), y) + a.distill_weight * F.mse_loss(adapted, target)
            loss = loss + a.config.residual_weight * F.mse_loss(adapted, x)
            loss.backward(); nn.utils.clip_grad_norm_(context.parameters(), 1); optimizer.step()
        score = validation_loss(context)
        log.append({'epoch': epoch+1, 'validation_ce': score})
        print(f'Context warm start {epoch+1}/{a.context_epochs}: validation CE {score:.5f}', flush=True)
        if score < best: best = score; best_state = deepcopy(context.state_dict()); stale = 0
        else: stale += 1
        if stale >= a.patience: break
    write_json(a.output / 'context_warm_start.json', dict(teacher_validation_ce=teacher_ce,
        selected_validation_ce=best, epochs=log, selection='lowest validation CE including identity initialization',
        source='TRAIN labels and unchanged source Context teacher; no TEST data'))
    return best_state


def new_agent(a, cnn, artifact, offline=False, saved=None):
    classes = len(artifact['label_encoder'].classes_); embedding_width = cnn.dense_layers[-1].in_features
    return EnhancementAgent(embedding_width + 4*classes + 14, embedding_width, classes,
        legacy.core.device, a.seed, a.config, offline=offline,
        policy_state=None if saved is None else saved['policy_state_dict'],
        prerouter_state=None if saved is None else saved.get('prerouter_state_dict'), gamma=a.gamma)


def train(a):
    if a.source_run is None: raise ValueError('--source-run is required for train.')
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Training requires a fresh output directory; existing runs are never overwritten.')
    source = a.source_run / 'Models'
    if any(not (source/name).is_file() for name in SOURCE_FILES): raise ValueError('Source Models are incomplete.')
    if a.warm_start_selector and not (source/'structured_bank_ddqn.pth').is_file():
        raise ValueError('--warm-start-selector requires the source structured_bank_ddqn.pth checkpoint.')
    models = a.output / 'Models'; models.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES: shutil.copy2(source/name, models/name)
    artifact = load_artifact(models, a.burst_dir)
    legacy.core.SELECTION_COST = a.selection_cost
    legacy.core.GAIN_WEIGHT, legacy.core.RETENTION_WEIGHT = a.gain_weight, a.retention_weight
    a.classes = [str(c) for c in artifact['label_encoder'].classes_]
    cnn = load_cnn(models, artifact)
    domains = {(level, split): legacy.read_domain(a.burst_dir, artifact, level, split)
               for level in a.levels for split in ('train_ids', 'val_ids')}
    anchors = legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')[:2]
    initial = pretrain_context(a, models, artifact, cnn, domains)
    torch.save({'architecture': a.config.architecture, 'initial_state_dict': initial}, models/'enhanced_context.pth')
    bank = build_bank(models, artifact, a.config, initial)
    agent = new_agent(a, cnn, artifact, offline=True)
    if a.warm_start_selector:
        old = torch.load(source/'structured_bank_ddqn.pth', map_location='cpu', weights_only=False)
        if old.get('version') != 1 or old.get('route_width') != agent.policy.route_width:
            raise ValueError('Warm-start selector must be the compatible original v1 bank.')
        for name, digest in old['model_hashes'].items():
            if legacy.core.file_sha256(models/name) != digest: raise ValueError('Warm-start selector provenance mismatch.')
        agent.policy.load_state_dict(old['policy_state_dict']); agent.target.load_state_dict(agent.policy.state_dict())
    # Novelty thresholds use TRAIN statistics and SOURCE validation only.
    _, train_h = legacy.base_cache(cnn, anchors[0])
    source_validation = legacy.read_domain(a.burst_dir, artifact, 0, 'val_ids')[:2]
    _, val_h = legacy.base_cache(cnn, source_validation[0])
    agent.pre_router.calibrate(train_h, val_h)
    for warm in range(a.warmup_episodes):
        level = a.levels[warm % len(a.levels)]
        m, _ = episode(a, cnn, bank, initial, agent, domains[level, 'train_ids'], anchors,
            mode='random', training=True, warmup=True, rollout_seed=a.seed+100003+1009*warm)
        print(f'Correctness warmup {warm+1}: labels {m["labels"]}/{m["label_cap"]}', flush=True)
    hashes = {name: legacy.core.file_sha256(models/name) for name in (*SOURCE_FILES, 'enhanced_context.pth')}
    provenance = {str(level): legacy.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
                  for level in a.levels}
    best = -float('inf'); stale = 0; log = []
    for epoch in range(a.epochs):
        agent.td_memory.clear(); agent.pair_memory.clear()
        level = a.levels[epoch % len(a.levels)]
        m, _ = episode(a, cnn, bank, initial, agent, domains[level, 'train_ids'], anchors,
                       training=True, rollout_seed=a.seed+200003+1009*epoch)
        snapshot = dict(policy_state_dict=deepcopy(agent.policy.state_dict()),
                        prerouter_state_dict=deepcopy(agent.pre_router.state_dict()))
        validation = []
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for validation_level in a.levels:
                temporary = new_agent(a, cnn, artifact, saved=snapshot)
                result, _ = episode(a, cnn, bank, initial, temporary,
                    domains[validation_level, 'val_ids'], anchors, reference=source_validation)
                validation.append(result)
        accuracy = float(np.mean([row['accuracy'] for row in validation]))
        drop = float(np.mean([row['reference']['drop_pp'] for row in validation]))
        score = accuracy - a.retention_selection_weight * max(0., (drop-a.max_reference_drop)/100)
        log.append(dict(epoch=epoch+1, domain=level, validation_accuracy=accuracy,
                        reference_drop_pp=drop, selection_score=score, train_labels=m['labels']))
        print(f'Epoch {epoch+1}/{a.epochs}: validation {100*accuracy:.2f}% | source drop {drop:+.2f} pp | labels {m["labels"]}', flush=True)
        if score > best:
            best, stale = score, 0
            saved = dict(format_version=FORMAT_VERSION, config=a.config.as_dict(), training_levels=a.levels,
                model_hashes=hashes, training_domain_sha256=provenance,
                training_args={key: getattr(a, key) for key in ('gamma', 'context_lr', 'context_alpha',
                    'selection_cost', 'gain_weight', 'retention_weight', 'seed', 'budget_fraction')},
                selected_epoch=epoch+1, validation_accuracy=accuracy, selection_score=score,
                validation_selection=dict(max_reference_drop_pp=a.max_reference_drop,
                    penalty_weight=a.retention_selection_weight), **snapshot)
            torch.save(saved, models/'final_enhancement.pth')
        else:
            stale += 1
            if stale >= a.patience: break
    write_json(a.output/'enhancement_training.json', dict(config=a.config.as_dict(), epochs=log,
        source_run=str(a.source_run), model_hashes=hashes, training_domains=provenance,
        warm_start_selector=a.warm_start_selector, evaluation_policy='source-disjoint, pre-feedback',
        min_max='selector maximizes correctness/transfer return; selected Context minimizes current and protected-memory loss'))
    print('Saved:', models/'final_enhancement.pth', flush=True)


def summary_row(level, mode, m):
    ref = m.get('reference', {})
    return dict(Level=level, Mode=mode, N=m['samples'], Accuracy=100*m['accuracy'], MacroF1=100*m['macro_f1'],
        Labels=m['labels'], Cap=m['label_cap'], LabelPct=100*m['actual_label_fraction'],
        Oracle=None if m['oracle_all'] is None else 100*m['oracle_all'],
        FullBankPct=100*m['full_bank_fraction'], SourceDrop=ref.get('drop_pp'),
        ContextDrop=ref.get('mean_context_drop_pp'), MsPerRow=m['online_ms_per_sample'],
        ResponseP50=m['response_ms_percentiles']['p50'], ResponseP95=m['response_ms_percentiles']['p95'],
        ResponseP99=m['response_ms_percentiles']['p99'])


def evaluate(a):
    models = a.output / 'Models'
    saved = torch.load(models/'final_enhancement.pth', map_location='cpu', weights_only=False)
    if saved['format_version'] != FORMAT_VERSION: raise ValueError('Unsupported enhancement checkpoint format.')
    config = EnhancementConfig(**saved['config']).validated()
    if a.profile is not None and a.profile != config.profile: raise ValueError('Profile differs from training.')
    if a.disable: raise ValueError('Use separate TRAIN runs for enhancement ablations; evaluation loads saved settings.')
    a.config = config
    for key, value in saved['training_args'].items():
        if key in ('seed', 'budget_fraction'): continue
        if key in ('context_lr', 'context_alpha') and getattr(a, key) is not None: continue
        setattr(a, key, value)
    legacy.core.SELECTION_COST = a.selection_cost
    legacy.core.GAIN_WEIGHT, legacy.core.RETENTION_WEIGHT = a.gain_weight, a.retention_weight
    if any(level in saved['training_levels'] for level in a.levels):
        raise ValueError('Evaluation levels must be held-out intensities; TRAIN levels are saved in the checkpoint.')
    for name, digest in saved['model_hashes'].items():
        if legacy.core.file_sha256(models/name) != digest: raise ValueError('Model provenance changed: ' + name)
    artifact = load_artifact(models, a.burst_dir)
    a.classes = [str(c) for c in artifact['label_encoder'].classes_]
    anchors = legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')[:2]
    clean = legacy.read_domain(a.burst_dir, artifact, 0, 'val_ids')[:2]
    domains = {level: legacy.read_domain(a.burst_dir, artifact, level, 'test_ids') for level in a.levels}
    audits = None
    if a.continual:
        common = sorted(set.intersection(*(set(map(int, d[2])) for d in domains.values())))
        if len(common) < 10: raise ValueError('Insufficient common IDs for an independent persistent audit.')
        order = np.random.default_rng(a.seed+8121).permutation(common)
        audit_ids = set(map(int, order[:max(1, int(len(order)*a.audit_fraction))]))
        audits = {}
        for level, (x, y, ids) in domains.items():
            mask = np.array([int(i) in audit_ids for i in ids])
            audits[level] = x[mask], y[mask]
            domains[level] = x[~mask], y[~mask], ids[~mask]
    execution = dict(seed=a.seed, budget_fraction=a.budget_fraction, context_lr=a.context_lr,
                     context_alpha=a.context_alpha, epsilon=a.eval_epsilon, modes=a.modes,
                     query_schedule=a.query_schedule, update_policy=a.update_policy,
                     config=config.as_dict(), checkpoint_sha256=legacy.core.file_sha256(models/'final_enhancement.pth'),
                     levels=a.levels, continual=a.continual, audit_fraction=a.audit_fraction,
                     evaluation_tag=a.evaluation_tag,
                     runtime=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                         numpy=np.__version__, device=str(legacy.core.device), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(legacy.core.device) if legacy.core.device.type == 'cuda' else None))
    run_id = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()[:12]
    root = a.output / 'Results' / ('evaluation_'+run_id)
    if root.exists(): raise ValueError('This exact evaluation already exists; change --seed or use a fresh run copy.')
    root.mkdir(parents=True, exist_ok=False)
    rows = []; audit_rows = []; forgetting_rows = []
    for mode in a.modes:
        legacy.core.set_seed(a.seed)
        cnn = load_cnn(models, artifact)
        frozen_weights = {name: value.detach().cpu().clone() for name, value in cnn.state_dict().items()}
        initial = torch.load(models/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
        bank = build_bank(models, artifact, config, initial)
        agent = new_agent(a, cnn, artifact, saved=saved)
        continuation = {} if a.continual else None
        audit_matrix = []
        for stage, level in enumerate(a.levels, start=1):
            if not a.continual: agent = new_agent(a, cnn, artifact, saved=saved)
            m, traces = episode(a, cnn, bank, initial, agent, domains[level], anchors,
                mode=mode, reference=clean, same_level=True, continuation=continuation)
            m.update(level=level, config_fingerprint=config.fingerprint, execution=execution,
                     classifier_unchanged=not cnn.training and all(not p.requires_grad for p in cnn.parameters())
                     and all(torch.equal(value.detach().cpu(), frozen_weights[name]) for name, value in cnn.state_dict().items()))
            if not m['classifier_unchanged']: raise RuntimeError('Frozen CNN parameters or buffers changed.')
            write_json(root/f'{mode}_level{level}_metrics.json', m)
            write_csv(root/f'{mode}_level{level}_trace.csv', traces)
            rows.append(summary_row(level, mode, m))
            print(f'Level {level} | {mode}: accuracy {100*m["accuracy"]:.2f}% | labels {m["labels"]}/{m["label_cap"]} | full bank {100*m["full_bank_fraction"]:.1f}% | {m["online_ms_per_sample"]:.2f} ms/row', flush=True)
            if audits is not None:
                history = continuation['history']; scores = []
                for audit_level, (ax, ay) in audits.items():
                    result = observer(cnn, bank, agent, ax, ay, history, m['labels'], m['label_cap'],
                                      m['samples'], a.seed, config, mode)
                    accuracy = float(np.mean(result[0] == ay)); scores.append(accuracy)
                    audit_rows.append(dict(Mode=mode, Stage=stage, AfterLevel=level, AuditLevel=audit_level,
                                           N=len(ay), Accuracy=100*accuracy))
                audit_matrix.append(scores)
        if audits is not None:
            matrix = np.asarray(audit_matrix); count = len(a.levels)
            forgetting = np.mean([matrix[j:, j].max()-matrix[-1, j] for j in range(count-1)])
            bwt = np.mean([matrix[-1, j]-matrix[j, j] for j in range(count-1)])
            forgetting_rows.append(dict(Mode=mode, FinalAudit=100*float(matrix[-1].mean()),
                Forgetting=100*float(forgetting), BWT=100*float(bwt),
                Labels=sum(row['Labels'] for row in rows if row['Mode'] == mode)))
    write_csv(root/'combined_results.csv', rows)
    if audits is not None:
        write_csv(root/'sequential_audit_matrix.csv', audit_rows)
        write_csv(root/'sequential_summary.csv', forgetting_rows)
        execution['audit_source_ids'] = sorted(audit_ids)
    write_json(root/'execution.json', execution)
    print('Results:', root, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate'])
    p.add_argument('--source-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--burst-dir', type=Path, required=True)
    p.add_argument('--profile', choices=['accuracy', 'compact', 'shared', 'cascade'])
    p.add_argument('--disable', nargs='+', choices=['ranking', 'paired', 'prior', 'memory', 'replay', 'damage'], default=[])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--levels', nargs='+', type=int)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--context-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--warmup-episodes', type=int, default=3)
    p.add_argument('--warm-start-selector', action='store_true')
    p.add_argument('--distill-weight', type=float, default=.2)
    p.add_argument('--context-lr', type=float)
    p.add_argument('--context-alpha', type=float)
    p.add_argument('--budget-fraction', type=float, default=.2)
    p.add_argument('--gamma', type=float, default=.95)
    p.add_argument('--selection-cost', type=float, default=.0001)
    p.add_argument('--gain-weight', type=float, default=1.)
    p.add_argument('--retention-weight', type=float, default=1.)
    p.add_argument('--train-epsilon', type=float, default=.2)
    p.add_argument('--eval-epsilon', type=float, default=.05)
    p.add_argument('--modes', nargs='+', choices=['learned', 'random'], default=['learned', 'random'])
    p.add_argument('--query-schedule', choices=['policy', 'uniform'], default='policy')
    p.add_argument('--update-policy', choices=['learned', 'fixed'], default='learned')
    p.add_argument('--max-reference-drop', type=float, default=2.)
    p.add_argument('--retention-selection-weight', type=float, default=1.)
    p.add_argument('--bank-executor', choices=['sequential', 'vmap'])
    p.add_argument('--routed-utility', action='store_true')
    p.add_argument('--continual', action='store_true')
    p.add_argument('--audit-fraction', type=float, default=.2)
    p.add_argument('--evaluation-tag', default='', help='Unique repeat label for identical-trajectory timing repetitions.')
    for name, value_type in CONFIG_OPTIONS.items():
        p.add_argument('--'+name.replace('_', '-'), type=value_type)
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.update_policy == 'fixed' and a.query_schedule != 'uniform': p.error('Fixed destinations require uniform feedback.')
    if not 0 <= a.budget_fraction <= 1 or not 0 <= a.gamma <= 1: p.error('Budget and gamma must be in [0,1].')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (a.train_epsilon, a.eval_epsilon)):
        p.error('Exploration probabilities must be in [0,1].')
    if min(a.epochs, a.context_epochs, a.patience) < 1 or a.warmup_episodes < 0: p.error('Invalid training schedule.')
    if a.context_lr is not None and (not np.isfinite(a.context_lr) or a.context_lr <= 0): p.error('Context LR must be positive.')
    if a.context_alpha is not None and not 0 <= a.context_alpha <= 1: p.error('Context alpha must be in [0,1].')
    if any(not np.isfinite(getattr(a, name)) or getattr(a, name) < 0 for name in
           ('distill_weight', 'selection_cost', 'gain_weight', 'retention_weight',
            'max_reference_drop', 'retention_selection_weight')):
        p.error('Loss, cost and retention settings must be finite and nonnegative.')
    if not .05 <= a.audit_fraction <= .5: p.error('Audit fraction must be .05.. .5.')
    if a.continual and a.command != 'evaluate': p.error('--continual is evaluation-only.')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA is unavailable.')
    legacy.core.device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else 'cpu' if a.device == 'auto' else a.device)
    legacy.core.set_seed(a.seed)
    a.output, a.burst_dir = a.output.resolve(), a.burst_dir.resolve()
    if a.source_run is not None: a.source_run = a.source_run.resolve()
    a.levels = a.levels or ([0, 1, 3] if a.command == 'train' else [5, 7, 10, 12, 15, 17, 20])
    if len(a.levels) != len(set(a.levels)) or min(a.levels) < 0: p.error('Levels must be unique nonnegative integers.')
    if a.continual and len(a.levels) < 2: p.error('A continual audit requires at least two domains.')
    a.bank_utility = False; a.balanced_reward_reference = True
    if a.command == 'train':
        from dataclasses import replace
        config = EnhancementConfig.preset(a.profile or 'compact')
        config = replace(config, bank_executor=a.bank_executor or config.bank_executor, routed_utility=a.routed_utility,
                         **{name: getattr(a, name) for name in CONFIG_OPTIONS if getattr(a, name) is not None})
        a.config = config.without(a.disable)
        a.context_lr = .0005 if a.context_lr is None else a.context_lr
        a.context_alpha = .8 if a.context_alpha is None else a.context_alpha
        train(a)
    else:
        if a.bank_executor is not None or a.routed_utility or any(getattr(a, name) is not None for name in CONFIG_OPTIONS):
            p.error('Enhancement settings are loaded from the trained checkpoint; use a separate TRAIN run to change them.')
        evaluate(a)


if __name__ == '__main__': main()
