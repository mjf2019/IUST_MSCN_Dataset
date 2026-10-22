"""Isolated AMCAL enhancement experiments: train, evaluate and persistent audit.

The original structured-bank entry point, checkpoints and results are not edited.
See FINAL_ENHANCEMENT.md for matched controls, profiles and Windows commands.
"""
import argparse
from copy import deepcopy, copy
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
from enhancements.agent import EnhancementAgent, DeviceReliability, load_policy_state, protected_routing_fraction
from enhancements.config import EnhancementConfig
from enhancements.models import ResidualContext, residual_bank
from enhancements.stream import episode, observer
from enhancements.sequence import split_validation, fit_sequence, validate_sequence, next_sequence_features
from enhancements.utility_normalization import fit_scale, validate_calibration

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


def build_bank(models, artifact, config, initial, slots=3):
    width = len(artifact['feature_names'])
    if config.architecture != 'legacy': return residual_bank(width, config, initial, legacy.core.device)
    bank = [legacy.core.ContextAwareNetwork(width).to(legacy.core.device).eval() for _ in range(slots)]
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


def new_agent(a, cnn, artifact, offline=False, saved=None, query_value_only=False, paired_query_only=False):
    classes = len(artifact['label_encoder'].classes_); embedding_width = cnn.dense_layers[-1].in_features
    agent = EnhancementAgent(embedding_width + 4*classes + 14, embedding_width, classes,
        legacy.core.device, a.seed, a.config, offline=offline,
        policy_state=None if saved is None else saved['policy_state_dict'],
        prerouter_state=None if saved is None else saved.get('prerouter_state_dict'), gamma=a.gamma,
        informative_weight=getattr(a, 'router_informative_weight', 1.), slots=getattr(a, 'context_count', 3),
        query_value_only=query_value_only, paired_query_only=paired_query_only)
    agent.policy.utility_normalization = getattr(a, 'utility_normalization', None)
    return agent


def calibrate_utility(a):
    if a.utility_calibration is None: raise ValueError('--utility-calibration output path is required.')
    if a.utility_calibration.exists(): raise ValueError('Calibration already exists; use evaluation or a fresh path.')
    models = a.output/'Models'
    saved = torch.load(models/'final_enhancement.pth', map_location='cpu', weights_only=False)
    if saved['format_version'] != FORMAT_VERSION: raise ValueError('Unsupported checkpoint format.')
    a.config = EnhancementConfig(**saved['config']).validated()
    if not set(a.levels).issubset(saved['training_levels']): raise ValueError('Calibration must use TRAIN intensities only.')
    for name, digest in saved['model_hashes'].items():
        if legacy.core.file_sha256(models/name) != digest: raise ValueError('Model provenance mismatch: '+name)
    artifact = load_artifact(models, a.burst_dir)
    a.classes = [str(c) for c in artifact['label_encoder'].classes_]
    for key, value in saved['training_args'].items():
        if key not in ('seed', 'budget_fraction'): setattr(a, key, value)
    legacy.core.SELECTION_COST = a.selection_cost
    legacy.core.GAIN_WEIGHT, legacy.core.RETENTION_WEIGHT = a.gain_weight, a.retention_weight
    a.utility_normalization = None; a.collect_utility_calibration = True
    cnn = load_cnn(models, artifact)
    initial = torch.load(models/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
    bank = build_bank(models, artifact, a.config, initial)
    agent = new_agent(a, cnn, artifact, saved=saved)
    anchors = legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')[:2]
    continuation = {}; updates = []; probabilities = []; source_ids = set(); stages = []
    for level in a.levels:
        domain = legacy.read_domain(a.burst_dir, artifact, level, 'train_ids')
        if legacy.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != saved['training_domain_sha256'][str(level)]:
            raise ValueError('Calibration TRAIN provenance mismatch.')
        result, traces = episode(a, cnn, bank, initial, agent, domain, anchors, continuation=continuation)
        for row in traces:
            if row['raw_update_values'] is not None:
                updates.append(row['raw_update_values']); probabilities.append(row['scores'][8:11])
        source_ids.update(map(int, domain[2]))
        stages.append(dict(level=level, rows=result['samples'], labels=result['labels']))
        print(f'Utility calibration TRAIN level {level} | rows {result["samples"]}', flush=True)
    calibration = fit_scale(updates, probabilities)
    calibration.update(checkpoint_sha256=legacy.core.file_sha256(models/'final_enhancement.pth'),
        levels=a.levels, seed=a.seed, stages=stages, model_hashes=saved['model_hashes'],
        source_ids_sha256=hashlib.sha256(json.dumps(sorted(source_ids)).encode()).hexdigest(),
        query_schedule=a.query_schedule, update_policy=a.update_policy,
        protocol='temporary persistent TRAIN rollout, paid feedback only; no VAL/TEST; original checkpoint unchanged')
    a.utility_calibration.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.utility_calibration, calibration)
    print(f'TRAIN utility scale: {calibration["scale"]:.6g} | saved {a.utility_calibration}', flush=True)


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
    warm = None
    if a.warm_start_enhancement:
        from dataclasses import replace
        warm = torch.load(source/'final_enhancement.pth', map_location='cpu', weights_only=False)
        if warm.get('format_version') != FORMAT_VERSION:
            raise ValueError('Unsupported enhancement warm-start format.')
        warm_config = EnhancementConfig(**warm['config'])
        if replace(warm_config, joint_router=a.config.joint_router) != a.config:
            raise ValueError('Warm-start configuration must match except for the joint-router extension.')
        if set(warm['training_levels']) != set(a.levels):
            raise ValueError('Warm-start TRAIN levels differ.')
        for name, digest in warm['model_hashes'].items():
            if legacy.core.file_sha256(source/name) != digest:
                raise ValueError('Enhancement warm-start provenance mismatch: '+name)
        for level, digest in warm['training_domain_sha256'].items():
            if legacy.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != digest:
                raise ValueError('Enhancement warm-start TRAIN domain mismatch: '+level)
        initial = torch.load(source/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
        shutil.copy2(source/'enhanced_context.pth', models/'enhanced_context.pth')
    else:
        initial = pretrain_context(a, models, artifact, cnn, domains)
        torch.save({'architecture': a.config.architecture, 'initial_state_dict': initial}, models/'enhanced_context.pth')
    bank = build_bank(models, artifact, a.config, initial, slots=a.context_count)
    agent = new_agent(a, cnn, artifact, offline=not a.aligned_sequence, saved=warm,
                      query_value_only=a.query_value_only_training, paired_query_only=a.paired_query_only_training)
    if a.warm_start_selector:
        old = torch.load(source/'structured_bank_ddqn.pth', map_location='cpu', weights_only=False)
        if old.get('version') != 1 or old.get('route_width') != agent.policy.route_width:
            raise ValueError('Warm-start selector must be the compatible original v1 bank.')
        for name, digest in old['model_hashes'].items():
            if legacy.core.file_sha256(models/name) != digest: raise ValueError('Warm-start selector provenance mismatch.')
        load_policy_state(agent.policy, old['policy_state_dict']); agent.target.load_state_dict(agent.policy.state_dict())
    # Novelty thresholds use TRAIN statistics and SOURCE validation only.
    _, train_h = legacy.base_cache(cnn, anchors[0])
    clean_validation_domain = legacy.read_domain(a.burst_dir, artifact, 0, 'val_ids')
    source_validation = clean_validation_domain[:2]
    calibration_x = source_validation[0]
    training_protocol = None
    if a.aligned_sequence:
        validation_domains = {level: domains[level, 'val_ids'] for level in a.levels}
        val_streams, val_audits, val_clean, val_audit_ids = split_validation(
            validation_domains, clean_validation_domain, a.seed, a.audit_fraction)
        val_audit_set = set(val_audit_ids)
        calibration_x = clean_validation_domain[0][np.array([
            int(source_id) not in val_audit_set for source_id in clean_validation_domain[2]])]
        training_protocol = dict(version=1, mode='aligned_sequence', levels=a.levels,
            query_schedule=a.query_schedule, update_policy='learned', paid_only_supervision=True,
            agent_optimizer='online settings', train_epsilon=a.eval_epsilon,
            validation_audit_source_ids=val_audit_ids, audit_fraction=a.audit_fraction,
            validation_adaptation_source_ids={str(level): list(map(int, d[2])) for level, d in val_streams.items()},
            selection_metric='final persistent VAL audit accuracy minus excess clean-drop penalty',
            reset='bank/memory reset per TRAIN pass; all adaptive state retained within each sequence')
        if a.one_context_rl_training:
            training_protocol['context_count'] = 1
            training_protocol['selector_specialization'] = dict(
                version=1, mode='one_context_policy_feedback',
                warm_start='inherited shared selector weights and initial Context',
                trained_parameters='correctness, future and update_value heads',
                context='paid-label adaptation within each sequence; reset to inherited initial state per epoch',
                supervision='TRAIN anchors and acquired labels only; no unqueried-label supervision',
                validation='temporary policy-feedback learner; disjoint observer-only VAL audit',
                damage_guard=False)
            if a.query_value_only_training:
                training_protocol['selector_specialization'].update(
                    mode='one_context_query_value_only', trained_parameters='update_value only',
                    frozen_parameters='correctness and future heads; exact inherited weights',
                    optimizer='Adam update_value only; existing TD and paired objectives',
                    evaluation='ordinary paid-feedback online adaptation of all selector heads')
    if a.paired_query_only_training:
        training_protocol['selector_specialization'].update(
            optimizer='Adam update_value only; paired objective only; absolute TD diagnostic has no gradient',
            target='utility + gamma * (V(after query) - V(skip))', paired_weight=a.config.paired_weight)
        training_protocol['temporal_credit'] = dict(version=1, paired_only=True,
            connected_levels=a.connected_sequence_td, gamma=a.gamma,
            boundary='next-domain first adaptation features; budget recharged; no next label',
            evaluation='ordinary online selector losses; connected boundaries only for persistent sequences',
            diagnostics='sampled TRAIN replay targets; rows may be counted repeatedly')
    if a.counterfactual_utility:
        training_protocol['utility_supervision'] = dict(version=1, scope='paid TRAIN only',
            alternatives=3, trained_parameters='update_value only', learning_rate=.0001,
            absolute_loss='SmoothL1 raw paired return', relative_loss='0.5 * centered SmoothL1',
            target='utility + gamma * (raw target V(updated) - raw target V(skip))',
            target_network='inherited critic fixed throughout offline fitting; Double-DQN action choice',
            reference='same pre-feedback protected reference and replay for all alternatives',
            isolation='copied Context and Adam per alternative; original bank/history/RNG unchanged',
            normalization=False, evaluation_simulations=False)
    _, val_h = legacy.base_cache(cnn, calibration_x)
    agent.pre_router.calibrate(train_h, val_h)
    for warm in range(0 if a.warm_start_enhancement else a.warmup_episodes):
        level = a.levels[warm % len(a.levels)]
        m, _ = episode(a, cnn, bank, initial, agent, domains[level, 'train_ids'], anchors,
            mode='random', training=True, warmup=True, rollout_seed=a.seed+100003+1009*warm)
        print(f'Correctness warmup {warm+1}: labels {m["labels"]}/{m["label_cap"]}', flush=True)
    hashes = {name: legacy.core.file_sha256(models/name) for name in (*SOURCE_FILES, 'enhanced_context.pth')}
    provenance = {str(level): legacy.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
                  for level in a.levels}
    best = -float('inf'); stale = 0; log = []
    # Include the inherited selector as an epoch-zero validation candidate.
    for epoch in range(-int(a.warm_start_enhancement), a.epochs):
        level = None; m = {'labels': 0}
        if epoch >= 0:
            agent.td_memory.clear(); agent.pair_memory.clear()
            agent.counterfactual_memory.clear()
            if a.aligned_sequence:
                agent.route_memory.clear()
                rollout_args = copy(a); rollout_args.train_epsilon = a.eval_epsilon
                train_domains = {level: domains[level, 'train_ids'] for level in a.levels}
                m = fit_sequence(rollout_args, cnn, bank, initial, agent, train_domains, anchors,
                                 a.seed+200003+1009*epoch)
                level = list(a.levels)
            else:
                level = a.levels[epoch % len(a.levels)]
                m, _ = episode(a, cnn, bank, initial, agent, domains[level, 'train_ids'], anchors,
                               training=True, rollout_seed=a.seed+200003+1009*epoch)
        snapshot = dict(policy_state_dict=deepcopy(agent.policy.state_dict()),
                        prerouter_state_dict=deepcopy(agent.pre_router.state_dict()))
        if a.query_value_only_training:
            frozen_unchanged = all(torch.equal(value.detach().cpu(), warm['policy_state_dict'][key].detach().cpu())
                for key, value in snapshot['policy_state_dict'].items() if not key.startswith('update_value.'))
            if not frozen_unchanged: raise RuntimeError('A frozen selector head changed during TRAIN.')
        validation = []
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            if a.aligned_sequence:
                temporary = new_agent(a, cnn, artifact, saved=snapshot)
                validation_bank = build_bank(models, artifact, a.config, initial, slots=a.context_count)
                val_sequence = validate_sequence(a, cnn, validation_bank, initial, temporary,
                    val_streams, val_audits, val_clean, anchors)
                accuracy, drop = val_sequence['accuracy'], val_sequence['drop_pp']
            else:
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
        if a.aligned_sequence:
            log[-1].update(validation_sequence=val_sequence, train_stages=m.get('stages', []))
        if a.query_value_only_training:
            log[-1]['frozen_selector_unchanged'] = frozen_unchanged
        if a.counterfactual_utility:
            log[-1]['counterfactual_updates_total'] = agent.counterfactual_updates
            log[-1]['counterfactual_loss'] = agent.last_losses.get('counterfactual')
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
            if a.warm_start_enhancement:
                saved['parent_checkpoint_sha256'] = legacy.core.file_sha256(source/'final_enhancement.pth')
            if a.one_context_rl_training:
                saved['training_context_count'] = 1
            if training_protocol is not None: saved['training_protocol'] = training_protocol
            torch.save(saved, models/'final_enhancement.pth')
        else:
            stale += 1
            if stale >= a.patience: break
    write_json(a.output/'enhancement_training.json', dict(config=a.config.as_dict(), epochs=log,
        source_run=str(a.source_run), model_hashes=hashes, training_domains=provenance,
        warm_start_selector=a.warm_start_selector, warm_start_enhancement=a.warm_start_enhancement,
        selected_epoch=saved['selected_epoch'],
        parent_checkpoint_sha256=saved.get('parent_checkpoint_sha256'),
        training_protocol=training_protocol,
        evaluation_policy='source-disjoint, pre-feedback',
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


def completed_evaluation(root, execution):
    """Only reuse an evaluation with matching metadata and all expected rows."""
    def row_count(path, columns):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not set(columns).issubset(reader.fieldnames or []): return -1
            return sum(1 for _ in reader)
    try:
        recorded = json.loads((root/'execution.json').read_text(encoding='utf-8-sig'))
        if not isinstance(recorded, dict): return False
        if any(recorded.get(key) != value for key, value in execution.items()): return False
        modes, levels = execution['modes'], execution['levels']
        if row_count(root/'combined_results.csv', ['Level','Mode','Accuracy']) != len(modes)*len(levels): return False
        for mode in modes:
            for level in levels:
                metric = json.loads((root/f'{mode}_level{level}_metrics.json').read_text(encoding='utf-8-sig'))
                if metric['mode'] != mode or metric['level'] != level or metric['samples'] <= 0: return False
                if not 0 <= metric['labels'] <= metric['label_cap']: return False
                if row_count(root/f'{mode}_level{level}_trace.csv', ['source_row','prediction','queried']) != metric['samples']: return False
        if execution['continual']:
            if row_count(root/'sequential_summary.csv', ['Mode','FinalAudit']) != len(modes): return False
            expected = len(modes)*(len(levels)+1)**2
            if row_count(root/'sequential_audit_matrix.csv', ['Mode','Stage','AuditLevel','Accuracy']) != expected: return False
        if 'routing_oracle_audit' in execution:
            if row_count(root/'routing_oracle_summary.csv', ['Mode','MeanStreamRoutingOracle']) != len(modes): return False
        return True
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return False


def evaluate(a):
    models = a.output / 'Models'
    saved = torch.load(models/'final_enhancement.pth', map_location='cpu', weights_only=False)
    if saved.get('training_context_count') is not None and a.context_count != saved['training_context_count']:
        raise ValueError('Context count must match this specialized training checkpoint.')
    if saved['format_version'] != FORMAT_VERSION: raise ValueError('Unsupported enhancement checkpoint format.')
    config = EnhancementConfig(**saved['config']).validated()
    if a.profile is not None and a.profile != config.profile: raise ValueError('Profile differs from training.')
    if a.joint_router and not config.joint_router: raise ValueError('Joint router requires a separately trained checkpoint.')
    if a.disable: raise ValueError('Use separate TRAIN runs for enhancement ablations; evaluation loads saved settings.')
    if a.online_memory_ce_weight is not None:
        if not config.protected_reference or config.memory_replay_size < 1:
            raise ValueError('Online replay weighting requires protected memory and nonempty replay.')
        from dataclasses import replace
        config = replace(config, memory_ce_weight=a.online_memory_ce_weight).validated()
    if a.context_damage_guard and (not config.protected_reference or config.routed_utility
            or config.memory_replay_size < 1 or config.memory_ce_weight != .1):
        raise ValueError('Damage-guard ablation requires protected replay with the original CE weight 0.1.')
    if a.routing_oracle_audit and config.cascade:
        raise ValueError('Routing oracle audit requires a full-bank evaluation without cascade.')
    a.config = config
    temporal_credit = saved.get('training_protocol', {}).get('temporal_credit')
    if temporal_credit is not None:
        a.connected_sequence_td = temporal_credit['connected_levels']
        if a.connected_sequence_td and (a.query_schedule != 'policy' or a.update_policy != 'learned'):
            raise ValueError('Connected-sequence checkpoints require policy feedback with learned updates.')
    if a.context_count != 3 and (config.architecture != 'legacy' or config.cascade or config.joint_router):
        raise ValueError('Context-count ablation requires the accuracy legacy architecture without cascade or joint routing.')
    calibration = None
    if a.utility_calibration is not None:
        calibration = validate_calibration(json.loads(a.utility_calibration.read_text()),
            legacy.core.file_sha256(models/'final_enhancement.pth'), saved['training_levels'])
        a.utility_normalization = dict(scale=calibration['scale'], weight=a.utility_weight, clip=a.utility_clip)
    if a.router_protected_replay and (not config.protected_reference or config.routed_utility
            or config.cascade or config.architecture != 'legacy' or config.joint_router):
        raise ValueError('Protected routing replay requires unchanged legacy Context with local protected memory.')
    if a.router_refresh and not config.protected_reference:
        raise ValueError('Router refresh requires protected TRAIN/paid-label references.')
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
    if a.paid_replay_mode != 'off':
        execution['paid_replay'] = dict(version=1, mode=a.paid_replay_mode,
            count='all_retained_paid' if a.paid_replay_mode == 'batch_all' else 2,
            sampling=('all unique retained recent/historical paid rows; excludes current'
                      if a.paid_replay_mode == 'batch_all' else
                      'uniform without replacement; recent/historical paid rows; excludes current'),
            protected_replay='unchanged; reused for all steps', utility='whole update')
    if temporal_credit is not None:
        execution['temporal_credit'] = dict(gamma=a.gamma, connected_levels=a.connected_sequence_td,
            active_boundary_links=bool(a.continual and a.connected_sequence_td))
    if 'random_query_online' in a.modes:
        execution['random_query_online_control'] = dict(version=1,
            query='matched full-random legal-action draws; query probability 1/3 until cap',
            prediction='correctness head argmax without exploration',
            learning='routing optimizer on paid pre-feedback correctness only; future/update/target fixed',
            context='paid adaptation and protected replay retained')
    if a.routing_oracle_audit:
        execution['routing_oracle_audit'] = dict(version=1,
            protocol='observer-only any-correct CNN/Context prediction before feedback; labels never select actions',
            scope='routing ceiling conditional on the evaluated query/adaptation trajectory; not global capacity')
    if 'random_query' in a.modes:
        execution['random_query_control'] = dict(version=1,
            query='same RNG and legal-action query draws as full random; probability 1/3 until cap',
            prediction='frozen correctness head argmax; no exploration; independent of query',
            learning='no selector learning; paid Context adaptation and memory retained',
            trace='action encodes feedback destination on queried rows; route encodes prediction')
    if 'random' in a.modes:
        execution['random_control'] = dict(version=2,
            actions='uniform among legal actions; random prediction routes after budget exhaustion',
            selector='no selector action/observer forward or learning in random evaluation; Context adaptation retained')
    if a.context_damage_guard:
        execution['context_damage_guard'] = dict(version=1, max_harm=config.damage_limit,
            step_scales=[1., .5, .25], reference='source/historical rows of the same pre-feedback reward reference',
            statistic='class-macro mean negative normalized CE change',
            optimizer='one accepted Adam step; original moments on every retry; base LR restored',
            rejected='live Context/Adam/version unchanged; acquired label still charged and stored',
            missing_reference='reject Context update', scope='online only; no VAL/TEST audit labels')
    if a.online_memory_ce_weight is not None:
        execution['online_adaptation_override'] = dict(version=1,
            memory_ce_weight=dict(checkpoint=saved['config']['memory_ce_weight'], online=config.memory_ce_weight),
            scope='online Context protected-replay CE only; inherited checkpoint unchanged')
    if a.context_count != 3:
        execution['context_count'] = a.context_count
        execution['action_count'] = 2*a.context_count+1
        execution['context_count_protocol'] = 'evaluation-only bank reduction; inherited shared weights; aggregates use active routes only'
        if saved.get('training_context_count') == a.context_count:
            execution['context_count_protocol'] = 'matched Context count from dedicated policy-feedback sequence training'
        if a.context_count > 3:
            execution['context_count_protocol'] = 'evaluation-only bank expansion; inherited shared weights; aggregates use active routes only'
    if a.continual:
        execution['retention_protocol_version'] = 2
        execution['audit_protocol'] = 'seed+8121 fixed source IDs excluded from every adaptation domain; observer-only labels'
        execution['continuation_protocol'] = 'all adaptive state retained; label budget recharged per domain'
    if 'training_protocol' in saved: execution['training_protocol'] = saved['training_protocol']
    if calibration is not None:
        execution['utility_normalization'] = dict(calibration=calibration,
            weight=a.utility_weight, clip=a.utility_clip, application='actor only; raw reward and critic units retained')
    if a.router_level_balanced_history:
        execution['router_level_balanced_history'] = dict(version=1,
            storage='unchanged original class reservoirs; acquisition-level tags only',
            sampler='routing-only historical draw: equal available levels then classes; separate RNG; current input excluded',
            weights='equal source/history mass; historical level/class hierarchy',
            capacity='unchanged; unavailable or evicted levels cannot be recovered',
            labels='acquired TEST feedback only for target history; source labels remain TRAIN anchors')
    if a.router_protected_replay:
        execution['router_protected_replay'] = dict(version=1,
            batch='source/historical subset of existing reward reference; current sample excluded',
            targets='fresh current-bank correctness after paid Context update',
            weights='equal source/historical mass; equal class mass within segment',
            loss='0.5 paid-replay routing + 0.5 protected routing + original prior; one optimizer step',
            labels='existing TRAIN anchors and paid history only; no new queries for replay',
            history='pre-feedback snapshot; no protected samples stored as stale route replay')
        if a.router_protected_weight != .5:
            execution['router_protected_replay']['protected_fraction'] = a.router_protected_weight
            execution['router_protected_replay']['loss'] = (
                f'{1-a.router_protected_weight:g} paid-replay routing + '
                f'{a.router_protected_weight:g} protected routing + original prior; one optimizer step')
    if a.router_refresh:
        execution['router_refresh'] = dict(version=1, batch='existing pre-update reward reference',
            targets='current bank after paid Context update', steps_per_paid_label=1,
            history='snapshot before current feedback', extra_label_capacity=0)
    if a.router_informative_weight != 1.:
        execution['router_informative_weight'] = a.router_informative_weight
        execution['router_objective_version'] = 1
    run_id = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()[:12]
    root = a.output / 'Results' / ('evaluation_'+run_id)
    if root.exists():
        if a.skip_existing_completed and completed_evaluation(root, execution):
            print(f'Skipping completed evaluation: {root}', flush=True)
            return
        if a.skip_existing_completed:
            raise ValueError(f'Existing evaluation is incomplete or mismatched: {root}. '
                'Preserved without overwriting; use --evaluation-tag retry1 for a fresh attempt with the same seed.')
        raise ValueError(f'This exact evaluation already exists: {root}; use --skip-existing-completed '
            'to reuse complete results, or --evaluation-tag retry1 for a fresh same-seed attempt.')

    root.mkdir(parents=True, exist_ok=False)
    rows = []; audit_rows = []; forgetting_rows = []; oracle_summaries = []
    for mode in a.modes:
        legacy.core.set_seed(a.seed)
        cnn = load_cnn(models, artifact)
        frozen_weights = {name: value.detach().cpu().clone() for name, value in cnn.state_dict().items()}
        initial = torch.load(models/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
        bank = build_bank(models, artifact, config, initial, slots=a.context_count)
        agent = new_agent(a, cnn, artifact, saved=saved)
        continuation = {} if a.continual else None
        audit_matrix = []
        clean_scores = []
        oracle_audit_matrix = []; oracle_clean_scores = []
        def observe_stage(stage, after_level, history, used, cap, position):
            scores = []; oracle_scores = []
            for audit_level, (ax, ay) in [(0, clean), *audits.items()]:
                result = observer(cnn, bank, agent, ax, ay, history, used, cap,
                                  position, a.seed, config, mode)
                accuracy = float(np.mean(result[0] == ay))
                audit_rows.append(dict(Mode=mode, Stage=stage, AfterLevel=after_level,
                                       AuditLevel=audit_level, N=len(ay), Accuracy=100*accuracy))
                if a.routing_oracle_audit:
                    oracle_accuracy = float(np.any(result[2] == np.asarray(ay)[:, None], axis=1).mean())
                    audit_rows[-1]['OracleAccuracy'] = 100*oracle_accuracy
                    for k, value in enumerate(result[1]):
                        audit_rows[-1][f'Route{k}Accuracy'] = 100*value
                    if audit_level == 0: oracle_clean_scores.append(oracle_accuracy)
                    else: oracle_scores.append(oracle_accuracy)
                if audit_level == 0: clean_scores.append(accuracy)
                else: scores.append(accuracy)
            audit_matrix.append(scores)
            if a.routing_oracle_audit: oracle_audit_matrix.append(oracle_scores)
        if audits is not None:
            observe_stage(0, 0, DeviceReliability(len(a.classes), legacy.core.device, a.context_count+1),
                          0, int(a.budget_fraction*len(domains[a.levels[0]][1])), 0)
        for stage, level in enumerate(a.levels, start=1):
            if not a.continual: agent = new_agent(a, cnn, artifact, saved=saved)
            m, traces = episode(a, cnn, bank, initial, agent, domains[level], anchors,
                mode=mode, reference=clean, same_level=True, continuation=continuation,
                next_domain_features=next_sequence_features(a, domains, stage-1, a.seed) if a.continual else None,
                **({'domain_level': level} if a.router_level_balanced_history else {}))
            m.update(level=level, config_fingerprint=config.fingerprint, execution=execution,
                     classifier_unchanged=not cnn.training and all(not p.requires_grad for p in cnn.parameters())
                     and all(torch.equal(value.detach().cpu(), frozen_weights[name]) for name, value in cnn.state_dict().items()))
            if not m['classifier_unchanged']: raise RuntimeError('Frozen CNN parameters or buffers changed.')
            write_json(root/f'{mode}_level{level}_metrics.json', m)
            write_csv(root/f'{mode}_level{level}_trace.csv', traces)
            rows.append(summary_row(level, mode, m))
            print(f'Level {level} | {mode}: accuracy {100*m["accuracy"]:.2f}% | labels {m["labels"]}/{m["label_cap"]} | full bank {100*m["full_bank_fraction"]:.1f}% | {m["online_ms_per_sample"]:.2f} ms/row', flush=True)
            if audits is not None:
                observe_stage(stage, level, continuation['history'], m['labels'],
                              m['label_cap'], m['samples'])
        if audits is not None:
            matrix = np.asarray(audit_matrix); count = len(a.levels)
            forgetting = np.mean([matrix[j+1:, j].max()-matrix[-1, j] for j in range(count-1)])
            bwt = np.mean([matrix[-1, j]-matrix[j+1, j] for j in range(count-1)])
            forgetting_rows.append(dict(Mode=mode, FinalAudit=100*float(matrix[-1].mean()),
                Forgetting=100*float(forgetting), BWT=100*float(bwt),
                CleanInitial=100*clean_scores[0], CleanFinal=100*clean_scores[-1],
                CleanDrop=100*(clean_scores[0]-clean_scores[-1]),
                MeanOnlineAccuracy=float(np.mean([row['Accuracy'] for row in rows if row['Mode'] == mode])),
                Labels=sum(row['Labels'] for row in rows if row['Mode'] == mode)))
        if a.routing_oracle_audit:
            selected_rows = [row for row in rows if row['Mode'] == mode]
            observed = float(np.mean([row['Accuracy'] for row in selected_rows]))
            ceiling = float(np.mean([row['Oracle'] for row in selected_rows]))
            diagnostic = dict(Mode=mode, MeanStreamAccuracy=observed,
                MeanStreamRoutingOracle=ceiling, StreamRoutingGap=ceiling-observed,
                Labels=sum(row['Labels'] for row in selected_rows))
            if audits is not None:
                final_ceiling = 100*float(np.mean(oracle_audit_matrix[-1]))
                diagnostic.update(FinalAudit=forgetting_rows[-1]['FinalAudit'],
                    FinalRoutingOracleAudit=final_ceiling,
                    FinalRoutingGap=final_ceiling-forgetting_rows[-1]['FinalAudit'],
                    CleanRoutingOracleFinal=100*oracle_clean_scores[-1])
            oracle_summaries.append(diagnostic)
    if a.routing_oracle_audit:
        write_csv(root/'routing_oracle_summary.csv', oracle_summaries)
    write_csv(root/'combined_results.csv', rows)
    if audits is not None:
        write_csv(root/'sequential_audit_matrix.csv', audit_rows)
        write_csv(root/'sequential_summary.csv', forgetting_rows)
        execution['audit_source_ids'] = sorted(audit_ids)
    write_json(root/'execution.json', execution)
    print('Results:', root, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate', 'calibrate-utility'])
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
    p.add_argument('--warm-start-enhancement', action='store_true',
                   help='Reuse source enhancement Context and selector; skip Context pretraining and routing warmup.')
    p.add_argument('--joint-router', action='store_true',
                   help='Train a zero-initialized cross-route attention residual in the correctness head.')
    p.add_argument('--aligned-sequence', action='store_true',
                   help='Continue training with paid-only persistent TRAIN sequences and independent persistent VAL audits.')
    p.add_argument('--one-context-rl-training', action='store_true',
                   help='Specialize an inherited accuracy selector using one-Context paid-only policy-feedback sequences.')
    p.add_argument('--query-value-only-training', action='store_true',
                   help='Dedicated one-Context TRAIN: fit only update_value; freeze inherited correctness and future heads.')
    p.add_argument('--paired-query-only-training', action='store_true',
                   help='Query-value-only TRAIN: optimize the paired advantage objective without absolute TD.')
    p.add_argument('--connected-sequence-td', action='store_true',
                   help='Paired-only TRAIN: bridge persistent levels using next-domain adaptation features only.')
    p.add_argument('--counterfactual-utility', action='store_true',
                   help='TRAIN only: fit update values using three isolated paid-label Context updates; freeze other selector heads.')
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
    p.add_argument('--router-protected-weight', type=float, default=.5, help='Protected share, or relative weight when --router-normal-weight is supplied.')
    p.add_argument('--router-normal-weight', type=float, help='Optional nonnegative relative paid-replay weight; both weights are normalized to sum to one.')
    p.add_argument('--router-level-balanced-history', action='store_true', help='Balance routing-only historical replay by known acquisition level and class.')
    p.add_argument('--router-protected-replay', action='store_true', help='Mix current-bank source/historical supervision into the paid routing optimizer step.')
    p.add_argument('--routing-oracle-audit', action='store_true', help='Observer-only routing ceiling on unchanged learner trajectories.')
    p.add_argument('--paid-replay-mode', choices=['off', 'sequential', 'batch', 'batch_all'], default='off')
    p.add_argument('--modes', nargs='+', choices=['learned', 'random', 'random_query', 'random_query_online'], default=['learned', 'random'])
    p.add_argument('--query-schedule', choices=['policy', 'uniform'], default='policy')
    p.add_argument('--update-policy', choices=['learned', 'fixed'], default='learned')
    p.add_argument('--max-reference-drop', type=float, default=2.)
    p.add_argument('--retention-selection-weight', type=float, default=1.)
    p.add_argument('--bank-executor', choices=['sequential', 'vmap'])
    p.add_argument('--routed-utility', action='store_true')
    p.add_argument('--context-count', type=int, choices=[1,2,3,4,5], default=3, help='Context-count evaluation ablation, or one-Context dedicated RL training.')
    p.add_argument('--context-damage-guard', action='store_true', help='Single-Context evaluation: accept only protected-reference-safe online Context steps.')
    p.add_argument('--online-memory-ce-weight', type=float,
                   help='Single-Context uniform-feedback evaluation: override protected-replay CE weight online only.')
    p.add_argument('--continual', action='store_true')
    p.add_argument('--audit-fraction', type=float, default=.2)
    p.add_argument('--skip-existing-completed', action='store_true', help='Skip only matching complete evaluations; preserve and reject partial results.')
    p.add_argument('--evaluation-tag', default='', help='Unique repeat label for identical-trajectory timing repetitions.')
    p.add_argument('--router-refresh', action='store_true',
                   help='After paid feedback, recalibrate routing on existing references with current-bank targets.')
    p.add_argument('--router-informative-weight', type=float, default=1.,
                   help='Online routing loss multiplier for paid rows with both correct and incorrect routes (default 1).')
    p.add_argument('--utility-calibration', type=Path,
                   help='TRAIN-only scale JSON: output for calibrate-utility, input for evaluate.')
    p.add_argument('--utility-weight', type=float, default=1.)
    p.add_argument('--utility-clip', type=float, default=1.)
    for name, value_type in CONFIG_OPTIONS.items():
        p.add_argument('--'+name.replace('_', '-'), type=value_type)
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if any(m in a.modes for m in ('random_query', 'random_query_online')) and (a.command != 'evaluate' or a.context_count != 1
            or a.query_schedule != 'policy' or a.update_policy != 'learned'
            or a.router_refresh or a.context_damage_guard):
        p.error('random_query requires single-Context policy-schedule evaluation without other routing ablations.')
    if (a.router_normal_weight is not None or a.router_protected_weight != .5) and not a.router_protected_replay:
        p.error('Routing weights require --router-protected-replay.')
    try:
        a.router_protected_weight = protected_routing_fraction(a.router_protected_weight, a.router_normal_weight)
    except ValueError as error:
        p.error(str(error))
    if a.router_level_balanced_history and (not a.router_protected_replay or a.router_protected_weight <= 0):
        p.error('--router-level-balanced-history requires active protected routing replay.')
    if a.router_protected_replay and (a.command != 'evaluate' or a.context_count != 1
            or a.modes != ['learned'] or a.query_schedule != 'policy' or a.update_policy != 'learned'
            or a.router_refresh or a.context_damage_guard or a.utility_calibration is not None
            or a.online_memory_ce_weight is not None or a.router_informative_weight != 1.):
        p.error('Protected routing replay is an isolated single-Context learned policy evaluation ablation.')
    if a.routing_oracle_audit and a.command != 'evaluate':
        p.error('--routing-oracle-audit is evaluation-only.')
    if a.paid_replay_mode != 'off':
        if (a.command != 'evaluate' or a.context_count != 1 or a.modes != ['learned']
                or a.query_schedule != 'policy' or a.update_policy != 'learned'
                or a.context_damage_guard or a.router_protected_replay or a.routed_utility
                or a.online_memory_ce_weight is not None):
            p.error('Paid replay requires isolated one-Context learned policy evaluation.')
    if a.paired_query_only_training and not a.query_value_only_training:
        p.error('--paired-query-only-training requires --query-value-only-training.')
    if a.connected_sequence_td and not a.paired_query_only_training:
        p.error('--connected-sequence-td is configured through paired-only training checkpoints.')
    if a.query_value_only_training and not a.one_context_rl_training:
        p.error('--query-value-only-training requires --one-context-rl-training.')
    if a.one_context_rl_training:
        if (a.command != 'train' or not a.warm_start_enhancement or not a.aligned_sequence
                or a.context_count != 1 or a.query_schedule != 'policy' or a.update_policy != 'learned'):
            p.error('One-Context RL training requires train --warm-start-enhancement --aligned-sequence --context-count 1 --query-schedule policy --update-policy learned.')
        if (a.profile != 'accuracy' or a.joint_router or a.disable or a.routed_utility
                or a.counterfactual_utility or a.utility_calibration is not None):
            p.error('Specialize the unchanged accuracy selector without other architecture or utility ablations.')
    if a.context_damage_guard:
        if a.command != 'evaluate' or a.context_count != 1 or a.update_policy != 'learned':
            p.error('Damage guard is a single-Context evaluation ablation with learned updates.')
        if a.online_memory_ce_weight is not None or a.utility_calibration is not None or a.router_refresh or a.router_informative_weight != 1.:
            p.error('Test damage guard separately, retaining the original replay weight and routing settings.')
    if a.online_memory_ce_weight is not None:
        if not np.isfinite(a.online_memory_ce_weight) or a.online_memory_ce_weight < 0:
            p.error('Online memory CE weight must be finite and nonnegative.')
        if a.command != 'evaluate' or a.context_count != 1 or a.query_schedule != 'uniform' or a.update_policy != 'learned':
            p.error('Online replay weighting is a single-Context uniform-feedback evaluation ablation.')
        if a.utility_calibration is not None or a.router_refresh or a.router_informative_weight != 1.:
            p.error('Test online replay weighting separately from other routing/utility ablations.')
    if a.context_count != 3 and not a.one_context_rl_training and (a.command != 'evaluate' or a.utility_calibration is not None):
        p.error('--context-count other than 3 is an evaluation ablation without utility normalization.')
    if a.counterfactual_utility:
        if a.command != 'train' or not a.warm_start_enhancement or not a.aligned_sequence:
            p.error('--counterfactual-utility requires train --warm-start-enhancement --aligned-sequence.')
        if a.profile != 'accuracy' or a.joint_router or a.disable or a.routed_utility:
            p.error('Counterfactual utility requires the unchanged accuracy architecture and protected references.')
        if a.utility_calibration is not None or a.router_refresh or a.router_informative_weight != 1.:
            p.error('Test counterfactual utility independently of actor scaling and routing ablations.')
    if not np.isfinite(a.utility_weight) or a.utility_weight < 0 or not np.isfinite(a.utility_clip) or a.utility_clip <= 0:
        p.error('Utility weight must be finite/nonnegative; clip finite/positive.')
    if a.utility_calibration is not None or a.command == 'calibrate-utility':
        if a.command == 'train' or a.query_schedule != 'uniform' or a.update_policy != 'learned':
            p.error('Utility normalization requires uniform feedback, learned destinations and an existing checkpoint.')
        if a.router_refresh or a.router_informative_weight != 1.:
            p.error('Test utility normalization separately from routing ablations.')
    elif a.utility_weight != 1. or a.utility_clip != 1.:
        p.error('Utility normalization weights require --utility-calibration.')
    if a.aligned_sequence and (a.command != 'train' or not a.warm_start_enhancement):
        p.error('--aligned-sequence requires train --warm-start-enhancement.')
    if a.aligned_sequence and not a.one_context_rl_training and (a.query_schedule != 'uniform' or a.update_policy != 'learned'):
        p.error('Aligned sequences require uniform queries and learned update destinations.')
    if a.warm_start_enhancement and (a.command != 'train' or a.warm_start_selector):
        p.error('--warm-start-enhancement is train-only and cannot combine with --warm-start-selector.')
    if not np.isfinite(a.router_informative_weight) or a.router_informative_weight < 1:
        p.error('--router-informative-weight must be finite and at least 1.')
    if a.router_informative_weight != 1. and a.command != 'evaluate':
        p.error('--router-informative-weight is an online evaluation adaptation ablation.')
    if a.router_informative_weight != 1. and a.router_refresh:
        p.error('Test informative routing separately from --router-refresh.')
    if a.router_refresh and a.command != 'evaluate': p.error('--router-refresh is an online evaluation adaptation ablation.')
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
    a.levels = a.levels or ([0, 1, 3] if a.command in ('train','calibrate-utility') else [5, 7, 10, 12, 15, 17, 20])
    if len(a.levels) != len(set(a.levels)) or min(a.levels) < 0: p.error('Levels must be unique nonnegative integers.')
    if a.continual and len(a.levels) < 2: p.error('A continual audit requires at least two domains.')
    if a.aligned_sequence and len(a.levels) < 2: p.error('Aligned training requires at least two TRAIN domains.')
    a.bank_utility = False; a.balanced_reward_reference = True
    if a.command == 'calibrate-utility':
        calibrate_utility(a)
    elif a.command == 'train':
        from dataclasses import replace
        config = EnhancementConfig.preset(a.profile or 'compact')
        config = replace(config, bank_executor=a.bank_executor or config.bank_executor, routed_utility=a.routed_utility,
                         joint_router=a.joint_router,
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

