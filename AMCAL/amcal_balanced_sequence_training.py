"""Persistent continuation training for the ORIGINAL balanced-reference v1 bank.

No enhancement agent, protected source-anchor memory, attention or new rewards.
TRAIN uses the original online paid-feedback mechanics. VAL audit identities
are excluded from adaptation at every level; checkpoint selection never uses TEST.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import joblib
import numpy as np
import torch
import amcal_burst_structured_bank_protocol as bank

SOURCE_FILES = ('initial_cnn_model.pth', 'adversarial_best_context.pth', 'protocol_preprocessing.joblib')


def split_validation(domains, clean, seed, fraction):
    common = sorted(set.intersection(*(set(map(int, d[2])) for d in domains.values()),
                                     set(map(int, clean[2]))))
    if len(common) < 10: raise ValueError('Insufficient common VAL source identities.')
    order = np.random.default_rng(seed+8121).permutation(common)
    audit_ids = set(map(int, order[:max(1, int(len(order)*fraction))]))
    streams, audits = {}, {}
    for level, (x, y, ids) in domains.items():
        mask = np.array([int(i) in audit_ids for i in ids])
        streams[level] = x[~mask], np.asarray(y)[~mask], np.asarray(ids)[~mask]
        audits[level] = x[mask], np.asarray(y)[mask]
        if not mask.any() or mask.all(): raise ValueError('Empty validation adaptation/audit split.')
    x, y, ids = clean
    mask = np.array([int(i) in audit_ids for i in ids])
    return streams, audits, (x[mask], np.asarray(y)[mask]), sorted(audit_ids)


def run_sequence(settings, cnn, contexts, initial, agent, domains, levels, seed):
    continuation = {}; stages = []
    for level in levels:
        # This is a TRAIN rollout with the unchanged online learning gates:
        # paid labels only, PaidHistory reference, and online optimizer settings.
        m, _ = bank.episode(settings, cnn, contexts, initial, agent, domains[level],
                            learning=True, rollout_seed=seed, continuation=continuation)
        stages.append(dict(level=level, accuracy=m['accuracy'], labels=m['labels'],
            cap=m['label_cap'], route_updates=m['route_updates'], td_updates=m['td_updates']))
    return stages, continuation


def validation_sequence(settings, models, artifact, policy, streams, audits, clean, levels):
    cnn, contexts, initial, width = bank.models_from_source(models, artifact)
    agent = bank.Agent(width, bank.core.device, settings.seed, policy_state=policy, gamma=settings.gamma)
    continuation = {}; scores = []; clean_scores = []; stages = []
    def observe(history, used, cap, position):
        row = []
        for level in levels:
            x, y = audits[level]
            pred, _ = bank.reference_observer(cnn, contexts, agent, x, y, history,
                                               used, cap, position, 'learned', settings.seed)
            row.append(float(np.mean(pred == y)))
        pred, _ = bank.reference_observer(cnn, contexts, agent, *clean, history,
                                           used, cap, position, 'learned', settings.seed)
        scores.append(row); clean_scores.append(float(np.mean(pred == clean[1])))
    observe(bank.Reliability(len(settings.classes)), 0,
            int(settings.budget_fraction*len(streams[levels[0]][1])), 0)
    for level in levels:
        m, _ = bank.episode(settings, cnn, contexts, initial, agent, streams[level], continuation=continuation)
        stages.append(dict(level=level, accuracy=m['accuracy'], labels=m['labels'], cap=m['label_cap']))
        observe(continuation['history'], m['labels'], m['label_cap'], m['samples'])
    matrix = np.asarray(scores); k = len(levels)
    return dict(final_audit_accuracy=float(matrix[-1].mean()),
        online_accuracy=float(np.mean([s['accuracy'] for s in stages])),
        clean_drop_pp=100*(clean_scores[0]-clean_scores[-1]),
        forgetting_pp=100*float(np.mean([matrix[j+1:, j].max()-matrix[-1, j] for j in range(k-1)])),
        bwt_pp=100*float(np.mean([matrix[-1, j]-matrix[j+1, j] for j in range(k-1)])),
        labels=sum(s['labels'] for s in stages), stages=stages,
        audit_accuracy_by_stage=scores, clean_accuracy_by_stage=clean_scores)


def train(a):
    source = a.source_run.resolve()/'Models'
    parent = torch.load(source/'structured_bank_ddqn.pth', map_location='cpu', weights_only=False)
    if parent.get('version') != 1 or parent.get('config', {}).get('decoupled_query', False):
        raise ValueError('Use the original coupled balanced-reference v1 checkpoint.')
    if not parent.get('config', {}).get('balanced_reward_reference', False):
        raise ValueError('Source checkpoint is not the finalized balanced-reference method.')
    if set(parent['training_levels']) != set(a.levels): raise ValueError('TRAIN levels must match the parent.')
    for name, digest in parent['model_hashes'].items():
        if bank.core.file_sha256(source/name) != digest: raise ValueError('Source model hash mismatch: '+name)
    artifact = joblib.load(source/'protocol_preprocessing.joblib')
    partitions = [np.asarray(artifact[k], dtype=np.int64) for k in ('train_ids', 'val_ids', 'test_ids')]
    ids = np.concatenate(partitions)
    if (any(not len(d) for d in partitions) or len(np.unique(ids)) != len(ids)
            or (ids < 0).any() or (ids >= artifact['source_rows']).any()):
        raise ValueError('Invalid or overlapping source partitions.')
    clean_file = a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'
    if bank.core.file_sha256(clean_file) != artifact['source_sha256']: raise ValueError('Clean source provenance mismatch.')
    provenance = {str(level): bank.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
                  for level in a.levels}
    if provenance != parent['training_domain_sha256']: raise ValueError('TRAIN domain provenance mismatch.')
    if a.output.exists(): raise ValueError('Use a fresh output directory.')
    models = a.output/'Models'; models.mkdir(parents=True)
    for name in SOURCE_FILES: shutil.copy2(source/name, models/name)
    cfg = dict(parent['config'])
    cfg.update(seed=a.seed, levels=a.levels, epochs=a.epochs, patience=a.patience,
        command='train_sequence', burst_dir=str(a.burst_dir.resolve()), audit_fraction=a.audit_fraction,
        router_warmup_episodes=0, resume_selector=True, train_epsilon=.05,
        classes=[str(c) for c in artifact['label_encoder'].classes_],
        context_lr=.0005, budget_fraction=.2, eval_epsilon=.05, context_alpha=cfg.get('context_alpha', .8),
        query_schedule=a.query_schedule, update_policy='learned', bank_utility=False,
        budget_aware_policy=False, decoupled_query=False, balanced_reward_reference=True,
        source_run=str(a.source_run.resolve()), output=str(a.output.resolve()), persistent_training=True)
    settings = SimpleNamespace(**cfg)
    bank.core.SELECTION_COST = settings.selection_cost
    bank.core.GAIN_WEIGHT, bank.core.RETENTION_WEIGHT = settings.gain_weight, settings.retention_weight
    bank.core.REFERENCE_BATCH_SIZE, bank.core.REFERENCE_HISTORY_SIZE = 32, 256
    cnn, contexts, initial, width = bank.models_from_source(models, artifact)
    frozen = {k: v.detach().cpu().clone() for k, v in cnn.state_dict().items()}
    agent = bank.Agent(width, bank.core.device, a.seed, policy_state=parent['policy_state_dict'], gamma=settings.gamma)
    domains = {level: bank.read_domain(a.burst_dir, artifact, level, 'train_ids') for level in a.levels}
    val = {level: bank.read_domain(a.burst_dir, artifact, level, 'val_ids') for level in a.levels}
    clean = bank.read_domain(a.burst_dir, artifact, 0, 'val_ids')
    streams, audits, clean_audit, audit_ids = split_validation(val, clean, a.seed, a.audit_fraction)
    protocol = dict(version=1, method='original balanced-reference coupled v1',
        train_order=a.levels, query_schedule=a.query_schedule, paid_only_supervision=True,
        optimizer='original online settings', reference='original balanced PaidHistory, capacity 256, sample 32',
        reset='Context/optimizers/reference reset per pass; retained across domains within a pass',
        validation_audit_source_ids=audit_ids,
        validation_adaptation_source_ids={str(level): list(map(int, d[2])) for level, d in streams.items()},
        selection='final persistent VAL audit accuracy; no TEST selection')
    hashes = {name: bank.core.file_sha256(models/name) for name in SOURCE_FILES}
    parent_hash = bank.core.file_sha256(source/'structured_bank_ddqn.pth')
    best = -float('inf'); stale = 0; log = []
    for epoch in range(a.epochs+1):
        stages = []
        if epoch:
            agent.route_memory.clear(); agent.td_memory.clear(); agent.pair_memory.clear()
            stages, _ = run_sequence(settings, cnn, contexts, initial, agent, domains, a.levels,
                                     a.seed+200003+1009*(epoch-1))
        snapshot = deepcopy(agent.policy.state_dict())
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            validation = validation_sequence(settings, models, artifact, snapshot, streams, audits,
                                               clean_audit, a.levels)
        # The original balanced method has no added retention-selection penalty.
        score = validation['final_audit_accuracy']
        log.append(dict(epoch=epoch, train_stages=stages, train_labels=sum(s['labels'] for s in stages),
                        validation_accuracy=score, selection_score=score, validation_sequence=validation))
        if not all(torch.equal(v.detach().cpu(), frozen[k]) for k, v in cnn.state_dict().items()):
            raise RuntimeError('Frozen CNN parameters or buffers changed.')
        print(f'Balanced sequence pass {epoch}/{a.epochs} | final VAL audit {100*score:.2f}% | '
              f'VAL online {100*validation["online_accuracy"]:.2f}% | TRAIN labels {log[-1]["train_labels"]}', flush=True)
        if score > best:
            best = score; stale = 0
            checkpoint = dict(parent)
            checkpoint.update(version=1, slots=bank.SLOTS, actions=bank.ACTIONS, route_width=width,
                policy_state_dict=snapshot, config=cfg, model_hashes=hashes,
                training_levels=a.levels, training_domain_sha256=provenance, validation_accuracy=score,
                selected_epoch=epoch, parent_checkpoint_sha256=parent_hash, training_protocol=protocol,
                supervision='TRAIN continuation / validation / online: paid labels only; original balanced PaidHistory')
            torch.save(checkpoint, models/'structured_bank_ddqn.pth')
        else:
            stale += 1
            if stale >= a.patience: break
    (a.output/'structured_sequence_training.json').write_text(json.dumps(dict(
        config=cfg, training_protocol=protocol, epochs=log, best_validation=best,
        selected_epoch=checkpoint['selected_epoch'], parent_checkpoint_sha256=parent_hash), indent=2))
    print('Saved original balanced-reference selector:', models/'structured_bank_ddqn.pth', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--burst-dir', type=Path, required=True)
    p.add_argument('--levels', nargs='+', type=int, default=[0, 1, 3])
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--audit-fraction', type=float, default=.2)
    p.add_argument('--query-schedule', choices=['policy', 'uniform'], default='uniform')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    a = p.parse_args(argv)
    if min(a.epochs, a.patience) < 1: p.error('Epochs and patience must be positive.')
    if len(set(a.levels)) != len(a.levels) or len(a.levels) < 2 or min(a.levels) < 0:
        p.error('At least two distinct nonnegative TRAIN levels required.')
    if not .05 <= a.audit_fraction <= .5: p.error('Audit fraction must be .05.. .5.')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA is unavailable.')
    bank.core.device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                                    else 'cpu' if a.device == 'auto' else a.device)
    bank.core.set_seed(a.seed)
    train(a)


if __name__ == '__main__': main()
