"""Opt-in adversarial query experiment; existing PairOnly entry points are untouched.

TRAIN 0/1/3, VAL-only checkpoint selection, independent TEST 5/7/10/12/15/17/20.
This is a new method variant, not a drop-in evaluation of an old checkpoint.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import amcal_final_enhancement as base
from enhancements.agent import DeviceReliability
from enhancements.config import EnhancementConfig
from enhancements.models import BankExecutor
from enhancements.state import build_state
from enhancements.memory import ProtectedMemory
from enhancements.minimax_query import (QueryMaximizer, maximize_step, context_min_step,
    counterfactual_step, CounterfactualWorkspace, query_target_allowed)

VERSION = 2
TRAIN_LEVELS = [0, 1, 3]
TEST_LEVELS = [5, 7, 10, 12, 15, 17, 20]
SETTING_KEYS = ['horizon', 'cost', 'gain_weight', 'forgetting_weight', 'context_lr',
    'query_lr', 'replay_weight', 'budget_fraction', 'seed', 'validation_retention_weight',
    'epochs', 'patience', 'reference_size']


def digest(path):
    return base.legacy.core.file_sha256(path)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    base.write_json(temporary, value)
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    base.write_csv(temporary, rows)
    temporary.replace(path)


def metrics(y, predicted, classes):
    pr, rc, f1, _ = precision_recall_fscore_support(y, predicted,
        labels=list(range(classes)), average='macro', zero_division=0)
    return dict(AC=100 * accuracy_score(y, predicted), PR=100 * pr,
        RC=100 * rc, F1=100 * f1)


def subset(domain, mask):
    x, y, ids = domain
    at = torch.as_tensor(np.flatnonzero(mask), device=x.device)
    return x[at], np.asarray(y)[mask], np.asarray(ids)[mask]


def pilot_indices(labels, limit, seed):
    """Seeded pilot subset, retaining up to three rows per available class."""
    labels = np.asarray(labels)
    if limit < 0: raise ValueError('Pilot row limit must be nonnegative.')
    if limit == 0 or len(labels) <= limit: return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        chosen.extend(rng.permutation(indices)[:min(3, len(indices))].tolist())
    if len(chosen) > limit: raise ValueError('Pilot row limit is too small to retain the classes.')
    remaining = np.setdiff1d(np.arange(len(labels)), chosen)
    chosen.extend(rng.permutation(remaining)[:limit-len(chosen)].tolist())
    return np.sort(chosen)


def pilot_domain(a, domain, offset):
    limit = getattr(a, 'pilot_rows', 0)
    if not limit: return domain
    selected = pilot_indices(domain[1], limit, a.seed+offset)
    mask = np.zeros(len(domain[1]), dtype=bool); mask[selected] = True
    return subset(domain, mask)


def pilot_spec(a):
    rows = getattr(a, 'pilot_rows', 0)
    return dict(version=1, rows_per_domain=rows,
        scope='small-data diagnostic, not full benchmark; TRAIN/VAL/TEST capped separately',
        sampling='fixed seeded subset with up to three guaranteed rows per class; remaining slots sampled uniformly',
        cap='floor(0.2 * actual sampled stream size) under the 20% budget',
        initialization='existing pretrained CNN/Context/router; only the experimental streams are reduced') if rows else None


def reference_partition(domain, seed, fraction=.1):
    """Reserve source IDs across all training intensities, stratified by class."""
    _, y, ids = domain
    rng = np.random.default_rng(seed + 901)
    chosen = []
    for cls in np.unique(y):
        indices = np.flatnonzero(np.asarray(y) == cls)
        if len(indices) < 3:
            raise ValueError('At least three source TRAIN rows per class are required.')
        chosen.extend(rng.permutation(indices)[:max(1, int(len(indices) * fraction))])
    selected = set(int(ids[i]) for i in chosen)
    mask = np.asarray([int(i) in selected for i in ids])
    held = subset(domain, mask)
    return subset(domain, ~mask), held, selected


def feature_key(row):
    return row.detach().cpu().contiguous().numpy().tobytes()


def feature_keys(x):
    """One device transfer for a domain, rather than one per feature comparison."""
    return [row.tobytes() for row in x.detach().cpu().contiguous().numpy()]


def balanced_draw(labels, count, rng):
    labels = np.asarray(labels)
    queues = {int(c): rng.permutation(np.flatnonzero(labels == c)).tolist() for c in np.unique(labels)}
    chosen = []
    while len(chosen) < min(count, len(labels)):
        for cls in rng.permutation(list(queues)):
            if queues[cls]: chosen.append(queues[cls].pop())
            if len(chosen) == min(count, len(labels)): break
    return chosen


def validate_trace(rows, cap):
    used = 0
    for position, row in enumerate(rows, start=1):
        if row['step'] != position or row['route'] not in (0, 1):
            raise RuntimeError('Invalid stream step or route.')
        if row['queried'] != (row['action'] == 2):
            raise RuntimeError('Query action and feedback flag disagree.')
        if not row['queried'] and row['action'] != row['route']:
            raise RuntimeError('Skip action and prediction route disagree.')
        used += int(row['queried'])
        if used > cap or row['labels_used'] != used:
            raise RuntimeError('Feedback usage violates the hard cap or causal accounting.')
        best = 0 if row['frozen_route_score'] >= row['context_route_score'] else 1
        if row['route'] != best:
            raise RuntimeError('Prediction did not use the best pre-feedback route score.')
        key = 'frozen_prediction' if row['route'] == 0 else 'context_prediction'
        if row['prediction'] != row[key]:
            raise RuntimeError('Recorded prediction differs from its pre-feedback route.')


class Replay:
    """TRAIN anchors and paid feedback only; exclude IDs and feature duplicates."""
    def __init__(self, anchors, seed, config, classes):
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.storage = ProtectedMemory(anchors[:2], classes, config, seed)
        self.ids = {}
        for key, source_id in zip(feature_keys(anchors[0]), anchors[2]):
            self.ids.setdefault(key, set()).add(int(source_id))

    def inventory(self, rows):
        return [(self.ids[r[2]], r[0], r[1], r[2]) for r in rows]

    def draw(self, current_id, current, future_ids=(), future_x=None, reference_keys=(),
             current_key=None, future_keys=None):
        blocked_ids = {int(current_id), *map(int, future_ids)}
        blocked_keys = {current_key if current_key is not None else feature_key(current), *reference_keys}
        if future_keys is not None: blocked_keys.update(future_keys)
        elif future_x is not None: blocked_keys.update(feature_keys(future_x))
        def take(pool, count, balanced=False):
            eligible = []
            seen = set(blocked_keys)
            for row in pool:
                if not row[0].intersection(blocked_ids) and row[3] not in seen:
                    eligible.append(row); seen.add(row[3])
            if not eligible: return []
            indices = (balanced_draw([r[2] for r in eligible], count, self.rng) if balanced else
                self.rng.choice(len(eligible), min(count, len(eligible)), replace=False))
            return [eligible[int(i)] for i in indices]
        stored_paid = [*self.storage.recent, *(r for group in self.storage.historical for r in group)]
        paid = take(self.inventory(stored_paid), 4)
        # The two replay pools must also be disjoint from each other.
        blocked_ids.update(i for r in paid for i in r[0])
        blocked_keys.update(r[3] for r in paid)
        old = [*self.storage.source, *(r for group in self.storage.historical for r in group)]
        source = take(self.inventory(old), self.config.memory_replay_size, balanced=True)
        def tensors(rows):
            if not rows: return None
            return torch.cat([r[1] for r in rows]).to(current.device), torch.tensor([r[2] for r in rows], device=current.device)
        return [(r[1].to(current.device), torch.tensor([r[2]], device=current.device)) for r in paid], tensors(source)

    def add(self, source_id, x, truth):
        key = feature_key(x)
        self.storage.add(x, torch.tensor([truth], device=x.device))
        self.ids.setdefault(key, set()).add(int(source_id))
        retained = {r[2] for r in self.storage.source}
        retained.update(r[2] for r in self.storage.recent)
        retained.update(r[2] for group in self.storage.historical for r in group)
        self.ids = {k: v for k, v in self.ids.items() if k in retained}


def new_runtime(a, artifact, saved, context_state, query_state=None):
    cnn = base.load_cnn(a.models, artifact)
    context = base.build_bank(a.models, artifact, a.config, context_state, slots=1)[0]
    agent = base.new_agent(a, cnn, artifact, saved=saved)
    # Only routing heads may change online. There is no TD/future learning here.
    agent.policy.requires_grad_(False)
    for parameter in agent.policy.routing_parameters(): parameter.requires_grad_(True)
    width = 2 * agent.policy.route_width + 3
    experiment = getattr(a, 'actor_experiment', None)
    if experiment:
        from enhancements.standardized_actor import StandardizedQueryMaximizer
        query = StandardizedQueryMaximizer(width, experiment['normalizer']).to(base.legacy.core.device)
    else:
        query = QueryMaximizer(width).to(base.legacy.core.device)
    if query_state is not None: query.load_state_dict(query_state)
    return cnn, context, agent, query


@torch.no_grad()
def endpoint(a, cnn, context, agent, domain, history, used, cap, position):
    x, y = domain[:2]
    z, h = base.legacy.base_cache(cnn, x)
    executor = BankExecutor(cnn, [context])
    predicted = []
    route_rng = np.random.default_rng(getattr(a, 'seed', 0)+811)
    for start in range(0, len(x), 64):
        out = executor.outputs(x[start:start+64], z[start:start+64])
        state = build_state(out, h[start:start+64], history, used, cap, position, max(position, 1), -1)
        route = prediction_route_scores(a, agent, state, out[0], route_rng).argmax(1)
        per_route = out[0].argmax(2)
        predicted.extend(per_route[torch.arange(len(route), device=x.device), route].cpu().tolist())
    return metrics(y, predicted, len(a.classes))


def uniform_query_at(position, count, cap):
    """Deterministic evenly spaced requests, exact cap, no labels/features used."""
    if count < 1 or not 0 <= cap <= count or not 0 <= position < count:
        raise ValueError('Invalid uniform query schedule arguments.')
    before = (2*position*cap+count)//(2*count)
    after = (2*(position+1)*cap+count)//(2*count)
    return after > before


def normalized_entropy(logits):
    """Predictive entropy in [0,1], with no labels or future rows."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError('Entropy requires finite logits for at least two classes.')
    shifted = values - values.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return np.clip(-(np.exp(logp)*logp).sum(axis=-1)/math.log(values.shape[-1]), 0., 1.)


def prediction_route_scores(a, agent, state, logits, rng=None):
    """Larger is better; routing controls never call the learned router."""
    if getattr(a, 'eval_routing', 'policy') == 'random':
        if rng is None: raise ValueError('Random routing requires its own seeded generator.')
        chosen = rng.integers(0, 2, size=len(logits))
        values = np.column_stack((chosen == 0, chosen == 1))
        return torch.as_tensor(values, dtype=logits.dtype, device=logits.device)
    if getattr(a, 'eval_routing', 'policy') == 'uncertainty':
        values = -normalized_entropy(logits.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=logits.dtype, device=logits.device)
    return agent.policy.route_logits(state)


def entropy_cutoff(scores, percentile):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Calibration requires a nonempty finite entropy vector.')
    if not math.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError('Uncertainty percentile must be between zero and 100.')
    return float(np.percentile(values, percentile))


@torch.no_grad()
def calibrate_uncertainty(a, runtime, domains):
    """Fixed initial-runtime VAL predictions; no labels or adaptive updates."""
    cnn, context, agent, _ = runtime
    scores, domain_ids = [], {}
    executor = BankExecutor(cnn, [context])
    for level, domain in domains.items():
        x, _, ids = domain
        n, cap = len(x), int(a.budget_fraction*len(x))
        history = DeviceReliability(len(a.classes), x.device, routes=2)
        z, h = base.legacy.base_cache(cnn, x)
        for start in range(0, n, 64):
            out = executor.outputs(x[start:start+64], z[start:start+64])
            state = build_state(out, h[start:start+64], history, 0, cap, 0, n, -1)
            routing = prediction_route_scores(a, agent, state, out[0])
            if not torch.isfinite(routing).all(): raise ValueError('Nonfinite calibration routing scores.')
            route = routing.argmax(1)
            logits = out[0][torch.arange(len(route), device=x.device), route]
            scores.extend(normalized_entropy(logits.cpu().numpy()).tolist())
        domain_ids[str(level)] = list(map(int, ids))
    threshold = entropy_cutoff(scores, a.uncertainty_percentile)
    return dict(mode='uncertainty', version=1, score='normalized entropy of selected prediction route',
        percentile=a.uncertainty_percentile, threshold=threshold, calibration_rows=len(scores),
        calibration_levels=list(domains), calibration_ids=domain_ids,
        calibration='VAL features only, saved initial checkpoint; empty feedback history, used=0, position=0; no updates',
        rule='request when current selected-route entropy >= fixed VAL threshold, subject to remaining cap',
        scope=('query and routing via entropy; paid-feedback Context updates; learned router fixed and unused'
            if getattr(a, 'eval_routing', 'policy') == 'uncertainty' else
            'query decision only; existing route argmax and paid-feedback Context/router updates'),
        actor_probability=('not evaluated; trace query_probability=0 is an unused placeholder'
            if getattr(a, 'eval_routing', 'policy') == 'uncertainty' else
            'observer only; ignored for requests'),
        budget='20% maximum per intensity; no forced consumption; all adaptive state reset')


def stream(a, runtime, domain, anchors, reference, clean, seed, training=False,
           training_state=None, target_endpoints=True, diagnostic_sink=None, max_steps=None):
    cnn, context, agent, query = runtime
    core = base.legacy.core
    x, y, ids = domain
    order = np.random.default_rng(seed).permutation(len(y))
    x, y, ids = x[torch.as_tensor(order, device=x.device)], np.asarray(y)[order], np.asarray(ids)[order]
    n, cap = len(y), int(a.budget_fraction * len(y))
    if n == 0: raise ValueError('Empty stream.')
    evaluation_schedule = getattr(a, 'eval_query_schedule', 'policy')
    training_schedule = getattr(a, 'train_query_schedule', 'policy')
    if evaluation_schedule not in ('policy', 'uniform', 'uncertainty', 'frontloaded') or training_schedule not in ('policy', 'uniform'):
        raise ValueError('Unknown query schedule.')
    if training and evaluation_schedule != 'policy':
        raise ValueError('Query overrides are evaluation-only; TRAIN must use its configured collection policy.')
    if training and getattr(a, 'eval_routing', 'policy') != 'policy':
        raise ValueError('Routing overrides are evaluation-only.')
    query_schedule = training_schedule if training else evaluation_schedule
    if query_schedule == 'uncertainty':
        threshold = getattr(a, 'uncertainty_threshold', None)
        if threshold is None or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('A finite VAL-calibrated uncertainty threshold is required.')
    if diagnostic_sink is not None and not training:
        raise ValueError('The TRAIN diagnostic recorder must not collect TEST/VAL targets.')
    steps = n if max_steps is None else min(n, max_steps)
    if steps < 1: raise ValueError('Diagnostic prefix must be positive.')
    if training and reference is None: raise ValueError('TRAIN requires an independent forgetting reference.')
    keys = feature_keys(x)
    history = DeviceReliability(len(a.classes), x.device, routes=2)
    memory = Replay(anchors, seed + 61, a.config, len(a.classes))
    if training_state is None: training_state = {}
    optimizer = training_state.get('context_optimizer') if training else None
    if optimizer is None: optimizer = torch.optim.Adam(context.parameters(), lr=a.context_lr, weight_decay=1e-4)
    query_optimizer = training_state.get('query_optimizer') if training else None
    if training and query_optimizer is None:
        query_optimizer = torch.optim.Adam(query.parameters(), lr=a.query_lr, weight_decay=1e-4)
    workspace = training_state.get('workspace', CounterfactualWorkspace()) if training else None
    query.requires_grad_(training)
    query.eval()
    rng = np.random.default_rng(seed + 71)
    route_rng = np.random.default_rng(seed + 181)
    z, h = base.legacy.base_cache(cnn, x)
    used, trace, predicted = 0, [], []
    reference_keys = set(feature_keys(reference[0])) if reference is not None else set()
    reference_probe = (reference[0], torch.as_tensor(reference[1], dtype=torch.long, device=x.device)) if reference is not None else None
    reference_labels = np.asarray(reference[1]) if reference is not None else None
    probe_rng = np.random.default_rng(seed + 131)
    source_before = endpoint(a, cnn, context, agent, clean, history, used, cap, 0) if not training else {}
    target_before = endpoint(a, cnn, context, agent, (x, y), history, used, cap, 0) if not training and target_endpoints else {}
    frozen_state = deepcopy(cnn.state_dict())
    future_state = deepcopy(agent.policy.future.state_dict())
    utility_state = deepcopy(agent.policy.update_value.state_dict())
    query_initial = deepcopy(query.state_dict()) if not training else None
    for pos in range(steps):
        inputs = x[pos:pos+1]
        versions = agent.versions.copy()
        with torch.no_grad():
            outputs = BankExecutor(cnn, [context]).outputs(inputs, z[pos:pos+1])
            state = build_state(outputs, h[pos:pos+1], history, used, cap, pos, n, -1)
            route_scores = prediction_route_scores(a, agent, state, outputs[0], route_rng)[0].cpu().tolist()
            if not all(math.isfinite(value) for value in route_scores):
                raise ValueError('Nonfinite routing scores.')
            route = int(route_scores[1] > route_scores[0])
            if used < cap and getattr(a, 'eval_routing', 'policy') != 'policy':
                probability, estimated_score = 0., None
            elif used < cap:
                logit, estimate = query(state)
                probability, estimated_score = float(logit.sigmoid()[0]), float(estimate[0])
                if not math.isfinite(probability) or not math.isfinite(estimated_score):
                    raise ValueError('Nonfinite query scores.')
            else:
                probability, estimated_score = 0., None
            entropy = None
            if query_schedule == 'uncertainty':
                entropy = float(normalized_entropy(outputs[0][0, route].cpu().numpy()))
                paid = used < cap and entropy >= threshold
            elif query_schedule == 'frontloaded':
                paid = used < cap and pos < cap
            elif query_schedule == 'uniform':
                paid = used < cap and uniform_query_at(pos, n, cap)
            else:
                paid = used < cap and (rng.random() < probability if training else probability >= .5)
            # Route choice is identical for query and skip, and precedes truth access.
            prediction = int(outputs[0][0, route].argmax())
            correct_predictions = outputs[0].argmax(2)
            route_predictions = correct_predictions[0].cpu().tolist()
        predicted.append(prediction)
        truth = int(y[pos])
        record = dict(step=pos+1, source_row=int(ids[pos]), truth=truth, prediction=prediction,
            route=route, action=2 if paid else route, queried=paid,
            frozen_prediction=route_predictions[0], context_prediction=route_predictions[1],
            frozen_route_score=route_scores[0], context_route_score=route_scores[1],
            query_probability=probability, estimated_score=estimated_score,
            gap=None, gain=None, harm=None, score=None, replay_loss=None, actor_loss=None, value_loss=None)
        if getattr(a, 'eval_routing', 'policy') == 'random':
            record.update(routing_rule='random')
        if getattr(a, 'eval_routing', 'policy') == 'uncertainty':
            route_entropies = normalized_entropy(outputs[0][0].cpu().numpy())
            record.update(routing_rule='uncertainty', frozen_entropy=float(route_entropies[0]),
                context_entropy=float(route_entropies[1]))
        if query_schedule == 'uncertainty':
            record.update(query_entropy=entropy, uncertainty_threshold=threshold)
        if diagnostic_sink is not None:
            record.update(label_cap=cap, domain_samples=n, budget_used_before=used,
                budget_remaining_before=cap-used, query_legal=used < cap,
                request_cost=a.cost, weighted_gain=None, weighted_harm=None,
                frozen_correct=int(route_predictions[0] == truth), context_correct=int(route_predictions[1] == truth),
                routing_supervision='paid_feedback' if paid else 'observer_only',
                actor_logit_before=float(logit[0]) if used < cap else None,
                actor_probability_after=None, actor_grad_norm=None,
                target_status='budget_exhausted' if used >= cap else 'insufficient_future')
        target = None
        if getattr(a, 'query_target', 'competitive') == 'update_utility':
            record.update(target_definition='update_utility', competitive_score=None)
        if training and getattr(a, 'actor_logit_penalty', 0) > 0:
            record.update(actor_score_loss=None, actor_logit_penalty_loss=None, actor_logit=None)
        label = torch.tensor([truth], device=x.device)
        # Only the offline TRAIN path can ever access future labels.
        if query_target_allowed(training, used, cap, pos, n, a.horizon):
            future_x = x[pos+1:pos+1+a.horizon]
            future_ids = ids[pos+1:pos+1+a.horizon]
            blocked = {keys[pos], *reference_keys}
            block_keys = keys[pos+1:pos+1+a.horizon]
            future_mask = np.asarray([int(i) != int(ids[pos]) and block_keys[j] not in blocked
                for j, i in enumerate(future_ids)])
            future_at = torch.as_tensor(np.flatnonzero(future_mask), device=x.device)
            future_x, future_ids = future_x[future_at], future_ids[future_mask]
        else:
            future_x = None
        if diagnostic_sink is not None and future_x is not None and len(future_x) == 0:
            record['target_status'] = 'empty_future_probe'
        if future_x is not None and len(future_x):
            paid_rows, replay = memory.draw(ids[pos], inputs, future_ids, reference_keys=reference_keys,
                current_key=keys[pos], future_keys=[k for k, keep in zip(block_keys, future_mask) if keep])
            ref_at = torch.as_tensor(balanced_draw(reference_labels, a.reference_size, probe_rng), device=x.device)
            reference_batch = reference_probe[0][ref_at], reference_probe[1][ref_at]
            candidate, candidate_optimizer, target = counterfactual_step(context, optimizer, cnn,
                (inputs, label), paid_rows, replay,
                (future_x, torch.as_tensor(y[pos+1:pos+1+a.horizon][future_mask], device=x.device)),
                reference_batch, a.replay_weight, a.gain_weight, a.forgetting_weight, a.cost, workspace)
            if getattr(a, 'query_target', 'competitive') == 'update_utility':
                target = dict(target, competitive_score=target['score'],
                    score=a.gain_weight*target['gain']-a.forgetting_weight*target['harm']-a.cost)
            # Full-information TRAIN actor max step; no successor/bootstrap term.
            record.update(target)
            experiment = getattr(a, 'actor_experiment', None)
            if getattr(a, 'actor_logit_penalty', 0) > 0:
                from enhancements.standardized_actor import regularized_actor_step
                record.update(regularized_actor_step(query, query_optimizer, state,
                    state.new_tensor([target['score']]), a.actor_logit_penalty))
            elif experiment and experiment['control'] == 'balanced':
                from enhancements.standardized_actor import balanced_actor_step
                record.update(balanced_actor_step(query, query_optimizer, state,
                    state.new_tensor([target['score']]), experiment['normalizer']['positive_fraction']))
            else:
                record.update(maximize_step(query, query_optimizer, state,
                    state.new_tensor([target['score']])))
            if diagnostic_sink is not None:
                with torch.no_grad():
                    record.update(weighted_gain=a.gain_weight*target['gain'],
                        weighted_harm=a.forgetting_weight*target['harm'], target_status='measured',
                        actor_probability_after=float(query(state)[0].sigmoid()[0]),
                        actor_grad_norm=float(sum(p.grad.detach().square().sum()
                            for p in query.actor.parameters() if p.grad is not None).sqrt()))
        if paid:
            history.observe_paid(outputs[0].softmax(2), truth)
            if target is not None:
                # Commit precisely the measured branch, including Adam moments.
                old_context, old_optimizer = context, optimizer
                context, optimizer = candidate, candidate_optimizer
                workspace.recycle(old_context, old_optimizer)
            else:
                paid_rows, replay = memory.draw(ids[pos], inputs, reference_keys=reference_keys, current_key=keys[pos])
                context_min_step(context, optimizer, cnn, (inputs, label), paid_rows, replay, a.replay_weight)
            with torch.no_grad():
                new_probs = cnn(context(inputs)).softmax(1)
                change = (new_probs - outputs[0][:, 1].softmax(1)).abs().mean()
            history.updated(1, pos+1, change)
            used += 1
            memory.add(ids[pos], inputs, truth)
            agent.versions[0] += 1
            # Correctness labels refer to predictions made before feedback/update.
            if getattr(a, 'eval_routing', 'policy') == 'policy':
                agent.observe_routes(state, (correct_predictions[0] == truth).float(), versions=versions)
        agent.clock += 1
        record['labels_used'] = used
        trace.append(record)
        if diagnostic_sink is not None: diagnostic_sink.observe(state, record)
        if pos == 0 or (pos+1) % a.progress_every == 0 or pos+1 == steps:
            print(f'{"TRAIN" if training else "EVAL"}: {pos+1}/{steps} | labels {used}/{cap}', flush=True)
    runtime = cnn, context, agent, query
    validate_trace(trace, cap)
    if query_schedule in ('uniform', 'frontloaded') and steps == n and used != cap:
        raise RuntimeError('Uniform collection did not consume its exact per-level label cap.')
    if training:
        training_state.update(context_optimizer=optimizer, query_optimizer=query_optimizer, workspace=workspace)
    source_after = endpoint(a, cnn, context, agent, clean, history, used, cap, n) if not training else {}
    target_after = endpoint(a, cnn, context, agent, (x, y), history, used, cap, n) if not training and target_endpoints else {}
    for name, before in (('future', future_state), ('update_value', utility_state)):
        if any(not torch.equal(v, getattr(agent.policy, name).state_dict()[k]) for k, v in before.items()):
            raise RuntimeError('A frozen legacy head changed: ' + name)
    if any(not torch.equal(v, cnn.state_dict()[k]) for k, v in frozen_state.items()):
        raise RuntimeError('Frozen classifier changed.')
    if query_initial is not None and any(not torch.equal(v, query.state_dict()[k]) for k, v in query_initial.items()):
        raise RuntimeError('The query head changed online.')
    result = dict(online=metrics(y[:steps], predicted, len(a.classes)), samples=steps, labels=used, cap=cap,
        labels_percent=100 * used/steps, source_before=source_before, source_after=source_after,
        original_domain_forgetting={k: source_before[k]-source_after[k] for k in source_before},
        target_before=target_before, target_after=target_after,
        target_gain={k: target_after[k]-target_before[k] for k in target_before},
        target_endpoint_protocol='same scored stream, descriptive endpoint; not independent generalization',
        classifier_unchanged=True, query_head_unchanged=not training,
        legacy_future_and_utility_unchanged=True)
    if steps != n: result.update(diagnostic_prefix=True, domain_samples=n, cap_scope='original full domain')
    if query_schedule in ('uniform', 'frontloaded'): result['query_schedule'] = query_schedule
    return runtime, result, trace


def initialize(a):
    source = a.source_run / 'Models'
    a.models = source
    artifact = base.load_artifact(source, a.burst_dir)
    classifier = torch.load(source/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
    from amcal_light_cnn import ARCHITECTURE
    is_light = classifier.get('architecture') == ARCHITECTURE
    if is_light != (a.dataset == 'PS'):
        raise ValueError('Expected original CNN for IAT and Light CNN for PS; source architecture differs.')
    saved = torch.load(source/'final_enhancement.pth', map_location='cpu', weights_only=False)
    if saved.get('format_version') != base.FORMAT_VERSION:
        raise ValueError('Unsupported source enhancement checkpoint format.')
    if saved.get('training_context_count') != 1 or set(saved['training_levels']) != set(TRAIN_LEVELS):
        raise ValueError('Source must be the one-Context PairOnly run trained on 0/1/3.')
    for name, expected in saved['model_hashes'].items():
        if digest(source/name) != expected: raise ValueError('Source provenance mismatch: ' + name)
    for level, expected in saved['training_domain_sha256'].items():
        if digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != expected:
            raise ValueError('Source training-domain provenance mismatch.')
    a.config = replace(EnhancementConfig(**saved['config']), memory_ce_weight=a.replay_weight).validated()
    if (a.config.architecture != 'legacy' or a.config.cascade or a.config.joint_router
            or a.config.routed_utility or not a.config.protected_reference):
        raise ValueError('This pilot requires protected full-bank legacy Contexts without routed utility.')
    a.classes = list(artifact['label_encoder'].classes_)
    if a.reference_size < len(a.classes):
        raise ValueError('Reference size must cover every source class.')
    a.context_count, a.gamma = 1, .95
    initial = torch.load(source/'enhanced_context.pth', map_location='cpu', weights_only=False)['initial_state_dict']
    return artifact, saved, initial


def train(a):
    artifact, inherited, initial = initialize(a)
    a.actor_experiment = None
    if getattr(a, 'actor_train_cache', None):
        from enhancements.standardized_actor import prepare_experiment
        a.actor_experiment = prepare_experiment(a)
        if getattr(a, 'query_target', 'competitive') == 'update_utility':
            a.actor_experiment['request_target'] = dict(version=1, mode='update_utility',
                formula='gain_weight*G - forgetting_weight*F; zero cost', cost=0.,
                gap='recorded for diagnosis and retained in Context minimization, excluded from query target',
                scope='TRAIN targets only; VAL/TEST use pre-feedback Actor estimates',
                theory='separate utility-max query and unchanged gap-min Context objectives; original shared min-max equivalence not claimed')
            a.actor_experiment['reward'] = 'signed update utility G-F; no competitive gap or request cost'
            a.actor_experiment['balance_scope'] = 'natural TRAIN weighting; utility-only query objective'
        if getattr(a, 'actor_logit_penalty', 0) > 0:
            a.actor_experiment['actor_regularizer'] = dict(version=1,
                type='quadratic_logit', coefficient=a.actor_logit_penalty,
                loss='-mean(sigmoid(z)*stopgrad(S)) + coefficient/2 * mean(z**2)',
                scope='TRAIN Actor only; S, Context update, value-head targets and routing unchanged',
                inference='Actor threshold 0.5; no forced labels or test-time bonus',
                objective='regularized max surrogate, not identical to the original unregularized objective')
        if getattr(a, 'train_query_schedule', 'policy') == 'uniform':
            a.actor_experiment['training_collection'] = dict(version=1, query_schedule='uniform',
                budget_fraction=a.budget_fraction,
                scope='TRAIN only; evenly spaced committed feedback independent of Actor probability',
                actor='same original signed full-information max loss; no importance ratio or bootstrap',
                validation='Actor policy at threshold 0.5; fresh adaptive state per VAL intensity',
                objective_scope='same per-state surrogate; behavior/state distribution differs from on-policy training')
            if getattr(a, 'actor_logit_penalty', 0) > 0:
                a.actor_experiment['training_collection']['actor'] = 'original signed max term plus recorded quadratic logit regularizer'
            if getattr(a, 'query_target', 'competitive') == 'update_utility':
                a.actor_experiment['training_collection']['actor'] = 'signed G-F utility max; zero request cost, no logit penalty'
    if (a.output/'Models'/'minimax_query.pth').exists():
        if not (a.output/'training_complete.json').exists():
            raise ValueError('Interrupted training preserved; use a fresh --output directory.')
        if getattr(a, 'train_diagnostics', False) and not (a.output/'TrainDiagnostics'/'target_summary.json').exists():
            raise ValueError('This completed run has no TRAIN diagnostics; use a fresh --output.')
        completed = torch.load(a.output/'Models'/'minimax_query.pth', map_location='cpu', weights_only=False)
        completion = json.loads((a.output/'training_complete.json').read_text())
        if completion['checkpoint_sha256'] != digest(a.output/'Models'/'minimax_query.pth'):
            raise ValueError('Completed checkpoint checksum mismatch.')
        if (completed.get('version') != VERSION
                or completed['settings'] != {key: getattr(a, key) for key in SETTING_KEYS}
                or completed['source_checkpoint_sha256'] != digest(a.models/'final_enhancement.pth')
                or completed.get('dataset') != a.dataset
                or completed.get('actor_experiment') != a.actor_experiment
                or completed.get('pilot') != pilot_spec(a)):
            raise ValueError('Completed checkpoint settings differ; use a fresh --output directory.')
        print('Using completed experimental checkpoint; train was not repeated.', flush=True)
        return
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Partial run preserved. Use a fresh --output directory.')
    models = a.output/'Models'
    models.mkdir(parents=True)
    if a.actor_experiment:
        write_json(a.output/'actor_experiment.json', a.actor_experiment)
    for name in ['protocol_preprocessing.joblib', 'initial_cnn_model.pth', 'enhanced_context.pth']:
        shutil.copy2(a.models/name, models/name)
    source_checkpoint_hash = digest(a.models/'final_enhancement.pth')
    source_domain = base.legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')
    source_domain = pilot_domain(a, source_domain, 701)
    anchors, reference, reserved = reference_partition(source_domain, a.seed)
    # A source reference cannot be a future probe or update at another intensity.
    reference_keys = set(feature_keys(reference[0]))
    train_domains = {}
    for level in TRAIN_LEVELS:
        domain = base.legacy.read_domain(a.burst_dir, artifact, level, 'train_ids')
        mask = np.asarray([int(i) not in reserved and key not in reference_keys
            for i, key in zip(domain[2], feature_keys(domain[0]))])
        train_domains[level] = pilot_domain(a, subset(domain, mask), 702)
        if len(train_domains[level][1]) <= a.horizon:
            raise ValueError('TRAIN stream is too short for the requested future horizon.')
    clean_val = pilot_domain(a, base.legacy.read_domain(a.burst_dir, artifact, 0, 'val_ids'), 703)
    validations = {level: pilot_domain(a, base.legacy.read_domain(a.burst_dir, artifact, level, 'val_ids'), 703) for level in TRAIN_LEVELS}
    if pilot_spec(a):
        write_json(a.output/'pilot_sampling.json', dict(protocol=pilot_spec(a),
            train_ids={str(k): d[2].tolist() for k,d in train_domains.items()},
            val_ids={str(k): d[2].tolist() for k,d in validations.items()},
            anchors=anchors[2].tolist(), reference=reference[2].tolist()))
    runtime = new_runtime(a, artifact, inherited, initial)
    training_state = {}
    recorder = None
    if getattr(a, 'train_diagnostics', False):
        from enhancements.minimax_diagnostics import TrainingRecorder
        recorder = TrainingRecorder(a.output/'TrainDiagnostics', a.seed, a.diagnostic_cache_rows,
            deepcopy(runtime[3].state_dict()))
    best, stale, log = -math.inf, 0, []
    for epoch in range(a.epochs):
        train_rows = []
        for level in TRAIN_LEVELS:
            print(f'Epoch {epoch+1}/{a.epochs} | TRAIN level {level}', flush=True)
            if recorder is not None: recorder.start_stage(epoch+1, level)
            try:
                runtime, result, _ = stream(a, runtime, train_domains[level], anchors, reference, clean_val,
                    a.seed + epoch*1009 + level*101, training=True, training_state=training_state,
                    diagnostic_sink=recorder)
                train_rows.append(dict(level=level, samples=result['samples'], labels=result['labels'],
                    cap=result['cap'], labels_percent=result['labels_percent'], online=result['online']))
            finally:
                if recorder is not None: recorder.finish_stage()
        cnn, context, agent, query = runtime
        snapshot = dict(policy_state_dict=deepcopy(agent.policy.state_dict()),
            prerouter_state_dict=deepcopy(agent.pre_router.state_dict()))
        context_state, query_state = deepcopy(context.state_dict()), deepcopy(query.state_dict())
        val_rows = []
        for level in TRAIN_LEVELS:
            temporary = new_runtime(a, artifact, snapshot, context_state, query_state)
            _, val, _ = stream(a, temporary, validations[level], anchors, None, clean_val,
                a.seed + 50003 + level, target_endpoints=False)
            val_rows.append(val)
        accuracy = float(np.mean([r['online']['AC'] for r in val_rows]))
        drop = float(np.mean([r['original_domain_forgetting']['AC'] for r in val_rows]))
        score = accuracy - a.validation_retention_weight * max(0., drop)
        entry = dict(epoch=epoch+1, validation_AC=accuracy, original_domain_forgetting=drop,
            selection_score=score, validation=val_rows)
        if a.actor_experiment: entry['training'] = train_rows
        log.append(entry)
        print(f'Epoch {epoch+1}/{a.epochs}: VAL AC {accuracy:.2f}% | source forgetting {drop:+.2f} pp', flush=True)
        write_json(a.output/'training.json', dict(version=VERSION, dataset=a.dataset,
            settings={key: getattr(a, key) for key in SETTING_KEYS}, epochs=log,
            **({'pilot': pilot_spec(a)} if pilot_spec(a) else {}),
            **({'actor_experiment': a.actor_experiment} if a.actor_experiment else {})))
        if score > best:
            best, stale = score, 0
            saved = dict(version=VERSION, dataset=a.dataset, config=a.config.as_dict(), **snapshot,
                context_state_dict=context_state, query_state_dict=query_state,
                training_levels=TRAIN_LEVELS, selected_epoch=epoch+1, selection_score=score,
                source_checkpoint_sha256=source_checkpoint_hash,
                model_hashes={name: digest(models/name) for name in
                    ['protocol_preprocessing.joblib', 'initial_cnn_model.pth', 'enhanced_context.pth']},
                training_domain_sha256={str(level): digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
                    for level in TRAIN_LEVELS},
                settings={key: getattr(a, key) for key in SETTING_KEYS},
                provenance=dict(reference_source_ids=sorted(reserved), reference_rows=len(reference[1]),
                    pretrained_reference_exposure=True, reference_scope='independent of experimental updates, not of source pretraining',
                    initialization='existing PairOnly routing; pretrained initial Context; new query actor/value',
                    future_scope='next TRAIN block, no updates within block',
                    test_used_for_selection=False, query_bonus=0, bootstrap=False,
                    objective='adversarial equal-weight batch5 gap plus detached gain/harm/cost; protected replay regularizer',
                    reference_sampling='class-balanced TRAIN reference mini-probes, excluded from all experimental updates',
                    minmax_scope='shared batch5 gap; protected replay is an auxiliary Context regularizer'))
            temporary = models/'minimax_query.pth.tmp'
            if a.actor_experiment: saved['actor_experiment'] = a.actor_experiment
            if getattr(a, 'query_target', 'competitive') == 'update_utility':
                saved['provenance']['objective'] = 'query Actor maximizes signed update utility G-F; Context gap/replay update unchanged'
                saved['provenance']['minmax_scope'] = 'separate local objectives; original shared on-policy min-max equivalence not claimed'
            if pilot_spec(a): saved['pilot'] = pilot_spec(a)
            torch.save(saved, temporary)
            temporary.replace(models/'minimax_query.pth')
        else:
            stale += 1
            if stale >= a.patience: break
    if recorder is not None: recorder.finalize()
    write_json(a.output/'training_complete.json', dict(epochs=len(log), best_score=best,
        checkpoint_sha256=digest(models/'minimax_query.pth')))


def evaluate(a):
    a.models = a.output/'Models'
    saved = torch.load(a.models/'minimax_query.pth', map_location='cpu', weights_only=False)
    if saved['version'] != VERSION: raise ValueError('Unsupported minimax pilot version.')
    if saved['dataset'] != a.dataset: raise ValueError('Checkpoint dataset mismatch.')
    if saved['settings']['seed'] != a.seed: raise ValueError('Requested seed differs from the trained checkpoint.')
    saved_pilot_rows = (saved.get('pilot') or {}).get('rows_per_domain', 0)
    if getattr(a, 'pilot_rows', 0) not in (0, saved_pilot_rows):
        raise ValueError('Requested pilot size differs from the checkpoint.')
    a.pilot_rows = saved_pilot_rows
    for key, value in saved['settings'].items(): setattr(a, key, value)
    a.actor_experiment = saved.get('actor_experiment')
    a.query_target = (a.actor_experiment or {}).get('request_target', {}).get('mode', 'competitive')
    a.config = EnhancementConfig(**saved['config']).validated()
    a.context_count, a.gamma = 1, .95
    artifact = base.load_artifact(a.models, a.burst_dir)
    a.classes = list(artifact['label_encoder'].classes_)
    for name, expected in saved['model_hashes'].items():
        if digest(a.models/name) != expected: raise ValueError('Experimental model provenance mismatch.')
    for level, expected in saved['training_domain_sha256'].items():
        if digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != expected:
            raise ValueError('Training data changed after experimental training.')
    source = base.legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')
    source = pilot_domain(a, source, 701)
    reserved = set(saved['provenance']['reference_source_ids'])
    anchors = subset(source, np.asarray([int(i) not in reserved for i in source[2]]))
    clean = pilot_domain(a, base.legacy.read_domain(a.burst_dir, artifact, 0, 'test_ids'), 704)
    execution = dict(version=VERSION, seed=a.seed, settings=saved['settings'], levels=TEST_LEVELS,
        checkpoint_sha256=digest(a.models/'minimax_query.pth'), prediction='before feedback; best scored route',
        reset='all model, optimizer, reliability and feedback memory state at each intensity',
        query_head_online='fixed', future_head_online='fixed and unused',
        route_head_online='paid labels only', context_online='current plus up to four previous paid labels and source replay',
        metrics='macro PR/RC/F1; pre-feedback online; source forgetting in percentage points')
    if getattr(a, 'eval_routing', 'policy') == 'uncertainty':
        execution['prediction'] = 'before feedback; minimum normalized entropy route; Frozen wins ties'
        execution['routing'] = dict(mode='uncertainty', version=1,
            score='negative normalized predictive entropy; larger is better',
            heads='learned routing and query heads fixed and bypassed',
            trace='frozen/context_route_score are negative entropy; individual entropies also recorded')
        execution['route_head_online'] = 'fixed and unused'
        execution['query_head_online'] = 'fixed and unused'
    if getattr(a, 'eval_routing', 'policy') == 'random':
        execution['prediction'] = 'before feedback; independent uniform random Frozen/Context route'
        execution['routing'] = dict(mode='random', version=1,
            score='one-hot random route indicator; larger is better',
            rng='dedicated NumPy generator: stream seed+181; endpoints seed+811, restarted per endpoint',
            endpoint='same seeded route assignments before/after; separate from online query/order RNG',
            heads='learned routing and query heads fixed and bypassed',
            trace='frozen/context_route_score are random route indicators')
        execution['route_head_online'] = 'fixed and unused'
        execution['query_head_online'] = 'fixed and unused'
    if a.actor_experiment: execution['actor_experiment'] = a.actor_experiment
    if pilot_spec(a): execution['pilot'] = pilot_spec(a)
    if getattr(a, 'eval_query_schedule', 'policy') == 'uniform':
        execution['query_schedule'] = dict(mode='uniform', version=1,
            rule='round-half-up cumulative cap*seen/n; evenly spaced exact cap',
            scope='query decision only; same checkpoint, route argmax and paid-feedback Context/router updates',
            actor_probability='logged for observation only; ignored for requests',
            information='positions and known stream length only; no current or future labels',
            budget='floor(budget_fraction*n) per intensity; all adaptive state reset')
    if getattr(a, 'eval_query_schedule', 'policy') == 'frontloaded':
        execution['query_schedule'] = dict(mode='frontloaded', version=1,
            rule='request on the first floor(budget_fraction*n) stream rows; no later requests',
            scope=('query override only; learned routing and paid-feedback Context/router updates retained'
                if getattr(a, 'eval_routing', 'policy') == 'policy' else
                'front-loaded feedback with specified routing control; Context updates retained, RL heads bypassed'),
            prediction='each row predicted before its own label; all rows including queried prefix are scored',
            information='current stream position only; no future features or labels',
            budget='exact cap per intensity; all adaptive state reset')
    if getattr(a, 'eval_query_schedule', 'policy') == 'uncertainty':
        calibration_domains = {level: pilot_domain(a, base.legacy.read_domain(
            a.burst_dir, artifact, level, 'val_ids'), 703) for level in TRAIN_LEVELS}
        base.legacy.core.set_seed(a.seed)
        calibration_runtime = new_runtime(a, artifact, saved, saved['context_state_dict'], saved['query_state_dict'])
        calibration = calibrate_uncertainty(a, calibration_runtime, calibration_domains)
        a.uncertainty_threshold = calibration['threshold']
        execution['query_schedule'] = calibration
        del calibration_runtime
        print(f'Uncertainty VAL q{a.uncertainty_percentile:g}: threshold {a.uncertainty_threshold:.6f}', flush=True)
    root = a.output/'Results'/('evaluation_' + hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()[:12])
    if (root/'completed.json').exists():
        if json.loads((root/'execution.json').read_text()) != execution: raise ValueError('Execution mismatch.')
        completed = json.loads((root/'completed.json').read_text())
        if digest(root/'summary.csv') != completed.get('summary_sha256'):
            raise ValueError('Completed evaluation summary is missing or changed.')
        for level in TEST_LEVELS:
            metrics_path = root/f'level_{level}_metrics.json'
            if digest(metrics_path) != completed.get('metrics_sha256', {}).get(str(level)):
                raise ValueError('Completed evaluation metrics are missing or changed.')
            row = json.loads(metrics_path.read_text())
            if digest(root/f'level_{level}_trace.csv') != row.get('trace_sha256'):
                raise ValueError('Completed evaluation trace is missing or changed.')
        print(f'Using completed evaluation: {root}', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'execution.json', execution)
    rows = []
    for level in TEST_LEVELS:
        metrics_path = root/f'level_{level}_metrics.json'
        if metrics_path.exists() and (root/f'level_{level}_trace.csv').exists():
            result = json.loads(metrics_path.read_text())
            if result.get('trace_sha256') != digest(root/f'level_{level}_trace.csv'):
                raise ValueError('Saved trace differs from its completed metrics: ' + str(level))
        else:
            base.legacy.core.set_seed(a.seed)
            runtime = new_runtime(a, artifact, saved, saved['context_state_dict'], saved['query_state_dict'])
            domain = base.legacy.read_domain(a.burst_dir, artifact, level, 'test_ids')
            domain = pilot_domain(a, domain, 704)
            _, result, trace = stream(a, runtime, domain, anchors, None, clean, a.seed)
            result['level'] = level
            write_csv(root/f'level_{level}_trace.csv', trace)
            result['trace_sha256'] = digest(root/f'level_{level}_trace.csv')
            write_json(metrics_path, result)
        rows.append(dict(level=level, **result['online'], labels=result['labels'],
            cap=result['cap'], labels_percent=result['labels_percent'],
            **{'forgetting_'+k: v for k, v in result['original_domain_forgetting'].items()},
            **{'gain_'+k: v for k, v in result['target_gain'].items()}))
        print(f'Level {level}: AC {result["online"]["AC"]:.2f}% | labels {result["labels"]}/{result["cap"]} | '
              f'forgetting {result["original_domain_forgetting"]["AC"]:+.2f} pp', flush=True)
    write_csv(root/'summary.csv', rows)
    write_json(root/'completed.json', dict(levels=len(rows), checkpoint_sha256=execution['checkpoint_sha256'],
        summary_sha256=digest(root/'summary.csv'),
        metrics_sha256={str(level): digest(root/f'level_{level}_metrics.json') for level in TEST_LEVELS}))
    print(f'Results: {root}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    p.add_argument('--action', choices=['All', 'Train', 'Evaluate', 'Diagnose'], default='All')
    p.add_argument('--source-run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--burst-dir', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--reference-size', type=int, default=32)
    p.add_argument('--progress-every', type=int, default=250)
    p.add_argument('--cost', type=float, default=.05)
    p.add_argument('--gain-weight', type=float, default=1.)
    p.add_argument('--forgetting-weight', type=float, default=1.)
    p.add_argument('--context-lr', type=float, default=.00025)
    p.add_argument('--query-lr', type=float, default=.0001)
    p.add_argument('--replay-weight', type=float, default=.1)
    p.add_argument('--budget-fraction', type=float, default=.2)
    p.add_argument('--validation-retention-weight', type=float, default=1.)
    p.add_argument('--train-diagnostics', action='store_true')
    p.add_argument('--diagnostic-phase', choices=['All', 'Collect', 'Actor', 'ActorScale', 'Routing', 'FixedRouter'], default='All')
    p.add_argument('--diagnostic-rows', type=int, default=256)
    p.add_argument('--diagnostic-cache-rows', type=int, default=1024)
    p.add_argument('--diagnostic-fit-steps', type=int, default=300)
    p.add_argument('--v2-run', type=Path)
    p.add_argument('--actor-train-cache', type=Path)
    p.add_argument('--actor-control', choices=['natural', 'balanced'], default='natural')
    p.add_argument('--eval-query-schedule', choices=['policy', 'uniform', 'uncertainty', 'frontloaded'], default='policy')
    p.add_argument('--uncertainty-percentile', type=float, default=75.)
    p.add_argument('--eval-routing', choices=['policy', 'uncertainty', 'random'], default='policy')
    p.add_argument('--train-query-schedule', choices=['policy', 'uniform'], default='policy')
    p.add_argument('--pilot-rows', type=int, default=0)
    p.add_argument('--actor-logit-penalty', type=float, default=0.)
    p.add_argument('--query-target', choices=['competitive', 'update_utility'], default='competitive')
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.query_target == 'update_utility':
        if a.cost != 0 or a.actor_logit_penalty != 0 or a.actor_control != 'natural':
            p.error('Utility-target control requires zero cost, zero logit penalty and Natural weighting.')
        if a.action == 'Diagnose': p.error('Utility-target control requires Train/Evaluate, not Diagnose.')
        if a.action in ('Train', 'All') and a.actor_train_cache is None:
            p.error('Utility-target training requires the existing TRAIN Actor normalizer cache.')
    if not math.isfinite(a.actor_logit_penalty) or a.actor_logit_penalty < 0:
        p.error('Actor logit penalty must be finite and nonnegative.')
    if a.actor_logit_penalty > 0:
        if a.actor_control != 'natural': p.error('Logit regularization is a separate Natural control; do not combine it with Balanced.')
        if a.action == 'Diagnose': p.error('Logit regularization requires Train/Evaluate, not Diagnose.')
        if a.action in ('Train', 'All') and a.actor_train_cache is None:
            p.error('Regularized Actor training requires --actor-train-cache.')
    if a.pilot_rows < 0: p.error('Pilot rows must be nonnegative; zero means the full domain.')
    if a.pilot_rows and a.pilot_rows <= a.horizon: p.error('Pilot rows must exceed the future horizon.')
    if a.pilot_rows and a.action == 'Diagnose': p.error('Pilot row limits apply to Train/Evaluate, not Diagnose.')
    if a.eval_routing != 'policy' and a.action != 'Evaluate':
        p.error('Routing controls require Evaluate; no retraining.')
    if a.eval_routing == 'uncertainty' and a.eval_query_schedule not in ('uncertainty', 'frontloaded'):
        p.error('Uncertainty routing requires uncertainty or frontloaded requests.')
    if a.eval_routing == 'random' and a.eval_query_schedule != 'frontloaded':
        p.error('Random routing control requires frontloaded requests.')
    if not math.isfinite(a.uncertainty_percentile) or not 0 < a.uncertainty_percentile < 100:
        p.error('Uncertainty percentile must be between zero and 100.')
    if a.eval_query_schedule != 'policy' and a.action != 'Evaluate':
        p.error('Query controls require --action Evaluate; they must not retrain or change VAL checkpoint selection.')
    if a.train_query_schedule == 'uniform' and a.action in ('Train', 'All') and a.actor_train_cache is None:
        p.error('Uniform TRAIN collection requires --actor-train-cache for the standardized Actor experiment.')
    if a.train_query_schedule == 'uniform' and a.action == 'Diagnose':
        p.error('Uniform TRAIN collection must use the training entry point, not Diagnose.')
    if a.actor_control == 'balanced' and a.actor_train_cache is None and a.action != 'Evaluate':
        p.error('Balanced Actor control requires --actor-train-cache.')
    if a.actor_train_cache is not None and a.action == 'Diagnose':
        p.error('Actor standardization is a training experiment, not a diagnostic collection option.')
    if min(a.horizon, a.epochs, a.patience, a.reference_size, a.progress_every) < 1:
        p.error('Horizon/epochs/patience/reference-size/progress-every must be positive.')
    if not 0 < a.budget_fraction <= 1: p.error('Budget must be in (0,1].')
    for key in ['cost', 'gain_weight', 'forgetting_weight', 'context_lr', 'query_lr', 'replay_weight', 'validation_retention_weight']:
        if not math.isfinite(getattr(a, key)) or getattr(a, key) < 0: p.error(key + ' must be finite and nonnegative.')
    if a.context_lr == 0 or a.query_lr == 0: p.error('Learning rates must be positive.')
    if min(a.diagnostic_rows, a.diagnostic_cache_rows, a.diagnostic_fit_steps) < 1:
        p.error('Diagnostic sizes and fit steps must be positive.')
    if a.diagnostic_cache_rows < 2: p.error('Diagnostic cache must permit at least two rows.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    stem = 'light_ps_paironly_batch5' if a.dataset == 'PS' else 'paironly_batch5'
    a.source_run = a.source_run or root/'protocol_runs'/f'{stem}_trainseed{a.seed}'/'paironly_batch5'
    a.burst_dir = a.burst_dir or root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    regular = root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v{VERSION}_h{a.horizon}_c{a.cost:g}'
    if a.actor_train_cache is not None or a.train_query_schedule == 'uniform' or a.query_target == 'update_utility':
        regular = regular.with_name(regular.name+'_actorstd_'+a.actor_control)
    if a.train_query_schedule == 'uniform':
        regular = regular.with_name(regular.name+'_trainuniform')
    if a.pilot_rows:
        regular = regular.with_name(regular.name+f'_pilot{a.pilot_rows}_ep{a.epochs}')
    if a.actor_logit_penalty > 0:
        regular = regular.with_name(regular.name+'_logitreg'+f'{a.actor_logit_penalty:g}'.replace('.', 'p'))
    if a.query_target == 'update_utility':
        regular = regular.with_name(regular.name+'_utility')
    a.v2_run = a.v2_run or regular
    if a.action == 'Diagnose':
        a.output = a.output or root/'protocol_runs'/f'minimax_diagnostics_seed{a.seed}_n{a.diagnostic_rows}'
    else:
        a.output = a.output or (regular.with_name(regular.name+'_logged') if a.train_diagnostics else regular)
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA is unavailable; use --device cpu.')
    base.legacy.core.device = torch.device('cuda' if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    base.legacy.core.set_seed(a.seed)
    if a.action == 'Diagnose':
        from enhancements.minimax_diagnostics import diagnose
        diagnose(a, sys.modules[__name__])
        return
    if a.action in ('All', 'Train'): train(a)
    if a.action in ('All', 'Evaluate'): evaluate(a)


if __name__ == '__main__': main()



