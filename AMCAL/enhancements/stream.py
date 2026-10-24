"""Predict-before-feedback closed loop, including optional sparse inference.

Only TRAIN anchors and previously paid labels enter memory. Sparse rows never
create targets for unexecuted routes. Complete routing oracles are deliberately
unavailable for a sparse stream; full-row-only diagnostics are named explicitly.
"""
from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import amcal_burst_structured_bank_protocol as legacy
from baselines.stream_timing import StreamTimer
from .agent import DeviceReliability
from .memory import ProtectedMemory, transfer
from .models import BankExecutor
from .state import build_state
from .damage_guard import guarded_context_step
from .counterfactual import context_step, alternatives
from .paid_replay import paid_context_step


def cascade_decision(agent, embedding, frozen, config, rng, gap):
    with torch.no_grad():
        probabilities = agent.pre_router(embedding, frozen).sigmoid().squeeze(0)
        top = probabilities.topk(2)
        candidate = int(top.indices[0])
        force = (float(top.values[0]) < config.cascade_confidence
                 or float(top.values[0] - top.values[1]) < config.cascade_margin
                 or float(agent.pre_router.novelty(embedding)[0]) > float(agent.pre_router.novelty_limit)
                 or float(agent.pre_router.fallback_value(embedding, frozen)[0]) > 0
                 or gap >= config.cascade_max_gap
                 or rng.random() < config.cascade_audit_fraction)
    return force, candidate


class FrozenRoutesExecutor(BankExecutor):
    """Keep selector tensor shape, but never execute the removed Context."""
    def outputs(self,x,frozen_logits,slots=None):
        return (frozen_logits[:,None,:].expand(-1,len(self.bank)+1,-1),
                x.new_zeros((len(x),len(self.bank)+1)))


def observer(cnn, bank, agent, x, y, history, used, cap, position, seed, config, mode='learned'):
    """Observer-only endpoints; identical independent audit draws at both ends."""
    z, h = legacy.base_cache(cnn, x)
    ablation=getattr(agent,'architecture_ablation','none')
    executor = (FrozenRoutesExecutor if ablation in ('no_context','frozen') else BankExecutor)(cnn, bank, config.bank_executor)
    predictions, all_predictions = [], []
    rng = np.random.default_rng(seed + 30003)
    with torch.no_grad():
        for start in range(0, len(x), 64):
            bx = x[start:start + 64]
            outputs = executor.outputs(bx, z[start:start + 64])
            if ablation in ('no_context','frozen','no_rl'):
                routes=torch.full((len(bx),),1 if ablation=='no_rl' else 0,dtype=torch.long,device=x.device)
            elif getattr(agent, 'uncertainty_routing', False):
                routes = outputs[0].softmax(2).amax(2).argmax(1)
            elif mode in ('random', 'random_al'):
                routes = torch.tensor(rng.integers(0, len(bank)+1, len(bx)), device=x.device)
            else:
                state = build_state(outputs, h[start:start + 64], history, used, cap,
                                           position, max(position, 1), 0)
                routes = agent.policy.route_logits(state).argmax(1)
            if mode != 'random' and config.cascade:
                for k in range(len(bx)):
                    full, candidate = cascade_decision(agent, h[start+k:start+k+1], z[start+k:start+k+1],
                                                       config, rng, (start + k) % (config.cascade_max_gap + 1))
                    if not full: routes[k] = candidate
            per_route = outputs[0].argmax(2)
            predictions.extend(per_route[torch.arange(len(bx), device=x.device), routes].cpu().tolist())
            all_predictions.append(per_route.cpu().numpy())
    matrix = np.concatenate(all_predictions)
    return np.asarray(predictions), (matrix == np.asarray(y)[:, None]).mean(0).tolist(), matrix


def probe_before(a, cnn, bank, selected, batch, agent, history, used, cap, position, count):
    if batch is None: return None, None
    if not getattr(a.config, 'routed_utility', False):
        return legacy.core.probe_losses(cnn, bank[selected], (batch.x, batch.y)), None
    with torch.no_grad():
        z, h = legacy.base_cache(cnn, batch.x)
        out = BankExecutor(cnn, bank).outputs(batch.x, z)
        state = build_state(out, h, history, used, cap, position, count, 0)
        weights = (.9 * agent.policy.route_logits(state).softmax(1) + .1 / (len(bank)+1))[:, selected + 1]
        loss = F.cross_entropy(out[0][:, selected + 1], batch.y, reduction='none')
    return loss.cpu().numpy(), weights.cpu().numpy()


def episode(a, cnn, bank, initial, agent, domain, anchors, mode='learned', training=False,
            warmup=False, rollout_seed=None, reference=None, same_level=False, continuation=None,
            paid_supervision=False, next_domain_features=None, domain_level=None):
    config = a.config
    uncertainty_threshold = getattr(a, 'uncertainty_threshold', None)
    if uncertainty_threshold is not None and (training or warmup or mode != 'learned' or len(bank) != 1 or a.query_schedule != 'policy'):
        raise ValueError('Uncertainty querying is an evaluation-only one-Context policy control.')
    ablation=getattr(a,'architecture_ablation','none')
    uncertainty_routing = getattr(a, 'uncertainty_routing', False)
    no_selector=ablation in ('no_rl','frozen') or uncertainty_routing or mode == 'random_al'
    no_context=ablation in ('no_context','frozen')
    random_no_rl = no_selector or (mode == 'random' and not training and not warmup)
    random_queries = mode in ('random_query', 'random_query_online')
    if random_queries and (training or warmup or len(bank) != 1
            or a.query_schedule != 'policy' or a.update_policy != 'learned'
            or config.cascade or config.routed_utility):
        raise ValueError('random_query requires single-Context policy-schedule evaluation without cascade/routed utility.')
    protected_routing = getattr(a, 'router_protected_replay', False)
    protected_active = protected_routing and getattr(a, 'router_protected_weight', .5) > 0
    level_balanced = getattr(a, 'router_level_balanced_history', False)
    if level_balanced and (not protected_active or domain_level is None or int(domain_level) < 0):
        raise ValueError('Level-balanced routing requires active protected replay and a known acquisition level.')
    level_exposures = {}; level_class_exposures = {}
    if protected_routing and (training or warmup or mode != 'learned' or len(bank) != 1
            or not config.protected_reference or config.routed_utility or config.cascade):
        raise ValueError('Protected routing replay requires single-Context learned evaluation with local protected memory.')
    protected_updates0 = agent.protected_route_updates
    protected_rows0 = agent.protected_route_rows
    protected_segments = np.zeros(2, dtype=np.int64)
    paired_totals0 = dict(agent.paired_target_totals)
    if next_domain_features is not None and (continuation is None or warmup):
        raise ValueError('A boundary successor requires a persistent, non-warmup sequence.')
    slots = len(bank); routes = slots+1
    if agent.slots != slots: raise ValueError("Selector/bank Context count mismatch.")
    counterfactual = training and not warmup and getattr(a, 'counterfactual_utility', False)
    if counterfactual and (a.query_schedule != 'uniform' or a.update_policy != 'learned'
            or not config.protected_reference or config.routed_utility or config.architecture != 'legacy'):
        raise ValueError('Counterfactual TRAIN requires uniform learned destinations and protected legacy Contexts.')
    damage_guard = getattr(a, 'context_damage_guard', False)
    if damage_guard and (training or warmup or slots != 1 or config.routed_utility
            or not config.protected_reference or a.update_policy != 'learned'):
        raise ValueError('Damage guard requires single-Context protected evaluation with learned updates.')
    guard_records = []
    counterfactual0 = agent.counterfactual_updates
    seed = a.seed if rollout_seed is None else rollout_seed
    x, y, source_ids = domain
    order = np.random.default_rng(seed).permutation(len(y))
    x, y, source_ids = x[order], np.asarray(y)[order], np.asarray(source_ids)[order]
    n, cap = len(y), int(np.floor(a.budget_fraction * len(y)))
    if ablation=='frozen':cap=0
    continuing = bool(continuation)
    if continuing:
        optimizers, history, memory = (continuation[key] for key in ('optimizers', 'history', 'memory'))
        history.last_update -= continuation['previous_count']
    else:
        for context in bank: context.load_state_dict(initial); context.eval()
        agent.versions += 32  # A reset is a new hidden-bank generation for replay weighting.
        optimizers = [torch.optim.Adam([p for p in c.parameters() if p.requires_grad],
                                      lr=a.context_lr, weight_decay=1e-4) for c in bank]
        history = DeviceReliability(len(a.classes), x.device, routes)
        if config.protected_reference:
            memory = ProtectedMemory(anchors, len(a.classes), config, seed + 910033, track_levels=True) if level_balanced else ProtectedMemory(anchors, len(a.classes), config, seed + 910033)
        else:
            # The reference-only ablation uses the original paid-history sampler.
            source = legacy.core.TrainingProbe(x, torch.tensor(y), seed + 910033) if training else legacy.core.PaidHistory(seed + 910033)
            memory = legacy.BalancedRewardReference(source)
    if not hasattr(memory, 'paid_replay_rng'):
        memory.paid_replay_rng = np.random.default_rng(seed + 681923)
    executor = (FrozenRoutesExecutor if no_context else BankExecutor)(cnn, bank, config.bank_executor)
    schedule = set(np.random.default_rng(seed + 819003).choice(n, cap, replace=False).tolist()) if a.query_schedule == 'uniform' or ablation=='no_rl' else None
    destinations = np.random.default_rng(seed + 440003).integers(0, slots, n)
    rng = np.random.default_rng(seed + 719003)
    audit_rng = np.random.default_rng(seed + 260917)
    timer = StreamTimer(x.device, enabled=not training)
    with timer.stage('cache'): frozen, embeddings = legacy.base_cache(cnn, x)
    uncertainty_scores = None
    if uncertainty_threshold is not None and not uncertainty_routing:
        with timer.stage('cache'):
            uncertainty_scores = (1 - frozen.softmax(1).amax(1)).detach().cpu().numpy()
    boundary_cache = None
    if next_domain_features is not None:
        boundary_x, boundary_count = next_domain_features
        with timer.stage('cache'): boundary_z, boundary_h = legacy.base_cache(cnn, boundary_x)
        boundary_cache = boundary_x, boundary_z, boundary_h, boundary_count
    first_ref = observer(cnn, bank, agent, *reference, history, 0, cap, 0, a.seed, config, mode) if reference is not None else None
    first_current = observer(cnn, bank, agent, x, y, history, 0, cap, 0, a.seed, config, mode) if same_level else None
    used = 0; gap = 0; next_cached = None
    traces, predictions_all, known_all = [], [], []
    labels_mask = np.zeros(n, dtype=bool)
    action_counts = np.zeros(2*slots+1, dtype=int)
    slot_queries = np.zeros(slots, dtype=int)
    route_updates0, td_updates0 = agent.route_updates, agent.td_updates
    fresh_updates0, fresh_rows0 = agent.fresh_route_updates, agent.fresh_route_rows
    full_correct, full_legal_correct, full_rows = 0, 0, 0
    briers, row_times, response_times = [], [], []
    counterfactual_errors = []
    timer.start()
    for pos in range(n):
        row_start = time.perf_counter()
        query_bonus = 0.
        inputs = x[pos:pos+1]
        versions = agent.versions.copy()
        gate = int(pos in schedule) if schedule is not None else -1
        if uncertainty_scores is not None:
            gate = int(used < cap and uncertainty_scores[pos] >= uncertainty_threshold)
        state_gate = 2 + int(destinations[pos]) if a.update_policy == 'fixed' and gate == 1 else gate
        with timer.stage('selection'):
            full = True; candidate = 0
            if config.cascade and not training and mode == 'learned' and next_cached is None and gate != 1:
                full, candidate = cascade_decision(agent, embeddings[pos:pos+1], frozen[pos:pos+1], config, audit_rng, gap)
        with timer.stage('prediction'):
            executed_slots = tuple(range(slots)) if full else (() if candidate == 0 else (candidate - 1,))
            outputs = next_cached if next_cached is not None else executor.outputs(inputs, frozen[pos:pos+1], executed_slots)
            next_cached = None
        with timer.stage('selection'):
            state = None; values = None; raw_values = None; update_values = None; route_logits = None
            if full:
                if uncertainty_routing:
                    with torch.no_grad():
                        route_confidence = outputs[0].softmax(2).amax(2)[0]
                        uncertainty_route = int(route_confidence.argmax())
                        current_uncertainty = float(1-route_confidence[uncertainty_route])
                    gate = int(used < cap and current_uncertainty >= uncertainty_threshold)
                if not random_no_rl:
                    state = build_state(outputs, embeddings[pos:pos+1], history, used, cap, pos, n, state_gate)
                allowed = legacy.base.legal_actions(slots, cap-used, gate)
                if ablation=='no_context':allowed=[v for v in allowed if v!=1]
                if a.update_policy == 'fixed': allowed = [v for v in allowed if v < routes or v == routes + int(destinations[pos])]
                with torch.no_grad():
                    if random_queries:
                        route_logits = agent.policy.route_logits(state)[0]
                        greedy = int(route_logits.argmax())
                    elif random_no_rl:
                        greedy = None
                    else:
                        values, route_logits, raw_values, update_values = agent.policy.forward_for_action(state)
                        values, route_logits = values[0], route_logits[0]
                        raw_values, update_values = raw_values[0], update_values[0]
                        strength = getattr(a, 'query_budget_bonus', 0.)
                        if strength > 0 and not training and used < cap and cap > 0:
                            expected = cap * (pos+1) / n
                            deficit = max(0., expected-used)
                            query_bonus = strength * min(1., deficit/max(1., .1*cap))
                            values = values.clone()
                            values[routes:] += query_bonus
                        valid = torch.tensor(allowed, device=x.device)
                        greedy = int(valid[values[valid].argmax()])
                epsilon = a.train_epsilon if training else a.eval_epsilon
                action = greedy if random_queries else (int(rng.choice(allowed)) if mode == 'random' or warmup or rng.random() < epsilon else greedy)
                if mode == 'random_al':
                    query_draw = int(rng.choice(allowed))
                    random_route = int(rng.integers(routes))
                    action = routes if query_draw >= routes else random_route
                elif uncertainty_routing:
                    action = routes if gate == 1 else uncertainty_route
                elif no_selector:action=(routes if gate==1 else 1) if ablation=='no_rl' else 0
                route = uncertainty_route if uncertainty_routing else legacy.base.route_index(action, slots)
                if mode == 'random_al': route = random_route
                if random_queries:
                    # Match the full-random query draw; choose prediction independently.
                    draw = int(rng.choice(allowed))
                    route = greedy
                    action = routes if draw >= routes else route
                gap = 0
            else:
                route = action = candidate
                gap += 1
        response_times.append(1000 * (time.perf_counter() - row_start))
        # No current label has entered the policy, pre-router or state above.
        observed = torch.isfinite(outputs[0]).all(2)[0].cpu().numpy()
        route_predictions = outputs[0].nan_to_num().argmax(2)[0].cpu().numpy()
        prediction = int(route_predictions[route])
        truth = int(y[pos]); paid = action >= routes
        correct = route_predictions == truth
        if full:
            full_rows += 1; full_correct += bool(correct.any())
            legal_routes = {legacy.base.route_index(v, slots) for v in allowed}
            full_legal_correct += any(correct[r] for r in legal_routes)
            if route_logits is not None:
                briers.append(float(np.mean((route_logits.sigmoid().detach().cpu().numpy() - correct) ** 2)))
        action_counts[action] += 1; agent.clock += 1
        reward = gain = harm = None
        reference_counts = None; alternative_utilities = None; alternative_successors = None
        guard_decision = None
        with timer.stage('update'):
            successor_output = None
            bridge = pos+1 == n and boundary_cache is not None
            terminal = pos+1 == n and not bridge
            if bridge:
                next_x, next_z, next_h, next_count = boundary_cache
                next_pos = 0; next_cap = int(np.floor(a.budget_fraction*next_count))
            else:
                next_x, next_z, next_h = x[pos+1:pos+2], frozen[pos+1:pos+2], embeddings[pos+1:pos+2]
                next_count, next_pos, next_cap = n, pos+1, cap
            if not no_selector and not terminal and not warmup and (paid or (training and not paid_supervision)):
                successor_output = executor.outputs(next_x, next_z)
            zero = torch.zeros_like(state) if state is not None else None
            skip_next = zero
            next_gate = int(pos+1 in schedule) if schedule is not None else -1
            if uncertainty_scores is not None:
                next_gate = int(pos+1 < n and used < cap and uncertainty_scores[min(pos+1,n-1)] >= uncertainty_threshold)
            if bridge and schedule is not None:
                next_schedule = np.random.default_rng(seed+819003).choice(next_count,next_cap,replace=False)
                next_gate = int(0 in next_schedule)
            next_state_gate = 2 + int(destinations[min(pos+1, n-1)]) if a.update_policy == 'fixed' and next_gate == 1 else next_gate
            def successor_state(outputs):
                next_history = deepcopy(history) if bridge else history
                if bridge: next_history.last_update -= n
                return build_state(outputs, next_h, next_history, 0 if bridge else used,
                    next_cap, next_pos, next_count, next_state_gate)
            if successor_output is not None:
                skip_next = successor_state(successor_output)
            if paid:
                if used >= cap: raise RuntimeError('A query exceeded the hard label cap.')
                selected = action - routes
                label = torch.tensor([truth], device=x.device)
                batch = memory.sample(inputs)
                # Keep reliability features from before this label was observed.
                # The fresh batch reuses the reward reference; no extra sampling
                # or label storage can change a fixed-destination bank trajectory.
                fresh_history = (deepcopy(history) if (getattr(a, 'router_refresh', False) or protected_active)
                                 and mode == 'learned' and batch is not None else None)
                if config.protected_reference:
                    before, route_weight = probe_before(a, cnn, bank, selected, batch, agent, history, used, cap, pos, n)
                    replay = batch.protected(config.memory_replay_size, memory.replay_rng) if batch is not None else None
                    reference_counts = memory.counts()
                else:
                    before = legacy.core.probe_losses(cnn, bank[selected], batch)
                    route_weight = None; replay = None
                if counterfactual:
                    alternative_utilities, alternative_successors = alternatives(a, cnn, bank, optimizers,
                        inputs, label, outputs, batch, replay, history, used, cap, pos, n,
                        successor_output, x[pos+1:pos+2] if successor_output is not None else None,
                        embeddings[pos+1:pos+2] if successor_output is not None else None, next_state_gate)
                history.observe_paid(outputs[0].softmax(2), truth)
                committed = True
                if no_context:
                    committed=False
                elif damage_guard:
                    guard_decision, after = guarded_context_step(a, cnn, bank[selected], optimizers[selected],
                        inputs, label, outputs[0][:,0], replay, batch, before)
                    guard_records.append(guard_decision)
                    committed = guard_decision['accepted']
                elif getattr(a, 'paid_replay_mode', 'off') != 'off':
                    paid_context_step(a, cnn, bank[selected], optimizers[selected],
                        inputs, label, outputs[0][:,0], replay, memory.paid_replay(inputs,
                            count=None if a.paid_replay_mode == 'batch_all' else {'batch3': 2, 'batch5': 4, 'batch7': 6, 'batch8': 7}.get(a.paid_replay_mode, 2)))
                else:
                    context_step(a, cnn, bank[selected], optimizers[selected], inputs, label, outputs[0][:,0], replay)
                if committed:
                    agent.versions[selected] += 1; executor.refresh(selected)
                if config.protected_reference:
                    if not damage_guard:
                        after = legacy.core.probe_losses(cnn, bank[selected], (batch.x, batch.y)) if batch is not None else None
                    if route_weight is not None:
                        # Multiply signed normalized transfer without changing its denominator.
                        delta = route_weight * (before-after) / (before+after+1e-8)
                        before, after = 1 + delta, 1 - delta
                    utility, gain, harm = transfer(before, after, batch, a.gain_weight,
                                                   a.retention_weight, a.selection_cost, config)
                else:
                    utility, gain, harm = legacy.utility_after(a, cnn, bank, selected, batch, before)
                if counterfactual:
                    counterfactual_errors.append(abs(alternative_utilities[selected]-utility))
                if warmup: utility = 0.
                reward = (1. if prediction == truth else -1.) + utility
                used += 1; slot_queries[selected] += 1; labels_mask[pos] = True
                if committed:
                    changed = executor.refresh_output(inputs, outputs, selected)
                    change = (changed[0][:, selected+1].softmax(1) - outputs[0][:, selected+1].softmax(1)).abs().mean()
                    history.updated(selected+1, pos+1, change)
                # TRAIN episodes reproduce the online availability of protected memory.
                if config.protected_reference or not training:
                    if level_balanced: memory.add(inputs, label, level=domain_level)
                    else: memory.add(inputs, label)
                if committed and successor_output is not None:
                    successor_output = executor.refresh_output(next_x, successor_output, selected)
            if counterfactual and paid:
                if alternative_successors is None: alternative_successors = zero.expand(3,-1).clone()
                agent.observe_counterfactual(state, alternative_utilities, alternative_successors,
                    skip_next, pos+1 == n, versions)
            if mode == 'random_query_online' and paid:
                # Only the routing optimizer sees correctness of pre-feedback outputs.
                # No action-value/TD, future-head or update-value learning is performed.
                agent.observe_routes(state, torch.tensor(correct, dtype=torch.float32, device=x.device),
                                     versions=versions)
            may_learn = not no_selector and not counterfactual and ((training and not paid_supervision) or (mode == 'learned' and paid))
            if may_learn:
                if not full: raise RuntimeError('Sparse routes cannot supervise unexecuted predictions.')
                target = torch.tensor(correct, dtype=torch.float32, device=x.device)
                protected = None
                if protected_active and paid and batch is not None:
                    route_batch = memory.router_reference(inputs, batch) if level_balanced else batch
                    old_ids = np.flatnonzero(route_batch.segments != 2) if route_batch is not None else []
                    if len(old_ids):
                        old_at = torch.as_tensor(old_ids, device=x.device)
                        old_x, old_y = route_batch.x[old_at], route_batch.y[old_at]
                        old_segments = route_batch.segments[old_ids]
                        old_labels = old_y.detach().cpu().numpy()
                        weights = np.zeros(len(old_ids), dtype=np.float32)
                        # Equal segment mass, then equal class mass within each segment.
                        for segment in np.unique(old_segments):
                            segment_mask = old_segments == segment
                            level_ids = route_batch.levels[old_ids] if level_balanced else np.zeros(len(old_ids), dtype=int)
                            levels = np.unique(level_ids[segment_mask])
                            for level in levels:
                                level_mask = segment_mask & (level_ids == level)
                                classes = np.unique(old_labels[level_mask])
                                for cls in classes:
                                    mask = level_mask & (old_labels == cls)
                                    weights[mask] = 1./(len(levels)*len(classes)*int(mask.sum()))
                                    if level_balanced and segment == 1:
                                        key = str(int(level)); cell = f'{int(level)}:{int(cls)}'
                                        level_exposures[key] = level_exposures.get(key,0)+int(mask.sum())
                                        level_class_exposures[cell] = level_class_exposures.get(cell,0)+int(mask.sum())
                        with torch.no_grad():
                            old_z, old_h = legacy.base_cache(cnn, old_x)
                            old_outputs = executor.outputs(old_x, old_z)
                            old_states = build_state(old_outputs, old_h, fresh_history,
                                used-1, cap, pos, n, state_gate)
                            old_correct = old_outputs[0].argmax(2) == old_y[:, None]
                        protected = (old_states, old_correct, torch.as_tensor(weights, device=x.device))
                        protected_segments += np.bincount(old_segments, minlength=2)
                agent.observe_routes(state, target, step_now=not training or paid or pos % 16 == 15 or pos+1 == n,
                                     versions=versions, **({'protected': protected, 'protected_weight': getattr(a, 'router_protected_weight', .5)} if protected_routing else {}))
                if paid and getattr(a, 'router_refresh', False) and fresh_history is not None:
                    with torch.no_grad():
                        fresh_z, fresh_h = legacy.base_cache(cnn, batch.x)
                        fresh_outputs = executor.outputs(batch.x, fresh_z)
                        fresh_states = build_state(fresh_outputs, fresh_h, fresh_history,
                            used-1, cap, pos, n, state_gate)
                        fresh_correct = fresh_outputs[0].argmax(2) == batch.y[:, None]
                    agent.refresh_routes(fresh_states, fresh_correct)
                if config.cascade: agent.observe_pre_router(embeddings[pos:pos+1], frozen[pos:pos+1], target, route_logits)
                if not warmup:
                    skip_state = state.clone()
                    if schedule is not None: skip_state[:, -3] = 0
                    for r in range(routes):
                        agent.push(skip_state, r, 1. if correct[r] else -1., skip_next, terminal, versions)
                    if paid:
                        successor = zero if successor_output is None else successor_state(successor_output)
                        agent.push(state, action, reward, successor, terminal, versions)
                        agent.push_pair(state, action, utility, successor, skip_next, terminal, versions)
                    if paid or pos % 16 == 15 or pos+1 == n: agent.learn_td()
        next_cached = None if bridge else successor_output
        predictions_all.append(route_predictions); known_all.append(observed)
        # One batched observer transfer replaces individual scalar .item() calls.
        scores = None if raw_values is None else torch.cat([raw_values, route_logits.sigmoid()]).detach().cpu().tolist()
        traces.append(dict(step=pos+1, source_row=int(source_ids[pos]), truth=truth, prediction=prediction,
            action=action, route=route, queried=paid, labels_used=used, full_bank=full,
            reward=reward, gain=gain, harm=harm, reference_counts=reference_counts,
            predictions=[int(v) if known else None for v, known in zip(route_predictions, observed)],
            scores=scores))
        if uncertainty_routing:
            traces[-1]['uncertainty_score'] = current_uncertainty
            traces[-1]['uncertainty_threshold'] = uncertainty_threshold
            traces[-1]['route_confidences'] = route_confidence.detach().cpu().tolist()
        if uncertainty_scores is not None:
            traces[-1]['uncertainty_score'] = float(uncertainty_scores[pos])
            traces[-1]['uncertainty_threshold'] = uncertainty_threshold
        if getattr(a, 'query_budget_bonus', 0.) > 0:
            traces[-1]['query_budget_bonus'] = query_bonus
            traces[-1]['selection_scores'] = None if values is None else values.detach().cpu().tolist()
        if random_queries:
            traces[-1]['prediction_route_scores'] = route_logits.detach().cpu().tolist()
        if damage_guard: traces[-1]['damage_guard'] = guard_decision
        if getattr(a, 'connected_sequence_td', False):
            traces[-1]['td_terminal'] = terminal
            traces[-1]['td_boundary_successor'] = bridge
        if counterfactual:
            traces[-1]['counterfactual_utilities'] = alternative_utilities
        if getattr(a, 'collect_utility_calibration', False) or agent.policy.utility_normalization is not None:
            traces[-1]['raw_update_values'] = None if update_values is None else update_values.detach().cpu().tolist()
        if agent.policy.utility_normalization is not None:
            traces[-1]['selection_scores'] = None if values is None else values.detach().cpu().tolist()
        row_times.append(1000 * (time.perf_counter() - row_start))
    timer.stop()
    if schedule is not None and used != cap: raise RuntimeError('Uniform feedback failed to acquire its exact cap.')
    chosen = np.asarray([row['prediction'] for row in traces])
    matrix = np.asarray(predictions_all)
    metrics = dict(accuracy=float(legacy.core.accuracy_score(y, chosen)),
        macro_f1=float(legacy.core.f1_score(y, chosen, average='macro', zero_division=0)),
        frozen_accuracy=float(np.mean(matrix[:, 0] == y)), samples=n, labels=used, label_cap=cap,
        actual_label_fraction=used/n, slot_queries=slot_queries.tolist(), action_counts=action_counts.tolist(),
        route_updates=agent.route_updates-route_updates0, td_updates=agent.td_updates-td_updates0,
        fresh_route_updates=agent.fresh_route_updates-fresh_updates0,
        fresh_route_rows=agent.fresh_route_rows-fresh_rows0,
        architecture_ablation=ablation,
        supervision_protocol='paid feedback only' if paid_supervision or not training else 'dense TRAIN labels',
        oracle_all=full_correct/n if full_rows == n else None,
        oracle_legal=full_legal_correct/n if full_rows == n else None,
        oracle_on_full_rows=full_correct/full_rows if full_rows else None, oracle_full_rows=full_rows,
        full_bank_fraction=full_rows/n, route_brier_on_full_rows=float(np.mean(briers)) if briers else None,
        stream_context_forwards=executor.context_forwards, stream_trunk_forwards=executor.trunk_forwards,
        memory=memory.counts() if config.protected_reference else {'paid_history_only': True},
        oracle_protocol='observer on executed full-bank rows only; complete oracle is null for sparse streams',
        mode=mode, source_ids=source_ids.tolist(), queried_source_ids=source_ids[labels_mask].tolist(),
        latency_protocol='synchronized finite-stream runner with frozen cache and one-input lookahead TD; no annotation delay; not a live-service benchmark')
    if protected_routing:
        metrics['protected_routing_replay'] = dict(
            updates=agent.protected_route_updates-protected_updates0,
            rows=agent.protected_route_rows-protected_rows0,
            source_rows=int(protected_segments[0]), historical_rows=int(protected_segments[1]),
            protected_weight=getattr(a, 'router_protected_weight', .5),
            protocol='exposures, not unique rows; weighted paid replay and current-bank protected loss; one routing step')
    if level_balanced:
        metrics['router_level_history'] = dict(version=1, acquisition_level=int(domain_level),
            historical_exposures_by_level=level_exposures,
            historical_exposures_by_level_class=level_class_exposures,
            protocol='routing-only balanced sampling/weights over retained acquired levels; original reservoirs unchanged')
    if agent.paired_query_only:
        totals = {k: agent.paired_target_totals[k]-paired_totals0[k] for k in paired_totals0}
        rows = totals['rows']; steps = totals['steps']
        metrics['paired_target_diagnostics'] = dict(sampled_rows=rows, optimizer_steps=steps,
            **{k+'_mean': totals[k]/rows if rows else None for k in ('utility','future','target','predicted')},
            positive_utility_rows=totals['positive_utility'], positive_target_rows=totals['positive_target'],
            terminal_rows=totals['terminal'], absolute_td_loss_diagnostic=totals['absolute_loss']/steps if steps else None,
            protocol='TRAIN replay samples; repeated rows count repeatedly; absolute TD has no gradient')
    if getattr(a, 'connected_sequence_td', False):
        metrics['sequence_td'] = dict(boundary_successor=boundary_cache is not None,
            boundary_paid=bool(boundary_cache is not None and traces[-1]['queried']))
    if damage_guard:
        accepted = [row for row in guard_records if row['accepted']]
        metrics['damage_guard'] = dict(paid_rows=len(guard_records), accepted=len(accepted),
            rejected=len(guard_records)-len(accepted),
            backtracked=sum(row['scale'] < 1 for row in accepted),
            trial_steps=sum(row['attempts'] for row in guard_records),
            no_protected_reference=sum(row['reason'] == 'no_protected_reference' for row in guard_records),
            max_accepted_harm=max((row['accepted_harm'] for row in accepted), default=0.),
            mean_accepted_scale=float(np.mean([row['scale'] for row in accepted])) if accepted else None,
            max_harm=config.damage_limit)
    if counterfactual:
        metrics['counterfactual_rows'] = used
        metrics['counterfactual_context_steps'] = 3*used
        metrics['counterfactual_updates'] = agent.counterfactual_updates-counterfactual0
        metrics['counterfactual_loss'] = agent.last_losses.get('counterfactual')
        metrics['counterfactual_selected_max_error'] = max(counterfactual_errors, default=0.)
        metrics['supervision_protocol'] = 'paid TRAIN feedback; three isolated alternatives; update-value head only'
    if uncertainty_routing or mode == 'random_al':
        if metrics['route_updates'] or metrics['td_updates'] or metrics['fresh_route_updates']:
            raise RuntimeError('A full AL replacement performed selector learning.')
        metrics['al_replacement'] = 'uncertainty' if uncertainty_routing else 'random'
    metrics.update(timer.metrics(n, used))
    if not training:
        metrics['response_ms_percentiles'] = dict(zip(('p50', 'p95', 'p99'), np.percentile(response_times, [50, 95, 99]).tolist()))
        metrics['row_ms_percentiles'] = dict(zip(('p50', 'p95', 'p99'), np.percentile(row_times, [50, 95, 99]).tolist()))
    if reference is not None:
        final_ref = observer(cnn, bank, agent, *reference, history, used, cap, n, a.seed, config, mode)
        metrics['reference'] = legacy.retention_summary(first_ref, final_ref, reference[1], np.ones(len(reference[1]), dtype=bool))
    if same_level:
        final_current = observer(cnn, bank, agent, x, y, history, used, cap, n, a.seed, config, mode)
        metrics['same_level'] = {name: legacy.retention_summary(first_current, final_current, y, mask)
                                for name, mask in [('all_rows', np.ones(n, dtype=bool)), ('unqueried', ~labels_mask)]}
    if continuation is not None:
        continuation.update(optimizers=optimizers, history=history, memory=memory, previous_count=n)
    return metrics, traces



