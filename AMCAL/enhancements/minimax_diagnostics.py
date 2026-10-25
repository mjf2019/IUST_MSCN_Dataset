"""Opt-in observation and small TRAIN-only audits; no deployed objective changes."""
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import statistics

DIAGNOSTIC_VERSION = 1
TERMS = ('gap', 'weighted_gain', 'weighted_harm', 'request_cost', 'score',
    'query_probability', 'actor_probability_after', 'actor_logit_before', 'actor_grad_norm')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def number(value):
    if value is None or value == '': return None
    value = float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite diagnostic value.')
    return value


def truth(value):
    return value is True or str(value).lower() in ('true', '1')


def distribution(values):
    values = sorted(v for value in values if (v := number(value)) is not None)
    if not values: return dict(n=0)
    def quantile(p):
        at = p*(len(values)-1); low = int(at); high = min(low+1, len(values)-1)
        return values[low]+(at-low)*(values[high]-values[low])
    return dict(n=len(values), mean=statistics.fmean(values), std=statistics.pstdev(values),
        minimum=values[0], maximum=values[-1],
        **{f'p{int(p*100):02d}': quantile(p) for p in (.01, .1, .25, .5, .75, .9, .99)})


def summarize_targets(rows):
    rows = list(rows)
    targets = [r for r in rows if number(r.get('score')) is not None]
    positive = [r for r in targets if number(r['score']) > 0]
    negative = [r for r in targets if number(r['score']) < 0]
    for row in targets:
        mode = row.get('target_definition') or 'competitive'
        if mode not in ('competitive', 'update_utility'): raise ValueError('Unknown TRAIN target definition.')
        utility = number(row['weighted_gain'])-number(row['weighted_harm'])-number(row['request_cost'])
        expected = utility+(number(row['gap']) if mode == 'competitive' else 0.)
        if not math.isclose(expected, number(row['score']), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('Logged score disagrees with its declared target definition.')
        if mode == 'update_utility' and not math.isclose(number(row['competitive_score']), number(row['gap'])+utility, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('Counterfactual competitive diagnostic score is inconsistent.')
        if not truth(row['query_legal']): raise ValueError('A target was created for an illegal query.')
    return dict(rows=len(rows), eligible_targets=len(targets), positive=len(positive), negative=len(negative),
        zero=len(targets)-len(positive)-len(negative),
        positive_percent=100*len(positive)/len(targets) if targets else None,
        requests=sum(truth(r['queried']) for r in rows),
        skips=sum(not truth(r['queried']) for r in rows),
        legal_rows=sum(truth(r['query_legal']) for r in rows),
        positive_targets_requested=sum(truth(r['queried']) for r in positive),
        negative_targets_requested=sum(truth(r['queried']) for r in negative),
        status_counts=dict(Counter(r.get('target_status', 'unknown') for r in rows)),
        terms={key: distribution(r.get(key) for r in targets) for key in TERMS},
        probability_on_positive=distribution(r['query_probability'] for r in positive),
        probability_on_negative=distribution(r['query_probability'] for r in negative),
        positive_targets_with_p_below_0p01=sum(number(r['query_probability']) < .01 for r in positive),
        positive_targets_with_nonpositive_gain=sum(number(r['weighted_gain']) <= 0 for r in positive),
        positive_gain_but_negative_score=sum(number(r['weighted_gain']) > 0 and number(r['score']) < 0 for r in targets))


class TrainingRecorder:
    """Write every visited TRAIN row; reservoir-cache signed targets separately."""
    def __init__(self, root, seed, cache_rows, initial_query):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed+12317)
        self.capacity = max(1, cache_rows//2)
        self.samples = {'positive': [], 'negative': []}
        self.seen = Counter()
        self.initial_query = initial_query
        self.handle = self.writer = None
        self.stage = None; self.stages = []

    def start_stage(self, epoch, level):
        if self.handle is not None: raise RuntimeError('Previous diagnostic stage is still open.')
        self.stage = dict(epoch=epoch, level=level)
        self.path = self.root/f'train_epoch{epoch}_level{level}.csv'
        self.handle = self.path.open('x', newline='', encoding='utf-8')
        self.writer = None

    def observe(self, state, record):
        if self.handle is None: raise RuntimeError('Diagnostic stage was not opened.')
        row = dict(**self.stage, **record)
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row)); self.writer.writeheader()
        self.writer.writerow(row)
        if row['step'] % 50 == 0: self.handle.flush()
        score = row.get('score')
        if score is None or score == 0: return
        sign = 'positive' if score > 0 else 'negative'
        self.seen[sign] += 1
        index = len(self.samples[sign]) if len(self.samples[sign]) < self.capacity else self.rng.randrange(self.seen[sign])
        if index < self.capacity:
            sample = dict(state=state.detach().cpu().reshape(-1).clone(), score=float(score), row=row)
            if index == len(self.samples[sign]): self.samples[sign].append(sample)
            else: self.samples[sign][index] = sample

    def finish_stage(self):
        if self.handle is None: return
        self.handle.close(); self.handle = None
        with self.path.open(encoding='utf-8', newline='') as handle: rows = list(csv.DictReader(handle))
        report = dict(**self.stage, **summarize_targets(rows))
        self.stages.append(report)
        print(f'TRAIN target summary: epoch {self.stage["epoch"]}, level {self.stage["level"]} | '
              f'positive {report["positive"]}/{report["eligible_targets"]} | requests {report["requests"]}', flush=True)
        write_json(self.root/'target_summary.json', dict(version=DIAGNOSTIC_VERSION,
            stages=self.stages, cache_seen=dict(self.seen),
            cache_scope='per-sign reservoir; cached class prevalence differs from the complete trace'))
        self.save_cache()

    def save_cache(self):
        import torch
        samples = self.samples['positive']+self.samples['negative']
        temporary = self.root/'actor_cache.pt.tmp'
        torch.save(dict(version=DIAGNOSTIC_VERSION, samples=samples,
            initial_query=self.initial_query, seen=dict(self.seen),
            scope='TRAIN only; same-cache fitting is a capacity/optimization diagnostic, not generalization'), temporary)
        temporary.replace(self.root/'actor_cache.pt')

    def finalize(self):
        if self.handle is not None: raise RuntimeError('Finish the active TRAIN stage first.')
        rows = []
        for path in sorted(self.root.glob('train_epoch*_level*.csv')):
            with path.open(encoding='utf-8', newline='') as handle: rows.extend(csv.DictReader(handle))
        write_json(self.root/'target_summary.json', dict(version=DIAGNOSTIC_VERSION,
            aggregate=summarize_targets(rows), stages=self.stages, cache_seen=dict(self.seen),
            cache_scope='per-sign reservoir; cached class prevalence differs from the complete trace'))
        self.save_cache()


def binary_report(scores, probabilities):
    labels = [score > 0 for score in scores]
    predictions = [p >= .5 for p in probabilities]
    pos = sum(labels); neg = len(labels)-pos
    tp = sum(y and p for y, p in zip(labels, predictions))
    tn = sum(not y and not p for y, p in zip(labels, predictions))
    # Rank AUC with average ranks for ties, independent of the deployment threshold.
    pairs = sorted(zip(probabilities, labels)); rank_sum = 0.; at = 0
    while at < len(pairs):
        end = at+1
        while end < len(pairs) and pairs[end][0] == pairs[at][0]: end += 1
        rank_sum += ((at+1+end)/2)*sum(y for _, y in pairs[at:end]); at = end
    return dict(n=len(labels), positives=pos, negatives=neg,
        accuracy=(tp+tn)/len(labels) if labels else None,
        positive_recall=tp/pos if pos else None, negative_recall=tn/neg if neg else None,
        balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,
        auc=(rank_sum-pos*(pos+1)/2)/(pos*neg) if pos and neg else None,
        requested_fraction=sum(predictions)/len(labels) if labels else None,
        continuous_regret=statistics.fmean(max(s, 0)-p*s for s, p in zip(scores, probabilities)) if labels else None,
        probability=distribution(probabilities))


def actor_gradient_contract(width, device, seed):
    import torch
    from .minimax_query import QueryMaximizer, maximize_step
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    checks = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for sign in (1., -1.):
            head = QueryMaximizer(width).to(device).eval()
            score = torch.tensor([sign*.5], device=device, requires_grad=True)
            optimizer = torch.optim.SGD(head.parameters(), lr=0.)
            maximize_step(head, optimizer, torch.zeros(1, width, device=device), score)
            gradient = float(head.actor[-1].bias.grad[0])
            checks.append(dict(score=float(score.detach()[0]), bias_gradient=gradient,
                ascent_direction_ok=gradient*sign < 0, target_detached=score.grad is None))
    return dict(uses_actual_maximize_step=True, checks=checks,
        sign_ok=all(r['ascent_direction_ok'] for r in checks),
        target_detached=all(r['target_detached'] for r in checks))


def actor_audit(cache_path, output, seed, steps, learning_rate, device, trained_query=None):
    import torch
    from .minimax_query import QueryMaximizer, maximize_step
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    if cached.get('version') != DIAGNOSTIC_VERSION: raise ValueError('Unsupported diagnostic cache version.')
    samples = cached['samples']
    positives = [r for r in samples if r['score'] > 0]
    negatives = [r for r in samples if r['score'] < 0]
    report = dict(version=DIAGNOSTIC_VERSION, scope=cached['scope'], raw_target_counts=cached['seen'],
        cached_positive=len(positives), cached_negative=len(negatives),
        steps=steps, learning_rate=learning_rate, deployment_objective_changed=False,
        cache_sha256=__import__('hashlib').sha256(Path(cache_path).read_bytes()).hexdigest())
    report['gradient_contract'] = actor_gradient_contract(cached['initial_query']['actor.0.weight'].shape[1], device, seed)
    if not report['gradient_contract']['sign_ok'] or not report['gradient_contract']['target_detached']:
        report['status'] = 'gradient_contract_failed'; write_json(output, report)
        raise RuntimeError('The actual Actor optimizer failed its gradient/detach contract.')
    if samples:
        all_states = torch.stack([r['state'] for r in samples]).to(device)
        all_scores = [r['score'] for r in samples]
        if not torch.isfinite(all_states).all(): raise ValueError('Nonfinite cached TRAIN states.')
        devices = [all_states.device.index] if all_states.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            probe_head = QueryMaximizer(all_states.shape[1]).to(device).eval()
            probe_head.load_state_dict(cached['initial_query'])
            with torch.no_grad(): p = probe_head(all_states)[0].sigmoid().cpu().tolist()
            report['initialized_head_on_cache'] = binary_report(all_scores, p)
            if trained_query is not None:
                probe_head.load_state_dict(trained_query)
                with torch.no_grad(): p = probe_head(all_states)[0].sigmoid().cpu().tolist()
                report['existing_v2_head_on_cache'] = binary_report(all_scores, p)
    if min(len(positives), len(negatives)) < 8:
        report.update(status='insufficient_sign_diversity',
            interpretation='Do not diagnose failed binary learning from one-sided targets. Inspect TRAIN score components first.')
        write_json(output, report); return report
    rng = random.Random(seed+271)
    count = min(len(positives), len(negatives), 256)
    chosen = rng.sample(positives, count)+rng.sample(negatives, count)
    states = torch.stack([r['state'] for r in chosen]).to(device)
    scores = torch.tensor([r['score'] for r in chosen], device=device)
    if not torch.isfinite(states).all(): raise ValueError('Nonfinite cached TRAIN states.')
    report.update(status='measured', fitting_rows=2*count,
        selection='equal signed classes, no state normalization; same rows used for fitting and scoring')
    state_signs = {}
    for sample in chosen:
        state_signs.setdefault(sample['state'].contiguous().numpy().tobytes(), set()).add(sample['score'] > 0)
    report['identical_states_with_conflicting_signs'] = sum(len(signs) > 1 for signs in state_signs.values())
    devices = [states.device.index] if states.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        initial = QueryMaximizer(states.shape[1]).to(device).eval()
        initial.load_state_dict(cached['initial_query'])
        with torch.no_grad(): initial_p = initial(states)[0].sigmoid().cpu().tolist()
        report['initial'] = binary_report(scores.cpu().tolist(), initial_p)
        report['initial_logit_zero_check'] = all(abs(p-.5) < 1e-6 for p in initial_p)
        if trained_query is not None:
            trained = deepcopy(initial); trained.load_state_dict(trained_query)
            with torch.no_grad(): p = trained(states)[0].sigmoid().cpu().tolist()
            report['existing_v2_head_on_same_train_cache'] = binary_report(scores.cpu().tolist(), p)
        for name in ('original_max_objective', 'bce_sign_diagnostic_only'):
            head = deepcopy(initial)
            optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=1e-4)
            sample_rng = random.Random(seed+317)
            trajectory = []
            for step in range(steps):
                # Identical balanced minibatches for the two diagnostic objectives.
                ids = sample_rng.sample(range(count), min(16, count))
                ids += [count+i for i in sample_rng.sample(range(count), min(16, count))]
                at = torch.tensor(ids, device=device)
                if name == 'original_max_objective': maximize_step(head, optimizer, states[at], scores[at])
                else:
                    logits = head.actor(states[at]).squeeze(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, (scores[at] > 0).float())
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.actor.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                if step == 0 or (step+1) % 50 == 0 or step+1 == steps:
                    with torch.no_grad(): p = head(states)[0].sigmoid().cpu().tolist()
                    trajectory.append(dict(step=step+1, **binary_report(scores.cpu().tolist(), p)))
            report[name] = dict(final=trajectory[-1], trajectory=trajectory)
            if name == 'original_max_objective':
                with torch.no_grad(): predicted_score = head(states)[1].cpu().tolist()
                report[name]['value_head_MAE'] = statistics.fmean(abs(p-s) for p, s in zip(predicted_score, scores.cpu().tolist()))
    write_json(output, report)
    return report


def diagnostic_actor_step(actor, optimizer, states, targets, weights):
    """Actor-only version of the original loss, with explicit diagnostic weights."""
    import torch
    logits = actor(states.detach()).squeeze(-1)
    loss = -(logits.sigmoid()*targets.detach()*weights.detach()).mean()
    if not torch.isfinite(loss): raise ValueError('Nonfinite diagnostic Actor loss.')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(grad)


def actor_scale_audit(cache_path, output, seed, steps, learning_rate, device):
    """TRAIN-only factorial audit; no fitted model is saved or deployed."""
    import torch
    from .minimax_query import QueryMaximizer
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Fit steps and learning rate must be positive.')
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    if cached.get('version') != DIAGNOSTIC_VERSION:
        raise ValueError('Unsupported diagnostic cache version.')
    positives = [r for r in cached['samples'] if r['score'] > 0]
    negatives = [r for r in cached['samples'] if r['score'] < 0]
    report = dict(version=1, scope='TRAIN same-cache diagnostic; not generalization or traffic accuracy',
        steps=steps, learning_rate=learning_rate, deployment_changed=False,
        cache_sha256=__import__('hashlib').sha256(Path(cache_path).read_bytes()).hexdigest(),
        raw_target_counts=cached['seen'], controls={},
        limitations=['Natural prevalence is reconstructed from full trace sign counts, with reservoir samples within each sign.',
            'Sign targets change the diagnostic objective; they are not equivalent to maximizing the original signed utility.',
            'The separate value head is omitted; it has no shared Actor parameters.',
            'Same-cache fitting cannot demonstrate generalization or guarantee sufficient optimization.'])
    if min(len(positives), len(negatives)) < 8:
        report['status'] = 'insufficient_sign_diversity'; write_json(output, report); return report
    rng = random.Random(seed+271)
    count = min(len(positives), len(negatives), 256)
    chosen = rng.sample(positives, count)+rng.sample(negatives, count)
    states = torch.stack([r['state'] for r in chosen]).to(device)
    scores = torch.tensor([r['score'] for r in chosen], device=device)
    if not torch.isfinite(states).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite cached states or scores.')
    # Statistics are fitted exclusively on these cached TRAIN rows.
    mean = states.double().mean(0)
    std = states.double().std(0, unbiased=False)
    scale = torch.where(std < 1e-6, torch.ones_like(std), std)
    standardized = ((states.double()-mean)/scale).to(states.dtype)
    seen_positive, seen_negative = cached['seen'].get('positive', 0), cached['seen'].get('negative', 0)
    if min(seen_positive, seen_negative) < 1:
        raise ValueError('Full TRAIN sign counts must include both signs.')
    prevalence = seen_positive/(seen_positive+seen_negative)
    natural_weights = torch.where(scores > 0, 2*prevalence, 2*(1-prevalence))
    report.update(status='measured', fitting_rows=len(chosen), natural_positive_fraction=prevalence,
        input_profile=dict(dimensions=states.shape[1], constant_or_near_constant_dimensions=int((std < 1e-6).sum()),
            dimension_mean=mean.cpu().tolist(), dimension_std=std.cpu().tolist(),
            raw_max_abs=float(states.abs().max()), standardized_max_abs=float(standardized.abs().max())),
        score_profile=dict(positive=distribution(scores[:count].cpu().tolist()),
            negative=distribution(scores[count:].cpu().tolist())))
    devices = [states.device.index] if states.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        initial = QueryMaximizer(states.shape[1]).to(device).eval()
        initial.load_state_dict(cached['initial_query'])
        for input_name, inputs in [('raw', states), ('standardized', standardized)]:
            for target_name, target in [('utility', scores), ('sign', scores.sign())]:
                for balance_name, weights in [('natural', natural_weights), ('balanced', torch.ones_like(scores))]:
                    name = f'{input_name}_{target_name}_{balance_name}'
                    actor = deepcopy(initial.actor)
                    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=1e-4)
                    sample_rng = random.Random(seed+317)
                    trajectory = []
                    for step in range(steps):
                        ids = sample_rng.sample(range(count), min(16, count))
                        ids += [count+i for i in sample_rng.sample(range(count), min(16, count))]
                        at = torch.tensor(ids, device=device)
                        grad = diagnostic_actor_step(actor, optimizer, inputs[at], target[at], weights[at])
                        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
                            with torch.no_grad():
                                p = actor(inputs).squeeze(-1).sigmoid()
                                original_utility = (natural_weights*p*scores).mean()
                                oracle_utility = (natural_weights*scores.clamp_min(0)).mean()
                            trajectory.append(dict(step=step+1,
                                **binary_report(scores.cpu().tolist(), p.cpu().tolist()),
                                natural_original_utility=float(original_utility),
                                natural_original_utility_regret=float(oracle_utility-original_utility),
                                saturated_fraction=float(((p < .01) | (p > .99)).float().mean()),
                                last_preclip_gradient_norm=float(grad)))
                    report['controls'][name] = dict(final=trajectory[-1], trajectory=trajectory)
                    print(f'Actor scale audit: {name} completed ({steps} steps)', flush=True)
    write_json(output, report)
    return report


def routing_report(rows):
    rows = list(rows); count = len(rows)
    if not count: return dict(samples=0)
    informative = []; mapping_errors = 0
    for row in rows:
        route = int(row['route']); label = int(row['truth'])
        frozen, context = int(row['frozen_prediction']), int(row['context_prediction'])
        best = int(float(row['context_route_score']) > float(row['frozen_route_score']))
        mapping_errors += route != best or int(row['prediction']) != (frozen if route == 0 else context)
        if (frozen == label) != (context == label): informative.append(row)
    accuracy = lambda key: 100*sum(int(r[key]) == int(r['truth']) for r in rows)/count
    selected_right = sum(int(r['prediction']) == int(r['truth']) for r in informative)
    frozen_only = [r for r in informative if int(r['frozen_prediction']) == int(r['truth'])]
    context_only = [r for r in informative if int(r['context_prediction']) == int(r['truth'])]
    return dict(samples=count, mapping_errors=mapping_errors, observer_only=True,
        frozen_AC=accuracy('frozen_prediction'), context_AC=accuracy('context_prediction'),
        selected_AC=accuracy('prediction'), informative_rows=len(informative),
        informative_selection_accuracy=100*selected_right/len(informative) if informative else None,
        missed_correct_route=len(informative)-selected_right,
        context_minus_frozen_score_when_frozen_only_correct=distribution(float(r['context_route_score'])-float(r['frozen_route_score']) for r in frozen_only),
        context_minus_frozen_score_when_context_only_correct=distribution(float(r['context_route_score'])-float(r['frozen_route_score']) for r in context_only),
        route_counts=dict(Counter(int(r['route']) for r in rows)),
        interpretation='Posthoc route diagnosis; TEST labels do not train or tune any model.')


def inspect_existing_routes(a, api):
    rows = []
    for root in sorted((a.v2_run/'Results').glob('evaluation_*')):
        if not (root/'completed.json').exists(): continue
        execution = json.loads((root/'execution.json').read_text(encoding='utf-8'))
        if execution.get('version') != api.VERSION: continue
        if execution['seed'] != a.seed: continue
        for level in api.TEST_LEVELS:
            path = root/f'level_{level}_trace.csv'
            metric = json.loads((root/f'level_{level}_metrics.json').read_text(encoding='utf-8'))
            if api.digest(path) != metric['trace_sha256']: raise ValueError('v2 trace checksum mismatch.')
            with path.open(encoding='utf-8-sig', newline='') as handle: trace = list(csv.DictReader(handle))
            rows.append(dict(evaluation=root.name, level=level, **routing_report(trace)))
    report = dict(version=DIAGNOSTIC_VERSION, dataset=a.dataset, rows=rows,
        status='measured' if rows else 'no_completed_v2_traces', parameters_changed=False)
    write_json(a.output/'routing_diagnostics.json', report)
    return report


def wiring_audit(a, api, artifact, inherited, initial, domain):
    import torch
    from .agent import routing_objective
    from .models import BankExecutor
    cnn, context, agent, _ = api.new_runtime(a, artifact, inherited, initial)
    x = domain[0][:8]
    with torch.no_grad():
        z, _ = api.base.legacy.base_cache(cnn, x)
        logits = BankExecutor(cnn, [context]).outputs(x, z)[0]
        frozen_ok = bool(torch.allclose(logits[:, 0], cnn(x), atol=1e-5, rtol=1e-5))
        context_ok = bool(torch.allclose(logits[:, 1], cnn(context(x)), atol=1e-5, rtol=1e-5))
    test_logits = torch.zeros(4, 2, device=x.device, requires_grad=True)
    correct = x.new_tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    loss = routing_objective(test_logits, correct, torch.ones(4, device=x.device), a.config.rank_weight)
    gradient = torch.autograd.grad(loss, test_logits)[0]
    sign_ok = bool(gradient[0, 0] < 0 and gradient[0, 1] > 0 and gradient[1, 0] > 0 and gradient[1, 1] < 0)
    route_ids = {id(p) for p in agent.policy.routing_parameters()}
    other_ids = {id(p) for p in [*agent.policy.future.parameters(), *agent.policy.update_value.parameters()]}
    report = dict(frozen_is_route_zero=frozen_ok, context_is_route_one=context_ok,
        correctness_target_columns=['Frozen_correct', 'Context_correct'],
        correctness_loss_gradient_sign_ok=sign_ok, routing_and_other_heads_disjoint=not bool(route_ids & other_ids),
        diagnostic_logits_gradient=gradient.detach().cpu().tolist())
    if not all((frozen_ok, context_ok, sign_ok, report['routing_and_other_heads_disjoint'])):
        raise RuntimeError('Routing wiring/gradient contract failed: '+json.dumps(report))
    write_json(a.output/'routing_wiring.json', report)
    return report


def collect(a, api):
    import torch
    artifact, inherited, initial = api.initialize(a)
    protocol = dict(version=DIAGNOSTIC_VERSION, dataset=a.dataset, seed=a.seed,
        settings={k: getattr(a, k) for k in api.SETTING_KEYS},
        max_rows_per_level=a.diagnostic_rows, cache_rows=a.diagnostic_cache_rows,
        source_checkpoint_sha256=api.digest(a.models/'final_enhancement.pth'),
        objective_implementation_sha256=api.digest(Path(__file__).with_name('minimax_query.py')),
        stream_implementation_sha256=api.digest(Path(api.__file__)),
        scope='one short TRAIN pass over 0/1/3; prefixes of full-size shuffled domains, original full-domain cap',
        limitation='This is not a reconstruction of the complete original v2 training trajectory.',
        reward_unchanged=True, threshold=.5, test_used_for_targets=False)
    completed = a.output/'collection_complete.json'
    if completed.exists():
        if json.loads((a.output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Diagnostic settings differ; use a fresh --output.')
        if not (a.output/'TRAIN'/'actor_cache.pt').exists(): raise ValueError('TRAIN cache is missing.')
        manifest = json.loads(completed.read_text(encoding='utf-8'))
        if api.digest(a.output/'TRAIN'/'actor_cache.pt') != manifest['cache_sha256']:
            raise ValueError('Diagnostic TRAIN cache checksum mismatch.')
        print('Using completed diagnostic TRAIN cache.', flush=True); return
    if (a.output/'TRAIN').exists(): raise ValueError('Partial diagnostics preserved; use a fresh --output.')
    a.output.mkdir(parents=True, exist_ok=True); write_json(a.output/'protocol.json', protocol)
    source = api.base.legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')
    anchors, reference, reserved = api.reference_partition(source, a.seed)
    blocked = set(api.feature_keys(reference[0]))
    domains = {}
    for level in api.TRAIN_LEVELS:
        d = api.base.legacy.read_domain(a.burst_dir, artifact, level, 'train_ids')
        import numpy as np
        mask = np.asarray([int(i) not in reserved and key not in blocked for i, key in zip(d[2], api.feature_keys(d[0]))])
        domains[level] = api.subset(d, mask)
    wiring_audit(a, api, artifact, inherited, initial, domains[0])
    # The audit constructed modules; restore the experiment seed before initializing the trajectory.
    api.base.legacy.core.set_seed(a.seed)
    runtime = api.new_runtime(a, artifact, inherited, initial)
    recorder = TrainingRecorder(a.output/'TRAIN', a.seed, a.diagnostic_cache_rows, deepcopy(runtime[3].state_dict()))
    training_state = {}
    clean = api.base.legacy.read_domain(a.burst_dir, artifact, 0, 'val_ids')
    for level in api.TRAIN_LEVELS:
        print(f'Diagnostic TRAIN level {level}: at most {a.diagnostic_rows} rows', flush=True)
        recorder.start_stage(1, level)
        try:
            runtime, _, _ = api.stream(a, runtime, domains[level], anchors, reference, clean,
                a.seed+level*101, training=True, training_state=training_state, target_endpoints=False,
                diagnostic_sink=recorder, max_steps=a.diagnostic_rows)
        finally: recorder.finish_stage()
    recorder.finalize()
    write_json(completed, dict(protocol=protocol, completed=True,
        cache_sha256=api.digest(a.output/'TRAIN'/'actor_cache.pt')))


def fixed_old_router(a, api):
    """Optional paired TEST control: old routing weights, v2 Context/query unchanged."""
    import torch
    artifact, old, _ = api.initialize(a)
    source_models = a.models
    models = a.v2_run/'Models'
    saved = torch.load(models/'minimax_query.pth', map_location='cpu', weights_only=False)
    if saved['version'] != api.VERSION or saved['dataset'] != a.dataset: raise ValueError('v2 model mismatch.')
    if saved['settings']['seed'] != a.seed: raise ValueError('v2 seed mismatch.')
    if saved['source_checkpoint_sha256'] != api.digest(source_models/'final_enhancement.pth'):
        raise ValueError('The old routing source is not the source used by this v2 checkpoint.')
    for name in ('protocol_preprocessing.joblib', 'initial_cnn_model.pth'):
        if api.digest(models/name) != api.digest(source_models/name): raise ValueError('Source/v2 model provenance differs.')
    for key, value in saved['settings'].items(): setattr(a, key, value)
    a.models = models
    from .config import EnhancementConfig
    a.config = EnhancementConfig(**saved['config']).validated()
    source = api.base.legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')
    import numpy as np
    reserved = set(saved['provenance']['reference_source_ids'])
    anchors = api.subset(source, np.asarray([int(i) not in reserved for i in source[2]]))
    clean = api.base.legacy.read_domain(a.burst_dir, artifact, 0, 'test_ids')
    root = a.output/'fixed_old_router'; root.mkdir(parents=True, exist_ok=True)
    if (root/'completed.json').exists(): raise ValueError('Fixed-router control already exists; preserve it and use a fresh diagnostic --output.')
    matching = []
    for path in (a.v2_run/'Results').glob('evaluation_*'):
        if not (path/'completed.json').exists(): continue
        execution = json.loads((path/'execution.json').read_text(encoding='utf-8'))
        if execution['checkpoint_sha256'] == api.digest(models/'minimax_query.pth') and execution['seed'] == a.seed:
            matching.append(path)
    if len(matching) != 1: raise ValueError('A unique completed matching v2 evaluation is required for the paired routing control.')
    reports = []
    for level in api.TEST_LEVELS:
        api.base.legacy.core.set_seed(a.seed)
        runtime = api.new_runtime(a, artifact, saved, saved['context_state_dict'], saved['query_state_dict'])
        agent = runtime[2]
        with torch.no_grad():
            for name, parameter in agent.policy.named_routing_parameters(): parameter.copy_(old['policy_state_dict'][name].to(parameter.device))
        agent.policy.requires_grad_(False)
        agent.observe_routes = lambda *args, **kwargs: None
        routing_before = {name: p.detach().clone() for name, p in agent.policy.named_routing_parameters()}
        domain = api.base.legacy.read_domain(a.burst_dir, artifact, level, 'test_ids')
        runtime, result, trace = api.stream(a, runtime, domain, anchors, None, clean, a.seed)
        if any(not torch.equal(p, routing_before[name]) for name, p in agent.policy.named_routing_parameters()):
            raise RuntimeError('Fixed old routing weights changed.')
        original_path = matching[0]/f'level_{level}_trace.csv'
        original_metric = json.loads((matching[0]/f'level_{level}_metrics.json').read_text(encoding='utf-8'))
        if api.digest(original_path) != original_metric['trace_sha256']: raise ValueError('Original v2 trace checksum mismatch.')
        with original_path.open(encoding='utf-8-sig', newline='') as handle: original = list(csv.DictReader(handle))
        def identity(row):
            return (int(row['source_row']), truth(row['queried']), int(row['labels_used']),
                int(row['frozen_prediction']), int(row['context_prediction']))
        if [identity(r) for r in trace] != [identity(r) for r in original]:
            raise RuntimeError('The routing control changed query/Context trajectories; it is not a paired routing-only comparison.')
        result.update(level=level, routing='old source routing weights, fixed online')
        api.write_csv(root/f'level_{level}_trace.csv', trace)
        api.write_json(root/f'level_{level}_metrics.json', result)
        reports.append(dict(level=level, **result['online'], labels=result['labels']))
    write_json(root/'completed.json', dict(reports=reports, v2_checkpoint_sha256=api.digest(models/'minimax_query.pth'),
        old_routing_checkpoint_sha256=api.digest(source_models/'final_enhancement.pth'),
        query_and_context_trajectories_match=True,
        scope='paired diagnostic control; changes routing initialization and freezes its online updates; no TEST-based tuning'))


def diagnose(a, api):
    protocol_path = a.output/'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol['dataset'] != a.dataset or protocol['seed'] != a.seed:
            raise ValueError('Diagnostic directory belongs to another dataset/seed.')
    if a.diagnostic_phase in ('All', 'Collect'): collect(a, api)
    if a.diagnostic_phase in ('All', 'Actor'):
        import torch
        if not (a.output/'collection_complete.json').exists():
            raise ValueError('Complete the diagnostic TRAIN collection before the Actor audit.')
        completion = json.loads((a.output/'collection_complete.json').read_text(encoding='utf-8'))
        if api.digest(a.output/'TRAIN'/'actor_cache.pt') != completion['cache_sha256']:
            raise ValueError('Diagnostic TRAIN cache checksum mismatch.')
        trained = None
        path = a.v2_run/'Models'/'minimax_query.pth'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['version'] != api.VERSION or saved['dataset'] != a.dataset or saved['settings']['seed'] != a.seed:
                raise ValueError('Existing v2 query head provenance mismatch.')
            protocol = json.loads((a.output/'protocol.json').read_text(encoding='utf-8'))
            if saved['source_checkpoint_sha256'] != protocol['source_checkpoint_sha256']:
                raise ValueError('Cached TRAIN states and v2 query head have different source models.')
            trained = saved['query_state_dict']
        report = actor_audit(a.output/'TRAIN'/'actor_cache.pt', a.output/'actor_fit.json',
            a.seed, a.diagnostic_fit_steps, a.query_lr, api.base.legacy.core.device, trained)
        print('Actor diagnostic: '+report['status'], flush=True)
    if a.diagnostic_phase in ('All', 'Routing'):
        report = inspect_existing_routes(a, api)
        print('Routing diagnostic: '+report['status'], flush=True)
    if a.diagnostic_phase == 'FixedRouter': fixed_old_router(a, api)
    if a.diagnostic_phase == 'ActorScale':
        path = a.output/'TRAIN'/'actor_cache.pt'
        if not protocol_path.exists() or not (a.output/'collection_complete.json').exists() or not path.exists():
            raise ValueError('Complete TRAIN collection first; the local actor_cache.pt is required.')
        completion = json.loads((a.output/'collection_complete.json').read_text(encoding='utf-8'))
        if api.digest(path) != completion['cache_sha256']:
            raise ValueError('Diagnostic TRAIN cache checksum mismatch.')
        report = actor_scale_audit(path, a.output/f'actor_scale_steps{a.diagnostic_fit_steps}.json',
            a.seed, a.diagnostic_fit_steps, a.query_lr, api.base.legacy.core.device)
        print('Actor scale diagnostic: '+report['status'], flush=True)
    print(f'Diagnostics: {a.output}', flush=True)

