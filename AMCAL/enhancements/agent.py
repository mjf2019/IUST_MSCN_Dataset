"""Ranking-supervised routing and paired update advantages; no Context gradients."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import amcal_burst_structured_bank_protocol as legacy
from .models import PreRouter
from .utility_normalization import relative_utility
from .state import masked_q


class JointRouteComparison(nn.Module):
    """Shared cross-route attention; no Context-slot positional bias.

    Zero output initialization makes the residual neutral at warm start.
    Each route can inspect every competing route's full state, including its
    predicted distribution, embedding and reliability, rather than pooled means.
    """
    def __init__(self, route_width):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(route_width, 64), nn.GELU())
        self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.output = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, routes):
        encoded = self.encode(routes)
        compared, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        return self.output(torch.cat([encoded, compared], -1)).squeeze(-1)


def load_policy_state(policy, state):
    """Only an entirely absent new comparison head may be warm-started.

    Missing legacy weights, unexpected keys and partial comparison checkpoints
    are errors. A zero residual is kept when loading a complete legacy policy.
    """
    expected = set(policy.state_dict())
    supplied = set(state)
    comparison = {key for key in expected if key.startswith('comparison.')}
    missing, extra = expected-supplied, supplied-expected
    if extra or (missing and missing != comparison):
        raise ValueError('Incompatible selector checkpoint: missing or unexpected policy weights.')
    if missing:
        nn.init.zeros_(policy.comparison.output[-1].weight)
        nn.init.zeros_(policy.comparison.output[-1].bias)
    policy.load_state_dict(state, strict=not missing)


class EfficientQ(legacy.StructuredQ):
    def __init__(self, route_width, joint_router=False, slots=3):
        super().__init__(route_width)
        self.slots = slots
        self.routes = slots+1
        self.comparison = JointRouteComparison(route_width) if joint_router else None
        self.utility_normalization = None

    def route_logits(self, states):
        route_x = states[:, :self.routes*self.route_width].reshape(-1, self.routes, self.route_width)
        logits = self.correctness(route_x).squeeze(-1)
        return logits if self.comparison is None else logits + self.comparison(route_x)

    def named_routing_parameters(self):
        for name, value in self.correctness.named_parameters(): yield 'correctness.'+name, value
        if self.comparison is not None:
            for name, value in self.comparison.named_parameters(): yield 'comparison.'+name, value

    def routing_parameters(self): return [p for _, p in self.named_routing_parameters()]

    def forward_components(self, states, detach_routes=False):
        route_x = states[:, :self.routes * self.route_width].reshape(-1, self.routes, self.route_width)
        logits = self.route_logits(states)
        reward = 2 * logits.sigmoid() - 1
        if detach_routes: reward = reward.detach()
        pooled = torch.cat([route_x.mean(1), route_x.max(1).values, states[:, -2:]], 1)
        future = self.future(pooled)
        features = torch.cat([route_x[:, 1:], pooled[:, None].expand(-1, self.slots, -1)], 2)
        update = self.update_value(features).squeeze(-1)
        return torch.cat([future + reward, future + reward[:, 1:] + update], 1), logits, update

    def forward_with_logits(self, states, detach_routes=False):
        values, logits, _ = self.forward_components(states, detach_routes)
        return values, logits

    def forward_for_action(self, states):
        raw, logits, update = self.forward_components(states)
        if self.utility_normalization is None: return raw, logits, raw, update
        normalized = relative_utility(update, **self.utility_normalization)
        selected = raw.clone()
        selected[:, self.routes:] = raw[:, self.routes:] - update + normalized
        return selected, logits, raw, update

    def forward(self, states, detach_routes=False):
        return self.forward_with_logits(states, detach_routes)[0]


def ranking_loss(logits, correct):
    pairs = correct[:, :, None].bool() & ~correct[:, None, :].bool()
    losses = F.softplus(logits[:, None, :] - logits[:, :, None])
    return (losses * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)


def routing_objective(logits, correct, weights, rank_weight, informative_weight=1.):
    """Emphasize rows where route selection can change correctness.

    Unanimously correct/incorrect rows retain BCE supervision for calibration.
    Weights are normalized; the number of optimizer steps does not change.
    """
    informative = correct.bool().any(1) & ~correct.bool().all(1)
    weights = weights * torch.where(informative, informative_weight, 1.)
    bce = F.binary_cross_entropy_with_logits(logits, correct, reduction='none').mean(1)
    return ((bce + rank_weight * ranking_loss(logits, correct)) * weights).sum() / weights.sum()


class EnhancementAgent(legacy.Agent):
    def __init__(self, route_width, embedding_width, classes, device, seed, config,
                 offline=False, policy_state=None, prerouter_state=None, gamma=.95,
                 informative_weight=1., slots=3, query_value_only=False, paired_query_only=False):
        if paired_query_only and (not query_value_only or config.paired_weight <= 0):
            raise ValueError('Paired-only training requires frozen heads and a positive paired weight.')
        base_state = ({k: v for k, v in policy_state.items() if not k.startswith('comparison.')}
                      if config.joint_router and policy_state is not None else policy_state)
        super().__init__(route_width, device, seed, offline=offline, policy_state=base_state, gamma=gamma)
        self.slots = slots
        policy = EfficientQ(route_width, config.joint_router, slots).to(device)
        load_policy_state(policy, self.policy.state_dict() if policy_state is None else policy_state)
        self.policy = policy.eval()
        self.target = deepcopy(policy).eval()
        self.query_value_only = query_value_only
        self.paired_query_only = paired_query_only
        self.paired_target_totals = dict(rows=0, steps=0, utility=0., future=0., target=0.,
            predicted=0., positive_utility=0, positive_target=0, terminal=0, absolute_loss=0.)
        if query_value_only:
            self.policy.requires_grad_(False)
            self.policy.update_value.requires_grad_(True)
        self.config = config
        self.informative_weight = informative_weight
        self.versions = np.zeros(slots, dtype=np.int64)
        self.route_optimizer = torch.optim.Adam(self.policy.routing_parameters(),
            lr=.001 if offline else config.route_lr_online, weight_decay=1e-4)
        td_parameters = list(self.policy.update_value.parameters())
        if not query_value_only: td_parameters = list(self.policy.future.parameters()) + td_parameters
        self.td_optimizer = torch.optim.Adam(td_parameters,
                                             lr=.0001, weight_decay=1e-4)
        self.prior = {name: value.detach().clone() for name, value in self.policy.named_routing_parameters()}
        self.pre_router = PreRouter(embedding_width, classes).to(device).eval()
        if prerouter_state is not None: self.pre_router.load_state_dict(prerouter_state)
        self.pre_optimizer = torch.optim.Adam(self.pre_router.parameters(), lr=.001 if offline else config.route_lr_online)
        self.last_losses = {}
        self.fresh_route_updates = 0
        self.fresh_route_rows = 0
        self.protected_route_updates = 0
        self.protected_route_rows = 0
        self.counterfactual_memory = deque(maxlen=512)
        self.counterfactual_optimizer = None
        self.counterfactual_updates = 0

    def _next_values(self, states):
        selection, _, _, _ = self.policy.forward_for_action(states)
        chosen = masked_q(selection, states, self.slots).argmax(1)
        # Choose with actor scores; evaluate in raw target-critic units.
        return self.target(states).gather(1, chosen[:, None]).squeeze(1)

    def refresh_routes(self, states, correct):
        """Fit current-bank labels directly, without replaying obsolete targets.

        Inputs must be reconstructed from previously available reference labels.
        This optimizer owns correctness parameters only; Context/TD heads are fixed.
        """
        if self.query_value_only: raise RuntimeError('Routing is frozen during query-value-only training.')
        logits = self.policy.route_logits(states.detach())
        correct = correct.detach().float()
        loss = F.binary_cross_entropy_with_logits(logits, correct)
        loss = loss + self.config.rank_weight * ranking_loss(logits, correct).mean()
        if not self.offline and self.config.route_prior_weight:
            loss = loss + self.config.route_prior_weight * sum(
                (value-self.prior[name]).square().mean()
                for name, value in self.policy.named_routing_parameters())
        self.route_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.routing_parameters(), 1)
        self.route_optimizer.step()
        self.fresh_route_updates += 1
        self.fresh_route_rows += len(states)
        self.last_losses['fresh_routing'] = float(loss.detach())

    def _sample(self, memory, half_life):
        samples, weights = super()._sample(memory, half_life)
        versions = np.asarray([row[-2] for row in samples])
        distance = np.abs(versions - self.versions).sum(1)
        weights *= torch.tensor(np.maximum(.05, np.exp2(-distance / self.config.version_half_life)),
                                device=self.device, dtype=weights.dtype)
        return samples, weights

    def observe_routes(self, state, correct, step_now=True, importance=1., versions=None, protected=None):
        if self.query_value_only: return
        version = tuple(self.versions if versions is None else versions)
        self.route_memory.append((state.detach().cpu().reshape(-1).clone(),
            correct.detach().cpu().float().reshape(self.slots+1).clone(), float(importance), version, self.clock))
        if not step_now: return
        samples, weight = self._sample(self.route_memory, 4096 if self.offline else 128)
        x = torch.stack([row[0] for row in samples]).to(self.device)
        y = torch.stack([row[1] for row in samples]).to(self.device)
        logits = self.policy.route_logits(x)
        weight *= torch.tensor([row[2] for row in samples], device=self.device)
        loss = routing_objective(logits, y, weight, self.config.rank_weight, self.informative_weight)
        if protected is not None:
            protected_states, protected_correct, protected_weights = protected
            protected_logits = self.policy.route_logits(protected_states.detach())
            protected_loss = routing_objective(protected_logits, protected_correct.detach().float(),
                protected_weights, self.config.rank_weight, self.informative_weight)
            loss = .5*(loss + protected_loss)
            self.protected_route_updates += 1
            self.protected_route_rows += len(protected_states)
            self.last_losses['protected_routing'] = float(protected_loss.detach())
        if not self.offline and self.config.route_prior_weight:
            prior = sum((value - self.prior[name]).square().mean()
                        for name, value in self.policy.named_routing_parameters())
            loss = loss + self.config.route_prior_weight * prior
        self.route_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.routing_parameters(), 1)
        self.route_optimizer.step(); self.route_updates += 1
        self.last_losses['routing'] = float(loss.detach())

    def push(self, state, action, reward, successor, terminal, versions=None):
        self.td_memory.append((state.detach().cpu().reshape(-1).clone(), action, float(reward),
            successor.detach().cpu().reshape(-1).clone(), float(terminal),
            tuple(self.versions if versions is None else versions), self.clock))

    def push_pair(self, state, action, utility, actual_next, skip_next, terminal, versions=None):
        self.pair_memory.append((state.detach().cpu().reshape(-1).clone(), action, float(utility),
            actual_next.detach().cpu().reshape(-1).clone(), skip_next.detach().cpu().reshape(-1).clone(),
            float(terminal), tuple(self.versions if versions is None else versions), self.clock))

    def learn_td(self):
        if len(self.td_memory) < 32: return
        if self.paired_query_only and not self.pair_memory: return
        rows, weight = self._sample(self.td_memory, 1024 if self.offline else 128)
        states = torch.stack([row[0] for row in rows]).to(self.device)
        next_states = torch.stack([row[3] for row in rows]).to(self.device)
        actions = torch.tensor([row[1] for row in rows], device=self.device)
        rewards = torch.tensor([row[2] for row in rows], device=self.device)
        terminal = torch.tensor([row[4] for row in rows], device=self.device)
        with torch.set_grad_enabled(not self.paired_query_only):
            current = self.policy(states, detach_routes=True).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad(): target = rewards + self.gamma * (1 - terminal) * self._next_values(next_states)
        loss = (F.smooth_l1_loss(current, target, reduction='none') * weight).sum() / weight.sum()
        if self.config.paired_weight and self.pair_memory:
            pairs, pw = self._sample(self.pair_memory, 1024 if self.offline else 128)
            px = torch.stack([row[0] for row in pairs]).to(self.device)
            pa = torch.tensor([row[1] for row in pairs], device=self.device)
            pu = torch.tensor([row[2] for row in pairs], device=self.device)
            pn = torch.stack([row[3] for row in pairs]).to(self.device)
            ps = torch.stack([row[4] for row in pairs]).to(self.device)
            pt = torch.tensor([row[5] for row in pairs], device=self.device)
            q = self.policy(px, detach_routes=True)
            # Each coupled query maps to its Context prediction-only route.
            advantage = q.gather(1, pa[:, None]).squeeze(1) - q.gather(1, (pa - self.slots)[:, None]).squeeze(1)
            with torch.no_grad():
                future = self.gamma * (1 - pt) * (self._next_values(pn) - self._next_values(ps))
                target = pu + future
            pair_loss = (F.smooth_l1_loss(advantage, target, reduction='none') * pw).sum() / pw.sum()
            if self.paired_query_only:
                totals = self.paired_target_totals
                totals['rows'] += len(pairs); totals['steps'] += 1
                for key, values in (('utility',pu),('future',future),('target',target),('predicted',advantage)):
                    totals[key] += float(values.detach().sum())
                totals['positive_utility'] += int((pu > 0).sum())
                totals['positive_target'] += int((target > 0).sum())
                totals['terminal'] += int(pt.sum())
                totals['absolute_loss'] += float(loss.detach())
                loss = self.config.paired_weight * pair_loss
            else:
                loss = loss + self.config.paired_weight * pair_loss
            self.last_losses['paired'] = float(pair_loss.detach())
        self.td_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.future.parameters()) + list(self.policy.update_value.parameters()), 1)
        self.td_optimizer.step(); self.td_updates += 1
        if self.td_updates % 15 == 0: self.target.load_state_dict(self.policy.state_dict())

    def observe_counterfactual(self, state, utilities, successors, skip_next, terminal, versions):
        """Fit only raw update values; routing, future head and target stay fixed.

        Each paid TRAIN row supplies all three alternatives at the same bank
        version. Centered regression supplies direct between-Context comparisons.
        """
        self.counterfactual_memory.append((state.detach().cpu().reshape(-1).clone(),
            torch.as_tensor(utilities, dtype=state.dtype).cpu().clone(),
            successors.detach().cpu().clone(), skip_next.detach().cpu().reshape(-1).clone(),
            float(terminal), tuple(versions), self.clock))
        if len(self.counterfactual_memory) < 8: return
        if self.counterfactual_optimizer is None:
            self.counterfactual_optimizer = torch.optim.Adam(self.policy.update_value.parameters(),
                                                            lr=.0001, weight_decay=1e-4)
        rows, weight = self._sample(self.counterfactual_memory, 1024)
        states = torch.stack([r[0] for r in rows]).to(self.device)
        utilities = torch.stack([r[1] for r in rows]).to(self.device)
        successors = torch.stack([r[2] for r in rows]).to(self.device)
        skipped = torch.stack([r[3] for r in rows]).to(self.device)
        terminal = torch.tensor([r[4] for r in rows], device=self.device)
        with torch.no_grad():
            continuation = self._next_values(successors.flatten(0,1)).reshape(-1,3)
            baseline = self._next_values(skipped)[:,None]
            targets = utilities + self.gamma*(1-terminal[:,None])*(continuation-baseline)
        predicted = self.policy.forward_components(states, detach_routes=True)[2]
        absolute = F.smooth_l1_loss(predicted, targets, reduction='none').mean(1)
        relative = F.smooth_l1_loss(predicted-predicted.mean(1,keepdim=True),
            targets-targets.mean(1,keepdim=True), reduction='none').mean(1)
        loss = ((absolute + .5*relative)*weight).sum()/weight.sum()
        self.counterfactual_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.update_value.parameters(), 1)
        self.counterfactual_optimizer.step()
        self.counterfactual_updates += 1
        self.last_losses['counterfactual'] = float(loss.detach())
        self.last_losses['counterfactual_relative'] = float(relative.mean().detach())

    def observe_pre_router(self, embedding, frozen, correct, teacher_logits):
        logits = self.pre_router(embedding.detach(), frozen.detach())
        target = (1 - self.config.prerouter_weight) * correct.reshape_as(logits).float()
        target = target + self.config.prerouter_weight * teacher_logits.detach().sigmoid()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        with torch.no_grad():
            candidate = logits.argmax(1)
            teacher = teacher_logits.reshape_as(logits).argmax(1)
            truth = correct.reshape_as(logits).float()
            recovered = truth.gather(1, teacher[:, None]).squeeze(1) - truth.gather(1, candidate[:, None]).squeeze(1)
            extra_contexts = 3 - (candidate != 0).float()
            compute_target = 2 * recovered - self.config.compute_cost * extra_contexts
        compute_value = self.pre_router.fallback_value(embedding.detach(), frozen.detach())
        loss = loss + F.smooth_l1_loss(compute_value, compute_target)
        self.pre_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.pre_router.parameters(), 1); self.pre_optimizer.step()


class DeviceReliability:
    """Original reliability semantics implemented on the inference device."""
    def __init__(self, classes, device, routes=4):
        self.routes = routes
        self.accuracy = torch.full((routes,), .5, device=device)
        self.ce = torch.full((routes,), .5, device=device)
        self.change = torch.zeros(routes, device=device)
        self.updates = torch.zeros(routes, device=device)
        self.last_update = torch.zeros(routes, device=device)
        self.class_good = torch.zeros(routes, classes, device=device)
        self.class_count = torch.zeros(routes, classes, device=device)
        self.paid = 0
        self.budget = None

    def features(self, probs, cap, position, count):
        predicted = probs.argmax(2).transpose(0, 1)
        support = self.class_count.gather(1, predicted).transpose(0, 1)
        good = self.class_good.gather(1, predicted).transpose(0, 1)
        reliability = (good + 4 * self.accuracy) / (support + 4)
        shared = torch.stack([self.accuracy, self.ce, self.updates / max(cap, 1),
            torch.full_like(self.accuracy, self.paid / max(cap, 1)),
            (position - self.last_update) / max(count, 1), self.change], 1)
        return torch.cat([shared[None].expand(len(probs), -1, -1), reliability[:, :, None],
                          (support / (support + 8))[:, :, None]], 2).to(probs.dtype)

    @torch.no_grad()
    def observe_paid(self, probs, truth):
        probs = probs.reshape(self.routes, -1)
        predicted = probs.argmax(1)
        good = predicted == truth
        ce = -probs[:, truth].clamp_min(1e-8).log()
        self.accuracy.mul_(.9).add_(good, alpha=.1)
        self.ce.mul_(.9).add_(ce / (1 + ce), alpha=.1)
        self.class_count.mul_(.99); self.class_good.mul_(.99)
        rows = torch.arange(self.routes, device=probs.device)
        self.class_count[rows, predicted] += 1
        self.class_good[rows, predicted] += good
        self.paid += 1

    def updated(self, route, position, change):
        self.updates[route] += 1; self.last_update[route] = position
        self.change[route] = .9 * self.change[route] + .1 * change
