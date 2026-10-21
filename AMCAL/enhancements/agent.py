"""Ranking-supervised routing and paired update advantages; no Context gradients."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import amcal_burst_structured_bank_protocol as legacy
from .models import PreRouter


class EfficientQ(legacy.StructuredQ):
    def forward_with_logits(self, states, detach_routes=False):
        route_x = states[:, :4 * self.route_width].reshape(-1, 4, self.route_width)
        logits = self.correctness(route_x).squeeze(-1)
        reward = 2 * logits.sigmoid() - 1
        if detach_routes: reward = reward.detach()
        pooled = torch.cat([route_x.mean(1), route_x.max(1).values, states[:, -2:]], 1)
        future = self.future(pooled)
        features = torch.cat([route_x[:, 1:], pooled[:, None].expand(-1, 3, -1)], 2)
        update = self.update_value(features).squeeze(-1)
        return torch.cat([future + reward, future + reward[:, 1:] + update], 1), logits

    def forward(self, states, detach_routes=False):
        return self.forward_with_logits(states, detach_routes)[0]


def ranking_loss(logits, correct):
    pairs = correct[:, :, None].bool() & ~correct[:, None, :].bool()
    losses = F.softplus(logits[:, None, :] - logits[:, :, None])
    return (losses * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)


class EnhancementAgent(legacy.Agent):
    def __init__(self, route_width, embedding_width, classes, device, seed, config,
                 offline=False, policy_state=None, prerouter_state=None, gamma=.95):
        super().__init__(route_width, device, seed, offline=offline, policy_state=policy_state, gamma=gamma)
        policy = EfficientQ(route_width).to(device)
        policy.load_state_dict(self.policy.state_dict())
        self.policy = policy.eval()
        self.target = deepcopy(policy).eval()
        self.config = config
        self.versions = np.zeros(3, dtype=np.int64)
        self.route_optimizer = torch.optim.Adam(self.policy.correctness.parameters(),
            lr=.001 if offline else config.route_lr_online, weight_decay=1e-4)
        self.td_optimizer = torch.optim.Adam(list(self.policy.future.parameters()) + list(self.policy.update_value.parameters()),
                                             lr=.0001, weight_decay=1e-4)
        self.prior = {name: value.detach().clone() for name, value in self.policy.correctness.named_parameters()}
        self.pre_router = PreRouter(embedding_width, classes).to(device).eval()
        if prerouter_state is not None: self.pre_router.load_state_dict(prerouter_state)
        self.pre_optimizer = torch.optim.Adam(self.pre_router.parameters(), lr=.001 if offline else config.route_lr_online)
        self.last_losses = {}

    def _sample(self, memory, half_life):
        samples, weights = super()._sample(memory, half_life)
        versions = np.asarray([row[-2] for row in samples])
        distance = np.abs(versions - self.versions).sum(1)
        weights *= torch.tensor(np.maximum(.05, np.exp2(-distance / self.config.version_half_life)),
                                device=self.device, dtype=weights.dtype)
        return samples, weights

    def observe_routes(self, state, correct, step_now=True, importance=1., versions=None):
        version = tuple(self.versions if versions is None else versions)
        self.route_memory.append((state.detach().cpu().reshape(-1).clone(),
            correct.detach().cpu().float().reshape(4).clone(), float(importance), version, self.clock))
        if not step_now: return
        samples, weight = self._sample(self.route_memory, 4096 if self.offline else 128)
        x = torch.stack([row[0] for row in samples]).to(self.device)
        y = torch.stack([row[1] for row in samples]).to(self.device)
        logits = self.policy.route_logits(x)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(1)
        rank = ranking_loss(logits, y)
        weight *= torch.tensor([row[2] for row in samples], device=self.device)
        loss = ((bce + self.config.rank_weight * rank) * weight).sum() / weight.sum()
        if not self.offline and self.config.route_prior_weight:
            prior = sum((value - self.prior[name]).square().mean()
                        for name, value in self.policy.correctness.named_parameters())
            loss = loss + self.config.route_prior_weight * prior
        self.route_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.correctness.parameters(), 1)
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
        rows, weight = self._sample(self.td_memory, 1024 if self.offline else 128)
        states = torch.stack([row[0] for row in rows]).to(self.device)
        next_states = torch.stack([row[3] for row in rows]).to(self.device)
        actions = torch.tensor([row[1] for row in rows], device=self.device)
        rewards = torch.tensor([row[2] for row in rows], device=self.device)
        terminal = torch.tensor([row[4] for row in rows], device=self.device)
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
            # Coupled query action 4/5/6 predicts through route 1/2/3.
            advantage = q.gather(1, pa[:, None]).squeeze(1) - q.gather(1, (pa - 3)[:, None]).squeeze(1)
            with torch.no_grad(): target = pu + self.gamma * (1 - pt) * (self._next_values(pn) - self._next_values(ps))
            pair_loss = (F.smooth_l1_loss(advantage, target, reduction='none') * pw).sum() / pw.sum()
            loss = loss + self.config.paired_weight * pair_loss
            self.last_losses['paired'] = float(pair_loss.detach())
        self.td_optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.future.parameters()) + list(self.policy.update_value.parameters()), 1)
        self.td_optimizer.step(); self.td_updates += 1
        if self.td_updates % 15 == 0: self.target.load_state_dict(self.policy.state_dict())

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
    def __init__(self, classes, device):
        self.accuracy = torch.full((4,), .5, device=device)
        self.ce = torch.full((4,), .5, device=device)
        self.change = torch.zeros(4, device=device)
        self.updates = torch.zeros(4, device=device)
        self.last_update = torch.zeros(4, device=device)
        self.class_good = torch.zeros(4, classes, device=device)
        self.class_count = torch.zeros(4, classes, device=device)
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
        probs = probs.reshape(4, -1)
        predicted = probs.argmax(1)
        good = predicted == truth
        ce = -probs[:, truth].clamp_min(1e-8).log()
        self.accuracy.mul_(.9).add_(good, alpha=.1)
        self.ce.mul_(.9).add_(ce / (1 + ce), alpha=.1)
        self.class_count.mul_(.99); self.class_good.mul_(.99)
        rows = torch.arange(4, device=probs.device)
        self.class_count[rows, predicted] += 1
        self.class_good[rows, predicted] += good
        self.paid += 1

    def updated(self, route, position, change):
        self.updates[route] += 1; self.last_update[route] = position
        self.change[route] = .9 * self.change[route] + .1 * change
