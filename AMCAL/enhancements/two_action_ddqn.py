"""Two-action Double-DQN with the existing compact Actor's TRAIN features."""
from collections import deque
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QueryQ(nn.Module):
    def __init__(self, spec, actor_state=None):
        super().__init__()
        mean = torch.as_tensor(spec['mean'], dtype=torch.float32)
        scale = torch.as_tensor(spec['scale'], dtype=torch.float32)
        if mean.ndim != 1 or mean.shape != scale.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid frozen TRAIN feature normalization.')
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.network = nn.Sequential(nn.Linear(len(mean), 16), nn.GELU(), nn.Linear(16, 2))
        if actor_state is not None:
            with torch.no_grad():
                self.network[0].weight.copy_(actor_state['actor.0.weight'])
                self.network[0].bias.copy_(actor_state['actor.0.bias'])
        # Actor logits are not discounted returns; transfer only the encoder.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state):
        if state.ndim == 1: state = state.unsqueeze(0)
        return self.network(((state-self.mean)/self.scale).clamp(-10, 10))


class QueryDDQN:
    def __init__(self, spec, actor_state, device, seed=42, gamma=.95, lr=.0001,
                 batch_size=32, learn_every=4, target_every=15, capacity=4096):
        if not 0 <= gamma <= 1 or not math.isfinite(lr) or lr <= 0 or min(batch_size, learn_every, target_every, capacity) < 1:
            raise ValueError('Invalid DDQN settings.')
        self.device = torch.device(device)
        self.gamma, self.batch_size = gamma, batch_size
        self.learn_every, self.target_every = learn_every, target_every
        self.policy = QueryQ(spec, actor_state).to(self.device).eval()
        self.target = deepcopy(self.policy).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed+8927)
        self.steps = self.updates = 0

    @torch.no_grad()
    def values(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        values = self.policy(state).reshape(-1, 2)[0]
        if not torch.isfinite(values).all(): raise ValueError('Nonfinite DDQN values.')
        return values.cpu().numpy().copy()

    def choose(self, state, can_query, epsilon, rng):
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('Invalid epsilon.')
        values = self.values(state)
        if not can_query: return 0, values
        action = int(rng.integers(2)) if epsilon and rng.random() < epsilon else int(values.argmax())
        return action, values

    def _state(self, state):
        value = torch.as_tensor(state, dtype=torch.float32).detach().cpu().reshape(-1).clone()
        if value.shape != self.policy.mean.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid replay state.')
        return value

    def remember(self, state, action, reward, successor, terminal, next_can_query):
        if action not in (0, 1) or not math.isfinite(reward) or (terminal and next_can_query):
            raise ValueError('Invalid real transition.')
        self.memory.append((self._state(state), int(action), float(reward), self._state(successor),
                            bool(terminal), bool(next_can_query)))
        self.steps += 1
        return self.learn() if self.steps % self.learn_every == 0 else None

    def learn(self):
        if len(self.memory) < self.batch_size: return None
        at = self.rng.choice(len(self.memory), self.batch_size, replace=False)
        rows = [self.memory[int(i)] for i in at]
        states = torch.stack([r[0] for r in rows]).to(self.device)
        successor = torch.stack([r[3] for r in rows]).to(self.device)
        actions = torch.tensor([r[1] for r in rows], device=self.device)
        rewards = torch.tensor([r[2] for r in rows], device=self.device)
        terminal = torch.tensor([r[4] for r in rows], device=self.device)
        can_query = torch.tensor([r[5] for r in rows], device=self.device)
        current = self.policy(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online = self.policy(successor).clone()
            online[~can_query, 1] = -torch.inf
            selected = online.argmax(1)
            future = self.target(successor).gather(1, selected[:, None]).squeeze(1)
            target = rewards + self.gamma*torch.where(terminal, torch.zeros_like(future), future)
        loss = F.smooth_l1_loss(current, target)
        if not torch.isfinite(loss): raise ValueError('Nonfinite DDQN loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step(); self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return dict(loss=float(loss.detach()), reward=float(rewards.mean()), target=float(target.mean()),
                    predicted=float(current.detach().mean()), updates=self.updates)

    def snapshot(self):
        return dict(policy=deepcopy(self.policy.state_dict()), target=deepcopy(self.target.state_dict()),
            optimizer=deepcopy(self.optimizer.state_dict()), replay=list(self.memory),
            rng=deepcopy(self.rng.bit_generator.state), steps=self.steps, updates=self.updates)

    def restore(self, saved):
        self.policy.load_state_dict(saved['policy']); self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.memory = deque(saved['replay'], maxlen=self.memory.maxlen)
        self.rng.bit_generator.state = deepcopy(saved['rng'])
        self.steps, self.updates = int(saved['steps']), int(saved['updates'])
