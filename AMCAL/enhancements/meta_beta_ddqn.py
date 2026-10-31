"""TRAIN-only bilevel reward weight for the existing two-action Double-DQN."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from .two_action_ddqn import QueryDDQN
from .paper_gain_reward import SameDomainGain, additive_reward


class ComponentReward(float):
    def __new__(cls, paper, gain, beta):
        value = float.__new__(cls, float(paper)+float(beta)*float(gain))
        value.paper, value.gain = float(paper), float(gain)
        return value


class LearnedGainReward(SameDomainGain):
    def __init__(self, probe, excluded_reference, learner, args):
        super().__init__(probe, excluded_reference, 1.)
        self.learner, self.args = learner, args

    def finish(self, cnn, context, before, paper_reward):
        after = self.measure(cnn, context)
        beta = self.learner.beta_value()
        self.args.beta = beta
        _, parts = additive_reward(paper_reward, before-after, beta)
        parts.update(generalization_loss_before=float(before),generalization_loss_after=float(after))
        return ComponentReward(paper_reward,before-after,beta), parts


class MetaRecorder:
    """Collect independent TRAIN trajectories without changing the live Q or beta."""
    def __init__(self, learner):
        self.policy = deepcopy(learner.policy).eval().requires_grad_(False)
        self.updates = self.steps = 0
        self.rows = []

    @torch.no_grad()
    def choose(self, state, can_query, epsilon, rng):
        q = self.policy(torch.as_tensor(state,device=next(self.policy.parameters()).device)).reshape(-1,2)[0].cpu().numpy()
        if not np.isfinite(q).all(): raise ValueError('Nonfinite meta-collection Q.')
        if not can_query: return 0,q
        action = int(rng.integers(2)) if rng.random() < epsilon else int(q.argmax())
        return action,q

    def remember(self,state,action,reward,successor,terminal,next_can_query):
        if action == 1:
            if not isinstance(reward,ComponentReward): raise ValueError('Missing meta TRAIN gain.')
            self.rows.append((torch.as_tensor(state).detach().cpu().clone(),reward.paper,reward.gain))
        return None


class MetaBetaDDQN(QueryDDQN):
    """Q uses DDQN TD; beta uses held-out TRAIN utility through a virtual SGD step.

    The virtual step is a local SGD approximation to the actual Adam update.
    It never changes policy/target/critic optimizer. Historical rewards are
    reconstructed from their two raw components at the current detached beta.
    """
    def __init__(self,*args,meta_lr=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.log_beta = nn.Parameter(torch.zeros((),device=self.device))
        self.meta_optimizer = torch.optim.Adam([self.log_beta],lr=meta_lr)
        self.meta_rows = []
        self.meta_updates = 0
        self.calibrated = False
        self.gain_scale = 1.
        self.virtual_lr = self.optimizer.param_groups[0]['lr']
        self.meta_history = []

    def beta_value(self): return float(self.log_beta.detach().exp())

    def set_meta_rows(self,rows):
        if len(rows) < self.batch_size: raise ValueError('Insufficient paid independent TRAIN meta rows.')
        states = [self._state(row[0]) for row in rows]
        if any(not math.isfinite(row[1]) or not math.isfinite(row[2]) for row in rows):
            raise ValueError('Nonfinite TRAIN meta targets.')
        self.meta_rows = [(s,float(r[1]),float(r[2])) for s,r in zip(states,rows)]
        if not self.calibrated:
            paper = np.asarray([r[1] for r in rows]); gain = np.asarray([r[2] for r in rows])
            self.gain_scale = float(np.sqrt(np.mean(gain**2)))
            paper_scale = float(np.sqrt(np.mean(paper**2)))
            if self.gain_scale <= 1e-8:
                raise ValueError('TRAIN gain is effectively zero; beta is unidentifiable. No synthetic targets added.')
            initial = max(paper_scale,1.)/self.gain_scale
            with torch.no_grad(): self.log_beta.copy_(self.log_beta.new_tensor(math.log(initial)).clamp(-8,8))
            self.calibrated = True

    def remember(self,state,action,reward,successor,terminal,next_can_query):
        if action not in (0,1) or not math.isfinite(float(reward)) or (terminal and next_can_query):
            raise ValueError('Invalid real transition.')
        if action == 1:
            if not isinstance(reward,ComponentReward): raise ValueError('Paid TRAIN transition lacks raw components.')
            paper,gain = reward.paper,reward.gain
        else:
            if float(reward) != 0.: raise ValueError('Skip reward must be zero.')
            paper,gain = 0.,0.
        self.memory.append((self._state(state),int(action),float(paper),float(gain),
            self._state(successor),bool(terminal),bool(next_can_query)))
        self.steps += 1
        return self.learn() if self.steps % self.learn_every == 0 else None

    def batch(self):
        at = self.rng.choice(len(self.memory),self.batch_size,replace=False)
        rows = [self.memory[int(i)] for i in at]
        return (torch.stack([r[0] for r in rows]).to(self.device),
            torch.tensor([r[1] for r in rows],device=self.device),
            torch.tensor([r[2] for r in rows],device=self.device),
            torch.tensor([r[3] for r in rows],device=self.device),
            torch.stack([r[4] for r in rows]).to(self.device),
            torch.tensor([r[5] for r in rows],device=self.device),
            torch.tensor([r[6] for r in rows],device=self.device))

    def targets(self,batch,beta):
        _,_,paper,gain,successor,terminal,can_query = batch
        with torch.no_grad():
            online = self.policy(successor).clone()
            online[~can_query,1] = -torch.inf
            selected = online.argmax(1)
            future = self.target(successor).gather(1,selected[:,None]).squeeze(1)
            future = torch.where(terminal,torch.zeros_like(future),future)
        return paper+beta*gain+self.gamma*future

    def meta_step(self,batch):
        if not self.meta_rows or not self.calibrated: return None
        states,actions = batch[:2]
        params = dict(self.policy.named_parameters())
        buffers = dict(self.policy.named_buffers())
        predicted = self.policy(states).gather(1,actions[:,None]).squeeze(1)
        inner = F.smooth_l1_loss(predicted,self.targets(batch,self.log_beta.exp()))
        grads = torch.autograd.grad(inner,tuple(params.values()),create_graph=True)
        virtual = {key:weight-self.virtual_lr*grad for (key,weight),grad in zip(params.items(),grads)}
        at = self.rng.choice(len(self.meta_rows),self.batch_size,replace=False)
        rows = [self.meta_rows[int(i)] for i in at]
        meta_states = torch.stack([r[0] for r in rows]).to(self.device)
        gain = torch.tensor([r[2] for r in rows],device=self.device)/self.gain_scale
        q = functional_call(self.policy,(virtual,buffers),(meta_states,))
        advantage = q[:,1]-q[:,0]
        temperature = advantage.detach().square().mean().sqrt().clamp_min(1.)
        # Gain magnitude and sign are both preserved; no labels from VAL/TEST.
        objective = -(gain.detach()*torch.sigmoid(advantage/temperature)).mean()
        gradient, = torch.autograd.grad(objective,self.log_beta)
        if not torch.isfinite(gradient): raise ValueError('Nonfinite beta hypergradient.')
        before = self.beta_value()
        self.meta_optimizer.zero_grad(set_to_none=True)
        self.log_beta.grad = gradient.detach().clone()
        nn.utils.clip_grad_norm_([self.log_beta],1.,error_if_nonfinite=True)
        self.meta_optimizer.step()
        with torch.no_grad(): self.log_beta.clamp_(-8,8)
        self.meta_updates += 1
        report = dict(update=self.meta_updates,beta_before=before,beta_after=self.beta_value(),
            hypergradient=float(gradient.detach()),meta_loss=float(objective.detach()),
            mean_meta_gain=float(gain.mean())*self.gain_scale,
            positive_meta_fraction=float((gain>0).float().mean()),
            beta_at_numerical_guard=bool(abs(float(self.log_beta.detach()))>=8.-1e-6))
        self.meta_history.append(report)
        return report

    def learn(self):
        if len(self.memory) < self.batch_size or not self.calibrated: return None
        batch = self.batch()
        meta = self.meta_step(batch) if self.updates % self.target_every == 0 else None
        current = self.policy(batch[0]).gather(1,batch[1][:,None]).squeeze(1)
        target = self.targets(batch,self.log_beta.detach().exp())
        loss = F.smooth_l1_loss(current,target)
        if not torch.isfinite(loss): raise ValueError('Nonfinite learned-beta DDQN loss.')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step(); self.updates += 1
        if self.updates % self.target_every == 0: self.target.load_state_dict(self.policy.state_dict())
        return dict(loss=float(loss.detach()),updates=self.updates,beta=self.beta_value(),meta=meta)

    def snapshot(self):
        return dict(version=1,policy=deepcopy(self.policy.state_dict()),target=deepcopy(self.target.state_dict()),
            optimizer=deepcopy(self.optimizer.state_dict()),replay=list(self.memory),
            rng=deepcopy(self.rng.bit_generator.state),steps=self.steps,updates=self.updates,
            log_beta=self.log_beta.detach().cpu().clone(),meta_optimizer=deepcopy(self.meta_optimizer.state_dict()),
            meta_rows=self.meta_rows,meta_updates=self.meta_updates,calibrated=self.calibrated,
            gain_scale=self.gain_scale,meta_history=deepcopy(self.meta_history))

    def restore(self,saved):
        if saved['version'] != 1: raise ValueError('Unknown learned-beta trainer format.')
        self.policy.load_state_dict(saved['policy']); self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.memory = deque(saved['replay'],maxlen=self.memory.maxlen)
        self.rng.bit_generator.state = deepcopy(saved['rng'])
        self.steps,self.updates = int(saved['steps']),int(saved['updates'])
        with torch.no_grad(): self.log_beta.copy_(saved['log_beta'].to(self.device))
        self.meta_optimizer.load_state_dict(saved['meta_optimizer'])
        self.meta_rows = saved['meta_rows']; self.meta_updates = saved['meta_updates']
        self.calibrated = saved['calibrated']; self.gain_scale = saved['gain_scale']
        self.meta_history = saved['meta_history']

