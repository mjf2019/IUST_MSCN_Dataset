"""Structured 3-Context / 7-action DDQN with directly supervised route correctness.

Frozen CNN and initial Context come from an existing healthy source run.
Offline correctness targets use existing TRAIN labels densely; Context updates
remain budgeted. Online learners see only paid labels after the scored prediction.
Optional evaluation quality/yield feedback control: see QUALITY_BUDGET.md.
Optional learned budget policy (--budget-aware-policy): see RL_BUDGET_POLICY.md.
"""
import argparse
from collections import deque
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import amcal_burst_context_bank_protocol as base
from baselines.stream_timing import StreamTimer, timing_row, timing_report
from baselines.quality_budget import QualityBudget, VERSION as QUALITY_VERSION
from baselines.rl_budget import BudgetState, GLOBAL_WIDTH, SLOT_WIDTH, VERSION as BUDGET_VERSION
from baselines.balanced_reference import balanced_indices, macro_transfer
core = base.core
SLOTS = 3
ROUTES = 4
ACTIONS = 7


def masked_q(q, states):
    result = q.clone()
    gate, remaining = states[:, -3], states[:, -2]
    result[(remaining <= 0) | (gate == 0), 4:] = -torch.inf
    result[(remaining > 0) & (gate == 1), :4] = -torch.inf
    fixed=(remaining>0)&(gate>=2)
    result[fixed,:4]=-torch.inf
    for k in range(SLOTS):
        result[fixed&(gate!=k+2),k+4]=-torch.inf
    return result


class StructuredQ(nn.Module):
    def __init__(self, route_width, budget_aware=False, decoupled_query=False):
        super().__init__()
        self.route_width = route_width
        self.budget_aware = budget_aware
        self.decoupled_query = decoupled_query
        # This scorer is shared by all routes, with an explicit frozen-route flag.
        self.correctness = nn.Sequential(nn.Linear(route_width,128),nn.ReLU(),
                                         nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        pooled = 2*route_width + 2 + (GLOBAL_WIDTH+2*SLOT_WIDTH if budget_aware else 0)
        self.future = nn.Sequential(nn.Linear(pooled,128),nn.ReLU(),nn.Linear(128,1))
        self.update_value = nn.Sequential(nn.Linear(pooled+route_width+(SLOT_WIDTH if budget_aware else 0),128),nn.ReLU(),nn.Linear(128,1))
        nn.init.zeros_(self.future[-1].weight); nn.init.zeros_(self.future[-1].bias)
        nn.init.zeros_(self.update_value[-1].weight); nn.init.zeros_(self.update_value[-1].bias)

    def route_logits(self, states):
        route_x = states[:, :ROUTES*self.route_width].reshape(-1,ROUTES,self.route_width)
        return self.correctness(route_x).squeeze(-1)

    def forward(self, states, detach_routes=False):
        route_x = states[:, :ROUTES*self.route_width].reshape(-1,ROUTES,self.route_width)
        p = self.correctness(route_x).squeeze(-1).sigmoid()
        reward = 2*p - 1
        if detach_routes: reward = reward.detach()
        # Schedule gate masks actions; it is not a shortcut feature for scoring.
        pooled = torch.cat([route_x.mean(1),route_x.max(1).values,states[:,-2:]],dim=1)
        if self.budget_aware:
            extra = states[:,ROUTES*self.route_width:-3]
            slots = extra[:,GLOBAL_WIDTH:].reshape(-1,ROUTES,SLOT_WIDTH)
            pooled = torch.cat([pooled,extra[:,:GLOBAL_WIDTH],slots.mean(1),slots.max(1).values],dim=1)
        future = self.future(pooled)
        context_x = torch.cat([route_x[:,1:],pooled[:,None,:].expand(-1,SLOTS,-1)],dim=2)
        if self.budget_aware:context_x=torch.cat([context_x,slots[:,1:]],dim=2)
        update = self.update_value(context_x).squeeze(-1)
        query_reward=reward.max(1,keepdim=True).values if (self.budget_aware or self.decoupled_query) else reward[:,1:]
        return torch.cat([future+reward,future+query_reward+update],dim=1)


class Agent:
    def __init__(self, route_width, device, seed, offline=False, policy_state=None, gamma=.95, budget_aware=False, decoupled_query=False):
        self.device, self.gamma, self.offline = device,gamma,offline
        self.policy = StructuredQ(route_width,budget_aware=budget_aware,decoupled_query=decoupled_query).to(device)
        if policy_state is not None:self.policy.load_state_dict(policy_state)
        self.target = deepcopy(self.policy).eval()
        self.route_optimizer = torch.optim.Adam(self.policy.correctness.parameters(),lr=.001,weight_decay=1e-4)
        self.td_optimizer = torch.optim.Adam(list(self.policy.future.parameters())+list(self.policy.update_value.parameters()),lr=.0001,weight_decay=1e-4)
        self.route_memory = deque(maxlen=8192 if offline else 256)
        self.td_memory = deque(maxlen=4096 if offline else 512)
        self.pair_memory = deque(maxlen=4096 if offline else 512)
        self.rng = np.random.default_rng(seed+920017)
        self.clock,self.route_updates,self.td_updates = 0,0,0
        self.policy.eval()

    def _sample(self, memory, half_life):
        ids = self.rng.choice(len(memory),min(32,len(memory)),replace=False)
        samples = [memory[int(i)] for i in ids]
        ages = np.asarray([self.clock-row[-1] for row in samples])
        weights = torch.tensor(np.maximum(.05,np.exp2(-ages/half_life)),device=self.device,dtype=torch.float32)
        return samples,weights

    def observe_routes(self, state, correct, step_now=True, importance=1.):
        row=(state.detach().cpu().reshape(-1).clone(),
             correct.detach().cpu().float().reshape(ROUTES).clone())
        self.route_memory.append(row+(float(importance),self.clock) if self.policy.budget_aware else row+(self.clock,))
        if not step_now:return
        samples,weight = self._sample(self.route_memory,4096 if self.offline else 128)
        x = torch.stack([r[0] for r in samples]).to(self.device)
        y = torch.stack([r[1] for r in samples]).to(self.device)
        loss = F.binary_cross_entropy_with_logits(self.policy.route_logits(x),y,reduction='none').mean(1)
        if self.policy.budget_aware:
            weight=weight*torch.tensor([r[2] for r in samples],device=self.device)
        loss = (loss*weight).sum()/weight.sum()
        self.route_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(self.policy.correctness.parameters(),1)
        self.route_optimizer.step();self.route_updates += 1

    def push(self, state, action, reward, successor, terminal):
        self.td_memory.append((state.detach().cpu().reshape(-1).clone(),action,float(reward),
                               successor.detach().cpu().reshape(-1).clone(),float(terminal),self.clock))

    def push_pair(self,state,action,utility,actual_next,skip_next,terminal):
        self.pair_memory.append((state.detach().cpu().reshape(-1).clone(),action,float(utility),
            actual_next.detach().cpu().reshape(-1).clone(),skip_next.detach().cpu().reshape(-1).clone(),
            float(terminal),self.clock))

    def _next_values(self,states):
        chosen=masked_q(self.policy(states),states).argmax(1)
        return self.target(states).gather(1,chosen[:,None]).squeeze(1)

    def learn_td(self):
        if len(self.td_memory)<32:return
        samples,weight = self._sample(self.td_memory,1024 if self.offline else 128)
        states = torch.stack([r[0] for r in samples]).to(self.device)
        successor = torch.stack([r[3] for r in samples]).to(self.device)
        actions = torch.tensor([r[1] for r in samples],device=self.device)
        rewards = torch.tensor([r[2] for r in samples],device=self.device)
        terminal = torch.tensor([r[4] for r in samples],device=self.device)
        current = self.policy(states,detach_routes=True).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            chosen = masked_q(self.policy(successor),successor).argmax(1)
            value = self.target(successor).gather(1,chosen[:,None]).squeeze(1)
            target = rewards+self.gamma*(1-terminal)*value
        loss = F.smooth_l1_loss(current,target,reduction='none')
        loss = (loss*weight).sum()/weight.sum()
        if self.policy.budget_aware and self.pair_memory:
            pairs,pair_weight=self._sample(self.pair_memory,1024 if self.offline else 128)
            px=torch.stack([r[0] for r in pairs]).to(self.device)
            pa=torch.tensor([r[1] for r in pairs],device=self.device)
            pu=torch.tensor([r[2] for r in pairs],device=self.device)
            pn=torch.stack([r[3] for r in pairs]).to(self.device)
            ps=torch.stack([r[4] for r in pairs]).to(self.device)
            terminal_pair=torch.tensor([r[5] for r in pairs],device=self.device)
            q=self.policy(px,detach_routes=True)
            advantage=q.gather(1,pa[:,None]).squeeze(1)-q[:,:ROUTES].max(1).values
            with torch.no_grad():
                paired_target=pu+self.gamma*(1-terminal_pair)*(self._next_values(pn)-self._next_values(ps))
            paired_loss=F.smooth_l1_loss(advantage,paired_target,reduction='none')
            loss=loss+(paired_loss*pair_weight).sum()/pair_weight.sum()
        self.td_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.future.parameters())+list(self.policy.update_value.parameters()),1)
        self.td_optimizer.step();self.td_updates += 1
        if self.td_updates%15==0:self.target.load_state_dict(self.policy.state_dict())


class Reliability:
    def __init__(self, classes):
        self.accuracy = np.full(ROUTES,.5)
        self.ce = np.full(ROUTES,.5)
        self.change = np.zeros(ROUTES)
        self.updates = np.zeros(ROUTES)
        self.last_update = np.zeros(ROUTES)
        self.class_good = np.zeros((ROUTES,classes))
        self.class_count = np.zeros((ROUTES,classes))
        self.paid = 0
        self.budget = None

    def features(self, probs, cap, position, count):
        predicted = probs.argmax(2).detach().cpu().numpy()
        rows=[]
        for r in range(ROUTES):
            c = predicted[:,r]
            support = self.class_count[r,c]
            reliability = (self.class_good[r,c]+4*self.accuracy[r])/(support+4)
            shared = [self.accuracy[r],self.ce[r],self.updates[r]/max(cap,1),
                      self.paid/max(cap,1),(position-self.last_update[r])/max(count,1),self.change[r]]
            rows.append(np.column_stack([np.broadcast_to(shared,(len(c),6)),reliability,support/(support+8)]))
        return probs.new_tensor(np.stack(rows,axis=1))

    def observe_paid(self, probs, truth):
        probabilities = probs.detach().cpu().numpy().reshape(ROUTES,-1)
        prediction = probabilities.argmax(1)
        good = prediction==int(truth)
        ce = -np.log(np.maximum(probabilities[:,int(truth)],1e-8))
        self.accuracy = .9*self.accuracy+.1*good
        self.ce = .9*self.ce+.1*ce/(1+ce)
        self.class_count *= .99; self.class_good *= .99
        for r,c in enumerate(prediction):
            self.class_count[r,c] += 1; self.class_good[r,c] += good[r]
        self.paid += 1

    def updated(self, route, position, change):
        self.updates[route] += 1;self.last_update[route] = position
        self.change[route] = .9*self.change[route]+.1*float(change)


def cnn_with_embedding(cnn,x):
    h = cnn.flatten(cnn.conv_block3(cnn.conv_block2(cnn.conv_block1(x))))
    h = cnn.dense_layers[:-1](h)
    return cnn.dense_layers[-1](h),h


def base_cache(cnn,x):
    logits,embeddings=[],[]
    with torch.no_grad():
        for start in range(0,len(x),64):
            z,h = cnn_with_embedding(cnn,x[start:start+64]);logits.append(z);embeddings.append(h)
    return torch.cat(logits),torch.cat(embeddings)


def route_outputs(cnn,bank,x,base_logits):
    with torch.no_grad():
        adapted = [c(x) for c in bank]
        z = cnn(torch.cat(adapted)).reshape(SLOTS,len(x),-1).transpose(0,1)
        logits = torch.cat([base_logits[:,None,:],z],dim=1)
        distortion = x.new_zeros((len(x),ROUTES))
        denom = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        for k,value in enumerate(adapted,start=1):
            d = (value-x).flatten(1).norm(dim=1)/denom
            distortion[:,k] = d/(1+d)
    return logits,distortion


def refresh_slot(cnn,bank,inputs,outputs,route):
    logits,distortion = (v.clone() for v in outputs)
    with torch.no_grad():
        adapted = bank[route-1](inputs)
        logits[:,route] = cnn(adapted)
        d = (adapted-inputs).flatten(1).norm(dim=1)/inputs.flatten(1).norm(dim=1).clamp_min(1e-6)
        distortion[:,route] = d/(1+d)
    return logits,distortion


def build_state(outputs,embedding,history,used,cap,position,count,gate):
    logits,distortion = outputs
    probs = logits.softmax(2)
    top = probs.topk(2,dim=2).values
    confidence = top[:,:,0]
    margin = top[:,:,0]-top[:,:,1]
    entropy = -(probs*probs.clamp_min(1e-8).log()).sum(2)/np.log(probs.shape[2])
    prediction = probs.argmax(2)
    agreement = (prediction[:,:,None]==prediction[:,None,:]).float().mean(2)
    frozen_flag = torch.zeros_like(confidence);frozen_flag[:,0] = 1
    info = torch.stack([confidence,margin,entropy,agreement,distortion,frozen_flag],dim=2)
    route_x = torch.cat([embedding[:,None,:].expand(-1,ROUTES,-1),probs,
                         probs[:,:1,:].expand(-1,ROUTES,-1),
                         probs.mean(1,keepdim=True).expand(-1,ROUTES,-1),
                         probs.max(1,keepdim=True).values.expand(-1,ROUTES,-1),info,
                         history.features(probs,cap,position,count)],dim=2)
    extra = logits.new_tensor([[gate,(cap-used)/max(cap,1),position/max(count,1)]]).expand(len(logits),-1)
    if history.budget is not None:
        bx=logits.new_tensor(history.budget.features(used,cap,position,count)).expand(len(logits),-1)
        return torch.cat([route_x.flatten(1),bx,extra],dim=1)
    return torch.cat([route_x.flatten(1),extra],dim=1)


class BalancedRewardReference:
    """Wrap the existing TRAIN/paid reservoir; only probe sampling changes."""
    def __init__(self, source):
        self.source = source

    def sample(self, current):
        source = self.source
        current_cpu = current.detach().cpu()
        if hasattr(source, 'x'):
            eligible = torch.any(source.x != current_cpu, dim=tuple(range(1, source.x.ndim))).nonzero().flatten().numpy()
            labels = source.y.numpy().reshape(-1)
            ids = balanced_indices(labels, eligible, core.REFERENCE_BATCH_SIZE, source.rng)
            if not len(ids):return None
            return source.x[ids].to(current.device), source.y[ids].to(current.device)
        eligible = [i for i,row in enumerate(source.rows) if not torch.equal(row[0], current_cpu)]
        labels = [int(row[1].item()) for row in source.rows]
        ids = balanced_indices(labels, eligible, core.REFERENCE_BATCH_SIZE, source.rng)
        if not len(ids):return None
        return (torch.cat([source.rows[i][0] for i in ids]).to(current.device),
                torch.cat([source.rows[i][1] for i in ids]).to(current.device))

    def add(self, x, y):
        self.source.add(x, y)


def utility_before(a,cnn,bank,selected,batch):
    if batch is None:return None
    if not a.bank_utility:return core.probe_losses(cnn,bank[selected],batch)
    with torch.no_grad():
        x,y = batch
        return torch.stack([F.cross_entropy(cnn(x),y,reduction='none'),
                            *[F.cross_entropy(cnn(c(x)),y,reduction='none') for c in bank]])


def utility_after(a,cnn,bank,selected,batch,before):
    if before is None:return core.transfer_utility(None,None)
    if not a.bank_utility:
        after=core.probe_losses(cnn,bank[selected],batch)
        if getattr(a,'balanced_reward_reference',False):
            return macro_transfer(before,after,batch[1].detach().cpu().numpy(),
                                  a.gain_weight,a.retention_weight,a.selection_cost)
        return core.transfer_utility(before,after)
    with torch.no_grad():
        after = before.clone()
        after[selected+1] = F.cross_entropy(cnn(bank[selected](batch[0])),batch[1],reduction='none')
    return core.transfer_utility(before.min(0).values.cpu().numpy(),after.min(0).values.cpu().numpy())


class MixedBudgetProbe:
    """Fixed TRAIN anchors plus TRAIN-domain / previous-paid reference rows.

    Only the current paid row receives a Context gradient. Reference rows are
    used without gradients and feature-identical current rows are excluded.
    """
    def __init__(self,anchors,current,seed):
        ax,ay=anchors
        self.anchor=core.TrainingProbe(ax,torch.as_tensor(ay),seed+1)
        self.current=current

    def sample(self,inputs):
        anchor=self.anchor.sample(inputs)
        recent=self.current.sample(inputs)
        if anchor is None:return recent
        if recent is None:return anchor
        half=max(1,core.REFERENCE_BATCH_SIZE//2)
        return (torch.cat([anchor[0][:half],recent[0][:half]]),
                torch.cat([anchor[1][:half],recent[1][:half]]))

    def add(self,inputs,label):self.current.add(inputs,label)


def budget_probe_before(cnn,bank,agent,batch,history,used,cap,pos,count):
    if batch is None:return None
    with torch.no_grad():
        z,h=base_cache(cnn,batch[0])
        outputs=route_outputs(cnn,bank,batch[0],z)
        state=build_state(outputs,h,history,used,cap,pos,count,0)
        # Freeze the pre-update route weights for causal attribution to the
        # selected Context. A small uniform component keeps dormant routes
        # represented. This is a loss surrogate, not a label-based oracle.
        weights=.9*agent.policy.route_logits(state).softmax(1)+.1/ROUTES
        losses=torch.stack([F.cross_entropy(outputs[0][:,r],batch[1],reduction='none') for r in range(ROUTES)],1)
    return losses,weights


def budget_probe_after(cnn,bank,selected,batch,before):
    if before is None:return 0.,0.
    with torch.no_grad():
        losses,weights=before
        after=F.cross_entropy(cnn(bank[selected](batch[0])),batch[1],reduction='none')
        prior=losses[:,selected+1]
        delta=weights[:,selected+1]*(prior-after)/(prior+after+1e-8)
    return float(delta.clamp_min(0).mean()),float((-delta).clamp_min(0).mean())


def action_route(action,values,budget_aware,decoupled_query=False):
    # Query destination and pre-feedback prediction destination are separate
    # in v2 or the isolated v1 ablation. Neither uses the current true label.
    return int(values[:ROUTES].argmax().item()) if (budget_aware or decoupled_query) and action>=ROUTES else base.route_index(action,SLOTS)


def reference_observer(cnn,bank,agent,x,y,history,used,cap,clock,mode,seed,
                       return_route_predictions=False):
    z,h = base_cache(cnn,x)
    predictions,per_route=[],[]
    rng = np.random.default_rng(seed+30003)
    with torch.no_grad():
        for start in range(0,len(x),64):
            bx=x[start:start+64];out=route_outputs(cnn,bank,bx,z[start:start+64])
            state=build_state(out,h[start:start+64],history,used,cap,clock,max(clock,1),0)
            routes = torch.tensor(rng.integers(0,ROUTES,len(bx)),device=x.device) if mode=='random' else agent.policy.route_logits(state).argmax(1)
            all_predictions=out[0].argmax(2)
            predictions.extend(all_predictions[torch.arange(len(bx),device=x.device),routes].cpu().tolist())
            per_route.append(all_predictions.cpu().numpy())
    all_predictions=np.concatenate(per_route)
    result=(np.asarray(predictions),(all_predictions==np.asarray(y)[:,None]).mean(0).tolist())
    return result+(all_predictions,) if return_route_predictions else result


def retention_summary(before,after,y,mask):
    """Observer-only comparison on exactly the same rows at both endpoints."""
    mask=np.asarray(mask,dtype=bool)
    count=int(mask.sum())
    if count==0:
        return dict(samples=0,initial=None,final=None,drop_pp=None,broken=0,fixed=0,
                    per_route_initial=None,per_route_final=None,per_route_broken=None,
                    per_route_fixed=None,mean_context_drop_pp=None)
    truth=np.asarray(y)[mask]
    good0=before[0][mask]==truth;good1=after[0][mask]==truth
    routes0=before[2][mask]==truth[:,None];routes1=after[2][mask]==truth[:,None]
    return dict(samples=count,initial=float(good0.mean()),final=float(good1.mean()),
                drop_pp=float(100*(good0.mean()-good1.mean())),
                broken=int((good0&~good1).sum()),fixed=int((~good0&good1).sum()),
                per_route_initial=routes0.mean(0).tolist(),per_route_final=routes1.mean(0).tolist(),
                per_route_broken=(routes0&~routes1).sum(0).tolist(),
                per_route_fixed=(~routes0&routes1).sum(0).tolist(),
                mean_context_drop_pp=float(100*(routes0[:,1:].mean()-routes1[:,1:].mean())))


def episode(a,cnn,bank,initial,agent,domain,mode='learned',training=False,
            learning=True,warmup=False,rollout_seed=None,reference=None,same_level=False):
    x,y,source_ids=domain
    seed=a.seed if rollout_seed is None else rollout_seed
    for context in bank:
        context.load_state_dict(initial);context.eval()
    optimizers=[torch.optim.Adam(c.parameters(),lr=a.context_lr,weight_decay=1e-4) for c in bank]
    order=np.random.default_rng(seed).permutation(len(y))
    x,y,source_ids=x[order],np.asarray(y)[order],np.asarray(source_ids)[order]
    n=len(y);cap=int(np.floor(a.budget_fraction*n))
    budget_aware=getattr(a,'budget_aware_policy',False)
    decoupled=getattr(a,'decoupled_query',False)
    if budget_aware and (a.query_schedule!='policy' or a.update_policy!='learned'):
        raise ValueError('Budget-aware RL requires policy queries and learned destinations; no external gate.')
    if agent.policy.budget_aware!=budget_aware:raise ValueError('Budget policy/state mismatch.')
    if agent.policy.decoupled_query!=decoupled:raise ValueError('Query prediction semantics mismatch.')
    schedule=set(np.random.default_rng(seed+819003).choice(n,cap,replace=False).tolist()) if a.query_schedule=='uniform' else None
    quality=None
    if a.query_schedule=='quality':
        if training:raise ValueError('Quality control is an evaluation-only ablation on an existing router checkpoint.')
        quality=QualityBudget(n,cap,a.quality_reference_accuracy,seed=seed,
            **{key:getattr(a,'quality_'+key) for key in QUALITY_OPTIONS},
            gain_weight=a.gain_weight,retention_weight=a.retention_weight,cost=a.selection_cost)
        quality.decide(0,0)
    fixed_destinations=np.random.default_rng(seed+440003).integers(0,SLOTS,n)
    def gate(pos):
        if quality is not None:return int(quality.decisions[pos]['query']) if pos in quality.decisions else -1
        return int(pos in schedule) if schedule is not None else -1
    def state_gate(pos):
        return 2+int(fixed_destinations[pos]) if a.update_policy=='fixed' and gate(pos)==1 else gate(pos)
    rng=np.random.default_rng(seed+719003)
    quality_route_rng=np.random.default_rng(seed+601037) if quality is not None else None
    query_route_rng=np.random.default_rng(seed+601089) if decoupled else None
    history=Reliability(len(a.classes))
    if budget_aware:
        history.budget=BudgetState(a.budget_priors,**budget_settings(a))
    reward_reference=core.TrainingProbe(x,torch.tensor(y,device=x.device),seed+910033) if training else core.PaidHistory(seed+910033)
    if getattr(a,'balanced_reward_reference',False):
        reward_reference=BalancedRewardReference(reward_reference)
    if budget_aware:
        reward_reference=MixedBudgetProbe(a.budget_anchors,reward_reference,seed+910043)
    timer=StreamTimer(x.device,enabled=not training)
    with timer.stage('cache'):
        bz,embedding=base_cache(cnn,x)
    initial_ref=reference_observer(cnn,bank,agent,*reference,history,0,cap,0,mode,a.seed) if reference is not None else None
    initial_current=reference_observer(cnn,bank,agent,x,y,history,0,cap,0,mode,a.seed,
                                       return_route_predictions=True) if same_level else None
    used=0;actions=np.zeros(ACTIONS,dtype=int);slot_queries=np.zeros(SLOTS,dtype=int)
    prediction_rows=[];trace=[];route_probabilities=[]
    oracle_all,oracle_legal=0,0
    route_updates0,td_updates0=agent.route_updates,agent.td_updates
    next_cached=None
    timer.start()
    for pos in range(n):
        inputs=x[pos:pos+1]
        with timer.stage('prediction'):
            outputs=next_cached if next_cached is not None else route_outputs(cnn,bank,inputs,bz[pos:pos+1])
            next_cached=None
        with timer.stage('selection'):
            state=build_state(outputs,embedding[pos:pos+1],history,used,cap,pos,n,state_gate(pos))
            allowed=base.legal_actions(SLOTS,cap-used,gate(pos))
            if a.update_policy=='fixed':
                fixed_action=4+int(fixed_destinations[pos])
                allowed=[v for v in allowed if v<4 or v==fixed_action]
            with torch.no_grad():
                values=agent.policy(state).squeeze(0)
                probability=agent.policy.route_logits(state).sigmoid().squeeze(0)
                epsilon=a.train_epsilon if training else a.eval_epsilon
                valid=torch.tensor(allowed,device=x.device)
                greedy=int(valid[values[valid].argmax()].item())
                if mode=='random' or warmup or rng.random()<epsilon:
                    action=int(rng.choice(allowed))
                else:
                    action=greedy
                query_fraction=sum(v>=ROUTES for v in allowed)/len(allowed)
                propensity=query_fraction if mode=='random' or warmup else epsilon*query_fraction+(1-epsilon)*int(greedy>=ROUTES)
                if quality is not None:
                    # Shadow the prediction-only router BEFORE feedback. A
                    # forced update-slot prediction can be weak even when the
                    # system's no-query route is good; do not confuse these.
                    quality_probe_route=int(quality_route_rng.integers(ROUTES)) if mode=='random' or quality_route_rng.random()<epsilon else int(values[:ROUTES].argmax().item())
        route=action_route(action,values,budget_aware,decoupled)
        if decoupled and mode=='random' and action>=ROUTES:route=int(query_route_rng.integers(ROUTES))
        if budget_aware and mode=='random' and action>=ROUTES:route=int(rng.integers(ROUTES))
        predictions=outputs[0].argmax(2).squeeze(0).cpu().numpy()
        prediction_rows.append(predictions.copy())
        route_probabilities.append(probability.cpu().numpy().copy())
        # Score is fixed now; truth below is used by learners only under the gates.
        prediction=int(predictions[route]);paid=action>=4
        truth=int(y[pos]);correct=(predictions==truth)
        oracle_all+=bool(correct.any())
        legal_routes=set(range(ROUTES)) if (budget_aware or (decoupled and any(v>=ROUTES for v in allowed))) else {base.route_index(v,SLOTS) for v in allowed}
        oracle_legal+=any(correct[r] for r in legal_routes)
        actions[action]+=1
        agent.clock+=1
        with timer.stage('update'):
            successor_output=None
            if pos+1<n:
                # Cache once, then refresh just the updated slot for the actual successor.
                if not warmup and (paid or training):
                    successor_output=route_outputs(cnn,bank,x[pos+1:pos+2],bz[pos+1:pos+2])
            zero=torch.zeros_like(state)
            skip_next=zero
            if successor_output is not None:
                skip_history=history
                if budget_aware:
                    skip_history=deepcopy(history)
                    skip_history.budget.advance(False,used,cap,pos,n)
                skip_next=build_state(successor_output,embedding[pos+1:pos+2],skip_history,used,cap,pos+1,n,state_gate(pos+1))
            reward,gain,harm=None,None,None
            before=None;batch=None;budget_reward=None;importance=1.
            used_before=used
            if paid:
                assert used<cap
                selected=action-ROUTES if (budget_aware or decoupled) else route-1
                label=torch.tensor([truth],device=x.device)
                if not warmup or budget_aware:
                    batch=reward_reference.sample(inputs)
                    before=budget_probe_before(cnn,bank,agent,batch,history,used,cap,pos,n) if budget_aware else utility_before(a,cnn,bank,selected,batch)
                history.observe_paid(outputs[0].softmax(2),truth)
                optimizers[selected].zero_grad()
                loss=core.context_objective(F.cross_entropy(cnn(bank[selected](inputs)),label),
                                            F.cross_entropy(outputs[0][:,0],label))
                loss.backward();optimizers[selected].step()
                if budget_aware:
                    gain,harm=budget_probe_after(cnn,bank,selected,batch,before)
                    budget_reward=history.budget.query_adjustment(selected,gain,harm,used,cap,pos,n,
                        a.gain_weight,a.retention_weight,a.selection_cost)
                    utility=budget_reward['utility']
                    importance=history.budget.observe_paid(correct,int(values[:ROUTES].argmax().item()),propensity,pos)
                    if before is not None:history.budget.observe_transfer(selected,gain,harm,pos+1)
                else:
                    utility,gain,harm=(0.0,0.0,0.0) if warmup else utility_after(a,cnn,bank,selected,batch,before)
                reward=(1.0 if prediction==truth else -1.0)+utility
                used+=1;slot_queries[selected]+=1
                updated_route=selected+1
                after_current=refresh_slot(cnn,bank,inputs,outputs,updated_route)
                change=(after_current[0][:,updated_route].softmax(1)-outputs[0][:,updated_route].softmax(1)).abs().mean().item()
                history.updated(updated_route,pos+1,change)
                if not training:reward_reference.add(inputs,label)
                if successor_output is not None:
                    successor_output=refresh_slot(cnn,bank,x[pos+1:pos+2],successor_output,updated_route)
            if budget_aware:history.budget.advance(paid,used_before,cap,pos,n)
            if quality is not None:
                if paid:quality.observe(pos,int(predictions[quality_probe_route])==truth,gain if batch is not None else None,harm if batch is not None else None)
                if pos+1<n:
                    # Plan once, after current paid feedback. Use the SAME next
                    # gate for behavior and DDQN backups, without future truth.
                    quality.decide(pos+1,used)
                    if successor_output is not None:skip_next[:,-3]=state_gate(pos+1)
            # Offline TRAIN labels supervise all routes; validation/test only paid labels.
            may_learn=training or (learning and mode=='learned' and paid)
            if may_learn:
                target=torch.tensor(correct,device=x.device,dtype=torch.float32)
                step_now=(not training) or paid or pos%16==15 or pos+1==n
                agent.observe_routes(state,target,step_now=step_now,importance=1. if training else importance)
            if may_learn and not warmup:
                # Non-query actions share the same no-update successor and future value.
                skip_state=state.clone()
                if schedule is not None or quality is not None:skip_state[:,-3]=0
                for r in range(ROUTES):
                    agent.push(skip_state,r,1.0 if correct[r] else -1.0,skip_next,pos+1==n)
                if paid:
                    successor=zero if successor_output is None else build_state(successor_output,embedding[pos+1:pos+2],history,used,cap,pos+1,n,state_gate(pos+1))
                    # Budget exhaustion masks queries; it does NOT end future prediction value.
                    agent.push(state,action,reward,successor,pos+1==n)
                    if budget_aware:agent.push_pair(state,action,utility,successor,skip_next,pos+1==n)
                if paid or pos%16==15 or pos+1==n:agent.learn_td()
        next_cached=successor_output
        row=dict(step=pos+1,source_row=int(source_ids[pos]),truth=truth,prediction=prediction,
                 action=action,route=route,queried=paid,labels_used=used,reward=reward,gain=gain,harm=harm,
                 oracle_all=bool(correct.any()),oracle_legal=any(correct[r] for r in legal_routes))
        if getattr(a,'balanced_reward_reference',False):
            probe_labels=[] if batch is None else batch[1].detach().cpu().numpy().reshape(-1).tolist()
            row['reference_class_counts']=json.dumps({str(int(c)):probe_labels.count(c) for c in sorted(set(probe_labels))})
        if decoupled:row['update_slot']=selected if paid else None
        if budget_aware:
            row.update(query_propensity=propensity,update_slot=selected if paid else None,
                       budget_price_before=float(state[0,ROUTES*agent.policy.route_width+4].item())*history.budget.price_max)
            if budget_reward is not None:row.update({'budget_'+k:v for k,v in budget_reward.items()})
        for r in range(ROUTES):
            row[f'prediction_{r}']=int(predictions[r]);row[f'correctness_probability_{r}']=float(probability[r].item())
        for action_id in range(ACTIONS):row[f'q_{action_id}']=float(values[action_id].item())
        if quality is not None:
            row.update({'quality_'+key:value for key,value in quality.decisions[pos].items()})
            row.update(quality_monitor_route=quality_probe_route,quality_monitor_prediction=int(predictions[quality_probe_route]))
        trace.append(row)
    timer.stop()
    if schedule is not None:assert used==cap
    matrix=np.asarray(prediction_rows)
    chosen=np.asarray([r['prediction'] for r in trace])
    actual_queries=np.asarray([r['queried'] for r in trace])
    fixed_skip=[float(np.mean(np.where(actual_queries,chosen,matrix[:,r])==y)) for r in range(ROUTES)]
    route_correct=(matrix==y[:,None])
    metrics=dict(accuracy=float(core.accuracy_score(y,chosen)),macro_f1=float(core.f1_score(y,chosen,average='macro',zero_division=0)),
                 frozen_accuracy=float(np.mean(matrix[:,0]==y)),oracle_all=oracle_all/n,oracle_legal=oracle_legal/n,
                 best_fixed_skip_route_accuracy=max(fixed_skip),fixed_skip_route_accuracies=fixed_skip,
                 route_brier=float(np.mean((np.asarray(route_probabilities)-route_correct)**2)),
                 samples=n,labels=used,label_cap=cap,actual_label_fraction=used/n,
                 budget_utilization_pct=100*used/cap if cap else 0.,labels_remaining=cap-used,
                 slot_queries=slot_queries.tolist(),action_counts=actions.tolist(),
                 route_updates=agent.route_updates-route_updates0,td_updates=agent.td_updates-td_updates0,
                 mode=mode,context_lr=a.context_lr,query_schedule=a.query_schedule,update_policy=a.update_policy,
                 bank_utility=a.bank_utility,seed=a.seed,
                 oracle_protocol='observer only, along this policy bank trajectory; legal oracle obeys uniform gate and fixed-destination control',
                 fixed_skip_protocol='retrospective control: keep actual query actions/destinations; use one fixed route on all non-query rows')
    metrics.update(timer.metrics(n,used))
    if decoupled:
        metrics['query_prediction']='best learned pre-feedback route; random baseline uses independent random query routes; update slot is action minus four'
        metrics['oracle_protocol']='observer only along this bank trajectory; any query action permits all four prediction routes; no oracle labels used by policy'
    if quality is not None:
        assert quality.labels==used<=cap
        metrics['quality_budget']=quality.statistics()
        metrics['oracle_protocol']='observer only along this bank trajectory; legal oracle obeys the external quality gate'
    if budget_aware:
        metrics['rl_budget']=history.budget.statistics(used,cap,n,n)
        paid_rows=[row for row in trace if row['queried']]
        metrics['rl_budget'].update(
            mean_query_utility=float(np.mean([row['budget_utility'] for row in paid_rows])) if paid_rows else None,
            positive_utility_fraction=float(np.mean([row['budget_utility']>0 for row in paid_rows])) if paid_rows else None,
            first_query_step=paid_rows[0]['step'] if paid_rows else None,
            last_query_step=paid_rows[-1]['step'] if paid_rows else None)
        metrics['oracle_protocol']='observer only along this bank trajectory; all four prediction routes are permitted; query destination is independent of pre-feedback prediction route'
    if reference is not None:
        final,per_final=reference_observer(cnn,bank,agent,*reference,history,used,cap,n,mode,a.seed)
        first,per_first=initial_ref
        good0,good1=first==reference[1],final==reference[1]
        metrics['reference']=dict(initial=float(good0.mean()),final=float(good1.mean()),drop_pp=float(100*(good0.mean()-good1.mean())),
                                  broken=int((good0&~good1).sum()),per_route_initial=per_first,per_route_final=per_final,
                                  mean_context_drop_pp=100*float(np.mean(per_first[1:])-np.mean(per_final[1:])),
                                  protocol='observer only; current bank reliability/query-count state retained; gate set to prediction-only')
    if same_level:
        final_current=reference_observer(cnn,bank,agent,x,y,history,used,cap,n,mode,a.seed,
                                         return_route_predictions=True)
        metrics['same_level']=dict(
            all_rows=retention_summary(initial_current,final_current,y,np.ones(n,dtype=bool)),
            unqueried=retention_summary(initial_current,final_current,y,~actual_queries),
            protocol='observer only; initial/final prediction-only routed system and each route; '
                     'same rows at both endpoints; final score is retrospective, not online accuracy; '
                     'unqueried subset selected by this policy after the stream, not an independent holdout; '
                     'no observer labels, states or predictions enter learning or action selection')
    return metrics,trace


def read_domain(root,artifact,level,split):
    frame,ids,_=base.official.mapped_domain(root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv',artifact,root,False)
    lookup={int(v):i for i,v in enumerate(ids)}
    selected=np.asarray([i for i in artifact[split] if int(i) in lookup],dtype=np.int64)
    if len(selected)!=len(artifact[split]) and not artifact.get('allow_partial_domains',False):raise ValueError('Domain does not cover partition: '+split)
    if not len(selected):raise ValueError('No supplied-domain rows in partition: '+split)
    frame=frame.iloc[[lookup[int(i)] for i in selected]]
    features=frame.iloc[:,:-1]
    scaler=artifact['scaler']
    scaled=scaler.transform(features if hasattr(scaler,'feature_names_in_') else features.to_numpy()).astype(np.float32)
    if not np.isfinite(scaled).all():raise ValueError('Nonfinite domain features.')
    labels=artifact['label_encoder'].transform(frame.iloc[:,-1])
    return torch.tensor(scaled,device=core.device).unsqueeze(1),labels,selected


def models_from_source(models,artifact):
    width=len(artifact['feature_names']);classes=len(artifact['label_encoder'].classes_)
    cnn=core.CNN1D(width,classes).to(core.device)
    digest=core.file_sha256(models/'protocol_preprocessing.joblib')
    cnn_checkpoint=torch.load(models/'initial_cnn_model.pth',map_location='cpu',weights_only=False)
    context_checkpoint=torch.load(models/'adversarial_best_context.pth',map_location='cpu',weights_only=False)
    for saved in (cnn_checkpoint,context_checkpoint):
        if saved.get('preprocessing_sha256',digest)!=digest:raise ValueError('Source model/preprocessing mismatch.')
    cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn.eval()
    for parameter in cnn.parameters():parameter.requires_grad_(False)
    initial=context_checkpoint['context_model_state_dict']
    bank=[core.ContextAwareNetwork(width).to(core.device) for _ in range(SLOTS)]
    for context in bank:context.load_state_dict(initial);context.eval()
    # Dense CNN embedding + four probability vectors + six local + eight reliability features.
    route_width=cnn.dense_layers[-1].in_features+4*classes+14
    return cnn,bank,initial,route_width


def report_level(level,mode,m):
    print(f'Level {level} | {mode} | online {100*m["accuracy"]:.2f}% | legal oracle {100*m["oracle_legal"]:.2f}% | all-route oracle {100*m["oracle_all"]:.2f}% | labels {m["labels"]}/{m["label_cap"]}',flush=True)
    print(f'  Label budget: {m["labels"]}/{m["label_cap"]} used | {100*m["actual_label_fraction"]:.2f}% of {m["samples"]} stream rows | {m["budget_utilization_pct"]:.2f}% of cap | remaining {m["labels_remaining"]}')
    print(f'  Slot queries {m["slot_queries"]} | best fixed non-query route {100*m["best_fixed_skip_route_accuracy"]:.2f}% | route updates {m["route_updates"]} | TD updates {m["td_updates"]}')
    if 'rl_budget' in m:
        q=m['rl_budget'];values='/'.join(f'{100*v:.1f}' for v in q['estimated_route_accuracies'])
        print(f'  Learned budget: route accuracy estimates {values}% | effective support {q["effective_support"]:.1f} | resource price {q["resource_price"]:.4f} | no external query gate')
    if 'quality_budget' in m:
        q=m['quality_budget']
        net='unknown' if q['mean_net_transfer'] is None else f'{q["mean_net_transfer"]:+.5f}'
        print(f'  Quality gate: clean frozen target {100*q["reference_accuracy"]:.2f}% | monitored no-query accuracy estimate {100*q["estimated_accuracy"]:.2f}% | recent net transfer {net}')
        print(f'  Queries bootstrap/audit/adaptive: {q["bootstrap_labels"]}/{q["audit_labels"]}/{q["adaptive_labels"]} | need {q["need"]:.2f} | yield factor {q["efficiency"]:.2f}')
    ref=m.get('reference')
    if ref:
        print(f'  Routed clean reference: {100*ref["initial"]:.2f}% -> {100*ref["final"]:.2f}% | drop {ref["drop_pp"]:+.2f} pp')
        for k,(before,after) in enumerate(zip(ref['per_route_initial'][1:],ref['per_route_final'][1:]),1):
            print(f'  Clean Context {k}: {100*before:.2f}% -> {100*after:.2f}% | drop {100*(before-after):+.2f} pp')
    current=m.get('same_level')
    if current:
        for key,label in [('all_rows','all rows'),('unqueried','unqueried rows')]:
            result=current[key]
            if not result['samples']:
                print(f'  Same level ({label}): no rows; retention N/A')
                continue
            print(f'  Same level ({label}; n={result["samples"]}): {100*result["initial"]:.2f}% -> {100*result["final"]:.2f}% | drop {result["drop_pp"]:+.2f} pp | broken {result["broken"]} | fixed {result["fixed"]}')
        result=current['unqueried']
        if result['samples']:
            for k in range(1,ROUTES):
                before,after=result['per_route_initial'][k],result['per_route_final'][k]
                print(f'  Same-level unqueried Context {k}: {100*before:.2f}% -> {100*after:.2f}% | drop {100*(before-after):+.2f} pp | broken {result["per_route_broken"][k]}')


BUDGET_OPTIONS=('window','need_weight','price_step','price_max','propensity_clip')


def budget_settings(a):
    defaults=dict(window=64,need_weight=1.,price_step=.005,price_max=.2,propensity_clip=20.)
    return {key:getattr(a,'budget_'+key,defaults[key]) for key in BUDGET_OPTIONS}


def prepare_budget(a,cnn,bank,artifact,saved=None):
    """Only source TRAIN rows enter policy priors or reference anchors."""
    x,y,ids=read_domain(a.burst_dir,artifact,0,'train_ids')
    if saved is None:
        with torch.no_grad():
            frozen,_=base_cache(cnn,x)
            frozen_accuracy=float(np.mean(frozen.argmax(1).cpu().numpy()==y))
            cp=[]
            for start in range(0,len(x),64):cp.extend(cnn(bank[0](x[start:start+64])).argmax(1).cpu().tolist())
        a.budget_priors=[frozen_accuracy]+[float(np.mean(np.asarray(cp)==y))]*SLOTS
        rng=np.random.default_rng(a.seed+601079)
        # Balanced anchors protect small classes without changing stream scores.
        groups=[rng.permutation(np.flatnonzero(y==c)).tolist() for c in np.unique(y)]
        chosen=[]
        while len(chosen)<min(256,len(y)):
            for group in groups:
                if group and len(chosen)<256:chosen.append(group.pop())
                if len(chosen)==min(256,len(y)):break
        a.budget_anchor_source_rows=np.asarray(ids)[chosen].tolist()
    else:
        a.budget_priors=saved['budget_priors']
        a.budget_anchor_source_rows=saved['budget_anchor_source_rows']
    row_index={int(v):i for i,v in enumerate(ids)}
    if len(set(a.budget_anchor_source_rows))!=len(a.budget_anchor_source_rows) or not set(a.budget_anchor_source_rows)<=set(row_index):
        raise ValueError('Budget reference anchors must be unique source TRAIN IDs.')
    chosen=[row_index[v] for v in a.budget_anchor_source_rows]
    a.budget_anchors=x[chosen],np.asarray(y)[chosen]
    print(f'RL budget v{BUDGET_VERSION}: seven learned actions; {len(chosen)} fixed TRAIN reference anchors; '
          f'TRAIN Frozen quality prior {100*a.budget_priors[0]:.2f}%; no external query gate.')


def train(a,models,artifact):
    cnn,bank,initial,route_width=models_from_source(models,artifact)
    aware=getattr(a,'budget_aware_policy',False)
    agent=Agent(route_width,core.device,a.seed,offline=True,gamma=a.gamma,budget_aware=aware,decoupled_query=a.decoupled_query)
    if getattr(a,'resume_selector',False):
        previous=torch.load(models/'structured_bank_ddqn.pth',map_location='cpu',weights_only=False)
        if previous.get('version')!=1 or aware or previous['route_width']!=route_width:
            raise ValueError('Resume requires a matching original seven-action structured selector.')
        if set(previous['training_levels'])!=set(a.levels):
            raise ValueError('Resume training levels must match the existing selector.')
        for name,digest in previous['model_hashes'].items():
            if core.file_sha256(models/name)!=digest:raise ValueError('Resume source checkpoint mismatch: '+name)
        agent.policy.load_state_dict(previous['policy_state_dict'])
        agent.target.load_state_dict(agent.policy.state_dict())
        print('Resuming all selector weights; fresh optimizers/replay. Frozen CNN and initial Context unchanged.')
    if getattr(a,'balanced_reward_reference',False):
        print('Balanced reward reference: at most 32 unique rows; equal class-weight gain/damage. Online: previous paid labels only.')
    if a.decoupled_query:
        print('Decoupled query: current prediction uses best learned route; paid label updates the action-selected Context. Seven actions; original transfer reward.')
    if aware:
        prepare_budget(a,cnn,bank,artifact)
        prior=models/'structured_bank_ddqn.pth'
        if prior.exists():
            previous=torch.load(prior,map_location='cpu',weights_only=False)
            if previous['route_width']!=route_width:raise ValueError('Warm-start correctness width mismatch.')
            for name,digest in previous.get('model_hashes',{}).items():
                if core.file_sha256(models/name)!=digest:raise ValueError('Warm-start source checkpoint mismatch: '+name)
            weights={k.removeprefix('correctness.'):v for k,v in previous['policy_state_dict'].items() if k.startswith('correctness.')}
            agent.policy.correctness.load_state_dict(weights)
            agent.target.load_state_dict(agent.policy.state_dict())
            print('Reused trained correctness scorer; new budget/update value heads start fresh.')
    domains={(v,s):read_domain(a.burst_dir,artifact,v,s) for v in a.levels for s in ('train_ids','val_ids')}
    for warm in range(a.router_warmup_episodes):
        level=a.levels[warm%len(a.levels)]
        m,_=episode(a,cnn,bank,initial,agent,domains[level,'train_ids'],mode='random',training=True,warmup=True,
                    rollout_seed=a.seed+100003+1009*warm)
        print(f'Correctness warmup {warm+1}/{a.router_warmup_episodes} | domain {level} | Context labels {m["labels"]}/{m["label_cap"]} | dense TRAIN route targets',flush=True)
    best,stale=-1,0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='budget_anchors'}
    hashes={name:core.file_sha256(models/name) for name in ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib')}
    domain_hashes={str(level):core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') for level in a.levels}
    log=[]
    for epoch in range(a.epochs):
        level=a.levels[epoch%len(a.levels)]
        agent.td_memory.clear()  # Context bank reset: do not mix old hidden-bank transitions.
        agent.pair_memory.clear()
        m,_=episode(a,cnn,bank,initial,agent,domains[level,'train_ids'],training=True,
                    rollout_seed=a.seed+200003+1009*epoch)
        scores=[];validation_labels=[]
        # Validation learners are temporary; no validation parameters/replay survive.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            snapshot=deepcopy(agent.policy.state_dict())
            for validation_level in a.levels:
                temporary=Agent(route_width,core.device,a.seed,policy_state=snapshot,gamma=a.gamma,budget_aware=aware,decoupled_query=a.decoupled_query)
                result,_=episode(a,cnn,bank,initial,temporary,domains[validation_level,'val_ids'])
                scores.append(result['accuracy'])
                validation_labels.append(result['labels'])
        score=float(np.mean(scores))
        log.append(dict(epoch=epoch+1,level=level,train_accuracy=m['accuracy'],validation_accuracy=score,
                        labels=m['labels'],slot_queries=m['slot_queries'],validation_labels=validation_labels))
        print(f'Epoch {epoch+1}/{a.epochs} | domain {level} | validation {100*score:.2f}% | labels {m["labels"]}/{m["label_cap"]} | slot queries {m["slot_queries"]}',flush=True)
        if aware:
            utility=m['rl_budget']['mean_query_utility'];text='N/A' if utility is None else f'{utility:+.5f}'
            print(f'  Query utility {text} | validation labels {validation_labels} | training resource price {m["rl_budget"]["resource_price"]:.4f}',flush=True)
        if score>best:
            best,stale=score,0
            checkpoint=dict(version=2 if aware else 1,slots=SLOTS,actions=ACTIONS,route_width=route_width,
                            policy_state_dict=deepcopy(agent.policy.state_dict()),model_hashes=hashes,
                            training_levels=a.levels,training_domain_sha256=domain_hashes,config=config,validation_accuracy=best,
                            protocol='Qskip=B+2p-1; Qquery=B+2p_context-1+U; correctness BCE isolated from TD; shared expert scorer',
                            supervision='offline: existing TRAIN labels on all rows; online/validation: queried labels only',
                            initial_context='three independent clones of unchanged healthy source checkpoint')
            if a.decoupled_query:
                checkpoint['protocol']='Qskip=B+2p-1; Qquery=B+max(2p-1)+U; seven actions; query destination independent of current prediction; original transfer reward'
            if aware:
                checkpoint.update(budget_version=BUDGET_VERSION,budget_priors=a.budget_priors,
                    budget_anchor_source_rows=a.budget_anchor_source_rows,
                    protocol='Qskip=B+2p-1; Qquery=B+max(2p-1)+U; paired DDQN update advantage; '
                             'isolated correctness head; soft resource cost; bounded routing-weighted transfer')
            torch.save(checkpoint,models/'structured_bank_ddqn.pth')
        else:
            stale+=1
            if stale>=a.patience:break
    (a.output/'structured_training.json').write_text(json.dumps(dict(config=config,epochs=log,best_validation=best),indent=2))
    print(f'Training complete. Best validation {100*best:.2f}%. CNN/initial Context unchanged; no heldout test scored.')


def evaluate(a,models,artifact):
    saved=torch.load(models/'structured_bank_ddqn.pth',map_location='cpu',weights_only=False)
    if saved.get('version') not in (1,2):raise ValueError('Unsupported structured-bank checkpoint.')
    aware=getattr(a,'budget_aware_policy',False)
    if aware!=(saved['version']==2):raise ValueError('Checkpoint uses a different budget policy; pass --budget-aware-policy only for a v2 trained run.')
    if bool(saved.get('config',{}).get('balanced_reward_reference',False))!=a.balanced_reward_reference:
        raise ValueError('Reward reference differs from training; use --balanced-reward-reference for a matching checkpoint.')
    if bool(saved.get('config',{}).get('decoupled_query',False))!=a.decoupled_query:
        raise ValueError('Query semantics differ from training; use --decoupled-query only with a matching trained checkpoint.')
    if aware:
        if saved.get('budget_version')!=BUDGET_VERSION:raise ValueError('Unsupported RL budget policy version.')
        config=saved['config']
        matching=('selection_cost','gain_weight','retention_weight','gamma',*['budget_'+key for key in BUDGET_OPTIONS])
        if any(getattr(a,key)!=config[key] for key in matching):raise ValueError('RL budget/reward settings differ from training; use checkpoint settings.')
    for name,digest in saved['model_hashes'].items():
        if core.file_sha256(models/name)!=digest:raise ValueError('Source checkpoint mismatch: '+name)
    if set(a.levels)&set(saved['training_levels']) and not getattr(a,'include_seen_levels',False):
        raise ValueError('Heldout test levels must exclude training levels; --include-seen-levels explicitly reports seen-domain TEST rows separately.')
    cnn,bank,initial,route_width=models_from_source(models,artifact)
    if route_width!=saved['route_width']:raise ValueError('Router state width mismatch.')
    if aware:prepare_budget(a,cnn,bank,artifact,saved)
    reference=read_domain(a.burst_dir,artifact,0,'val_ids')[:2]
    if a.query_schedule=='quality':
        # One offline scalar from clean SOURCE validation. No target labels or
        # evolving retention observer outputs enter the controller.
        with torch.no_grad():
            logits,_=base_cache(cnn,reference[0])
            a.quality_reference_accuracy=float(np.mean(logits.argmax(1).cpu().numpy()==reference[1]))
        print(f'Quality budget reference: frozen CNN clean SOURCE validation {100*a.quality_reference_accuracy:.2f}%')
    root=a.output/'Results';root.mkdir(exist_ok=True)
    execution={key:getattr(a,key) for key in ('seed','context_lr','budget_fraction','selection_cost','gain_weight','retention_weight','gamma','eval_epsilon','query_schedule','update_policy','bank_utility')}
    execution['same_level_diagnostics_version']=1
    if a.balanced_reward_reference:execution['reward_reference']='class_balanced_macro_v1'
    if a.decoupled_query:execution['query_prediction']='decoupled_best_route_v1'
    execution['budget_report_version']=1
    execution['timing_version']=1
    if getattr(a,'include_seen_levels',False):execution['include_seen_levels']=True
    if aware:
        execution.update(budget_policy_version=BUDGET_VERSION,budget_aware_policy=True,
                         **{'budget_'+key:value for key,value in budget_settings(a).items()})
    if a.query_schedule=='quality':
        execution.update(quality_budget_version=QUALITY_VERSION,quality_reference_accuracy=a.quality_reference_accuracy,
                         **{'quality_'+key:getattr(a,'quality_'+key) for key in QUALITY_OPTIONS})
    run_id=hashlib.sha256(json.dumps(execution,sort_keys=True).encode()).hexdigest()[:8]
    rows=[];retention_rows=[];budget_rows=[];time_rows=[];evaluation_results=[]
    for level in a.levels:
        domain=read_domain(a.burst_dir,artifact,level,'test_ids')
        pair={}
        for mode in a.modes:
            core.set_seed(a.seed)
            agent=Agent(route_width,core.device,a.seed,policy_state=saved['policy_state_dict'],gamma=a.gamma,budget_aware=aware,decoupled_query=a.decoupled_query)
            m,trace=episode(a,cnn,bank,initial,agent,domain,mode=mode,learning=mode=='learned',reference=reference,same_level=True)
            m.update(level=level,classes=a.classes,execution_config=execution,classifier_unchanged=not cnn.training and all(not p.requires_grad for p in cnn.parameters()))
            m['domain_status']='seen_domain_test_rows' if level in saved['training_levels'] else 'unseen_domain_test_rows'
            evaluation_results.append(m)
            tag=f'structured_{mode}_per{level}_seed{a.seed}_lr{a.context_lr:g}_{a.update_policy}_{run_id}'
            (root/(tag+'_metrics.json')).write_text(json.dumps(m,indent=2))
            base.write_trace(root/(tag+'_trace.csv'),trace)
            report_level(level,mode,m);pair[mode]=m
            print(timing_report(m),flush=True)
            time_rows.append(timing_row(mode,level,m))
            budget_rows.append(dict(PER=level,Mode=mode,N=m['samples'],Labels=m['labels'],Cap=m['label_cap'],
                **{'Label%':100*m['actual_label_fraction'],'CapUsed%':m['budget_utilization_pct']},
                Left=m['labels_remaining']))
            for subset,label in [('all_rows','All'),('unqueried','Unqueried')]:
                current=m['same_level'][subset]
                retention_rows.append(dict(PER=level,Mode=mode,Subset=label,N=current['samples'],
                    Initial=None if current['initial'] is None else 100*current['initial'],
                    Final=None if current['final'] is None else 100*current['final'],
                    Drop=current['drop_pp'],Broken=current['broken'],Fixed=current['fixed'],
                    CtxDrop=current['mean_context_drop_pp']))
        row={'PER':level,'Frozen':100*next(iter(pair.values()))['frozen_accuracy']}
        for mode,prefix in [('learned','RL'),('random','Rnd')]:
            if mode not in pair:continue
            m=pair[mode];ref=m['reference']
            row.update({prefix:100*m['accuracy'],'Legal'+prefix:100*m['oracle_legal'],
                        'All'+prefix:100*m['oracle_all'],'Lab'+prefix:m['labels'],
                        'Ref'+prefix:100*ref['final'],'Drop'+prefix:ref['drop_pp'],
                        'CtxDrop'+prefix:ref['mean_context_drop_pp']})
            if mode=='learned':row['FixedRL']=100*m['best_fixed_skip_route_accuracy']
        if 'RL' in row and 'Rnd' in row:row['Gap']=row['RL']-row['Rnd']
        rows.append(row)
    preferred=['PER','Frozen','RL','Rnd','Gap','FixedRL','LegalRL','LegalRnd','AllRL','AllRnd','LabRL','LabRnd','RefRL','RefRnd','DropRL','DropRnd','CtxDropRL','CtxDropRnd']
    fields=[key for key in preferred if key in rows[0]]
    rows=[{key:row[key] for key in fields} for row in rows]
    mean={'PER':'Mean'}
    for key in fields[1:]:mean[key]=float(np.mean([r[key] for r in rows]))
    rows.append(mean)
    print('\nCombined structured-bank results (%; Drop/Gap in pp):')
    from amcal_burst_results_summary import display
    display(rows,fields)
    print('Legal: permitted-action oracle. All: all-route oracle. FixedRL: best fixed non-query route along the RL update trajectory. Ref: routed clean reference. CtxDrop: mean Context drop.')
    if a.decoupled_query:
        print('Decoupled query: learned predicts through best estimated route before feedback, then updates the action-selected Context. Random query prediction routes use a separate random generator.')
    filename=root/f'combined_structured_seed{a.seed}_lr{a.context_lr:g}_eps{a.eval_epsilon:g}_{a.update_policy}_{run_id}.csv'
    base.write_trace(filename,rows)
    print('Saved:',filename)
    budget_file=filename.with_name(filename.stem+'_budget.csv')
    for mode in a.modes:
        own=[row for row in budget_rows if row['Mode']==mode]
        budget_rows.append(dict(PER='Mean',Mode=mode,**{
            key:float(np.mean([row[key] for row in own])) for key in budget_rows[0] if key not in ('PER','Mode')}))
    print('\nLabel budget consumption (Label%: stream percentage; CapUsed%: cap percentage):')
    display(budget_rows,list(budget_rows[0]));base.write_trace(budget_file,budget_rows)
    print('Budget report saved:',budget_file)
    for mode in a.modes:
        own=[row for row in time_rows if row['Method']==mode]
        time_rows.append(dict(Method=mode,PER='Mean',**{
            key:float(np.mean([row[key] for row in own])) for key in time_rows[0] if key not in ('Method','PER')}))
    timing_file=filename.with_name(filename.stem+'_timing.csv')
    print('\nOnline runtime (seconds; synchronized CUDA; excludes endpoint observers and file I/O):')
    display(time_rows,list(time_rows[0]));base.write_trace(timing_file,time_rows)
    print('Pred includes Cache (not an extra charge). Update includes Context, reward probes and selector updates. Other includes bookkeeping. Run modes sequentially for timing comparisons.')
    print('Timing report saved:',timing_file)
    retention_file=filename.with_name(filename.stem+'_same_level.csv')
    base.write_trace(retention_file,retention_rows)
    print('\nSame-level retention on unqueried rows (%; positive Drop = net forgetting):')
    same_fields=['PER','Mode','N','Initial','Final','Drop','Broken','Fixed','CtxDrop']
    same_rows=[{key:row[key] for key in same_fields} for row in retention_rows if row['Subset']=='Unqueried']
    for mode in a.modes:
        valid=[row for row in same_rows if row['Mode']==mode and row['N']>0]
        if valid:
            same_rows.append(dict(PER='Mean',Mode=mode,**{
                key:float(np.mean([row[key] for row in valid])) for key in same_fields[2:]}))
    # display() accepts numeric missing values as strings; empty subsets are explicit N/A.
    display([{key:'N/A' if value is None else value for key,value in row.items()} for row in same_rows],same_fields)
    print('Initial/Final: retrospective prediction-only system; CtxDrop: mean of three Context drops. No labels from these rows used for updates.')
    print('Unqueried subsets can differ between policies; these are not independent heldout sets. All-row and unqueried results saved:',retention_file)
    if a.query_schedule=='quality':
        print('Quality is an external feedback controller; routing/update scores and rewards are unchanged. It is not a purely RL query policy.')
        print('Quality trajectories can acquire different rows/counts across modes. Use uniform feedback for the matched-resource routing control.')
    if aware:
        print('Budget-aware RL: all seven actions selected by policy/epsilon exploration; only exhausted-budget query masking. No minimum spending guarantee.')
        print('Query actions predict through the learned best route before feedback, then update their selected Context. Random uses random prediction routes.')
    return evaluation_results


QUALITY_OPTIONS=('bootstrap_labels','audit_fraction','window','return_window','min_return_updates',
                 'tolerance','gap_scale','gain_scale','burst')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--source-run',type=Path)
    p.add_argument('--include-seen-levels',action='store_true',help='Evaluation only: include seen intensities on disjoint TEST source IDs; reported separately from unseen domains.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--burst-dir',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--router-warmup-episodes',type=int,default=3)
    p.add_argument('--balanced-reward-reference',action='store_true',help='Class-balanced probe sampling and macro class gain/damage; online uses only previous paid labels.')
    p.add_argument('--resume-selector',action='store_true',help='Continue all original selector weights from --source-run, with fresh replay and optimizers.')
    p.add_argument('--decoupled-query',action='store_true',help='Isolated seven-action ablation: predict through best learned route when querying, then update action-selected Context; use in train and evaluate.')
    p.add_argument('--context-lr',type=float,default=.0005)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--selection-cost',type=float,default=.0001)
    p.add_argument('--gain-weight',type=float,default=1)
    p.add_argument('--retention-weight',type=float,default=1)
    p.add_argument('--gamma',type=float,default=.95)
    p.add_argument('--train-epsilon',type=float,default=.2)
    p.add_argument('--eval-epsilon',type=float,default=0)
    p.add_argument('--query-schedule',choices=['uniform','policy','quality'],default='uniform')
    p.add_argument('--budget-aware-policy',action='store_true',help='Train/use v2 seven-action RL; query prediction and update destination are separate; no external query gate.')
    p.add_argument('--budget-window',type=int,default=64)
    p.add_argument('--budget-need-weight',type=float,default=1.)
    p.add_argument('--budget-price-step',type=float,default=.005)
    p.add_argument('--budget-price-max',type=float,default=.2)
    p.add_argument('--budget-propensity-clip',type=float,default=20.)
    p.add_argument('--quality-bootstrap-labels',type=int,default=12)
    p.add_argument('--quality-audit-fraction',type=float,default=.1,help='Fraction of cap reserved for random monitors, in addition to bootstrap.')
    p.add_argument('--quality-window',type=int,default=32)
    p.add_argument('--quality-return-window',type=int,default=12)
    p.add_argument('--quality-min-return-updates',type=int,default=8)
    p.add_argument('--quality-tolerance',type=float,default=.02)
    p.add_argument('--quality-gap-scale',type=float,default=.15)
    p.add_argument('--quality-gain-scale',type=float,default=.001)
    p.add_argument('--quality-burst',type=int,default=4)
    p.add_argument('--update-policy',choices=['learned','fixed'],default='learned',help='fixed: label-blind shared destinations for prediction-routing control.')
    p.add_argument('--bank-utility',action='store_true',help='Optional second-stage experiment: reward best-bank reference loss rather than selected-slot loss.')
    p.add_argument('--modes',nargs='+',choices=['learned','random'],default=['learned','random'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if min(a.epochs,a.patience)<1 or a.router_warmup_episodes<0:p.error('Invalid epochs/patience/warmup.')
    if a.include_seen_levels and a.command!='evaluate':p.error('--include-seen-levels is evaluation only.')
    if not 0<=a.budget_fraction<=1 or not 0<=a.gamma<=1 or not 0<=a.train_epsilon<=1 or not 0<=a.eval_epsilon<=1:p.error('Budget, gamma and epsilon must be in [0,1].')
    if not np.isfinite(a.context_lr) or a.context_lr<=0 or not np.isfinite(a.selection_cost) or any(not np.isfinite(v) or v<0 for v in (a.gain_weight,a.retention_weight)):p.error('Invalid LR/reward.')
    if a.update_policy=='fixed' and a.query_schedule!='uniform':p.error('Fixed-destination control requires uniform feedback.')
    if a.balanced_reward_reference and (a.bank_utility or a.budget_aware_policy):
        p.error('Balanced reward reference is an isolated v1 selected-Context reward experiment.')
    if a.resume_selector and (a.command!='train' or a.router_warmup_episodes!=0):
        p.error('Resume selector requires train and --router-warmup-episodes 0.')
    if a.decoupled_query and (a.budget_aware_policy or a.query_schedule=='quality' or a.update_policy!='learned'):
        p.error('Decoupled query is an isolated v1 experiment with learned update destinations and no external quality gate.')
    if a.budget_aware_policy:
        if a.query_schedule!='policy' or a.update_policy!='learned' or a.bank_utility:
            p.error('Budget-aware RL requires --query-schedule policy --update-policy learned, without --bank-utility or an external gate.')
        try:BudgetState([.5]*ROUTES,**budget_settings(a))
        except ValueError as error:p.error(str(error))
    if a.query_schedule=='quality':
        if a.command!='evaluate':p.error('Quality control uses an existing trained checkpoint; use uniform/policy for training.')
        try:QualityBudget(1,0,.5,**{key:getattr(a,'quality_'+key) for key in QUALITY_OPTIONS})
        except ValueError as error:p.error(str(error))
    if a.device!='auto':core.device=torch.device(a.device)
    core.set_seed(a.seed)
    core.SELECTION_COST,core.GAIN_WEIGHT,core.RETENTION_WEIGHT=a.selection_cost,a.gain_weight,a.retention_weight
    core.REFERENCE_BATCH_SIZE,core.REFERENCE_HISTORY_SIZE=32,256
    a.output,a.burst_dir=a.output.resolve(),a.burst_dir.resolve()
    a.levels=a.levels or ([0,1,3] if a.command=='train' else [5,7,10,12,15,17,20])
    if any(v<0 for v in a.levels) or len(a.levels)!=len(set(a.levels)):p.error('Levels must be distinct nonnegative integers.')
    if a.command=='train':
        if a.source_run is None:p.error('Training requires a healthy --source-run.')
        if a.output.exists():p.error('Use a fresh output directory.')
        source=a.source_run.resolve()/'Models'
        old=torch.load(source/'adversarial_best_dqn.pth',map_location='cpu',weights_only=False)
        if set(a.levels)!=set(old.get('training_perturbations',[])):p.error('Training domains must match source CNN/Context development levels.')
        source_artifact=source/'protocol_preprocessing.joblib'
    else:
        source=a.output/'Models';source_artifact=source/'protocol_preprocessing.joblib'
    artifact=joblib.load(source_artifact)
    partitions=[np.asarray(artifact[k],dtype=np.int64) for k in ('train_ids','val_ids','test_ids')]
    all_ids=np.concatenate(partitions)
    if any(len(ids)==0 for ids in partitions) or len(np.unique(all_ids))!=len(all_ids) or (all_ids<0).any() or (all_ids>=artifact['source_rows']).any():p.error('Source train/validation/test IDs must be nonempty, unique and disjoint.')
    if core.file_sha256(a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv')!=artifact['source_sha256']:p.error('Clean source file differs from checkpoint provenance.')
    a.classes=[str(v) for v in artifact['label_encoder'].classes_]
    if len(a.classes)<2:p.error('At least two classes required.')
    if a.command=='train':
        a.output.mkdir(parents=True);shutil.copytree(source,a.output/'Models')
        train(a,a.output/'Models',artifact)
    else:evaluate(a,a.output/'Models',artifact)


if __name__=='__main__':main()


