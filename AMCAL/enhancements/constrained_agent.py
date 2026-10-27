"""Three-action DDQN with component rewards and primal-dual constraints.

All quality units are fractions. The min-max is policy cost versus nonnegative
constraint multipliers, not an adversarial Context-versus-selector game.
"""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def quality(confusion):
    c=np.asarray(confusion,dtype=np.float64);tp=np.diag(c)
    precision=np.divide(tp,c.sum(0),out=np.zeros_like(tp),where=c.sum(0)>0)
    recall=np.divide(tp,c.sum(1),out=np.zeros_like(tp),where=c.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)>0)
    return dict(AC=float(tp.sum()/max(c.sum(),1)),PR=float(precision.mean()),RC=float(recall.mean()),F1=float(f1.mean()))


def constraint_values(result,reference,tolerances):
    return np.array([reference['AC']-tolerances[0]-result['AC'],
        reference['F1']-tolerances[1]-result['F1'],
        result['forgetting']-reference['forgetting']-tolerances[2]],dtype=np.float64)


def dual_ascent(values,violations,lr,limit=100.):
    values=np.asarray(values,dtype=np.float64);violations=np.asarray(violations,dtype=np.float64)
    if values.shape!=(3,) or violations.shape!=(3,) or not np.isfinite([*values,*violations,lr,limit]).all() or lr<0 or limit<=0:
        raise ValueError('Invalid dual update.')
    return np.clip(values+lr*violations,0.,limit)


def training_indices(labels,losses,count,seed,hard_fraction=.5):
    """TRAIN-only curriculum: class-covered hard examples plus uniform remainder."""
    labels=np.asarray(labels);losses=np.asarray(losses,dtype=np.float64)
    if losses.shape!=labels.shape or not np.isfinite(losses).all() or not 0<=hard_fraction<=1 or count<1:
        raise ValueError('Invalid TRAIN curriculum inputs.')
    count=min(count,len(labels));classes=np.unique(labels);rng=np.random.default_rng(seed)
    mandatory=[];extra=[]
    for cls in classes:
        indices=np.flatnonzero(labels==cls)
        order=indices[np.lexsort((rng.random(len(indices)),-losses[indices]))] if hard_fraction else rng.permutation(indices)
        minimum=min(3,len(indices));mandatory.extend(order[:minimum].tolist())
        quota=max(minimum,int(np.ceil(count*hard_fraction/len(classes))))
        extra.extend(order[minimum:quota].tolist())
    if len(mandatory)>count:raise ValueError('Episode rows cannot cover the TRAIN classes.')
    extra=sorted(extra,key=lambda i:-losses[i])[:count-len(mandatory)]
    chosen=[*mandatory,*extra]
    remaining=np.setdiff1d(np.arange(len(labels)),chosen)
    chosen.extend(rng.permutation(remaining)[:count-len(chosen)].tolist())
    return np.sort(chosen)


def scalar_rewards(components,duals):
    """[-label_fraction, correct_fraction, macro_F1_increment, -forgetting]."""
    return components[...,0]+(components[...,1:]*duals).sum(-1)


def mask_q(values,can_query):
    result=values.clone()
    result[...,2]=torch.where(torch.as_tensor(can_query,device=values.device),result[...,2],-torch.inf)
    return result


class SmallQ(nn.Module):
    def __init__(self,width):
        super().__init__();self.network=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,state):return self.network(state)


class ConstrainedDDQN:
    def __init__(self,width,device,seed,lr=.001,capacity=20000):
        self.device=device;self.policy=SmallQ(width).to(device);self.target=deepcopy(self.policy).eval()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=lr,weight_decay=1e-4)
        self.memory=deque(maxlen=capacity);self.rng=np.random.default_rng(seed);self.updates=0

    @torch.no_grad()
    def values(self,state):
        return self.policy(torch.as_tensor(state,dtype=torch.float32,device=self.device)[None])[0].cpu().numpy()

    def choose(self,state,can_query,epsilon,rng):
        values=self.values(state)
        legal=[0,1,2] if can_query else [0,1]
        action=int(rng.choice(legal)) if rng.random()<epsilon else int(legal[np.argmax(values[legal])])
        return action,values

    def warm_start(self,examples,steps=100):
        if not examples:raise ValueError('No TRAIN uncertainty demonstrations.')
        self.policy.train();losses=[]
        for _ in range(steps):
            at=self.rng.choice(len(examples),min(128,len(examples)),replace=False)
            rows=[examples[int(i)] for i in at]
            states=torch.as_tensor(np.stack([r[0] for r in rows]),device=self.device)
            actions=torch.tensor([r[1] for r in rows],device=self.device)
            routes=torch.tensor([r[2] for r in rows],device=self.device)
            legal=torch.tensor([r[3] for r in rows],device=self.device)
            logits=self.policy(states)
            loss=F.cross_entropy(mask_q(logits,legal),actions)+.5*F.cross_entropy(logits[:,:2],routes)
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),1.,error_if_nonfinite=True);self.optimizer.step()
            losses.append(float(loss.detach()))
        self.target.load_state_dict(self.policy.state_dict())
        # Imitation is initialization; TD gets a fresh optimizer.
        lr=self.optimizer.param_groups[0]['lr']
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=lr,weight_decay=1e-4)
        return float(np.mean(losses))

    def add_episode(self,states,actions,components,legal):
        if not (len(states)==len(actions)==len(components)==len(legal)) or not states:
            raise ValueError('Malformed episode.')
        for i in range(len(states)):
            done=i+1==len(states)
            self.memory.append((np.asarray(states[i],dtype=np.float32).copy(),int(actions[i]),
                np.asarray(components[i],dtype=np.float32).copy(),
                np.zeros_like(states[i]) if done else np.asarray(states[i+1],dtype=np.float32).copy(),
                done,False if done else bool(legal[i+1])))

    def learn(self,duals,steps=20):
        if len(self.memory)<32:return None
        losses=[];self.policy.train()
        multiplier=torch.as_tensor(duals,dtype=torch.float32,device=self.device)
        for _ in range(steps):
            at=self.rng.choice(len(self.memory),min(128,len(self.memory)),replace=False)
            rows=[self.memory[int(i)] for i in at]
            states=torch.as_tensor(np.stack([r[0] for r in rows]),device=self.device)
            actions=torch.tensor([r[1] for r in rows],device=self.device)
            components=torch.as_tensor(np.stack([r[2] for r in rows]),device=self.device)
            next_states=torch.as_tensor(np.stack([r[3] for r in rows]),device=self.device)
            terminal=torch.tensor([r[4] for r in rows],device=self.device)
            legal=torch.tensor([r[5] for r in rows],device=self.device)
            with torch.no_grad():
                next_action=mask_q(self.policy(next_states),legal).argmax(1)
                future=self.target(next_states).gather(1,next_action[:,None]).squeeze(1)
                # Finite horizon, gamma=1: the reward sum equals the stated episode objective.
                target=scalar_rewards(components,multiplier)+torch.where(terminal,0.,future)
            prediction=self.policy(states).gather(1,actions[:,None]).squeeze(1)
            loss=F.smooth_l1_loss(prediction,target)
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),1.,error_if_nonfinite=True);self.optimizer.step()
            self.updates+=1
            if self.updates%20==0:self.target.load_state_dict(self.policy.state_dict())
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def snapshot(self):
        return dict(policy=deepcopy(self.policy.state_dict()),target=deepcopy(self.target.state_dict()),
            optimizer=deepcopy(self.optimizer.state_dict()),memory=list(self.memory),
            rng=deepcopy(self.rng.bit_generator.state),updates=self.updates)

    def restore(self,saved):
        self.policy.load_state_dict(saved['policy']);self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer']);self.memory=deque(saved['memory'],maxlen=self.memory.maxlen)
        self.rng.bit_generator.state=saved['rng'];self.updates=int(saved['updates'])
