"""TRAIN-only pairwise initialization; the subsequent quota max is separate."""
import numpy as np
import torch
from torch.nn import functional as F

from enhancements.actor_diagnostics import correlation


def pairwise_rank_step(head,optimizer,states,targets,groups):
    if len(states)>256:
        at=torch.randperm(len(states),device=states.device)[:256]
        groups=torch.as_tensor(groups,device=states.device)[at]
        states,targets=states[at],targets[at]
    logits,_=head(states.detach())
    target=targets.detach().reshape_as(logits)
    group=torch.as_tensor(groups,device=logits.device)
    if group.shape!=logits.shape or not torch.isfinite(target).all():
        raise ValueError('Finite targets and one TRAIN group per state are required.')
    difference=target[:,None]-target[None,:]
    legal=torch.triu(torch.ones_like(difference,dtype=torch.bool),diagonal=1)
    legal &= (group[:,None]==group[None,:]) & (difference.abs()>1e-8)
    if not legal.any():raise ValueError('No non-tied within-level TRAIN ranking pairs.')
    preference=difference[legal].sign()
    predicted=(logits[:,None]-logits[None,:])[legal]
    loss=F.softplus(-preference*predicted).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite pairwise TRAIN loss.')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(head.actor.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def rank_metrics(head,states,targets,groups,quota):
    logits=head(states)[0].detach().cpu().numpy()
    target=targets.detach().cpu().numpy();group=np.asarray(groups)
    rows=[]
    for level in np.unique(group):
        at=np.flatnonzero(group==level);z,s=logits[at],target[at]
        level_quota=quota[int(level)] if isinstance(quota,dict) else quota
        k=max(1,min(len(at),int(round(len(at)*level_quota))))
        top=np.argsort(-z,kind='stable')[:k];oracle=np.sort(s)[-k:]
        rows.append(dict(level=int(level),samples=len(at),top_k=k,
            spearman=correlation(z,s,True),target_regret=float(oracle.mean()-s[top].mean())))
    values=[r['spearman'] for r in rows]
    return dict(spearman=None if any(v is None for v in values) else float(np.mean(values)),
        target_regret=float(np.mean([r['target_regret'] for r in rows])),levels=rows)

