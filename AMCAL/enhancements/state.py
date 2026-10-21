"""Route-count aware state and legal-action mask; default v1 semantics retained."""
import numpy as np
import torch
import amcal_burst_structured_bank_protocol as legacy


def build_state(outputs, embedding, history, used, cap, position, count, gate):
    logits, distortion = outputs
    routes = logits.shape[1]
    if routes == 4:
        return legacy.build_state(outputs, embedding, history, used, cap, position, count, gate)
    probs = logits.softmax(2)
    top = probs.topk(2, dim=2).values
    confidence = top[:,:,0]; margin = top[:,:,0]-top[:,:,1]
    entropy = -(probs*probs.clamp_min(1e-8).log()).sum(2)/np.log(probs.shape[2])
    prediction = probs.argmax(2)
    agreement = (prediction[:,:,None] == prediction[:,None,:]).float().mean(2)
    frozen_flag = torch.zeros_like(confidence); frozen_flag[:,0] = 1
    info = torch.stack([confidence,margin,entropy,agreement,distortion,frozen_flag],dim=2)
    route_x = torch.cat([embedding[:,None,:].expand(-1,routes,-1), probs,
        probs[:,:1,:].expand(-1,routes,-1),probs.mean(1,keepdim=True).expand(-1,routes,-1),
        probs.max(1,keepdim=True).values.expand(-1,routes,-1),info,
        history.features(probs,cap,position,count)],dim=2)
    extra = logits.new_tensor([[gate,(cap-used)/max(cap,1),position/max(count,1)]]).expand(len(logits),-1)
    return torch.cat([route_x.flatten(1),extra],dim=1)


def masked_q(q, states, slots):
    routes = slots+1
    result = q.clone()
    gate, remaining = states[:,-3], states[:,-2]
    result[(remaining <= 0) | (gate == 0),routes:] = -torch.inf
    result[(remaining > 0) & (gate == 1),:routes] = -torch.inf
    fixed = (remaining > 0) & (gate >= 2)
    result[fixed,:routes] = -torch.inf
    for k in range(slots): result[fixed & (gate != k+2),k+routes] = -torch.inf
    return result
