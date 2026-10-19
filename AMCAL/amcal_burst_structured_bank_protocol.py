"""Structured 3-Context / 7-action DDQN with directly supervised route correctness.

Frozen CNN and initial Context come from an existing healthy source run.
Offline correctness targets use existing TRAIN labels densely; Context updates
remain budgeted. Online learners see only paid labels after the scored prediction.
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
    def __init__(self, route_width):
        super().__init__()
        self.route_width = route_width
        # This scorer is shared by all routes, with an explicit frozen-route flag.
        self.correctness = nn.Sequential(nn.Linear(route_width,128),nn.ReLU(),
                                         nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        pooled = 2*route_width + 2
        self.future = nn.Sequential(nn.Linear(pooled,128),nn.ReLU(),nn.Linear(128,1))
        self.update_value = nn.Sequential(nn.Linear(pooled+route_width,128),nn.ReLU(),nn.Linear(128,1))
        nn.init.zeros_(self.future[-1].weight); nn.init.zeros_(self.future[-1].bias)
        nn.init.zeros_(self.update_value[-1].weight); nn.init.zeros_(self.update_value[-1].bias)

    def route_logits(self, states):
        route_x = states[:, :-3].reshape(-1,ROUTES,self.route_width)
        return self.correctness(route_x).squeeze(-1)

    def forward(self, states, detach_routes=False):
        route_x = states[:, :-3].reshape(-1,ROUTES,self.route_width)
        p = self.correctness(route_x).squeeze(-1).sigmoid()
        reward = 2*p - 1
        if detach_routes: reward = reward.detach()
        # Schedule gate masks actions; it is not a shortcut feature for scoring.
        pooled = torch.cat([route_x.mean(1),route_x.max(1).values,states[:,-2:]],dim=1)
        future = self.future(pooled)
        context_x = torch.cat([route_x[:,1:],pooled[:,None,:].expand(-1,SLOTS,-1)],dim=2)
        update = self.update_value(context_x).squeeze(-1)
        return torch.cat([future+reward,future+reward[:,1:]+update],dim=1)


class Agent:
    def __init__(self, route_width, device, seed, offline=False, policy_state=None, gamma=.95):
        self.device, self.gamma, self.offline = device,gamma,offline
        self.policy = StructuredQ(route_width).to(device)
        if policy_state is not None:self.policy.load_state_dict(policy_state)
        self.target = deepcopy(self.policy).eval()
        self.route_optimizer = torch.optim.Adam(self.policy.correctness.parameters(),lr=.001,weight_decay=1e-4)
        self.td_optimizer = torch.optim.Adam(list(self.policy.future.parameters())+list(self.policy.update_value.parameters()),lr=.0001,weight_decay=1e-4)
        self.route_memory = deque(maxlen=8192 if offline else 256)
        self.td_memory = deque(maxlen=4096 if offline else 512)
        self.rng = np.random.default_rng(seed+920017)
        self.clock,self.route_updates,self.td_updates = 0,0,0
        self.policy.eval()

    def _sample(self, memory, half_life):
        ids = self.rng.choice(len(memory),min(32,len(memory)),replace=False)
        samples = [memory[int(i)] for i in ids]
        ages = np.asarray([self.clock-row[-1] for row in samples])
        weights = torch.tensor(np.maximum(.05,np.exp2(-ages/half_life)),device=self.device,dtype=torch.float32)
        return samples,weights

    def observe_routes(self, state, correct, step_now=True):
        self.route_memory.append((state.detach().cpu().reshape(-1).clone(),
                                  correct.detach().cpu().float().reshape(ROUTES).clone(),self.clock))
        if not step_now:return
        samples,weight = self._sample(self.route_memory,4096 if self.offline else 128)
        x = torch.stack([r[0] for r in samples]).to(self.device)
        y = torch.stack([r[1] for r in samples]).to(self.device)
        loss = F.binary_cross_entropy_with_logits(self.policy.route_logits(x),y,reduction='none').mean(1)
        loss = (loss*weight).sum()/weight.sum()
        self.route_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(self.policy.correctness.parameters(),1)
        self.route_optimizer.step();self.route_updates += 1

    def push(self, state, action, reward, successor, terminal):
        self.td_memory.append((state.detach().cpu().reshape(-1).clone(),action,float(reward),
                               successor.detach().cpu().reshape(-1).clone(),float(terminal),self.clock))

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
    return torch.cat([route_x.flatten(1),extra],dim=1)


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
        return core.transfer_utility(before,core.probe_losses(cnn,bank[selected],batch))
    with torch.no_grad():
        after = before.clone()
        after[selected+1] = F.cross_entropy(cnn(bank[selected](batch[0])),batch[1],reduction='none')
    return core.transfer_utility(before.min(0).values.cpu().numpy(),after.min(0).values.cpu().numpy())


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
    schedule=set(np.random.default_rng(seed+819003).choice(n,cap,replace=False).tolist()) if a.query_schedule=='uniform' else None
    fixed_destinations=np.random.default_rng(seed+440003).integers(0,SLOTS,n)
    def gate(pos):return int(pos in schedule) if schedule is not None else -1
    def state_gate(pos):
        return 2+int(fixed_destinations[pos]) if a.update_policy=='fixed' and gate(pos)==1 else gate(pos)
    rng=np.random.default_rng(seed+719003)
    history=Reliability(len(a.classes))
    reward_reference=core.TrainingProbe(x,torch.tensor(y,device=x.device),seed+910033) if training else core.PaidHistory(seed+910033)
    bz,embedding=base_cache(cnn,x)
    initial_ref=reference_observer(cnn,bank,agent,*reference,history,0,cap,0,mode,a.seed) if reference is not None else None
    initial_current=reference_observer(cnn,bank,agent,x,y,history,0,cap,0,mode,a.seed,
                                       return_route_predictions=True) if same_level else None
    used=0;actions=np.zeros(ACTIONS,dtype=int);slot_queries=np.zeros(SLOTS,dtype=int)
    prediction_rows=[];trace=[];route_probabilities=[]
    oracle_all,oracle_legal=0,0
    route_updates0,td_updates0=agent.route_updates,agent.td_updates
    next_cached=None
    for pos in range(n):
        inputs=x[pos:pos+1]
        outputs=next_cached if next_cached is not None else route_outputs(cnn,bank,inputs,bz[pos:pos+1])
        next_cached=None
        state=build_state(outputs,embedding[pos:pos+1],history,used,cap,pos,n,state_gate(pos))
        allowed=base.legal_actions(SLOTS,cap-used,gate(pos))
        if a.update_policy=='fixed':
            fixed_action=4+int(fixed_destinations[pos])
            allowed=[v for v in allowed if v<4 or v==fixed_action]
        with torch.no_grad():
            values=agent.policy(state).squeeze(0)
            probability=agent.policy.route_logits(state).sigmoid().squeeze(0)
            epsilon=a.train_epsilon if training else a.eval_epsilon
            if mode=='random' or warmup or rng.random()<epsilon:
                action=int(rng.choice(allowed))
            else:
                valid=torch.tensor(allowed,device=x.device)
                action=int(valid[values[valid].argmax()].item())
        route=base.route_index(action,SLOTS)
        predictions=outputs[0].argmax(2).squeeze(0).cpu().numpy()
        prediction_rows.append(predictions.copy())
        route_probabilities.append(probability.cpu().numpy().copy())
        # Score is fixed now; truth below is used by learners only under the gates.
        prediction=int(predictions[route]);paid=action>=4
        truth=int(y[pos]);correct=(predictions==truth)
        oracle_all+=bool(correct.any())
        oracle_legal+=any(correct[base.route_index(v,SLOTS)] for v in allowed)
        actions[action]+=1
        agent.clock+=1
        successor_output=None
        if pos+1<n:
            # Cache once, then refresh just the updated slot for the actual successor.
            if not warmup and (paid or training):
                successor_output=route_outputs(cnn,bank,x[pos+1:pos+2],bz[pos+1:pos+2])
        zero=torch.zeros_like(state)
        skip_next=zero
        if successor_output is not None:
            skip_next=build_state(successor_output,embedding[pos+1:pos+2],history,used,cap,pos+1,n,state_gate(pos+1))
        reward,gain,harm=None,None,None
        before=None;batch=None
        if paid:
            assert used<cap
            selected=route-1
            label=torch.tensor([truth],device=x.device)
            if not warmup:
                batch=reward_reference.sample(inputs)
                before=utility_before(a,cnn,bank,selected,batch)
            history.observe_paid(outputs[0].softmax(2),truth)
            optimizers[selected].zero_grad()
            loss=core.context_objective(F.cross_entropy(cnn(bank[selected](inputs)),label),
                                        F.cross_entropy(outputs[0][:,0],label))
            loss.backward();optimizers[selected].step()
            utility,gain,harm=(0.0,0.0,0.0) if warmup else utility_after(a,cnn,bank,selected,batch,before)
            reward=(1.0 if prediction==truth else -1.0)+utility
            used+=1;slot_queries[selected]+=1
            after_current=refresh_slot(cnn,bank,inputs,outputs,route)
            change=(after_current[0][:,route].softmax(1)-outputs[0][:,route].softmax(1)).abs().mean().item()
            history.updated(route,pos+1,change)
            if not training:reward_reference.add(inputs,label)
            if successor_output is not None:
                successor_output=refresh_slot(cnn,bank,x[pos+1:pos+2],successor_output,route)
        # Offline TRAIN labels supervise all routes; validation/test only paid labels.
        may_learn=training or (learning and mode=='learned' and paid)
        if may_learn:
            target=torch.tensor(correct,device=x.device,dtype=torch.float32)
            step_now=(not training) or paid or pos%16==15 or pos+1==n
            agent.observe_routes(state,target,step_now=step_now)
        if may_learn and not warmup:
            # Non-query actions share the same no-update successor and future value.
            skip_state=state.clone()
            if schedule is not None:skip_state[:,-3]=0
            for r in range(ROUTES):
                agent.push(skip_state,r,1.0 if correct[r] else -1.0,skip_next,pos+1==n)
            if paid:
                successor=zero if successor_output is None else build_state(successor_output,embedding[pos+1:pos+2],history,used,cap,pos+1,n,state_gate(pos+1))
                # Budget exhaustion masks queries; it does NOT end future prediction value.
                agent.push(state,action,reward,successor,pos+1==n)
            if paid or pos%16==15 or pos+1==n:agent.learn_td()
        next_cached=successor_output
        row=dict(step=pos+1,source_row=int(source_ids[pos]),truth=truth,prediction=prediction,
                 action=action,route=route,queried=paid,labels_used=used,reward=reward,gain=gain,harm=harm,
                 oracle_all=bool(correct.any()),oracle_legal=any(correct[base.route_index(v,SLOTS)] for v in allowed))
        for r in range(ROUTES):
            row[f'prediction_{r}']=int(predictions[r]);row[f'correctness_probability_{r}']=float(probability[r].item())
        for action_id in range(ACTIONS):row[f'q_{action_id}']=float(values[action_id].item())
        trace.append(row)
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
                 labels=used,label_cap=cap,slot_queries=slot_queries.tolist(),action_counts=actions.tolist(),
                 route_updates=agent.route_updates-route_updates0,td_updates=agent.td_updates-td_updates0,
                 mode=mode,context_lr=a.context_lr,query_schedule=a.query_schedule,update_policy=a.update_policy,
                 bank_utility=a.bank_utility,seed=a.seed,
                 oracle_protocol='observer only, along this policy bank trajectory; legal oracle obeys uniform gate and fixed-destination control',
                 fixed_skip_protocol='retrospective control: keep actual query actions/destinations; use one fixed route on all non-query rows')
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
    selected=np.asarray(artifact[split])
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
    print(f'  Slot queries {m["slot_queries"]} | best fixed non-query route {100*m["best_fixed_skip_route_accuracy"]:.2f}% | route updates {m["route_updates"]} | TD updates {m["td_updates"]}')
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


def train(a,models,artifact):
    cnn,bank,initial,route_width=models_from_source(models,artifact)
    agent=Agent(route_width,core.device,a.seed,offline=True,gamma=a.gamma)
    domains={(v,s):read_domain(a.burst_dir,artifact,v,s) for v in a.levels for s in ('train_ids','val_ids')}
    for warm in range(a.router_warmup_episodes):
        level=a.levels[warm%len(a.levels)]
        m,_=episode(a,cnn,bank,initial,agent,domains[level,'train_ids'],mode='random',training=True,warmup=True,
                    rollout_seed=a.seed+100003+1009*warm)
        print(f'Correctness warmup {warm+1}/{a.router_warmup_episodes} | domain {level} | Context labels {m["labels"]}/{m["label_cap"]} | dense TRAIN route targets',flush=True)
    best,stale=-1,0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    hashes={name:core.file_sha256(models/name) for name in ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib')}
    domain_hashes={str(level):core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') for level in a.levels}
    log=[]
    for epoch in range(a.epochs):
        level=a.levels[epoch%len(a.levels)]
        agent.td_memory.clear()  # Context bank reset: do not mix old hidden-bank transitions.
        m,_=episode(a,cnn,bank,initial,agent,domains[level,'train_ids'],training=True,
                    rollout_seed=a.seed+200003+1009*epoch)
        scores=[]
        # Validation learners are temporary; no validation parameters/replay survive.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            snapshot=deepcopy(agent.policy.state_dict())
            for validation_level in a.levels:
                temporary=Agent(route_width,core.device,a.seed,policy_state=snapshot,gamma=a.gamma)
                result,_=episode(a,cnn,bank,initial,temporary,domains[validation_level,'val_ids'])
                scores.append(result['accuracy'])
        score=float(np.mean(scores))
        log.append(dict(epoch=epoch+1,level=level,train_accuracy=m['accuracy'],validation_accuracy=score,
                        labels=m['labels'],slot_queries=m['slot_queries']))
        print(f'Epoch {epoch+1}/{a.epochs} | domain {level} | validation {100*score:.2f}% | labels {m["labels"]}/{m["label_cap"]} | slot queries {m["slot_queries"]}',flush=True)
        if score>best:
            best,stale=score,0
            torch.save(dict(version=1,slots=SLOTS,actions=ACTIONS,route_width=route_width,
                            policy_state_dict=deepcopy(agent.policy.state_dict()),model_hashes=hashes,
                            training_levels=a.levels,training_domain_sha256=domain_hashes,config=config,validation_accuracy=best,
                            protocol='Qskip=B+2p-1; Qquery=B+2p_context-1+U; correctness BCE isolated from TD; shared expert scorer',
                            supervision='offline: existing TRAIN labels on all rows; online/validation: queried labels only',
                            initial_context='three independent clones of unchanged healthy source checkpoint'),models/'structured_bank_ddqn.pth')
        else:
            stale+=1
            if stale>=a.patience:break
    (a.output/'structured_training.json').write_text(json.dumps(dict(config=config,epochs=log,best_validation=best),indent=2))
    print(f'Training complete. Best validation {100*best:.2f}%. CNN/initial Context unchanged; no heldout test scored.')


def evaluate(a,models,artifact):
    saved=torch.load(models/'structured_bank_ddqn.pth',map_location='cpu',weights_only=False)
    if saved.get('version')!=1:raise ValueError('Unsupported structured-bank checkpoint.')
    for name,digest in saved['model_hashes'].items():
        if core.file_sha256(models/name)!=digest:raise ValueError('Source checkpoint mismatch: '+name)
    if set(a.levels)&set(saved['training_levels']):raise ValueError('Heldout test levels must exclude training levels.')
    cnn,bank,initial,route_width=models_from_source(models,artifact)
    if route_width!=saved['route_width']:raise ValueError('Router state width mismatch.')
    reference=read_domain(a.burst_dir,artifact,0,'val_ids')[:2]
    root=a.output/'Results';root.mkdir(exist_ok=True)
    execution={key:getattr(a,key) for key in ('seed','context_lr','budget_fraction','selection_cost','gain_weight','retention_weight','gamma','eval_epsilon','query_schedule','update_policy','bank_utility')}
    execution['same_level_diagnostics_version']=1
    run_id=hashlib.sha256(json.dumps(execution,sort_keys=True).encode()).hexdigest()[:8]
    rows=[];retention_rows=[]
    for level in a.levels:
        domain=read_domain(a.burst_dir,artifact,level,'test_ids')
        pair={}
        for mode in a.modes:
            core.set_seed(a.seed)
            agent=Agent(route_width,core.device,a.seed,policy_state=saved['policy_state_dict'],gamma=a.gamma)
            m,trace=episode(a,cnn,bank,initial,agent,domain,mode=mode,learning=mode=='learned',reference=reference,same_level=True)
            m.update(level=level,classes=a.classes,execution_config=execution,classifier_unchanged=not cnn.training and all(not p.requires_grad for p in cnn.parameters()))
            tag=f'structured_{mode}_per{level}_seed{a.seed}_lr{a.context_lr:g}_{a.update_policy}_{run_id}'
            (root/(tag+'_metrics.json')).write_text(json.dumps(m,indent=2))
            base.write_trace(root/(tag+'_trace.csv'),trace)
            report_level(level,mode,m);pair[mode]=m
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
    filename=root/f'combined_structured_seed{a.seed}_lr{a.context_lr:g}_eps{a.eval_epsilon:g}_{a.update_policy}_{run_id}.csv'
    base.write_trace(filename,rows)
    print('Saved:',filename)
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--source-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--burst-dir',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--router-warmup-episodes',type=int,default=3)
    p.add_argument('--context-lr',type=float,default=.0005)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--selection-cost',type=float,default=.0001)
    p.add_argument('--gain-weight',type=float,default=1)
    p.add_argument('--retention-weight',type=float,default=1)
    p.add_argument('--gamma',type=float,default=.95)
    p.add_argument('--train-epsilon',type=float,default=.2)
    p.add_argument('--eval-epsilon',type=float,default=0)
    p.add_argument('--query-schedule',choices=['uniform','policy'],default='uniform')
    p.add_argument('--update-policy',choices=['learned','fixed'],default='learned',help='fixed: label-blind shared destinations for prediction-routing control.')
    p.add_argument('--bank-utility',action='store_true',help='Optional second-stage experiment: reward best-bank reference loss rather than selected-slot loss.')
    p.add_argument('--modes',nargs='+',choices=['learned','random'],default=['learned','random'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if min(a.epochs,a.patience)<1 or a.router_warmup_episodes<0:p.error('Invalid epochs/patience/warmup.')
    if not 0<=a.budget_fraction<=1 or not 0<=a.gamma<=1 or not 0<=a.train_epsilon<=1 or not 0<=a.eval_epsilon<=1:p.error('Budget, gamma and epsilon must be in [0,1].')
    if not np.isfinite(a.context_lr) or a.context_lr<=0 or any(not np.isfinite(v) or v<0 for v in (a.selection_cost,a.gain_weight,a.retention_weight)):p.error('Invalid LR/reward.')
    if a.update_policy=='fixed' and a.query_schedule!='uniform':p.error('Fixed-destination control requires uniform feedback.')
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

