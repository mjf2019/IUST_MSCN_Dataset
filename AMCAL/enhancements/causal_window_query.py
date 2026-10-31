"""Causal window ranking: predict the entire arrived window before paid feedback."""
import math
import time
import numpy as np
import torch
import amcal_two_action_paper_gain as base

WINDOW_MODES = ['UncertaintyWindow','RandomWindow','DDQNWindowRank']


def window_quota(start,end,fraction):
    if (not isinstance(start,int) or not isinstance(end,int) or start<0 or end<=start or
            not math.isfinite(fraction) or not 0<=fraction<=1):
        raise ValueError('Invalid causal window or budget.')
    return int(fraction*end)-int(fraction*start)


def rank_arrived(scores,quota):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<=quota<=len(values):
        raise ValueError('Invalid arrived-window scores/quota.')
    # Equal scores favor the earlier arrival, never a label or future sample.
    return sorted(sorted(range(len(values)),key=lambda i:(-values[i],i))[:quota])


def run_window_episode(experiment,learner,domain,reference,seed,mode,window_size=5):
    if mode not in WINDOW_MODES or not isinstance(window_size,int) or window_size<1:
        raise ValueError('Unknown causal-window policy or invalid window size.')
    a=experiment.a; cnn=experiment.cnn; context=experiment.context()
    optimizer=torch.optim.Adam(context.parameters(),lr=a.context_lr,weight_decay=1e-4)
    memory=experiment.memory(seed)
    order=np.random.default_rng(seed).permutation(len(domain[1]))
    x=domain[0][torch.as_tensor(order,device=domain[0].device)]
    y,ids=np.asarray(domain[1])[order],np.asarray(domain[2])[order]
    n=len(y); fraction=a.budget_fraction; cap=int(fraction*n)
    if not n or not math.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Nonempty stream and a legal feedback budget are required.')
    before=base.endpoint(experiment,context,reference)
    target_before=base.endpoint(experiment,context,(x,y))
    feature_keys=base.pilot.feature_keys(x)
    paid_embeddings=[]; paid_labels=[]; traces=[]; predictions=[]
    used=last_update=0; delays=[]
    query_hash=base.weight_hash(learner.policy.state_dict()); updates,steps=learner.updates,learner.steps
    cnn_hash=base.weight_hash(cnn.state_dict())
    rng=np.random.default_rng(seed+71)
    if x.is_cuda: torch.cuda.synchronize(x.device)
    started=time.perf_counter()
    frozen,embeddings=base.pilot.base.legacy.base_cache(cnn,x)
    executor=base.BankExecutor(cnn,[context])
    for start in range(0,n,window_size):
        end=min(n,start+window_size); pending=[]; scores=[]
        used_at_prediction=used
        # No label acquisition or update may occur until every window prediction is fixed.
        for position in range(start,end):
            inputs=x[position:position+1]
            with torch.no_grad():
                outputs=executor.outputs(inputs,frozen[position:position+1])
                entropy=float(base.pilot.normalized_entropy(outputs[0][0,1].cpu().numpy()))
                pred=int(outputs[0][0,1].argmax())
                frozen_pred=int(outputs[0][0,0].argmax())
                q=None
                if mode=='DDQNWindowRank':
                    state=base.query_features(outputs,embeddings[position:position+1],paid_embeddings,
                        paid_labels,used,cap,position,n,last_update).detach().cpu().reshape(-1)
                    q=learner.values(state)
            score=entropy if mode=='UncertaintyWindow' else (
                float(rng.random()) if mode=='RandomWindow' else float(q[1]-q[0]))
            scores.append(score)
            pending.append(dict(step=position+1,source_row=int(ids[position]),prediction=pred,
                frozen_prediction=frozen_pred,context_prediction=pred,route=1,
                context_entropy=entropy,rank_score=score,
                q_skip=None if q is None else float(q[0]),q_query=None if q is None else float(q[1]),
                labels_used_at_prediction=used_at_prediction,window_start=start+1,window_end=end))
            predictions.append(pred)
        quota=window_quota(start,end,fraction)
        selected=rank_arrived(scores,quota)
        selected_set=set(selected)
        feedback_counts={}
        # Only already-arrived rows can now be queried. Process selected rows in arrival order.
        for local in selected:
            position=start+local
            inputs=x[position:position+1]
            truth=int(y[position])
            label=torch.tensor([truth],device=x.device)
            paid,replay=memory.draw(ids[position],inputs,reference_keys=set(),current_key=feature_keys[position])
            base.pilot.context_min_step(context,optimizer,cnn,(inputs,label),paid,replay,a.replay_weight)
            memory.add(ids[position],inputs,truth)
            paid_embeddings.append(embeddings[position:position+1].detach().clone())
            paid_labels.append(truth); used+=1; last_update=end
            feedback_counts[local]=used
            delays.append(end-(position+1))
        if used!=int(fraction*end) or used>cap:
            raise RuntimeError('Causal-window quota accounting failed.')
        # Ground truth for unqueried rows is observer-only, appended after selection/updates.
        for local,row in enumerate(pending):
            queried=local in selected_set
            row.update(truth=int(y[start+local]),action=int(queried),queried=queried,
                cap=cap,window_quota=quota,labels_used=used_at_prediction,
                labels_used_after_window=used,labels_used_after_feedback=feedback_counts.get(local),
                feedback_step=end if queried else None,feedback_delay_rows=end-row['step'] if queried else None,
                prediction_before_feedback=True,query_rule=mode,threshold=None,epsilon=0.,
                terminal=row['step']==n)
            traces.append(row)
    if x.is_cuda: torch.cuda.synchronize(x.device)
    elapsed=time.perf_counter()-started
    after=base.endpoint(experiment,context,reference)
    target_after=base.endpoint(experiment,context,(x,y))
    if (query_hash!=base.weight_hash(learner.policy.state_dict()) or updates!=learner.updates or steps!=learner.steps
            or cnn_hash!=base.weight_hash(cnn.state_dict())):
        raise RuntimeError('Window evaluation changed the CNN or DDQN.')
    if used!=cap or used!=sum(r['queried'] for r in traces):
        raise RuntimeError('Window method did not use exactly the allowed labels.')
    return dict(online=base.pilot.metrics(y,predictions,len(a.classes)),samples=n,labels=used,
        cap=cap,labels_percent=100*used/n,source_before=before,source_after=after,
        original_domain_forgetting={k:before[k]-after[k] for k in before},
        target_before=target_before,target_after=target_after,td_updates=0,mean_TD_loss=None,
        milliseconds_per_row=1000*elapsed/n,window_size=window_size,
        mean_feedback_delay_rows=float(np.mean(delays)) if delays else 0.,
        max_feedback_delay_rows=max(delays,default=0),
        label_protocol='predict all arrived window rows, then rank/query; no window label used in its predictions',
        Q_learning='none; fixed DDQN; no TEST reward or threshold calibration',
        quota='floor(fraction*window_end)-floor(fraction*window_start); exact global floor cap',
        ranking='original arrival-time scores, no rescoring after window feedback'),traces

