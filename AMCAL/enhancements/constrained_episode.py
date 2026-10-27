"""Causal adaptation environment. Observer labels never enter the policy state."""
import time
import numpy as np
import torch
from .constrained_agent import quality


def probabilities(logits):
    z=np.asarray(logits,dtype=np.float64);e=np.exp(z-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def entropy(p):return -(p*np.log(np.maximum(p,1e-12))).sum(-1)/np.log(p.shape[-1])


class History:
    def __init__(self,classes):
        self.counts=np.zeros(classes);self.correct=np.full(2,.5);self.ce=np.full(2,.5)
        self.embeddings=[];self.last=0;self.change=0.

    def state(self,logits,embedding,stats,distortion,used,cap,position,count):
        p=probabilities(logits);top=np.sort(p,axis=1)[:,-2:];h=entropy(p)
        middle=p.mean(0)
        js=.5*sum(float(np.sum(row*np.log(np.maximum(row,1e-12)/np.maximum(middle,1e-12)))) for row in p)/np.log(2.)
        normalized=np.asarray(embedding,dtype=np.float64)
        norm=float(np.linalg.norm(normalized));unit=normalized/max(norm,1e-8)
        distance=float(np.sqrt(np.mean(((normalized-stats['mean'])/stats['scale'])**2)))
        if self.embeddings:
            stored=np.stack(self.embeddings)
            nearest=float(np.clip((1-(stored@unit).max())/2,0,1))
            average=float(np.clip((1-np.dot(stored.mean(0),unit))/2,0,1))
        else:nearest,average=1.,1.
        predicted=p.argmax(1)
        support=self.counts[predicted]/max(self.counts.sum(),1)
        summary=[*top[:,1],*(top[:,1]-top[:,0]),*h,js,float(predicted[0]==predicted[1]),
            float(distortion),distance/(1+distance),nearest,average,float((self.counts>0).mean()),
            *support,cap/(cap+32.),(cap-used)/max(cap,1),position/max(count,1),
            (position-self.last)/max(count,1),self.change,*self.correct,*self.ce]
        state=np.asarray([*summary,*p.flatten()],dtype=np.float32)
        if len(state)!=24+2*p.shape[1] or not np.isfinite(state).all():raise ValueError('Invalid compact policy state.')
        return state,p,h

    def paid(self,p,truth,embedding,position,change):
        self.counts[truth]+=1
        self.correct=.9*self.correct+.1*(p.argmax(1)==truth)
        ce=-np.log(np.maximum(p[:,truth],1e-12));self.ce=.9*self.ce+.1*ce/(1+ce)
        value=np.asarray(embedding,dtype=np.float64);self.embeddings.append(value/max(np.linalg.norm(value),1e-8))
        self.embeddings=self.embeddings[-64:];self.last=position;self.change=.9*self.change+.1*change


@torch.no_grad()
def context_accuracy(cnn,context,reference):
    x,y=reference[:2];correct=0
    for start in range(0,len(y),128):
        prediction=cnn(context(x[start:start+128])).argmax(1).cpu().numpy()
        correct+=int((prediction==np.asarray(y)[start:start+128]).sum())
    return correct/len(y)


@torch.no_grad()
def system_accuracy(cnn,context,cache,reference,history,stats,agent,mode,seed,used,cap,position,count):
    x,y=reference[:2];z,h=cache(cnn,x);predictions=[];rng=np.random.default_rng(seed+50001)
    for start in range(0,len(y),128):
        inputs=x[start:start+128];changed=context(inputs);adapted=cnn(changed)
        magnitude=(changed-inputs).flatten(1).norm(dim=1)/inputs.flatten(1).norm(dim=1).clamp_min(1e-6)
        for j in range(len(inputs)):
            logits=torch.stack([z[start+j],adapted[j]]).cpu().numpy()
            d=float(magnitude[j]);state,p,e=history.state(logits,h[start+j].cpu().numpy(),stats,d/(1+d),used,cap,position,count)
            if mode=='RL':
                values=agent.values(state);route=int(values[1]>values[0])
            elif mode=='Uncertainty':route=int(e[1]<e[0])
            elif mode=='Random':route=int(rng.integers(2))
            else:route=1
            predictions.append(int(p[route].argmax()))
    return float((np.asarray(predictions)==np.asarray(y)).mean())


def run_episode(cnn,make_context,make_memory,cache,update,domain,anchors,reference,stats,classes,
                agent,mode,seed,budget,threshold,context_lr,epsilon=0.,initial_reference_accuracy=None):
    if mode not in ('RL','Uncertainty','Random','FrontLoadedContext','NoQueryContext'):raise ValueError('Unknown policy mode.')
    context=make_context().eval();optimizer=torch.optim.Adam(context.parameters(),lr=context_lr,weight_decay=1e-4)
    memory=make_memory(anchors,seed);history=History(classes);rng=np.random.default_rng(seed+71)
    order=np.random.default_rng(seed).permutation(len(domain[1]))
    x=domain[0][torch.as_tensor(order,device=domain[0].device)]
    labels=np.asarray(domain[1])[order];ids=np.asarray(domain[2])[order]
    n=len(labels);cap=int(n*budget);used=0
    if not n or not 0<=budget<=1:raise ValueError('Invalid episode or budget.')
    before=context_accuracy(cnn,context,reference) if initial_reference_accuracy is None else float(initial_reference_accuracy)
    confusion=np.zeros((classes,classes),dtype=np.int64)
    trace=[];states=[];actions=[];components=[];legal=[];examples=[];previous_f1=0.
    if x.is_cuda:torch.cuda.synchronize(x.device)
    start_time=time.perf_counter()
    frozen,embeddings=cache(cnn,x);frozen=frozen.detach();embeddings=embeddings.detach()
    for pos in range(n):
        inputs=x[pos:pos+1]
        with torch.no_grad():
            changed=context(inputs);adapted=cnn(changed)
            denominator=inputs.flatten(1).norm().clamp_min(1e-6)
            magnitude=float((changed-inputs).flatten(1).norm()/denominator)
            logits=torch.cat([frozen[pos:pos+1],adapted]).cpu().numpy()
        embedding=embeddings[pos].cpu().numpy()
        state,p,h=history.state(logits,embedding,stats,magnitude/(1+magnitude),used,cap,pos,n)
        can_query=used<cap;uncertain_route=int(h[1]<h[0])
        teacher_action=2 if can_query and h[uncertain_route]>=threshold else uncertain_route
        values=np.zeros(3)
        if mode=='RL':action,values=agent.choose(state,can_query,epsilon,rng)
        elif mode=='Uncertainty':action=teacher_action
        elif mode=='FrontLoadedContext':action=2 if can_query and pos<cap else 1
        elif mode=='NoQueryContext':action=1
        else:action=int(rng.choice([0,1,2] if can_query else [0,1]))
        if action==2:
            # Fixed action semantics for DDQN: action 2 is Context + feedback.
            # Its prediction route must not change when Q0/Q1 parameters change.
            route=1 if mode=='RL' else (int(rng.integers(2)) if mode=='Random' else uncertain_route)
            if mode=='FrontLoadedContext':route=1
        else:route=action
        if action==2 and not can_query:raise RuntimeError('Policy requested feedback after budget exhaustion.')
        prediction=int(p[route].argmax())
        # Only now is current truth used for scoring/update; unqueried truth is observer-only.
        truth=int(labels[pos]);confusion[truth,prediction]+=1;current=quality(confusion)
        reward=np.array([-float(action==2)/n,float(prediction==truth)/n,current['F1']-previous_f1,0.],dtype=np.float32)
        previous_f1=current['F1']
        if action==2:
            paid,replay=memory.draw(ids[pos],inputs)
            update(context,optimizer,cnn,(inputs,torch.tensor([truth],device=x.device)),paid,replay)
            with torch.no_grad():new=cnn(context(inputs)).softmax(1)[0].cpu().numpy()
            history.paid(p,truth,embedding,pos+1,float(np.abs(new-p[1]).mean()))
            memory.add(ids[pos],inputs,truth);used+=1
        states.append(state);actions.append(action);components.append(reward);legal.append(can_query)
        examples.append((state,teacher_action,uncertain_route,can_query))
        trace.append(dict(step=pos+1,source_row=int(ids[pos]),truth=truth,prediction=prediction,route=route,
            action=action,queried=action==2,labels_used=used,cap=cap,
            frozen_entropy=float(h[0]),context_entropy=float(h[1]),novelty=float(state[10]),
            q0=float(values[0]),q1=float(values[1]),q2=float(values[2])))
    if x.is_cuda:torch.cuda.synchronize(x.device)
    elapsed=time.perf_counter()-start_time
    after=context_accuracy(cnn,context,reference);forgetting=before-after
    routed_source=system_accuracy(cnn,context,cache,reference,history,stats,agent,mode,seed,used,cap,n,n)
    components[-1][3]=-forgetting
    result=dict(**quality(confusion),samples=n,labels=used,cap=cap,label_fraction=used/n,
        source_initial_AC=before,source_final_AC=after,forgetting=forgetting,
        source_system_final_AC=routed_source,
        milliseconds_per_row=1000*elapsed/n,confusion=confusion.tolist(),
        forgetting_definition='fixed Context path accuracy before minus after on source audit; common path for every policy')
    if used>cap or used!=sum(r['queried'] for r in trace):raise RuntimeError('Invalid feedback accounting.')
    return result,trace,(states,actions,components,legal),examples
