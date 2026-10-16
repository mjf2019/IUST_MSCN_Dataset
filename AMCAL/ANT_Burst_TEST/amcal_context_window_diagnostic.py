"""Isolate shared-Context window effects using identical label-blind query positions.
No RL, flow/class routing or class quotas. Predictions always precede feedback.
"""
from pathlib import Path
from collections import deque, Counter
import argparse
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import amcal_burst_three_action as core


def query_positions(n, fraction, seed):
    cap = int(n * fraction)
    return set(int(i) for i in np.random.default_rng(seed + 1000003).choice(n, cap, replace=False))


def metrics(truth, pred):
    return dict(accuracy=float(accuracy_score(truth, pred)),
                macro_f1=float(f1_score(truth, pred, average='macro', zero_division=0)),
                weighted_f1=float(f1_score(truth, pred, average='weighted', zero_division=0)),
                prediction_counts=dict(Counter(int(i) for i in pred)))


def score(x, y, cnn, context=None):
    predictions = []
    with torch.no_grad():
        for start in range(0,len(y),128):
            batch = x[start:start+128]
            predictions.extend(cnn(batch if context is None else context(batch)).argmax(1).cpu().tolist())
    return metrics(y, predictions)


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir',type=Path,default=root/'Dataset')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--windows',type=int,nargs='+',default=[1,5])
    p.add_argument('--context-lr',type=float,default=.25)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--stream-order',choices=['input','shuffle'],default='shuffle')
    p.add_argument('--per',type=int,choices=core.LEVELS)
    args=p.parse_args()
    if not args.windows or any(w<1 for w in args.windows) or len(set(args.windows)) != len(args.windows):
        p.error('Windows must be distinct positive integers.')
    if not 0<=args.budget_fraction<=1 or not np.isfinite(args.context_lr) or args.context_lr<=0:
        p.error('Invalid budget fraction or Context learning rate.')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a new empty output directory to preserve prior results.')
    bundle=torch.load(args.checkpoint_output/'Models'/'three_action_best.pth',map_location='cpu',weights_only=False)
    if bundle.get('version') != core.VERSION:
        p.error('Expected a trained three-action checkpoint.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for level in ([args.per] if args.per is not None else core.LEVELS):
        path=args.data_dir/'AdvBurst_FTSC-IAT'/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
        frame=pd.read_csv(path).iloc[:1073]
        x,y=core.tensors(frame,bundle['scaler'],bundle['encoder'],bundle['columns'],device)
        order=(np.random.default_rng(args.seed).permutation(len(y)) if args.stream_order=='shuffle' else np.arange(len(y)))
        positions=query_positions(len(y),args.budget_fraction,args.seed)
        schedule_hash=hashlib.sha256(json.dumps(dict(order=order.tolist(),queries=sorted(positions))).encode()).hexdigest()
        for window_size in args.windows:
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
            cnn,context=core.build_models(bundle['cnn_checkpoint'],bundle['context_checkpoint'],device)
            frozen=score(x,y,cnn)
            initial=score(x,y,cnn,context)
            cnn_before={k:v.detach().cpu().clone() for k,v in cnn.state_dict().items()}
            optimizer=torch.optim.Adam(context.parameters(),lr=args.context_lr,weight_decay=1e-4)
            core.legacy.CONTEXT_LOSS_MODE=bundle['context_loss_mode']
            window=deque(maxlen=window_size)
            trace=[]
            for pos,index in enumerate(order):
                sample=x[index:index+1]
                with torch.no_grad():
                    logits=cnn(context(sample))
                    prediction=int(logits.argmax(1))
                    base_prediction=int(cnn(sample).argmax(1))
                # True labels are used below only if this fixed position is paid.
                paid=pos in positions
                before=after=None
                post_prediction=None
                if paid:
                    label=int(y[index])
                    target=torch.tensor([label],device=device)
                    before=float(nn.functional.cross_entropy(logits,target).item())
                    window.append((sample.detach().clone(),label))
                    wx=torch.cat([item[0] for item in window])
                    wy=torch.tensor([item[1] for item in window],device=device)
                    optimizer.zero_grad()
                    ce=nn.functional.cross_entropy(cnn(context(wx)),wy)
                    with torch.no_grad():
                        base_ce=nn.functional.cross_entropy(cnn(wx),wy)
                    loss=core.legacy.context_objective(ce,base_ce,online=True)
                    loss.backward();optimizer.step()
                    with torch.no_grad():
                        post_logits=cnn(context(sample))
                        after=float(nn.functional.cross_entropy(post_logits,target).item())
                        post_prediction=int(post_logits.argmax(1))
                trace.append(dict(step=pos+1,source_row=int(index)+1,true_label=int(y[index]),
                                  prediction_before_feedback=prediction,frozen_prediction=base_prediction,
                                  queried=paid,loss_before=before,loss_after=after,
                                  prediction_after_feedback=post_prediction,window_used=len(window)))
            assert sum(r['queried'] for r in trace)==int(len(y)*args.budget_fraction)
            assert all(torch.equal(cnn_before[k],v.detach().cpu()) for k,v in cnn.state_dict().items())
            truth=[r['true_label'] for r in trace]
            result=metrics(truth,[r['prediction_before_feedback'] for r in trace])
            result.update(perturbation_level=level,seed=args.seed,context_window=window_size,
                          context_lr=args.context_lr,labels_used=len(positions),label_cap=len(positions),
                          frozen_cnn=frozen,initial_context=initial,
                          final_context_retrospective=score(x,y,cnn,context),
                          retrospective_is_online_score=False,
                          schedule_sha256=schedule_hash,source_loss_mode=bundle['context_loss_mode'],
                          protocol='fixed uniform label-blind query positions; shared Context; no RL; before feedback')
            summaries.append(result)
            pd.DataFrame(trace).to_csv(args.output/f'PER_{level}_window{window_size}_trace.csv',index=False)
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
            print(json.dumps(result))
            del cnn,context,optimizer
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,dict)} for row in summaries]).to_csv(args.output/'combined_results.csv',index=False)
    (args.output/'manifest.json').write_text(json.dumps(vars(args),default=str,indent=2))
    print('Results:',args.output)


if __name__=='__main__':
    main()
