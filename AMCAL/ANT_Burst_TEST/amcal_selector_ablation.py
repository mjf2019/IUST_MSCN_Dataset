"""Selection-only diagnostic: common Context prediction route, exact global query count.
Uses existing three-action policy as a query policy; not a retrained binary selector.
No class labels are used for routing or query decisions.
"""
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
import random
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import amcal_burst_three_action as core


def query_gate(preferred, used, cap, remaining_samples):
    """Online exact quota: force only when every remaining sample is needed."""
    need = cap - used
    if need <= 0:
        return False, False
    mandatory = need >= remaining_samples
    return bool(preferred or mandatory), bool(mandatory and not preferred)


def block_schedule(n, block_size, fraction):
    """Allocate floor(n*fraction) queries by cumulative rounding, without labels."""
    return [(start, min(start+block_size,n),
             int(min(start+block_size,n)*fraction)-int(start*fraction))
            for start in range(0,n,block_size)]


def calibrate_entropy(bundle, args, device):
    """Label-free quantile calibration on validation features, never test data."""
    path = args.data_dir/"oversampled_validation_dataset.csv"
    frame = pd.read_csv(path)
    if list(frame.columns) != bundle["columns"] or len(frame) == 0:
        raise ValueError("Validation schema mismatch or empty validation set.")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    cnn, context = core.build_models(bundle["cnn_checkpoint"],bundle["context_checkpoint"],device)
    features = bundle["scaler"].transform(frame.iloc[:,:-1])
    values=[]
    with torch.no_grad():
        for start in range(0,len(features),32):
            x=torch.tensor(features[start:start+32],dtype=torch.float32,device=device).unsqueeze(1)
            probs=cnn(context(x)).softmax(1)
            entropy=-(probs*probs.clamp_min(1e-12).log()).sum(1)/max(np.log(bundle["num_classes"]),1e-12)
            values.extend(entropy.cpu().tolist())
    threshold=float(np.quantile(values,1-args.budget_fraction))
    return threshold, dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           samples=len(values),quantile=1-args.budget_fraction,
                           threshold=threshold,initial_scores=values,
                           protocol="frozen initial Context; validation features only")


class RollingEntropy:
    """Threshold uses only past observed scores; no test labels or future scores."""
    def __init__(self, initial_scores, window, fraction):
        self.history = deque(initial_scores, maxlen=window)
        self.fraction = fraction

    def decide(self, score):
        threshold = float(np.quantile(list(self.history), 1-self.fraction))
        preferred = score >= threshold
        self.history.append(float(score))
        return bool(preferred), threshold


def evaluate(frame, bundle, args, seed, method, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cnn, context = core.build_models(bundle["cnn_checkpoint"], bundle["context_checkpoint"], device)
    opts = SimpleNamespace(dqn_lr=args.dqn_lr, gamma=args.gamma,
                           batch_size=args.batch_size)
    size = bundle["cnn_checkpoint"]["input_length"] + 2*bundle["num_classes"] + 1
    agent = core.Agent(size, opts, device)
    agent.q.load_state_dict(bundle["router_state"])
    agent.target.load_state_dict(bundle["router_state"])
    agent.warmup = args.online_warmup
    optimizer = torch.optim.Adam(context.parameters(), lr=args.context_lr, weight_decay=1e-4)
    core.legacy.CONTEXT_LOSS_MODE = bundle["context_loss_mode"]
    x, labels = core.tensors(frame, bundle["scaler"], bundle["encoder"], bundle["columns"], device)
    n = len(labels)
    cap = int(n*args.budget_fraction)
    order = (np.random.default_rng(seed).permutation(n) if args.stream_order == "shuffle"
             else np.arange(n))
    # Independent RNGs prevent replay sampling from changing query exploration.
    rng = random.Random(seed + 1000003)
    schedule = block_schedule(n,args.budget_block_size,args.budget_fraction)
    block_used = [0]*len(schedule)
    random_rng = random.Random(seed + 2000003)
    random_positions = set()
    for start,end,quota in schedule:
        random_positions.update(random_rng.sample(range(start,end),quota))
    cnn_initial = {k:v.detach().cpu().clone() for k,v in cnn.state_dict().items()}
    router_initial = {k:v.detach().cpu().clone() for k,v in agent.q.state_dict().items()}
    rows, used, forced_count, losses = [], 0, 0, []
    entropy_selector = RollingEntropy(args.entropy_initial_scores,args.entropy_window,
                                      args.budget_fraction)
    for pos, idx in enumerate(order):
        block_id = pos//args.budget_block_size
        block_start,block_end,block_cap = schedule[block_id]
        sample = x[idx:idx+1]
        remaining = (cap-used)/max(cap,1)
        state, frozen, adapted = core.observe(sample, cnn, context, remaining)
        probabilities = adapted.softmax(1).squeeze(0)
        entropy = float(-(probabilities*probabilities.clamp_min(1e-12).log()).sum().item())
        entropy /= max(np.log(bundle["num_classes"]), 1e-12)
        with torch.no_grad():
            q = agent.q(state).squeeze(0)
            advantage = float((q[2]-q[:2].max()).item())
        threshold_used = args.entropy_threshold
        if method == "random":
            preferred = pos in random_positions
        elif method == "uncertainty":
            if args.uncertainty_mode == "rolling-quantile":
                preferred, threshold_used = entropy_selector.decide(entropy)
            else:
                preferred = entropy >= threshold_used
        elif rng.random() < args.eval_epsilon:
            preferred = rng.randrange(3) == 2
        else:
            preferred = advantage > 0
        paid, forced = query_gate(preferred, block_used[block_id], block_cap, block_end-pos)
        # Always score the Context route BEFORE any current-label feedback.
        prediction = int(adapted.argmax(1))
        before = after = None
        if paid and method == "adaptive_ddqn":
            if pos == n-1:
                pre_update_next, pre_update_allowed = state, False
            else:
                nxt = order[pos+1]
                pre_update_next = core.observe(x[nxt:nxt+1],cnn,context,remaining)[0]
                next_block = (pos+1)//args.budget_block_size
                pre_update_allowed = block_used[next_block] < schedule[next_block][2]
        if paid:
            used += 1
            block_used[block_id] += 1
            forced_count += int(forced)
            target = torch.tensor([int(labels[idx])], device=device)
            before = float(nn.functional.cross_entropy(adapted,target).item())
            optimizer.zero_grad()
            ce = nn.functional.cross_entropy(cnn(context(sample)),target)
            base_ce = nn.functional.cross_entropy(frozen,target)
            loss = core.legacy.context_objective(ce,base_ce,online=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                after = float(nn.functional.cross_entropy(cnn(context(sample)),target).item())
            if method == "adaptive_ddqn":
                done = pos == n-1
                if done:
                    ns, allowed = state, False
                else:
                    nxt = order[pos+1]
                    ns = core.observe(x[nxt:nxt+1],cnn,context,(cap-used)/max(cap,1))[0]
                    next_block = (pos+1)//args.budget_block_size
                    allowed = block_used[next_block] < schedule[next_block][2]
                fp, cp, truth = int(frozen.argmax(1)), prediction, int(labels[idx])
                # Preserve the existing policy reward; context routing is fixed only for scoring.
                # Unpaid experiences are never used for online learning.
                for candidate in (0,1):
                    reward = core.routing_reward(candidate,fp,cp,truth)
                    agent.remember(state,candidate,reward,pre_update_next,done,pre_update_allowed)
                reward = core.routing_reward(2,fp,cp,truth,before,after,
                                            bundle["selection_cost"],bundle["gain_weight"])
                agent.remember(state,2,reward,ns,done,allowed)
                value=agent.optimize()
                if value is not None: losses.append(value)
        rows.append(dict(step=pos+1,source_row=int(idx)+1,true_label=int(labels[idx]),
                         prediction_before_feedback=prediction,queried=paid,
                         forced_query=forced,preferred_query=preferred,labels_used=used,
                         budget_block=block_id,block_labels_used=block_used[block_id],
                         block_cap=block_cap,
                         normalized_entropy=entropy,entropy_threshold_used=threshold_used,
                         query_advantage=advantage,
                         loss_before=before,loss_after=after))
    assert used == cap
    assert block_used == [block[2] for block in schedule]
    assert all(torch.equal(cnn_initial[k],v.detach().cpu()) for k,v in cnn.state_dict().items())
    if method != "adaptive_ddqn":
        assert all(torch.equal(router_initial[k],v.detach().cpu()) for k,v in agent.q.state_dict().items())
    truth=[r["true_label"] for r in rows]
    pred=[r["prediction_before_feedback"] for r in rows]
    return dict(method=method,seed=seed,accuracy=float(accuracy_score(truth,pred)),
                macro_f1=float(f1_score(truth,pred,average="macro",zero_division=0)),
                weighted_f1=float(f1_score(truth,pred,average="weighted",zero_division=0)),
                labels_used=used,label_cap=cap,forced_queries=forced_count,
                dqn_updates=agent.steps,mean_dqn_loss=float(np.mean(losses)) if losses else None,
                prediction_route="Context -> frozen CNN",scoring="before feedback",
                oracle_context_routing=False,context_window=1,
                budget_protocol="exact label-blind block quotas; forced block completion",
                budget_block_size=args.budget_block_size,
                per_class_queried={str(c):sum(r["queried"] for r in rows if r["true_label"]==c)
                                   for c in range(bundle["num_classes"])},
                entropy_threshold=args.entropy_threshold,
                uncertainty_mode=args.uncertainty_mode,
                entropy_window=args.entropy_window),rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    p.add_argument("--checkpoint-output",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,default=root/"Dataset")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--per",type=int,nargs="+",choices=core.LEVELS,default=[0])
    p.add_argument("--methods",nargs="+",choices=["adaptive_ddqn","random","uncertainty","frozen_ddqn"],
                   default=["adaptive_ddqn","random","uncertainty","frozen_ddqn"])
    p.add_argument("--budget-fraction",type=float,default=.2)
    p.add_argument("--context-lr",type=float,default=.25)
    p.add_argument("--dqn-lr",type=float,default=.0001)
    p.add_argument("--gamma",type=float,default=.98)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--online-warmup",type=int,default=6)
    p.add_argument("--eval-epsilon",type=float,default=.1)
    p.add_argument("--entropy-threshold",type=float,default=None,
                   help="Explicit validation-tuned threshold; default calibrates validation entropy quantile.")
    p.add_argument("--budget-block-size",type=int,default=50,
                   help="Label-blind online allocation interval; Context persists across blocks.")
    p.add_argument("--uncertainty-mode",choices=["fixed","rolling-quantile"],default="rolling-quantile")
    p.add_argument("--entropy-window",type=int,default=200,
                   help="Number of past uncertainty scores; initialized from validation entropy.")
    p.add_argument("--stream-order",choices=["input","shuffle"],default="input")
    args=p.parse_args()
    if not all(np.isfinite(v) and 0<=v<=1 for v in
               [args.budget_fraction,args.eval_epsilon,args.gamma]):
        p.error("Fractions, epsilon, threshold and gamma must be finite in [0,1].")
    if any(not np.isfinite(v) or v<=0 for v in [args.context_lr,args.dqn_lr]):
        p.error("Learning rates must be positive and finite.")
    if args.entropy_threshold is not None and (not np.isfinite(args.entropy_threshold) or not 0<=args.entropy_threshold<=1):
        p.error("entropy-threshold must be finite in [0,1].")
    if args.entropy_window<1:p.error("entropy-window must be positive.")
    if args.budget_block_size<1:p.error("budget-block-size must be positive.")
    if args.online_warmup<1 or args.batch_size<1:p.error("Replay settings must be positive.")
    if args.output.exists():p.error("Choose a new output directory to preserve previous runs.")
    checkpoint=args.checkpoint_output/"Models"/"three_action_best.pth"
    bundle=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if bundle.get("version")!=core.VERSION:p.error("Three-action checkpoint required.")
    if bundle.get("training_args",{}).get("context_window",1)!=1:
        p.error("Use a single-sample Context checkpoint.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = {"source":"explicit threshold; user must tune on validation"}
    if args.entropy_threshold is None or args.uncertainty_mode == "rolling-quantile":
        calibrated_threshold, calibration = calibrate_entropy(bundle,args,device)
        if args.entropy_threshold is None:
            args.entropy_threshold = calibrated_threshold
    args.entropy_initial_scores = calibration.pop("initial_scores", [args.entropy_threshold])
    calibration["initial_scores_count"] = len(args.entropy_initial_scores)
    calibration["initial_scores_sha256"] = hashlib.sha256(
        json.dumps(args.entropy_initial_scores).encode()).hexdigest()
    print("Entropy calibration:",json.dumps(calibration),flush=True)
    args.output.mkdir(parents=True)
    manifest=dict({k:v for k,v in vars(args).items() if k != "entropy_initial_scores"},
                  entropy_calibration=calibration,protocol_version="selection_blocks_v3",checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  torch_version=torch.__version__,numpy_version=np.__version__,
                  device=str(device),training_seed=bundle.get("training_args",{}).get("seed"),
                  note="Seeds repeat online evaluation, not checkpoint training. Selection-only diagnostic; no oracle routing.")
    (args.output/"manifest.json").write_text(json.dumps(manifest,default=str,indent=2))
    summaries=[]
    for level in args.per:
        path=args.data_dir/"AdvBurst_FTSC-IAT"/f"AdvBurstTrainedBurst_size_{level}_Step_2000.csv"
        frame=pd.read_csv(path).iloc[:1073]
        for seed in args.seeds:
            for method in args.methods:
                summary,rows=evaluate(frame,bundle,args,seed,method,device)
                summary.update(perturbation_level=level,dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                summaries.append(summary)
                pd.DataFrame(rows).to_csv(args.output/f"PER_{level}_seed{seed}_{method}_trace.csv",index=False)
                (args.output/"summary.json").write_text(json.dumps(summaries,indent=2))
                pd.DataFrame(summaries).to_csv(args.output/"combined_results.csv",index=False)
                print(json.dumps(summary),flush=True)
    table=pd.DataFrame(summaries)
    table.groupby(["perturbation_level","method"])[["accuracy","macro_f1","labels_used","forced_queries"]].agg(
        ["mean","std","count"]).to_csv(args.output/"aggregate.csv")
    print("Results:",args.output)


if __name__=="__main__":
    main()
