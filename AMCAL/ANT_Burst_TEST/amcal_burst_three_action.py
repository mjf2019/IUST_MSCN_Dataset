"""Separate three-action routing ablation; never alters legacy checkpoints.

0: Frozen, no query. 1: Context -> Frozen, no query.
2: Context -> Frozen, paid feedback and Context update.
All reported stream predictions precede feedback. Class quota metadata is an
offline oracle controller, not an input feature or a deployment claim.
"""
from pathlib import Path
from collections import deque, Counter
import argparse
import copy
import csv
import json
import random
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import amcal_burst_legacy as legacy
from amcal_burst_stream import fractional_class_budget, stream_permutation
from amcal_matched_budget import matched_query_positions
from amcal_three_action_reward import routing_reward, prediction_route, available_actions

VERSION = "three_action_routing_v1"
LEVELS = (0, 1, 3, 5, 7, 10, 12, 15, 17, 20)


class Router(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.network(x)


class Agent:
    def __init__(self, size, args, device):
        self.q = Router(size).to(device)
        self.target = copy.deepcopy(self.q).eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=args.dqn_lr)
        self.buffer = deque(maxlen=5000)
        self.device, self.gamma, self.batch = device, args.gamma, args.batch_size
        self.steps = 0
        self.warmup = self.batch

    def action(self, state, allowed, epsilon):
        if random.random() < epsilon:
            return random.choice(available_actions(allowed))
        with torch.no_grad():
            values = self.q(state).squeeze(0).clone()
            if not allowed:
                values[2] = -torch.inf
            return int(values.argmax().item())

    def remember(self, state, action, reward, next_state, done, next_allowed):
        self.buffer.append((state.detach().cpu(), action, reward,
                            next_state.detach().cpu(), done, next_allowed))

    def optimize(self):
        if len(self.buffer) < self.warmup:
            return None
        entries = random.sample(self.buffer, min(self.batch, len(self.buffer)))
        s, a, r, ns, done, allowed = zip(*entries)
        s, ns = torch.cat(s).to(self.device), torch.cat(ns).to(self.device)
        actions = torch.tensor(a, device=self.device).unsqueeze(1)
        with torch.no_grad():
            online_next = self.q(ns)
            online_next[~torch.tensor(allowed, dtype=torch.bool, device=self.device), 2] = -torch.inf
            next_action = online_next.argmax(1, keepdim=True)
            future = self.target(ns).gather(1, next_action).squeeze(1)
            target = torch.tensor(r, device=self.device) + self.gamma * future * (
                1 - torch.tensor(done, dtype=torch.float32, device=self.device))
        loss = nn.functional.smooth_l1_loss(self.q(s).gather(1, actions).squeeze(1), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())


def observe(x, cnn, context, remaining):
    with torch.no_grad():
        frozen_logits, context_logits = cnn(x), cnn(context(x))
        state = torch.cat((x.squeeze(1), frozen_logits.softmax(1),
                           context_logits.softmax(1),
                           x.new_tensor([[remaining]])), dim=1)
    return state, frozen_logits, context_logits


def build_models(cnn_checkpoint, context_checkpoint, device):
    architecture = context_checkpoint.get("context_architecture", "legacy_single_token")
    if cnn_checkpoint.get("context_architecture", "legacy_single_token") != architecture:
        raise ValueError("Mixed source checkpoint architectures.")
    legacy.CONTEXT_ARCHITECTURE = architecture
    cnn = legacy.CNN1D(input_length=cnn_checkpoint["input_length"],
                       num_classes=cnn_checkpoint["num_classes"]).to(device)
    context = legacy.ContextAwareNetwork(input_size=cnn_checkpoint["input_length"], num_layers=4).to(device)
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    context.load_state_dict(context_checkpoint["context_model_state_dict"])
    cnn.eval().requires_grad_(False)
    context.eval()
    return cnn, context


def tensors(frame, scaler, encoder, columns, device):
    if list(frame.columns) != columns:
        raise ValueError("Feature schema differs from training.")
    x = torch.tensor(scaler.transform(frame.iloc[:, :-1]), dtype=torch.float32, device=device).unsqueeze(1)
    y = encoder.transform(frame.iloc[:, -1])
    return x, y


def run_stream(frame, args, bundle, cnn, context, agent, training=False):
    device = next(cnn.parameters()).device
    agent.warmup = args.batch_size if training else args.online_warmup
    steps_before = agent.steps
    first_update_at_labels = None
    x, labels = tensors(frame, bundle["scaler"], bundle["encoder"], bundle["columns"], device)
    quotas = fractional_class_budget(labels, bundle["num_classes"], args.budget_fraction)
    matched_caps = getattr(args, "matched_class_caps", None)
    if matched_caps is not None:
        matched_caps = {int(k): v for k,v in matched_caps.items()}
        if set(matched_caps) != set(quotas) or any(
                type(n) is not int or n < 0 or n > quotas[c] for c,n in matched_caps.items()):
            raise ValueError("Reference counts exceed this run's class budget or differ in classes.")
        quotas = matched_caps
    used = {c: 0 for c in quotas}
    total_cap = sum(quotas.values())
    optimizer = torch.optim.Adam(context.parameters(), lr=args.context_lr, weight_decay=1e-4)
    legacy.CONTEXT_LOSS_MODE = bundle["context_loss_mode"]
    order = stream_permutation(labels, args.stream_order, args.seed)
    query_positions = (matched_query_positions(labels, quotas, args.seed)
                       if matched_caps is not None else None)
    # Independent routing RNG: DDQN replay sampling cannot change random routes.
    routing_rng = random.Random(args.seed + 1000003)
    rows, losses = [], []
    epsilon = args.epsilon_start if training else args.eval_epsilon
    initial_cnn = {k: v.detach().cpu().clone() for k,v in cnn.state_dict().items()}
    for pos, index in enumerate(order):
        sample = x[index:index+1]
        class_id = int(labels[index])  # quota controller only; never exposed to policy
        allowed = used[class_id] < quotas[class_id]
        remaining = (total_cap - sum(used.values())) / max(total_cap, 1)
        state, frozen, adapted = observe(sample, cnn, context, remaining)
        frozen_pred, context_pred = int(frozen.argmax(1)), int(adapted.argmax(1))
        if query_positions is None:
            action = agent.action(state, allowed, epsilon)
        else:
            action = 2 if index in query_positions else routing_rng.choice((0, 1))
        prediction = frozen_pred if action == 0 else context_pred
        paid = action == 2
        before = after = None
        reward = None
        done = pos == len(order)-1
        # Counterfactual non-query actions preserve both Context and budget.
        if training or paid:
            if done:
                no_query_next, no_query_allowed = state, False
            else:
                next_index = order[pos+1]
                next_class = int(labels[next_index])
                no_query_allowed = used[next_class] < quotas[next_class]
                no_query_next = observe(x[next_index:next_index+1], cnn, context, remaining)[0]
        # Only this branch reveals a test label to the learner.
        if paid:
            used[class_id] += 1
            target = torch.tensor([class_id], device=device)
            before = float(nn.functional.cross_entropy(adapted, target).item())
            optimizer.zero_grad()
            ce = nn.functional.cross_entropy(cnn(context(sample)), target)
            base_ce = nn.functional.cross_entropy(frozen, target)
            loss = legacy.context_objective(ce, base_ce, online=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                after = float(nn.functional.cross_entropy(cnn(context(sample)), target).item())
            reward = routing_reward(2, frozen_pred, context_pred, class_id, before, after,
                                    args.selection_cost, args.gain_weight)
        if training or paid:
            if done:
                next_state, next_allowed = state, False
            else:
                next_index = order[pos+1]
                next_class = int(labels[next_index])  # controller mask, not a state feature
                next_allowed = used[next_class] < quotas[next_class]
                next_state = observe(x[next_index:next_index+1], cnn, context,
                                     (total_cap-sum(used.values()))/max(total_cap,1))[0]
            # In offline training all labels are available. At test these
            # counterfactual route rewards are computed ONLY for paid labels.
            for candidate in (0, 1):
                route_reward = routing_reward(candidate, frozen_pred, context_pred, class_id)
                agent.remember(state, candidate, route_reward, no_query_next, done, no_query_allowed)
            if paid:
                agent.remember(state, 2, reward, next_state, done, next_allowed)
            value = agent.optimize()
            if value is not None:
                if first_update_at_labels is None:
                    first_update_at_labels = sum(used.values())
                losses.append(value)
                if training:
                    epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        # Observer labels below never drive routing, Context loss or replay.
        rows.append(dict(step=pos+1, source_row=index+1, true_label=class_id,
                         action=action, route=prediction_route(action), queried=paid,
                         frozen_prediction=frozen_pred, context_prediction=context_pred,
                         prediction_before_feedback=prediction, labels_used=sum(used.values()),
                         class_labels_used=used[class_id], class_budget=quotas[class_id],
                         reward=reward, loss_before=before, loss_after=after))
    assert all(used[c] <= quotas[c] for c in quotas)
    if matched_caps is not None:
        assert used == quotas, "Matched random baseline did not consume exact reference counts."
    assert all(torch.equal(initial_cnn[k],v.detach().cpu()) for k,v in cnn.state_dict().items())
    truth = [r["true_label"] for r in rows]
    pred = [r["prediction_before_feedback"] for r in rows]
    summary = dict(accuracy=float(accuracy_score(truth,pred)),
                   weighted_f1=float(f1_score(truth,pred,average="weighted",zero_division=0)),
                   macro_f1=float(f1_score(truth,pred,average="macro",zero_division=0)),
                   labels_used=sum(used.values()), label_cap=total_cap,
                   per_class_used=used, per_class_caps=quotas,
                   actions=dict(Counter(r["action"] for r in rows)),
                   dqn_loss=float(np.mean(losses)) if losses else None,
                   dqn_updates=agent.steps-steps_before,
                   first_dqn_update_at_labels=first_update_at_labels,
                   replay_warmup=agent.warmup,
                   classifier_unchanged=True, scoring="before feedback",
                   budget_protocol="offline true-class quota controller",
                   reward_version=VERSION,
                   query_protocol=("uniform random positions with exact reference class counts"
                                   if matched_caps is not None else "epsilon-greedy"))
    return summary, rows


def validation(frame, bundle, cnn, context, agent, device, args):
    """Budgeted online validation, isolated from training and checkpoint state."""
    previous_loss_mode = legacy.CONTEXT_LOSS_MODE
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        validation_cnn = copy.deepcopy(cnn)
        validation_context = copy.deepcopy(context)
        validation_agent = Agent(agent.q.network[0].in_features, args, device)
        validation_agent.q.load_state_dict(agent.q.state_dict())
        validation_agent.target.load_state_dict(agent.q.state_dict())
        # Empty replay and fresh optimizers, exactly as at evaluation start.
        summary, _ = run_stream(frame, args, bundle, validation_cnn,
                                validation_context, validation_agent, training=False)
        summary["protocol"] = "online budgeted; predictions before feedback; isolated copies"
        return summary
    finally:
        legacy.CONTEXT_LOSS_MODE = previous_loss_mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def early_stopping_step(score, best, stale):
    """Strict validation improvement resets the patience counter."""
    if score > best:
        return score, 0, True
    return best, stale + 1, False


def main():
    root = Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("train","evaluate"))
    p.add_argument("--source-output",type=Path,help="Existing two-action CNN/Context run; training warm start only.")
    p.add_argument("--output",type=Path,required=True,help="Separate three-action run directory.")
    p.add_argument("--data-dir",type=Path,default=root/"Dataset")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=15,
                   help="Stop after this many consecutive epochs without a new best online validation accuracy.")
    p.add_argument("--selection-cost",type=float,default=None)
    p.add_argument("--gain-weight",type=float,default=None)
    p.add_argument("--gamma",type=float,default=0.98)
    p.add_argument("--context-lr",type=float,default=0.25)
    p.add_argument("--dqn-lr",type=float,default=0.0001)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--online-warmup",type=int,default=6,
                   help="Minimum replay experiences before online DDQN updates; default 6 = two paid labels.")
    p.add_argument("--budget-fraction",type=float,default=0.2)
    p.add_argument("--epsilon-start",type=float,default=0.3)
    p.add_argument("--epsilon-end",type=float,default=0.05)
    p.add_argument("--epsilon-decay",type=float,default=0.995)
    p.add_argument("--eval-epsilon",type=float,default=0.0)
    p.add_argument("--stream-order",choices=("input","class-interleaved"),default="input")
    p.add_argument("--per",type=int,choices=LEVELS)
    p.add_argument("--match-budget-from", type=Path,
                   help="RL summary.json; random routes and uniformly sampled queries matching each class count.")
    args=p.parse_args()
    if args.match_budget_from is not None and (args.command != "evaluate" or args.eval_epsilon != 1):
        p.error("--match-budget-from requires evaluate --eval-epsilon 1.")
    if not 0<=args.epsilon_end<=args.epsilon_start<=1 or not 0<=args.eval_epsilon<=1 or not 0<args.epsilon_decay<=1:
        p.error("Invalid exploration parameters.")
    if args.patience < 1:
        p.error("patience must be positive.")
    if args.online_warmup < 1:
        p.error("online-warmup must be positive.")
    if not 0<=args.budget_fraction<=1 or not 0<=args.gamma<=1 or args.batch_size<1 or args.epochs<1:
        p.error("Invalid budget, gamma, batch size or epochs.")
    if any(not np.isfinite(v) or v<=0 for v in (args.context_lr,args.dqn_lr)):
        p.error("Learning rates must be finite and positive.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path=args.output/"Models"/"three_action_best.pth"
    if args.command=="train":
        if checkpoint_path.exists(): p.error("Three-action checkpoint exists; choose a fresh output.")
        if args.source_output is None: p.error("Training requires --source-output.")
        if args.source_output.resolve()==args.output.resolve(): p.error("Use a separate output.")
        cnn_ck=torch.load(args.source_output/"Models"/"initial_cnn_model.pth",map_location="cpu",weights_only=False)
        context_ck=torch.load(args.source_output/"Models"/"adversarial_best_context.pth",map_location="cpu",weights_only=False)
        train=pd.read_csv(args.data_dir/"oversampled_train_dataset.csv")
        val=pd.read_csv(args.data_dir/"oversampled_validation_dataset.csv")
        bundle=dict(version=VERSION,cnn_checkpoint=cnn_ck,context_checkpoint=context_ck,
                    scaler=StandardScaler().fit(train.iloc[:,:-1]),encoder=LabelEncoder().fit(train.iloc[:,-1]),
                    columns=list(train.columns),num_classes=cnn_ck["num_classes"],
                    context_loss_mode=context_ck.get("context_loss_mode","legacy"))
        if train.shape[1]-1 != cnn_ck["input_length"] or len(bundle["encoder"].classes_) != bundle["num_classes"]:
            p.error("Training data and checkpoint dimensions differ.")
        args.selection_cost=0.2 if args.selection_cost is None else args.selection_cost
        args.gain_weight=1.0 if args.gain_weight is None else args.gain_weight
    else:
        bundle=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        if bundle.get("version")!=VERSION: p.error("A trained three-action checkpoint is required.")
        for key in ("selection_cost","gain_weight"):
            if getattr(args,key) is not None and getattr(args,key)!=bundle[key]:
                p.error(f"{key} differs from training; retrain for a new reward.")
            setattr(args,key,bundle[key])
    if any(not np.isfinite(v) or v<0 for v in (args.selection_cost,args.gain_weight)):
        p.error("Reward cost and gain weight must be finite and nonnegative.")
    print("Three-action routing; device:",device,"cost:",args.selection_cost,"lambda:",args.gain_weight)
    size=bundle["cnn_checkpoint"]["input_length"]+2*bundle["num_classes"]+1
    if args.command=="train":
        agent=Agent(size,args,device);best=-1.;history=[];stale=0;best_epoch=None
        for epoch in range(args.epochs):
            # Reset Context at episode boundary; the three-action policy persists.
            cnn,context=build_models(bundle["cnn_checkpoint"],bundle["context_checkpoint"],device)
            summary,_=run_stream(train,args,bundle,cnn,context,agent,training=True)
            validation_summary=validation(val,bundle,cnn,context,agent,device,args)
            score=validation_summary["accuracy"]
            best, stale, improved = early_stopping_step(score, best, stale)
            summary.update(epoch=epoch+1,validation_routing_accuracy=score,
                           validation_online=validation_summary,
                           epochs_without_improvement=stale);history.append(summary)
            print(json.dumps(summary))
            if improved:
                best_epoch=epoch+1
                saved=copy.deepcopy(bundle)
                saved["context_checkpoint"]["context_model_state_dict"]={k:v.detach().cpu().clone() for k,v in context.state_dict().items()}
                saved.update(router_state={k:v.detach().cpu().clone() for k,v in agent.q.state_dict().items()},
                             selection_cost=args.selection_cost,gain_weight=args.gain_weight,
                             training_args=vars(args),best_epoch=epoch+1,
                             validation_protocol="online_budgeted_before_feedback_v2",
                             checkpoint_validation=validation_summary)
                checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(saved,checkpoint_path)
            (args.output/"training_history.json").write_text(json.dumps(history,indent=2))
            print(f"Best online validation: {best:.6f} (epoch {best_epoch}); patience {stale}/{args.patience}")
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch+1}.")
                break
        (args.output/"training_status.json").write_text(json.dumps(dict(
            completed_epochs=len(history), best_epoch=best_epoch,
            best_online_validation_accuracy=best, patience=args.patience,
            stopped_early=stale >= args.patience),indent=2))
        (args.output/"training_history.json").write_text(json.dumps(history,indent=2))
        print("Saved:",checkpoint_path)
    else:
        tag=f"three_action_seed{args.seed}_budget{args.budget_fraction:g}_lr{args.context_lr:g}_eps{args.eval_epsilon:g}_{args.stream_order}_warmup{args.online_warmup}"
        references = {}
        if args.match_budget_from is not None:
            raw = args.match_budget_from.read_bytes()
            entries = json.loads(raw)
            if not isinstance(entries, list):
                p.error("Reference summary must be a list of per-level results.")
            for entry in entries:
                if entry.get("reward_version") != VERSION:
                    p.error("Reference reward version differs.")
                level = int(entry["perturbation_level"])
                if level in references:
                    p.error("Duplicate perturbation level in reference.")
                if sum(entry["per_class_used"].values()) != entry["labels_used"]:
                    p.error("Invalid reference label counts.")
                references[level] = entry
            tag += "_matched_random_uniform_" + hashlib.sha256(raw).hexdigest()[:8]
        results=args.output/"Results"/tag;results.mkdir(parents=True,exist_ok=True)
        summaries=[]
        for level in ([args.per] if args.per is not None else LEVELS):
            # Each level starts from the identical CNN/Context/router snapshot.
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            cnn,context=build_models(bundle["cnn_checkpoint"],bundle["context_checkpoint"],device)
            agent=Agent(size,args,device);agent.q.load_state_dict(bundle["router_state"]);agent.target.load_state_dict(bundle["router_state"])
            frame=pd.read_csv(args.data_dir/"AdvBurst_FTSC-IAT"/f"AdvBurstTrainedBurst_size_{level}_Step_2000.csv").iloc[:1073]
            if args.match_budget_from is not None:
                if level not in references:
                    p.error(f"Reference lacks perturbation level {level}.")
                args.matched_class_caps = references[level]["per_class_used"]
            summary,rows=run_stream(frame,args,bundle,cnn,context,agent)
            if args.match_budget_from is not None:
                summary["reference_rl_accuracy"] = references[level]["accuracy"]
                summary["reference_rl_labels_used"] = references[level]["labels_used"]
                summary["accuracy_delta_vs_rl"] = summary["accuracy"] - references[level]["accuracy"]
            summary["perturbation_level"]=level;summaries.append(summary)
            pd.DataFrame(rows).to_csv(results/f"PER_{level}_trace.csv",index=False)
            print(json.dumps(summary))
        (results/"summary.json").write_text(json.dumps(summaries,indent=2))
        pd.DataFrame(summaries).to_csv(results/"combined_results.csv",index=False)
        (results/"manifest.json").write_text(json.dumps(vars(args),default=str,indent=2))
        print("Results:",results)


if __name__=="__main__":
    main()
