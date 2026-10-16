"""Sparse learned Context bank: action 0 is frozen CNN, 1..K are unpaid
Context routes, K+1..2K query and update the selected Context.
Routing state uses frozen CNN probabilities and paid history; only the selected
Context is evaluated. No class or flow identifier drives routing or budget.
Requires a new checkpoint; existing three-action checkpoints remain untouched.
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
from amcal_burst_stream import stream_permutation
from amcal_three_action_reward import routing_reward

VERSION = "sparse_learned_context_bank_v1"
LEVELS = (0, 1, 3, 5, 7, 10, 12, 15, 17, 20)


class Router(nn.Module):
    def __init__(self, size, count):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1 + 2 * count))

    def forward(self, x):
        return self.network(x)


class Agent:
    def __init__(self, size, args, device):
        self.routing_rng = random.Random(args.seed + 1000003)
        self.count = args.num_contexts
        self.q = Router(size, self.count).to(device)
        self.target = copy.deepcopy(self.q).eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=args.dqn_lr)
        self.buffer = deque(maxlen=5000)
        self.device, self.gamma, self.batch = device, args.gamma, args.batch_size
        self.steps = 0
        self.warmup = self.batch

    def action(self, state, allowed, epsilon):
        if self.routing_rng.random() < epsilon:
            return self.routing_rng.choice(list(range(1 + 2*self.count)) if allowed else list(range(1+self.count)))
        with torch.no_grad():
            values = self.q(state).squeeze(0).clone()
            if not allowed:
                values[1+self.count:] = -torch.inf
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
            online_next[~torch.tensor(allowed, dtype=torch.bool, device=self.device), 1+self.count:] = -torch.inf
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


class LearnedContextBank(nn.Module):
    def __init__(self, template, count):
        super().__init__()
        self.models = nn.ModuleList([copy.deepcopy(template) for _ in range(count)])


def observe(x, cnn, stats, remaining, position, total):
    """No Context forward and no current label is used to build the routing state."""
    history = []
    for count, last, gain, correct in stats:
        history.extend((count/max(total,1), (position+1-last)/max(total,1), gain, correct))
    with torch.no_grad():
        frozen = cnn(x)
        state = torch.cat([x.squeeze(1), frozen.softmax(1),
                           x.new_tensor([history]), x.new_tensor([[remaining]])], dim=1)
    return state, frozen


def build_models(cnn_checkpoint, context_checkpoint, device, count, bank_states=None):
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
    context = LearnedContextBank(context, count).to(device).eval()
    if bank_states is not None:
        context.load_state_dict(bank_states)
    return cnn, context


def tensors(frame, scaler, encoder, columns, device):
    if list(frame.columns) != columns:
        raise ValueError("Feature schema differs from training.")
    x = torch.tensor(scaler.transform(frame.iloc[:, :-1]), dtype=torch.float32, device=device).unsqueeze(1)
    y = encoder.transform(frame.iloc[:, -1])
    return x, y


def diagnostic_accuracy(x, labels, cnn, model=None):
    """Observer-only accuracy, in small batches; never updates model state."""
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), 128):
            batch = x[start:start+128]
            logits = cnn(batch if model is None else model(batch))
            pred = logits.argmax(1).detach().cpu().numpy()
            correct += int((pred == labels[start:start+128]).sum())
    return correct / max(len(labels), 1)


def run_stream(frame, args, bundle, cnn, context, agent, training=False,
               context_optimizers=None, episode_epsilon=None):
    device = next(cnn.parameters()).device
    agent.warmup = args.batch_size if training else args.online_warmup
    learn_ddqn = training or args.policy_mode == 'adaptive'
    steps_before = agent.steps
    x, labels = tensors(frame, bundle['scaler'], bundle['encoder'], bundle['columns'], device)
    total = len(labels)
    diagnostics = getattr(args, 'context_diagnostics', False)
    if diagnostics and training:
        raise ValueError("Context diagnostics are evaluation-only.")
    initial_diagnostics = None
    if diagnostics:
        # The source Context before bank training, and each saved slot before
        # online adaptation, are assessed independently of the RL policy.
        template = copy.deepcopy(context.models[0])
        template.load_state_dict(bundle['context_checkpoint']['context_model_state_dict'])
        initial_diagnostics = dict(
            frozen_accuracy=diagnostic_accuracy(x, labels, cnn),
            source_context_accuracy=diagnostic_accuracy(x, labels, cnn, template),
            saved_context_accuracies=[diagnostic_accuracy(x, labels, cnn, model)
                                      for model in context.models])
        del template
    cap = int(total * args.budget_fraction)
    used = Counter({c: 0 for c in range(bundle['num_classes'])})
    if context_optimizers is not None and not training:
        raise ValueError("Validation/evaluation must use fresh Context optimizers.")
    optimizers = context_optimizers
    if optimizers is None:
        optimizers = [torch.optim.Adam(m.parameters(), lr=args.context_lr, weight_decay=1e-4)
                      for m in context.models]
    if len(optimizers) != len(context.models):
        raise ValueError("One optimizer is required per Context.")
    windows = [deque(maxlen=args.context_window) for _ in context.models]
    # Count, last paid update position, EMA of gain, EMA of pre-feedback correctness.
    stats = [[0, 0, 0.0, 0.0] for _ in context.models]
    legacy.CONTEXT_LOSS_MODE = bundle['context_loss_mode']
    order = (np.random.default_rng(args.seed).permutation(total) if args.stream_order == 'shuffle'
             else stream_permutation(labels, args.stream_order, args.seed))
    epsilon = (args.epsilon_start if episode_epsilon is None else episode_epsilon) if training else args.eval_epsilon
    rows, losses = [], []
    first_update = None
    initial_cnn = {k: v.detach().cpu().clone() for k,v in cnn.state_dict().items()}
    for pos, index in enumerate(order):
        sample = x[index:index+1]
        allowed = sum(used.values()) < cap
        remaining = (cap-sum(used.values())) / max(cap, 1)
        state, frozen = observe(sample, cnn, stats, remaining, pos, total)
        frozen_pred = int(frozen.argmax(1))
        action = agent.action(state, allowed, epsilon)
        preferred = action
        forced = False
        if args.consume_full_budget and action <= args.num_contexts and allowed and cap-sum(used.values()) >= total-pos:
            if action == 0:
                with torch.no_grad():
                    action = 1+args.num_contexts+int(agent.q(state)[0, 1+args.num_contexts:].argmax())
            else:
                action += args.num_contexts
            forced = True
        slot = None if action == 0 else (action-1) % args.num_contexts
        paid = action > args.num_contexts
        adapted = None
        context_pred = None
        if slot is not None:
            with torch.no_grad():
                adapted = cnn(context.models[slot](sample))
            context_pred = int(adapted.argmax(1))
        prediction = frozen_pred if slot is None else context_pred
        # Observer label is not used to form state or choose the route.
        class_id = int(labels[index])
        done = pos == total-1
        before = after = reward = None
        post_prediction = next_pre_prediction = next_post_prediction = next_truth = None
        no_query_next = state
        no_query_allowed = False
        if learn_ddqn and (training or paid) and not done:
            nxt = order[pos+1]
            no_query_next = observe(x[nxt:nxt+1], cnn, stats, remaining, pos+1, total)[0]
            no_query_allowed = allowed
        if paid:
            used[class_id] += 1
            model, optimizer, window = context.models[slot], optimizers[slot], windows[slot]
            target = torch.tensor([class_id], device=device)
            before = float(nn.functional.cross_entropy(adapted, target).item())
            window.append((sample.detach().clone(), class_id))
            wx = torch.cat([item[0] for item in window])
            wy = torch.tensor([item[1] for item in window], device=device)
            if diagnostics and not done:
                probe_index = order[pos+1]
                probe = x[probe_index:probe_index+1]
                with torch.no_grad():
                    next_pre_prediction = int(cnn(model(probe)).argmax(1))
                next_truth = int(labels[probe_index])  # observer only
            optimizer.zero_grad()
            ce = nn.functional.cross_entropy(cnn(model(wx)), wy)
            with torch.no_grad():
                base_ce = nn.functional.cross_entropy(cnn(wx), wy)
            loss = legacy.context_objective(ce, base_ce, online=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                post_logits = cnn(model(sample))
                after = float(nn.functional.cross_entropy(post_logits, target).item())
                if diagnostics:
                    post_prediction = int(post_logits.argmax(1))
                    if not done:
                        next_post_prediction = int(cnn(model(probe)).argmax(1))
            reward = routing_reward(2, frozen_pred, context_pred, class_id, before, after,
                                    args.selection_cost, args.gain_weight)
            gain = (before-after)/(before+after+1e-8)
            old_count, _, old_gain, old_correct = stats[slot]
            alpha = 1.0 if old_count == 0 else 0.1
            stats[slot] = [old_count+1, pos+1,
                           (1-alpha)*old_gain+alpha*gain,
                           (1-alpha)*old_correct+alpha*float(context_pred == class_id)]
        if learn_ddqn and (training or paid):
            next_allowed = sum(used.values()) < cap and not done
            next_state = (state if done else observe(x[order[pos+1]:order[pos+1]+1], cnn, stats,
                          (cap-sum(used.values()))/max(cap,1), pos+1,total)[0])
            # Only the executed Context is evaluated. No counterfactual forwards.
            agent.remember(state, 0, routing_reward(0, frozen_pred, frozen_pred, class_id),
                           no_query_next, done, no_query_allowed)
            if slot is not None:
                agent.remember(state, slot+1, routing_reward(1, frozen_pred, context_pred, class_id),
                               no_query_next, done, no_query_allowed)
            if paid:
                agent.remember(state, action, reward, next_state, done, next_allowed)
            value = agent.optimize()
            if value is not None:
                if first_update is None:
                    first_update = sum(used.values())
                losses.append(value)
        rows.append(dict(step=pos+1,source_row=int(index)+1,true_label=class_id,
                         action=action,preferred_action=preferred,forced_query=forced,
                         context_key=slot,queried=paid,frozen_prediction=frozen_pred,
                         context_prediction=context_pred,prediction_before_feedback=prediction,
                         labels_used=sum(used.values()),reward=reward,loss_before=before,loss_after=after,
                         diagnostic_current_post_prediction=post_prediction,
                         diagnostic_next_before=next_pre_prediction,
                         diagnostic_next_after=next_post_prediction,
                         diagnostic_next_truth=next_truth))
    assert sum(used.values()) <= cap
    if args.consume_full_budget:
        assert sum(used.values()) == cap
    if not learn_ddqn:
        assert agent.steps == steps_before
    assert all(torch.equal(initial_cnn[k],v.detach().cpu()) for k,v in cnn.state_dict().items())
    truth = [r['true_label'] for r in rows]
    pred = [r['prediction_before_feedback'] for r in rows]
    # Observer-only diagnostics: no new Context forwards or routing inputs.
    route_diagnostics = {}
    for key in [None] + list(range(args.num_contexts)):
        subset = [r for r in rows if r['context_key'] == key]
        name = 'direct_cnn' if key is None else f'context_{key}'
        n = len(subset)
        route_diagnostics[name] = dict(
            samples=n,
            accuracy=sum(r['prediction_before_feedback'] == r['true_label'] for r in subset)/n if n else None,
            frozen_accuracy_same_samples=sum(r['frozen_prediction'] == r['true_label'] for r in subset)/n if n else None,
            queries=sum(r['queried'] for r in subset))
    frozen_accuracy = sum(r['frozen_prediction'] == r['true_label'] for r in rows)/max(len(rows),1)
    summary = dict(frozen_cnn_accuracy=frozen_accuracy,
                   accuracy_delta_vs_frozen=float(accuracy_score(truth,pred))-frozen_accuracy,
                   selected_route_diagnostics=route_diagnostics,
                   accuracy=float(accuracy_score(truth,pred)),
                   weighted_f1=float(f1_score(truth,pred,average='weighted',zero_division=0)),
                   macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),
                   labels_used=sum(used.values()),label_cap=cap,per_class_used=dict(used),per_class_caps=None,
                   actions=dict(Counter(r['action'] for r in rows)),
                   dqn_loss=float(np.mean(losses)) if losses else None,dqn_updates=agent.steps-steps_before,
                   first_dqn_update_at_labels=first_update,replay_warmup=agent.warmup,
                   context_window=args.context_window,context_scope='sparse_learned_context_bank',
                   context_instances=args.num_contexts,oracle_context_routing=False,
                   context_route_counts=dict(Counter(r['context_key'] for r in rows if r['context_key'] is not None)),
                   context_query_counts=dict(Counter(r['context_key'] for r in rows if r['queried'])),
                   context_statistics=stats,policy_mode=args.policy_mode,
                   consume_full_budget=args.consume_full_budget,
                   forced_queries=sum(r['forced_query'] for r in rows),classifier_unchanged=True,
                   scoring='before feedback',budget_protocol='global label budget; no class quotas',
                   state_protocol='features; frozen CNN probabilities; per-slot paid history; remaining budget',
                   context_forward_protocol='selected slot only; additional forwards for paid update',
                   reward_version=VERSION,query_protocol='epsilon-greedy',final_epsilon=epsilon)
    if diagnostics:
        paid_rows = [r for r in rows if r['queried']]
        probes = [r for r in paid_rows if r['diagnostic_next_truth'] is not None]
        def mean_correct(items, prediction_key, truth_key):
            return (sum(r[prediction_key] == r[truth_key] for r in items)/len(items)
                    if items else None)
        summary['context_diagnostics'] = dict(
            initial=initial_diagnostics,
            paid_samples=len(paid_rows),
            paid_accuracy_before=mean_correct(paid_rows, 'context_prediction', 'true_label'),
            paid_accuracy_after=mean_correct(paid_rows, 'diagnostic_current_post_prediction', 'true_label'),
            next_sample_probes=len(probes),
            next_accuracy_before=mean_correct(probes, 'diagnostic_next_before', 'diagnostic_next_truth'),
            next_accuracy_after=mean_correct(probes, 'diagnostic_next_after', 'diagnostic_next_truth'),
            next_fixed=sum(r['diagnostic_next_before'] != r['diagnostic_next_truth'] and
                           r['diagnostic_next_after'] == r['diagnostic_next_truth'] for r in probes),
            next_broken=sum(r['diagnostic_next_before'] == r['diagnostic_next_truth'] and
                            r['diagnostic_next_after'] != r['diagnostic_next_truth'] for r in probes),
            protocol='observer only; next sample uses same selected Context before/after update; not necessarily its eventual RL route')
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
        summary["protocol"] = ("online unlimited" if getattr(args, "no_label_quota", False) else "online budgeted") + "; predictions before feedback; isolated copies"
        return summary
    finally:
        legacy.CONTEXT_LOSS_MODE = previous_loss_mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def prepare_training_epoch(agent, args, epoch):
    """Discard stale replay, retain learned Q/target weights and optimizer state.

    Epsilon is fixed within the epoch and decays once per epoch, without restart.
    Context parameters and Adam states are owned by the outer training loop.
    """
    agent.buffer.clear()
    return max(args.epsilon_end, args.epsilon_start * args.epsilon_decay ** epoch)


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
    p.add_argument("--num-contexts",type=int,default=None,
                   help="Number of learned Context slots; train default 3, evaluation uses checkpoint.")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=15,
                   help="Stop after this many consecutive epochs without a new best online validation accuracy.")
    p.add_argument("--selection-cost",type=float,default=None)
    p.add_argument("--gain-weight",type=float,default=None)
    p.add_argument("--gamma",type=float,default=0.98)
    p.add_argument("--context-lr",type=float,default=0.25)
    p.add_argument("--context-window", type=int, default=None,
                   help="FIFO size of paid labelled samples per Context update; default 1 for training, checkpoint value for evaluation.")
    p.add_argument("--dqn-lr",type=float,default=0.0001)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--online-warmup",type=int,default=6,
                   help="Minimum replay experiences before online DDQN updates; default 6 = two paid labels.")
    p.add_argument("--budget-fraction",type=float,default=0.2)
    p.add_argument("--no-label-quota", action="store_true",
                   help="Disable all query quotas; selection cost alone controls feedback requests.")
    p.add_argument("--global-label-budget", action="store_true",
                   help="Remove class quotas; enforce only the global --budget-fraction cap.")
    p.add_argument("--epsilon-start",type=float,default=0.3)
    p.add_argument("--epsilon-end",type=float,default=0.05)
    p.add_argument("--epsilon-decay",type=float,default=0.95,
                   help="Per-epoch epsilon multiplier; epsilon remains constant within each training epoch.")
    p.add_argument("--eval-epsilon",type=float,default=0.0)
    p.add_argument("--stream-order",choices=("input","class-interleaved","shuffle"),default="input")
    p.add_argument("--class-context-proxy", action="store_true",
                   help="Evaluation-only oracle: true class substitutes for Flow ID; independent Context/Adam, shared CNN/router; default per-class quotas, optional global budget.")
    p.add_argument("--consume-full-budget",action="store_true",
                   help="Evaluation with global budget: force paid action only when remaining positions equal unused budget.")
    p.add_argument("--policy-mode", choices=("adaptive","frozen","uncertainty"),default="adaptive",
                   help="Evaluation policy; uncertainty uses rolling entropy queries and frozen DDQN non-query routing.")
    p.add_argument("--uncertainty-window",type=int,default=200)
    p.add_argument("--uncertainty-initial-threshold",type=float,default=0.5,
                   help="First-sample normalized entropy threshold; subsequent thresholds use past scores.")
    p.add_argument("--per",type=int,choices=LEVELS)
    p.add_argument("--match-budget-from", type=Path,
                   help="RL summary.json; random routes and uniformly sampled queries matching each class count.")
    p.add_argument("--context-diagnostics", action="store_true",
                   help="Evaluation only: initial Context accuracy and current/next-sample update probes; extra forwards, observer labels only.")
    args=p.parse_args()
    if args.context_diagnostics and args.command != 'evaluate':
        p.error("--context-diagnostics is evaluation-only.")
    if args.num_contexts is not None and args.num_contexts < 1:
        p.error("num-contexts must be positive.")
    if args.class_context_proxy or args.match_budget_from or args.no_label_quota:
        p.error("Learned Context bank uses a global budget; proxy, matched and unlimited modes are unsupported.")
    if args.policy_mode == "uncertainty":
        p.error("Uncertainty baseline is not implemented for a learned bank.")
    args.global_label_budget = True
    if args.consume_full_budget and (args.command != "evaluate" or not args.global_label_budget or args.match_budget_from is not None):
        p.error("--consume-full-budget requires evaluate --global-label-budget without --match-budget-from.")
    if args.policy_mode != "adaptive" and args.command != "evaluate":
        p.error("Frozen and uncertainty modes are evaluation-only.")
    if args.policy_mode != "adaptive" and args.match_budget_from is not None:
        p.error("Baseline policy modes cannot be combined with --match-budget-from.")
    if args.uncertainty_window < 1 or not np.isfinite(args.uncertainty_initial_threshold) or not 0<=args.uncertainty_initial_threshold<=1:
        p.error("Invalid uncertainty window or initial threshold.")
    if args.class_context_proxy and args.command != "evaluate":
        p.error("--class-context-proxy requires evaluate.")
    if args.global_label_budget and args.no_label_quota:
        p.error("Choose --global-label-budget or --no-label-quota, not both.")
    if args.global_label_budget and args.match_budget_from is not None:
        p.error("--global-label-budget cannot be combined with --match-budget-from.")
    if args.no_label_quota and args.match_budget_from is not None:
        p.error("--no-label-quota cannot be combined with --match-budget-from.")
    if args.match_budget_from is not None and (args.command != "evaluate" or args.eval_epsilon != 1):
        p.error("--match-budget-from requires evaluate --eval-epsilon 1.")
    if not 0<=args.epsilon_end<=args.epsilon_start<=1 or not 0<=args.eval_epsilon<=1 or not 0<args.epsilon_decay<=1:
        p.error("Invalid exploration parameters.")
    if args.context_window is not None and args.context_window < 1:
        p.error("context-window must be positive.")
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
    checkpoint_path=args.output/"Models"/"sparse_context_bank_best.pth"
    if args.command=="train":
        args.num_contexts = 3 if args.num_contexts is None else args.num_contexts
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
        args.context_window = 1 if args.context_window is None else args.context_window
        args.selection_cost=0.2 if args.selection_cost is None else args.selection_cost
        args.gain_weight=1.0 if args.gain_weight is None else args.gain_weight
    else:
        bundle=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        if bundle.get("version")!=VERSION: p.error("A trained three-action checkpoint is required.")
        trained_count = bundle["num_contexts"]
        if args.num_contexts is not None and args.num_contexts != trained_count:
            p.error("num-contexts differs from training; train a separate bank.")
        args.num_contexts = trained_count
        trained_window = bundle.get("training_args", {}).get("context_window", 1)
        args.context_window = trained_window if args.context_window is None else args.context_window
        for key in ("selection_cost","gain_weight"):
            if getattr(args,key) is not None and getattr(args,key)!=bundle[key]:
                p.error(f"{key} differs from training; retrain for a new reward.")
            setattr(args,key,bundle[key])
    if any(not np.isfinite(v) or v<0 for v in (args.selection_cost,args.gain_weight)):
        p.error("Reward cost and gain weight must be finite and nonnegative.")
    print("Sparse learned Context bank; slots:",args.num_contexts,"device:",device,"cost:",args.selection_cost,"lambda:",args.gain_weight)
    size=bundle["cnn_checkpoint"]["input_length"]+bundle["num_classes"]+4*args.num_contexts+1
    if args.command=="train":
        agent=Agent(size,args,device);best=-1.;history=[];stale=0;best_epoch=None
        # Persistent bank and independent Adam states across training epochs.
        cnn,context=build_models(bundle["cnn_checkpoint"],bundle["context_checkpoint"],device,args.num_contexts)
        context_optimizers = [torch.optim.Adam(model.parameters(), lr=args.context_lr, weight_decay=1e-4)
                              for model in context.models]
        for epoch in range(args.epochs):
            epoch_epsilon = prepare_training_epoch(agent, args, epoch)
            # Feedback histories/windows and label budget are per-stream; model
            # parameters and optimizer momentum continue across epochs.
            summary,_=run_stream(train,args,bundle,cnn,context,agent,training=True,
                                 context_optimizers=context_optimizers,episode_epsilon=epoch_epsilon)
            summary.update(epoch_epsilon=epoch_epsilon,
                           training_protocol="persistent_context_adam; fresh_epoch_replay; epoch_epsilon_v2")
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
                saved["bank_state"]={k:v.detach().cpu().clone() for k,v in context.state_dict().items()}
                saved["num_contexts"]=args.num_contexts
                saved.update(router_state={k:v.detach().cpu().clone() for k,v in agent.q.state_dict().items()},
                             selection_cost=args.selection_cost,gain_weight=args.gain_weight,
                             training_args=vars(args),best_epoch=epoch+1,
                             training_protocol="persistent_context_adam; fresh_epoch_replay; epoch_epsilon_v2",
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
        tag=f"sparsebank{args.num_contexts}_seed{args.seed}_budget{args.budget_fraction:g}_lr{args.context_lr:g}_eps{args.eval_epsilon:g}_{args.stream_order}_warmup{args.online_warmup}"
        tag += f"_contextwindow{args.context_window}"
        if args.context_diagnostics:
            tag += "_contextdiagnostics"
        if args.no_label_quota:
            tag += "_no_label_quota"
        if args.global_label_budget:
            tag += "_global_label_budget"
        if args.class_context_proxy:
            tag += "_oracle_class_context_proxy"
        if args.policy_mode != "adaptive":
            tag += f"_policy{args.policy_mode}"
        if args.policy_mode == "uncertainty":
            tag += f"_entropywindow{args.uncertainty_window}_initial{args.uncertainty_initial_threshold:g}"
        if args.consume_full_budget:
            tag += "_fullbudget"
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
            cnn,context=build_models(bundle["cnn_checkpoint"],bundle["context_checkpoint"],device,args.num_contexts,
                                     None if args.command=="train" else bundle["bank_state"])
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
