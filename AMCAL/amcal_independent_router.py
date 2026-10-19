"""Supervised route-only control, trained from paid feedback on training IDs.

The CNN and initial Context are copied unchanged. Uniform query positions stay
independent of router predictions. Test labels never enter router training.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch
from torch import nn


class RouteHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, state):
        return self.network(state)


def route_features(inputs, base_prob, context_prob):
    return torch.cat([inputs.flatten(1), base_prob, context_prob], dim=1)



class OnlineRouteLearner:
    """Bounded replay of paid, pre-feedback route preferences; isolated RNG."""
    def __init__(self, head, lr, seed, capacity=256, batch_size=32):
        from collections import deque
        self.head = head
        self.optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self.batch_size, self.updates, self.examples = batch_size, 0, 0
        self.last_loss = None

    def observe(self, inputs, base_prob, context_prob, label):
        frozen_ok = int(base_prob.argmax(1).item()) == int(label.item())
        context_ok = int(context_prob.argmax(1).item()) == int(label.item())
        if frozen_ok != context_ok:
            state = route_features(inputs, base_prob, context_prob).detach().reshape(-1).cpu().clone()
            self.buffer.append((state, int(context_ok)))
            self.examples += 1
        if not self.buffer:
            return
        indices = self.rng.choice(len(self.buffer), min(self.batch_size, len(self.buffer)), replace=False)
        device = next(self.head.parameters()).device
        states = torch.stack([self.buffer[int(i)][0] for i in indices]).to(device)
        targets = torch.tensor([self.buffer[int(i)][1] for i in indices], dtype=torch.long, device=device)
        self.optimizer.zero_grad()
        loss = nn.functional.cross_entropy(self.head(states), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1)
        self.optimizer.step()
        self.updates += 1
        self.last_loss = float(loss.detach().item())


def main():
    import amcal_burst_official_protocol as official
    core = official.core
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--burst-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[0, 1, 3])
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--router-lr', type=float, default=0.001)
    p.add_argument('--context-lr', type=float, default=0.0001)
    p.add_argument('--budget-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.epochs, a.patience) < 1 or not 0 < a.budget_fraction <= 1 or min(a.router_lr, a.context_lr) <= 0:
        p.error('Invalid epochs, patience, budget or LR.')
    source, out, burst = a.source_run.resolve(), a.output.resolve(), a.burst_dir.resolve()
    if out.exists():
        p.error('Use a fresh output directory.')
    art = joblib.load(source/'Models/protocol_preprocessing.joblib')
    saved_policy = torch.load(source/'Models/adversarial_best_dqn.pth', map_location='cpu', weights_only=False)
    if saved_policy.get('context_replay_size', 0) != 0:
        p.error('This control currently requires context replay size zero.')
    levels = sorted(set(a.levels))
    if levels != sorted(saved_policy.get('training_perturbations', [])):
        p.error('Requested domains must match source-run training domains.')
    core.set_seed(a.seed)
    width, classes = len(art['feature_names']), len(art['label_encoder'].classes_)
    cnn = core.CNN1D(width, classes).to(core.device)
    cnn.load_state_dict(torch.load(source/'Models/initial_cnn_model.pth', map_location='cpu', weights_only=False)['model_state_dict'])
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    initial = torch.load(source/'Models/adversarial_best_context.pth', map_location='cpu', weights_only=False)['context_model_state_dict']
    context = core.ContextAwareNetwork(width).to(core.device)
    head = RouteHead(width+2*classes).to(core.device)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=a.router_lr, weight_decay=1e-4)
    # Exactly the existing Context objective; this module changes only routing.
    domains = {}
    for level in levels:
        path = burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
        frame, ids, _ = official.mapped_domain(path, art, burst, False)
        lookup = {int(v): i for i, v in enumerate(ids)}
        for split in ('train_ids', 'val_ids'):
            subset = frame.iloc[[lookup[int(i)] for i in art[split]]]
            x = art['scaler'].transform(subset.iloc[:, :-1]).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite router development features.')
            y = art['label_encoder'].transform(subset.iloc[:, -1])
            domains[level, split] = (torch.tensor(x, device=core.device).unsqueeze(1), y)
    out.mkdir(parents=True)
    shutil.copytree(source/'Models', out/'Models')
    replay_x, replay_y = [], []
    replay_rng = np.random.default_rng(a.seed+99003)

    def episode(level, split, learn):
        x, truths = domains[level, split]
        order = np.random.default_rng(a.seed).permutation(len(truths))
        cap = int(a.budget_fraction*len(truths))
        schedule = set(np.random.default_rng(a.seed+819003).choice(len(truths), cap, replace=False).tolist())
        context.load_state_dict(initial)
        context.eval()
        optimizer = torch.optim.Adam(context.parameters(), lr=a.context_lr, weight_decay=1e-4)
        correct, examples, queried = 0, 0, 0
        for step, idx in enumerate(order):
            inputs = x[idx:idx+1]
            with torch.no_grad():
                base, adapted = cnn(inputs), cnn(context(inputs))
                state = route_features(inputs, base.softmax(1), adapted.softmax(1))
                route = int(head(state).argmax(1).item())
                paid = step in schedule
                prediction = int((adapted if paid or route == 1 else base).argmax(1).item())
            correct += prediction == int(truths[idx])  # observer score, never a feature
            if not paid:
                continue
            queried += 1
            label = torch.tensor([int(truths[idx])], device=core.device)
            if learn:
                b_ok = int(base.argmax(1).item()) == int(label.item())
                c_ok = int(adapted.argmax(1).item()) == int(label.item())
                if b_ok != c_ok:
                    replay_x.append(state.detach().squeeze(0).cpu())
                    replay_y.append(int(c_ok))
                    examples += 1
                if replay_x:
                    indices = replay_rng.choice(len(replay_x), min(32, len(replay_x)), replace=False)
                    batch = torch.stack([replay_x[i] for i in indices]).to(core.device)
                    target = torch.tensor([replay_y[i] for i in indices], device=core.device)
                    loss = nn.functional.cross_entropy(head(batch), target)
                    head_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
                    head_optimizer.step()
            optimizer.zero_grad()
            loss = core.context_objective(nn.functional.cross_entropy(cnn(context(inputs)), label),
                                          nn.functional.cross_entropy(base, label))
            loss.backward()
            optimizer.step()
        assert queried == cap
        return correct/len(truths), examples

    best, stale, best_state = -1, 0, None
    for epoch in range(a.epochs):
        level = levels[epoch % len(levels)]
        _, examples = episode(level, 'train_ids', True)
        validation = [episode(v, 'val_ids', False)[0] for v in levels]
        score = float(np.mean(validation))
        print(f'Router epoch {epoch+1}/{a.epochs} | train domain {level} | paid route examples {examples} | validation {100*score:.2f}%', flush=True)
        if score > best:
            best, stale, best_state = score, 0, deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= a.patience:
                break
    metadata = dict(protocol='independent supervised route head; uniform paid training labels only; ties excluded',
                    input_width=width+2*classes, seed=a.seed, levels=levels, context_lr=a.context_lr,
                    budget_fraction=a.budget_fraction, validation_accuracy=best,
                    model_hashes={name:core.file_sha256(out/'Models'/name) for name in
                                  ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib')})
    torch.save(dict(**metadata, state_dict=best_state), out/'Models/independent_router.pth')
    (out/'independent_router_training.json').write_text(json.dumps(metadata, indent=2))
    print(f'Router saved. CNN and initial Context unchanged. No heldout test scored. Best validation: {100*best:.2f}%')


if __name__ == '__main__':
    main()
