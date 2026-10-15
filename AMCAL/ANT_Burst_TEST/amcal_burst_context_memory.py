"""Class-balanced Context rehearsal, separate from DQN replay."""
from collections import deque
import random


class ContextMemory:
    def __init__(self, per_class=32, seed=42):
        if per_class < 1:
            raise ValueError("memory-per-class must be positive.")
        self.per_class = per_class
        self.buckets = {}
        self.rng = random.Random(seed)
        self.total_added = 0
        self.replayed_samples = 0

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets.values())

    def add(self, feature, label, source_row=None):
        label = int(label)
        self.buckets.setdefault(label, deque(maxlen=self.per_class)).append(
            (feature, label, source_row))
        self.total_added += 1

    def add_batch(self, inputs, labels, source_rows=None):
        # Detached CPU snapshots; no live graph or alias to model inputs.
        for i in range(len(labels)):
            self.add(inputs[i].detach().cpu().clone(), labels[i].item(),
                     source_rows[i] if source_rows is not None else None)

    def sample_items(self, size):
        classes = list(self.buckets)
        self.rng.shuffle(classes)
        pools = {c: list(self.buckets[c]) for c in classes}
        for pool in pools.values():
            self.rng.shuffle(pool)
        chosen = []
        while len(chosen) < size:
            progressed = False
            for c in classes:
                if pools[c] and len(chosen) < size:
                    chosen.append(pools[c].pop())
                    progressed = True
            if not progressed:
                break
        self.replayed_samples += len(chosen)
        return chosen

    def stats(self):
        return dict(size=len(self), per_class={str(c):len(b) for c,b in self.buckets.items()},
                    added_samples=self.total_added, replayed_samples=self.replayed_samples)


def make_context_memory(mode, per_class=32, seed=42):
    if mode == "off":
        return None
    if mode != "per-class":
        raise ValueError("Unsupported Context memory mode.")
    return ContextMemory(per_class, seed)


def with_context_rehearsal(current_loss, memory, context, cnn, batch_size=16, weight=1.0):
    """Compute loss before insertion of the current paid example(s).

    Balanced replay uses unweighted CE to avoid reintroducing class-frequency
    bias. No detach/no_grad on the prediction path: gradients reach Context.
    """
    if memory is None or len(memory) == 0 or weight == 0:
        return current_loss, dict(memory_loss=0.0, replay_samples=0)
    import torch
    import torch.nn.functional as functional
    items = memory.sample_items(batch_size)
    features = torch.stack([item[0] for item in items]).to(next(context.parameters()).device)
    labels = torch.tensor([item[1] for item in items], dtype=torch.long, device=features.device)
    replay_loss = functional.cross_entropy(cnn(context(features)), labels)
    return current_loss + weight * replay_loss, dict(
        memory_loss=float(replay_loss.detach().item()), replay_samples=len(items))

def context_update_probe(context, cnn, inputs, labels, memory):
    """Observer only: all retained paid rows, no sampling or RNG consumption."""
    import torch
    import torch.nn.functional as F
    modules = list(context.modules()) + list(cnn.modules())
    modes = [m.training for m in modules]
    context.eval()
    cnn.eval()
    try:
        with torch.no_grad():
            current = float(F.cross_entropy(cnn(context(inputs)), labels).item())
            retained = []
            if memory is not None:
                retained = [item for bucket in memory.buckets.values() for item in bucket]
            result = dict(current_ce=current, memory_ce=None, memory_accuracy=None,
                          memory_per_class={}, memory_samples=len(retained))
            if retained:
                device = inputs.device
                predictions, losses, targets = [], [], []
                for start in range(0, len(retained), 64):
                    items = retained[start:start+64]
                    x = torch.stack([item[0] for item in items]).to(device)
                    y = torch.tensor([item[1] for item in items], device=device, dtype=torch.long)
                    logits = cnn(context(x))
                    predictions.extend(logits.argmax(1).cpu().tolist())
                    losses.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
                    targets.extend(y.cpu().tolist())
                for c in sorted(set(targets)):
                    indices = [i for i, y in enumerate(targets) if y == c]
                    result["memory_per_class"][str(c)] = dict(
                        samples=len(indices),
                        accuracy=sum(predictions[i] == c for i in indices)/len(indices),
                        ce=sum(losses[i] for i in indices)/len(indices))
                result["memory_ce"] = sum(losses)/len(losses)
                result["memory_accuracy"] = sum(p == y for p,y in zip(predictions,targets))/len(targets)
            return result
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode


def final_context_memory_report(context, cnn, memory):
    if memory is None or not len(memory):
        return None
    first = next(item for bucket in memory.buckets.values() for item in bucket)
    device = next(context.parameters()).device
    import torch
    report = context_update_probe(context, cnn, first[0].unsqueeze(0).to(device),
                                  torch.tensor([first[1]], device=device), memory)
    report.pop("current_ce")
    print("Final Context memory accuracy:", report["memory_accuracy"])
    print("Final Context memory per class:", report["memory_per_class"])
    return report


def retention_guard_step(optimizer, context, cnn, inputs, labels, memory,
                         before, tolerance=0.02, retries=4, factor=0.5):
    """Retry the already-computed gradient; restore parameters AND Adam state.

    Compare each previously observed class except the current queried class.
    No memory sampling, label query, backward pass, or holdout access here.
    """
    import copy
    import math
    snapshot = copy.deepcopy(context.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    rates = [group["lr"] for group in optimizer.param_groups]
    protected = {key: value["ce"] for key, value in before["memory_per_class"].items()
                 if int(key) not in set(labels.detach().cpu().tolist())}
    after = before
    for attempt in range(retries + 1):
        if attempt:
            context.load_state_dict(snapshot)
            optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * factor ** attempt
        optimizer.step()
        after = context_update_probe(context, cnn, inputs, labels, memory)
        finite = math.isfinite(after["current_ce"]) and all(
            math.isfinite(v["ce"]) for v in after["memory_per_class"].values())
        safe = finite and all(after["memory_per_class"][key]["ce"] <= value + tolerance
                              for key, value in protected.items())
        if safe:
            used_lr = [group["lr"] for group in optimizer.param_groups]
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate
            return after, dict(accepted=True, attempts=attempt+1,
                               protected_classes=list(protected), used_lr=used_lr)
    context.load_state_dict(snapshot)
    optimizer.load_state_dict(optimizer_snapshot)
    return before, dict(accepted=False, attempts=retries+1,
                        protected_classes=list(protected), used_lr=[])
