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
