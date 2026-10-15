"""Offline random baseline: match RL query counts without using its positions."""
import random


def matched_query_positions(labels, caps, seed):
    caps = {int(c): n for c, n in caps.items()}
    classes = set(map(int, labels))
    if classes - set(caps):
        raise ValueError("Reference budget lacks a class in this stream.")
    positions = {}
    for i, label in enumerate(labels):
        positions.setdefault(int(label), []).append(i)
    rng = random.Random(seed)
    chosen = set()
    for c in sorted(caps):
        n = caps[c]
        if type(n) is not int or n < 0 or n > len(positions.get(c, [])):
            raise ValueError("Reference count must be an integer within class size.")
        chosen.update(rng.sample(positions.get(c, []), n))
    return chosen
