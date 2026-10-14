"""Disjoint row holdout for final evaluation, excluded from online adaptation."""
import math


def split_holdout_indices(labels, fraction):
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("holdout fraction must be finite and in [0,1).")
    if fraction == 0:
        return list(range(len(labels))), []
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    heldout = set()
    for label, positions in groups.items():
        if len(positions) < 2:
            raise ValueError(f"Class {label} needs at least two rows for adaptation and holdout.")
        count = min(len(positions)-1, max(1, math.ceil(len(positions)*fraction)))
        heldout.update(positions[-count:])
    return ([i for i in range(len(labels)) if i not in heldout],
            [i for i in range(len(labels)) if i in heldout])


def evaluate_heldout(cnn, context, features, labels, device, source_rows, num_classes):
    """Pure inference on final models. No labels, rewards or optimizer in forward."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    modes = {m: m.training for model in (cnn, context) for m in model.modules()}
    base, adapted = [], []
    try:
        cnn.eval()
        context.eval()
        with torch.no_grad():
            for start in range(0, len(features), 128):
                x = torch.as_tensor(features[start:start+128], dtype=torch.float32,
                                    device=device).unsqueeze(1)
                base.extend(cnn(x).argmax(1).cpu().tolist())
                adapted.extend(cnn(context(x)).argmax(1).cpu().tolist())
    finally:
        for module, training in modes.items():
            module.training = training
    class_ids = list(range(num_classes))
    report = classification_report(labels, adapted, labels=class_ids, output_dict=True, zero_division=0)
    rows = [dict(source_row=int(source_rows[i])+1, true_label=int(labels[i]),
                 base_prediction=base[i], final_context_prediction=adapted[i])
            for i in range(len(labels))]
    summary = dict(samples=len(labels), accuracy=accuracy_score(labels, adapted),
                   base_accuracy=accuracy_score(labels, base),
                   macro_f1=f1_score(labels, adapted, average="macro", zero_division=0),
                   weighted_f1=f1_score(labels, adapted, average="weighted", zero_division=0),
                   classes=report, confusion_matrix=confusion_matrix(labels, adapted, labels=class_ids).tolist(),
                   model_updates=0, queried_labels=0, protocol="final frozen model; holdout excluded from online stream")
    return summary, rows
