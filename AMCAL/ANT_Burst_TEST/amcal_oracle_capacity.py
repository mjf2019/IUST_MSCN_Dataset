"""Full-information greedy look-ahead oracle, NOT a deployable policy.

Uses all remaining target labels to compare one real Context update with no
update. Budget counts committed updates only; oracle label reads are explicitly
unlimited. Not a globally optimal policy or a strict upper bound.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import amcal_burst_legacy as legacy
from amcal_burst_three_action import build_models, LEVELS


def measurements(cnn, context, x, y, batch_size):
    predictions, losses = [], []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = cnn(context(x[start:start+batch_size])) if context is not None else cnn(x[start:start+batch_size])
            predictions.extend(logits.argmax(1).cpu().tolist())
            losses.extend(nn.functional.cross_entropy(logits, y[start:start+batch_size], reduction='none').cpu().tolist())
    return np.asarray(predictions), np.asarray(losses)


def objective(pred, loss, truth, base_pred, base_loss, actions):
    if actions == 3:
        correct = (pred == truth) | (base_pred == truth)
        route_loss = np.minimum(loss, base_loss)
    else:
        correct, route_loss = pred == truth, loss
    return int(correct.sum()), float(route_loss.sum())


def prefer_update(old, new, immediate_penalty):
    gain = new[0] - old[0] - immediate_penalty
    return gain > 0 or (gain == 0 and new[1] < old[1] - 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actions', type=int, choices=(2, 3), required=True)
    p.add_argument('--checkpoint-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'Dataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--context-lr', type=float, default=0.0001)
    p.add_argument('--budget-fraction', type=float, default=0.2)
    p.add_argument('--stream-order', choices=('input', 'shuffle'), default='shuffle')
    p.add_argument('--per', type=int, choices=LEVELS)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--reference-max-drop', type=float, default=None,
                   help='Optional maximum validation-reference accuracy drop from initialization, e.g. 0.02. Oracle only.')
    args = p.parse_args()
    if not 0 <= args.budget_fraction <= 1 or not np.isfinite(args.context_lr) or args.context_lr <= 0 or args.batch_size < 1:
        p.error('Invalid budget, learning rate or batch size')
    if args.reference_max_drop is not None and not 0 <= args.reference_max_drop <= 1:
        p.error('reference-max-drop must be in [0,1]')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.actions == 3:
        bundle = torch.load(args.checkpoint_output/'Models'/'three_action_best.pth', map_location='cpu', weights_only=False)
        cnn_ck, context_ck = bundle['cnn_checkpoint'], bundle['context_checkpoint']
        scaler, encoder, columns = bundle['scaler'], bundle['encoder'], bundle['columns']
        loss_mode = bundle['context_loss_mode']
    else:
        cnn_ck = torch.load(args.checkpoint_output/'Models'/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
        context_ck = torch.load(args.checkpoint_output/'Models'/'adversarial_best_context.pth', map_location='cpu', weights_only=False)
        train = pd.read_csv(args.data_dir/'oversampled_train_dataset.csv')
        scaler, encoder = StandardScaler().fit(train.iloc[:, :-1]), LabelEncoder().fit(train.iloc[:, -1])
        columns = list(train.columns)
        loss_mode = context_ck.get('context_loss_mode', 'aligned')
    legacy.CONTEXT_LOSS_MODE = loss_mode
    def prepare(frame):
        if list(frame.columns) != columns:
            raise ValueError('Feature schema mismatch')
        return (torch.tensor(scaler.transform(frame.iloc[:, :-1]), dtype=torch.float32, device=device).unsqueeze(1),
                torch.tensor(encoder.transform(frame.iloc[:, -1]), dtype=torch.long, device=device))
    reference_path = args.data_dir/'oversampled_validation_dataset.csv'
    rx, ry = prepare(pd.read_csv(reference_path))
    summaries = []
    print('ORACLE: unlimited true-label access including future; committed updates alone obey global 20% cap.')
    for level in ([args.per] if args.per is not None else LEVELS):
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        cnn, context = build_models(cnn_ck, context_ck, device)
        optimizer = torch.optim.Adam(context.parameters(), lr=args.context_lr, weight_decay=1e-4)
        frame = pd.read_csv(args.data_dir/'AdvBurst_FTSC-IAT'/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').iloc[:1073]
        order = np.random.default_rng(args.seed).permutation(len(frame)) if args.stream_order == 'shuffle' else np.arange(len(frame))
        x, y = prepare(frame.iloc[order]); truth = y.cpu().numpy()
        base_pred, base_loss = measurements(cnn, None, x, y, args.batch_size)
        initial_pred, _ = measurements(cnn, context, x, y, args.batch_size)
        ref_initial, _ = measurements(cnn, context, rx, ry, args.batch_size)
        ref_initial_acc = accuracy_score(ry.cpu().numpy(), ref_initial)
        cap, used, trials = int(len(y)*args.budget_fraction), 0, 0
        rows = []
        for i in range(len(y)):
            current_pred, current_loss = measurements(cnn, context, x[i:], y[i:], args.batch_size)
            skip_prediction = int(current_pred[0])
            skip_action = 0
            if args.actions == 3:
                skip_action = 0 if base_pred[i] == truth[i] or current_pred[0] != truth[i] else 1
                skip_prediction = int(base_pred[i] if skip_action == 0 else current_pred[0])
            accepted = False
            if used < cap:
                trials += 1
                candidate = copy.deepcopy(context)
                candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=args.context_lr, weight_decay=1e-4)
                candidate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                candidate_optimizer.zero_grad()
                ce = nn.functional.cross_entropy(cnn(candidate(x[i:i+1])), y[i:i+1])
                with torch.no_grad():
                    base_ce = nn.functional.cross_entropy(cnn(x[i:i+1]), y[i:i+1])
                legacy.context_objective(ce, base_ce, online=True).backward()
                candidate_optimizer.step()
                new_pred, new_loss = measurements(cnn, candidate, x[i+1:], y[i+1:], args.batch_size)
                old_score = objective(current_pred[1:], current_loss[1:], truth[i+1:], base_pred[i+1:], base_loss[i+1:], args.actions)
                new_score = objective(new_pred, new_loss, truth[i+1:], base_pred[i+1:], base_loss[i+1:], args.actions)
                penalty = int(skip_prediction == truth[i])-int(current_pred[0] == truth[i])
                accepted = prefer_update(old_score, new_score, penalty)
                if accepted and args.reference_max_drop is not None:
                    candidate_ref, _ = measurements(cnn, candidate, rx, ry, args.batch_size)
                    accepted = accuracy_score(ry.cpu().numpy(), candidate_ref) >= ref_initial_acc-args.reference_max_drop-1e-12
                if accepted:
                    context, optimizer = candidate, candidate_optimizer
                    used += 1
                del candidate, candidate_optimizer
            prediction = int(current_pred[0]) if accepted else skip_prediction
            rows.append(dict(step=i+1, source_row=int(order[i])+1, true_label=int(truth[i]),
                             action=(2 if args.actions == 3 else 1) if accepted else skip_action,
                             prediction_before_feedback=prediction, queried=accepted, labels_used=used))
            if (i+1) % 100 == 0:
                print(f'PER {level}; {i+1}/{len(y)}; committed labels {used}/{cap}', flush=True)
        final_pred, _ = measurements(cnn, context, x, y, args.batch_size)
        ref_final, _ = measurements(cnn, context, rx, ry, args.batch_size)
        predictions = [r['prediction_before_feedback'] for r in rows]
        no_adapt = ((initial_pred == truth) | (base_pred == truth)).mean() if args.actions == 3 else (initial_pred == truth).mean()
        summary = dict(perturbation_level=level, actions=args.actions,
            online_accuracy=float(accuracy_score(truth, predictions)),
            online_macro_f1=float(f1_score(truth, predictions, average='macro', zero_division=0)),
            no_adaptation_oracle_routing_accuracy=float(no_adapt),
            labels_used=used, label_cap=cap, counterfactual_update_trials=trials,
            initial_context_accuracy=float(accuracy_score(truth, initial_pred)),
            final_context_accuracy=float(accuracy_score(truth, final_pred)),
            reference_initial_accuracy=float(ref_initial_acc),
            reference_final_accuracy=float(accuracy_score(ry.cpu().numpy(), ref_final)),
            reference_drop_pp=float(100*(ref_initial_acc-accuracy_score(ry.cpu().numpy(), ref_final))),
            reference_correct_to_wrong=int(np.sum((ref_initial==ry.cpu().numpy()) & (ref_final!=ry.cpu().numpy()))),
            oracle_label_access='all target labels including future; no label-efficiency claim',
            oracle_protocol='greedy one-update comparison on full remaining suffix; accuracy first, CE tie-break; not global optimum',
            reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
        summaries.append(summary)
        pd.DataFrame(rows).to_csv(args.output/f'PER_{level}_trace.csv', index=False)
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary), flush=True)
    keys = ['perturbation_level','online_accuracy','no_adaptation_oracle_routing_accuracy','final_context_accuracy',
            'reference_final_accuracy','reference_drop_pp','labels_used']
    table = pd.DataFrame(summaries)[keys]
    table.to_csv(args.output/'combined_results.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps(vars(args),default=str,indent=2))
    print(table.to_string(index=False,float_format=lambda v:f'{v:.4f}'))


if __name__ == '__main__':
    main()
