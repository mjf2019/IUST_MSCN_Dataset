"""Online Burst ablation: random Context updates with no DQN or replay.

Reuse the original frozen CNN and initial Context checkpoint. This ablates the
online selector/learner, not RL's contribution to offline Context pretraining.
No training or evaluation is executed when this module is imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reuse the exact original architecture and Context objective. Importing defines
# legacy classes, but this runner never instantiates a DQN, loads its checkpoint,
# computes Q/rewards, creates replay, or invokes a selector optimizer.
import amcal_burst_legacy as original

PER_LEVELS = (0, 1, 3, 5, 7, 10, 12, 15, 17, 20)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--data-dir", type=Path, default=root / "Dataset")
    parser.add_argument("--output", type=Path,
                        default=root / "protocol_runs" / "notebook_original",
                        help="Existing CNN/Context model folder, not a new training folder.")
    parser.add_argument("--results-output", type=Path,
                        help="Optional separate result folder.")
    parser.add_argument("--per", type=int, choices=PER_LEVELS,
                        help="One perturbation level; omit to evaluate all ten levels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=214)
    parser.add_argument("--max-samples", type=int, default=1073)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--context-loss-mode", choices=("legacy", "aligned"), default="legacy")
    parser.add_argument("--query-gate", choices=("none", "legacy"), default="none",
                        help="none: uniform fixed-budget queries; legacy: random 50%% actions "
                             "with original disagreement/confidence-gap filters.")
    args = parser.parse_args()
    if args.budget < 0 or args.max_samples < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("budget must be nonnegative, max-samples positive, and lr finite and positive.")
    return args


def evaluate_level(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = original.device
    original.CONTEXT_LOSS_MODE = args.context_loss_mode
    model_dir = args.output.resolve() / "Models"
    data_dir = args.data_dir.resolve()

    # Only TWO checkpoint files are read. No DQN checkpoint is required.
    cnn_path = model_dir / "initial_cnn_model.pth"
    context_path = model_dir / "adversarial_best_context.pth"
    cnn_checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
    context_checkpoint = torch.load(context_path, map_location=device, weights_only=False)
    for checkpoint in (cnn_checkpoint, context_checkpoint):
        if checkpoint.get("context_loss_mode", "legacy") != args.context_loss_mode:
            raise ValueError("Context loss setting must match the existing checkpoints.")
    if cnn_checkpoint["input_length"] != context_checkpoint["input_length"]:
        raise ValueError("CNN and Context feature counts differ.")
    if cnn_checkpoint["num_classes"] != context_checkpoint["num_classes"]:
        raise ValueError("CNN and Context class counts differ.")

    input_length = cnn_checkpoint["input_length"]
    num_classes = cnn_checkpoint["num_classes"]
    cnn = original.CNN1D(input_length=input_length, num_classes=num_classes).to(device)
    context = original.ContextAwareNetwork(input_size=input_length, num_layers=4).to(device)
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    context.load_state_dict(context_checkpoint["context_model_state_dict"])
    cnn.eval()
    context.eval()
    cnn_initial_state = {name: tensor.detach().cpu().clone()
                         for name, tensor in cnn.state_dict().items()}

    base_dir = data_dir / "AdvBurst_FTSC-IAT"
    reference_path = base_dir / "AdvBurstTrainedBurst_size_0_Step_2000.csv"
    attack_path = base_dir / f"AdvBurstTrainedBurst_size_{args.per}_Step_2000.csv"
    reference = pd.read_csv(reference_path)
    attack = pd.read_csv(attack_path).iloc[:args.max_samples]
    if len(attack) == 0:
        raise ValueError("Evaluation stream is empty.")
    if reference.shape[1] - 1 != input_length or attack.shape[1] - 1 != input_length:
        raise ValueError("CSV feature count does not match model checkpoints.")
    if list(reference.columns) != list(attack.columns):
        raise ValueError("Reference and attack CSV feature schemas differ.")

    # Same PER0-fitted scaler, label mapping and online class weights as the
    # original successful runner. No new split, scaling rule or oversampling.
    scaler = StandardScaler().fit(reference.iloc[:, :-1].to_numpy())
    encoder = LabelEncoder().fit(reference.iloc[:, -1].to_numpy())
    if len(encoder.classes_) != num_classes:
        raise ValueError("PER0 class count does not match the checkpoints.")
    reference_labels = encoder.transform(reference.iloc[:, -1].to_numpy())
    class_counts = np.bincount(reference_labels, minlength=num_classes)
    labels_array = encoder.transform(attack.iloc[:, -1].to_numpy())
    inputs = torch.tensor(scaler.transform(attack.iloc[:, :-1].to_numpy()),
                          dtype=torch.float32, device=device).unsqueeze(1)
    labels = torch.tensor(labels_array, dtype=torch.long, device=device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_counts, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(context.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(attack)
    label_budget = min(args.budget, n)
    rng = np.random.default_rng(args.seed)
    query_mask = np.zeros(n, dtype=bool)
    if args.query_gate == "none":
        # Uniform sample without replacement; depends only on N, B and seed.
        # Predictions/features/labels do not influence the chosen positions.
        query_mask[rng.choice(n, size=label_budget, replace=False)] = True
    threshold = 0.005 if args.query_gate == "legacy" else None
    updates = 0
    trace, before_predictions, after_predictions, base_predictions = [], [], [], []
    cumulative_correct = 0

    print(f"Loaded original CNN and Context from: {model_dir}")
    print("RL removed: no DQN construction/checkpoint, Q values, rewards, replay or optimizer.")
    print(f"Random query rule: {args.query_gate}; label budget: {label_budget}/{n}")
    for index in range(n):
        x = inputs[index:index + 1]
        y = labels[index:index + 1]
        with torch.no_grad():
            base_logits = cnn(x)
            base_probs = torch.softmax(base_logits, dim=1)
            base_prediction = int(base_probs.argmax(1).item())
            before_logits = cnn(context(x))
            before_probs = torch.softmax(before_logits, dim=1)
            before_prediction = int(before_probs.argmax(1).item())
            gap = float(abs(before_probs.max().item() - base_probs.max().item()))

        if args.query_gate == "none":
            chosen = bool(query_mask[index])
            requested = chosen
        else:
            # Matched-gate online ablation: retain the original 50/50 random
            # action, disagreement filter and threshold, but remove all DQN work.
            requested = bool(rng.random() < 0.5) if updates < label_budget else False
            chosen = (requested and before_prediction != base_prediction
                      and gap >= threshold and updates < label_budget)

        observed_loss = None
        if chosen:
            # Predict first, then query and update: no label is passed to the
            # optimizer for unselected samples. Context stays in eval mode,
            # exactly like the original online loop (fixed BN/Dropout behavior).
            optimizer.zero_grad()
            context_loss = criterion(cnn(context(x)), y)
            base_loss = criterion(base_logits.detach(), y)
            loss = original.context_objective(context_loss, base_loss, online=True)
            loss.backward()
            optimizer.step()
            updates += 1
            observed_loss = float(loss.detach().item())
            with torch.no_grad():
                after_prediction = int(cnn(context(x)).argmax(1).item())
        else:
            after_prediction = before_prediction

        true_label = int(y.item())  # observer/scoring, not selector input
        cumulative_correct += int(after_prediction == true_label)
        before_predictions.append(before_prediction)
        after_predictions.append(after_prediction)
        base_predictions.append(base_prediction)
        trace.append(dict(
            step=index + 1, true_label=true_label, base_prediction=base_prediction,
            prediction_before_update=before_prediction,
            prediction_after_update=after_prediction,
            random_requested=requested, updated=chosen, labels_used=updates,
            confidence_gap=gap, context_loss=observed_loss,
            cumulative_accuracy=cumulative_correct / (index + 1)))

    if args.query_gate == "none" and updates != label_budget:
        raise RuntimeError("Uniform random queries did not consume the requested unique-label budget.")
    classifier_unchanged = all(
        torch.equal(cnn_initial_state[name], tensor.detach().cpu())
        for name, tensor in cnn.state_dict().items())
    if not classifier_unchanged:
        raise RuntimeError("Frozen CNN weights or buffers changed.")

    diagnostics = dict(
        per=args.per, seed=args.seed, evaluated_rows=n,
        base_accuracy_legacy_scaler=float(accuracy_score(labels_array, base_predictions)),
        amcal_accuracy_before_update=float(accuracy_score(labels_array, before_predictions)),
        amcal_accuracy_after_update=float(accuracy_score(labels_array, after_predictions)),
        before_weighted_f1=float(f1_score(labels_array, before_predictions, average="weighted", zero_division=0)),
        after_weighted_f1=float(f1_score(labels_array, after_predictions, average="weighted", zero_division=0)),
        before_macro_f1=float(f1_score(labels_array, before_predictions, average="macro", zero_division=0)),
        after_macro_f1=float(f1_score(labels_array, after_predictions, average="macro", zero_division=0)),
        label_budget=label_budget, labels_used=updates, unique_labels_queried=updates,
        actual_label_fraction=updates / n, context_updates=updates,
        dqn_loaded=False, dqn_updates=0, replay_size=0, reward_computations=0,
        classifier_unchanged=classifier_unchanged,
        first_query_step=next((row["step"] for row in trace if row["updated"]), None),
        last_query_step=next((row["step"] for row in reversed(trace) if row["updated"]), None),
        selector_mode="random", query_gate=args.query_gate,
        confidence_gap_threshold=threshold, online_lr=args.lr,
        context_loss_mode=args.context_loss_mode,
        ablation_scope="online only; initial Context was trained by the original method",
        protocol="predict-score-query-update; also records legacy post-update scores",
        alignment="attack-file local row IDs; source overlap unverified",
        attack_sha256=sha256_file(attack_path),
        cnn_checkpoint_sha256=sha256_file(cnn_path),
        context_checkpoint_sha256=sha256_file(context_path),
        query_rule=("uniform fixed-budget sample without replacement over stream rows"
                    if args.query_gate == "none" else
                    "random 50/50 action, disagreement, confidence gap >= 0.005, budget remaining"),
        class_label_mapping=[str(label) for label in encoder.classes_],
        classes_before=classification_report(labels_array, before_predictions, output_dict=True, zero_division=0),
        classes_after=classification_report(labels_array, after_predictions, output_dict=True, zero_division=0))
    results_dir = (args.results_output.resolve() if args.results_output else
                   args.output.resolve() / "Results" / f"random_no_rl_{args.query_gate}_seed{args.seed}")
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"no_rl_PER_{args.per}"
    pd.DataFrame(trace).to_csv(results_dir / f"{prefix}_trace.csv", index=False)
    (results_dir / f"{prefix}_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    (results_dir / f"{prefix}_manifest.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, indent=2), encoding="utf-8")
    # Keep detailed diagnostics in JSON; console follows the original notebook.
    weighted = diagnostics["classes_after"]["weighted avg"]
    row = {
        "Perturbation Level": args.per,
        "Test Accuracy": diagnostics["amcal_accuracy_after_update"],
        "Precision": float(weighted["precision"]),
        "Recall": float(weighted["recall"]),
        "F1 Score": diagnostics["after_weighted_f1"],
        "Context Updates Used": updates,
        "DQN Updates Used": 0,
        "DQN Loss": 0.0,
    }
    print("Diagnostic frozen CNN / pre-update AMCAL / post-update AMCAL:",
          diagnostics["base_accuracy_legacy_scaler"],
          diagnostics["amcal_accuracy_before_update"],
          diagnostics["amcal_accuracy_after_update"])
    print(f"\nResults for Perturbation Level {args.per}:")
    print(f"  Test Accuracy: {row['Test Accuracy']:.4f}")
    print(f"  Precision (weighted): {row['Precision']:.4f}")
    print(f"  Recall (weighted): {row['Recall']:.4f}")
    print(f"  F1 Score (weighted): {row['F1 Score']:.4f}")
    print(f"  Context Updates Used: {updates}/{label_budget}")
    print("  DQN Updates Used: 0 (RL removed)")
    print("  Average DQN Loss: 0.0000 (RL removed)")
    return row, [item["cumulative_accuracy"] for item in trace], results_dir


def evaluate(args):
    levels = list(PER_LEVELS) if args.per is None else [args.per]
    rows, accuracy_by_level = [], {}
    results_dir = None
    for per in levels:
        print("\n" + "=" * 50)
        print(f"Starting processing for perturbation level {per}")
        print("=" * 50)
        level_args = argparse.Namespace(**vars(args))
        level_args.per = per
        # Reload CNN/Context AND recreate optimizer/budget inside each level.
        row, cumulative_accuracy, results_dir = evaluate_level(level_args)
        rows.append(row)
        accuracy_by_level[per] = cumulative_accuracy

    results = pd.DataFrame(rows)
    results_path = results_dir / "AMCAL_20_cumulative_accuracy.csv"
    results.to_csv(results_path, index=False)
    print("\nCombined Results Table:")
    print(results.to_string(index=False))
    print(f"\nCombined results saved to '{results_path}'.")

    max_samples = max(len(values) for values in accuracy_by_level.values())
    combined_columns = {"Size": list(accuracy_by_level)}
    for index in range(max_samples):
        combined_columns[str(index + 1)] = [
            values[index] if index < len(values) else np.nan
            for values in accuracy_by_level.values()]
    cumulative_path = results_dir / "Micfoal_Conf_cumulative_accuracy_all_perturbations_transposed.csv"
    pd.DataFrame(combined_columns).to_csv(cumulative_path, index=False)
    print(f"Combined cumulative accuracy saved to '{cumulative_path}'")

    # Same cumulative-accuracy chart as original, saved without opening a window.
    original.plt.figure(figsize=(12, 8))
    for per, values in accuracy_by_level.items():
        original.plt.plot(range(1, len(values) + 1), values,
                          label=f"Perturbation {per}", marker="o",
                          markersize=4, linewidth=1.5)
    original.plt.xlabel("Sample Index")
    original.plt.ylabel("Cumulative Accuracy")
    original.plt.title("Cumulative Accuracy per Sample Across Perturbation Levels")
    original.plt.legend()
    original.plt.grid(True, linestyle="--", alpha=0.7)
    figure_path = results_dir / "AMCAL_20_cumulative_accuracy.png"
    original.plt.savefig(figure_path, dpi=300)
    original.plt.close()
    print(f"Plot saved to '{figure_path}'")

    manifest = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    manifest["perturbation_levels"] = levels
    suffix = "all_levels" if args.per is None else f"PER_{args.per}"
    (results_dir / f"no_rl_{suffix}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return results



if __name__ == "__main__":
    evaluate(parse_args())
