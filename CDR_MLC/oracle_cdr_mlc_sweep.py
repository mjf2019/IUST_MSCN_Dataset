"""Perfect congestion-level routing with 0%/20% level-specific experts.

Low expert uses all Low records. Medium/High experts use chronological
calibration prefixes. The oracle sends held-out tails to the matching level
expert. This is a diagnostic upper bound, not deployable inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import TIMING, fit_rf, metrics, predict_rf


LEVELS = ("Low", "Medium", "High")


def split_prefix(frame: pd.DataFrame, fraction: float):
    prefix, tail = [], []
    for _, group in frame.groupby("capture_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * fraction)
        if fraction > 0 and not 0 < cut < len(group):
            raise ValueError(f"capture too short for fraction {fraction}")
        prefix.append(group.iloc[:cut].copy())
        tail.append(group.iloc[cut:].copy())
    empty = frame.iloc[:0].copy()
    return (
        pd.concat(prefix, ignore_index=True) if fraction > 0 else empty,
        pd.concat(tail, ignore_index=True),
    )


def prepare_protocol(data: pd.DataFrame, fraction: float):
    low = data[data.congestion_level.eq("Low")].copy().reset_index(drop=True)
    medium_train, medium_test = split_prefix(
        data[data.congestion_level.eq("Medium")].copy(), fraction
    )
    high_train, high_test = split_prefix(
        data[data.congestion_level.eq("High")].copy(), fraction
    )
    return {
        "train": {"Low": low, "Medium": medium_train, "High": high_train},
        "test": {"Medium": medium_test, "High": high_test},
    }


def train_models(protocol, seed: int, trees: int):
    models = {"Low": fit_rf(protocol["train"]["Low"], TIMING, seed, trees)}
    for level in ("Medium", "High"):
        frame = protocol["train"][level]
        if len(frame):
            models[level] = fit_rf(frame, TIMING, seed, trees)
    pooled = pd.concat(
        [frame for frame in protocol["train"].values() if len(frame)],
        ignore_index=True,
    )
    models["Pooled"] = fit_rf(pooled, TIMING, seed, trees)
    return models, pooled


def evaluate_fraction(data, fraction, output, seed, trees):
    protocol = prepare_protocol(data, fraction)
    models, pooled = train_models(protocol, seed, trees)
    fraction_dir = output / f"cal_{int(round(fraction * 100)):02d}"
    fraction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = {
        "fraction": fraction,
        "training_rows": {level: len(frame) for level, frame in protocol["train"].items()},
        "pooled_training_rows": len(pooled),
        "test_rows": {level: len(frame) for level, frame in protocol["test"].items()},
        "level_expert_available": {level: level in models for level in LEVELS},
    }
    for level in ("Medium", "High"):
        test = protocol["test"][level].reset_index(drop=True)
        truth = test.traffic_label.to_numpy()
        expert_available = level in models
        routed_model = models[level] if expert_available else models["Low"]
        predictions = {
            "Oracle_Level_Expert": predict_rf(routed_model, test),
            "Low_Expert": predict_rf(models["Low"], test),
            "Pooled_RF": predict_rf(models["Pooled"], test),
        }
        details = {}
        for method, prediction in predictions.items():
            result = metrics(truth, prediction, APPLICATIONS)
            details[method] = result
            if method == "Oracle_Level_Expert":
                routed_expert = level if expert_available else "Low_fallback"
            else:
                routed_expert = method
            rows.append({
                "adaptation_fraction": fraction,
                "test_level": level,
                "method": method,
                "routed_expert": routed_expert,
                "level_expert_available": expert_available,
                **{key: result[key] for key in (
                    "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
                )},
            })
        prediction_frame = test[
            ["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]
        ].copy()
        for method, prediction in predictions.items():
            prediction_frame[f"prediction_{method}"] = prediction
            prediction_frame[f"correct_{method}"] = prediction == truth
        prediction_frame["oracle_routed_expert"] = (
            level if expert_available else "Low_fallback_no_level_calibration"
        )
        level_dir = fraction_dir / f"test_{level.lower()}"
        level_dir.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(level_dir / "predictions.csv", index=False)
        (level_dir / "metrics.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )
    (fraction_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/oracle_level_experts")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()
    if any(not 0 <= fraction < 1 for fraction in args.fractions):
        parser.error("fractions must be in [0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(TIMING))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for fraction in args.fractions:
        fraction_rows, audit = evaluate_fraction(
            data, fraction, args.output, args.seed, args.trees
        )
        rows.extend(fraction_rows)
        audits[str(fraction)] = audit
    summary = pd.DataFrame(rows).sort_values(
        ["adaptation_fraction", "test_level", "method"]
    )
    summary.to_csv(args.output / "oracle_level_expert_summary.csv", index=False)
    manifest = {
        "diagnostic_only": True,
        "routing": "true congestion level selects its matching level-specific expert",
        "training": {
            "Low": "100% of every Low capture",
            "Medium": "chronological calibration prefix of every Medium capture",
            "High": "chronological calibration prefix of every High capture",
        },
        "zero_fraction_behavior": (
            "Medium/High experts cannot be trained at 0%; Oracle_Level_Expert "
            "explicitly falls back to the Low expert."
        ),
        "expert_inputs_exclude": TIMING,
        "fractions": args.fractions,
        "seed": args.seed,
        "trees": args.trees,
        "audits": audits,
        "limitations": [
            "True congestion level is used at inference; this is not deployable.",
            "The 20% prefix and 80% tail may share TCP connections.",
            "This estimates perfect level routing plus level-specific calibration.",
        ],
    }
    (args.output / "oracle_level_expert_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
