"""Evaluate the strict temporal meta stacker on scenario-specific fixed tails.

For every target capture, the final ``test_fraction`` is always reserved as
test.  Adaptation takes a prefix of the same capture; any rows between the
adaptation prefix and fixed test tail are deliberately unused.  Consequently,
all adaptation fractions are evaluated on identical target records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import SCENARIOS, TIMING, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all


def fixed_tail_calibration(frame: pd.DataFrame, adaptation_fraction: float,
                           test_fraction: float):
    """Return a chronological prefix and an immutable chronological tail."""
    calibration, test, audit = [], [], []
    for capture_id, group in frame.groupby("capture_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        calibration_end = int(len(group) * adaptation_fraction)
        test_start = int(len(group) * (1.0 - test_fraction))
        if calibration_end > test_start:
            raise ValueError(
                f"{capture_id}: adaptation prefix overlaps fixed test tail"
            )
        if adaptation_fraction > 0 and calibration_end == 0:
            raise ValueError(f"{capture_id}: empty nonzero calibration prefix")
        calibration.append(group.iloc[:calibration_end].copy())
        test.append(group.iloc[test_start:].copy())
        audit.append({
            "capture_id": capture_id,
            "total_rows": len(group),
            "calibration_rows": calibration_end,
            "unused_rows": test_start - calibration_end,
            "fixed_test_rows": len(group) - test_start,
            "calibration_last_source_row": (
                int(group.iloc[calibration_end - 1].source_row)
                if calibration_end else None
            ),
            "test_first_source_row": int(group.iloc[test_start].source_row),
        })
    empty = frame.iloc[:0].copy()
    calibration_frame = (
        pd.concat(calibration, ignore_index=True)
        if adaptation_fraction > 0 else empty
    )
    test_frame = pd.concat(test, ignore_index=True)
    return calibration_frame, test_frame, audit


def frame_identity(frame: pd.DataFrame) -> str:
    """Stable identity of ordered test records, independent of model output."""
    ordered = frame.sort_values(
        ["capture_id", "sequence_id", "timestamp", "source_row"], kind="stable"
    )[["capture_id", "sequence_id", "source_file", "source_row"]]
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_scenario_fixed_test_protocol(
    data, source_level, target_level, adaptation_fraction, test_fraction
):
    """Build one independent source-to-target protocol.

    Only the requested target's chronological prefix may be added to
    development.  No calibration rows from another scenario/level are used.
    """
    if source_level == target_level:
        raise ValueError("source and target levels must differ")
    source = data[data.congestion_level.eq(source_level)].copy()
    target = data[data.congestion_level.eq(target_level)].copy()
    calibration, fixed_test, capture_audit = fixed_tail_calibration(
        target, adaptation_fraction, test_fraction
    )
    development_parts = [source]
    if adaptation_fraction > 0:
        development_parts.append(calibration)
    audit = [
        {
            "level": source_level,
            "role": "full_source_level",
            "development_rows": len(source),
        },
        {
            "level": target_level,
            "role": "scenario_target_fixed_test",
            "calibration_rows": len(calibration),
            "fixed_test_rows": len(fixed_test),
            "captures": capture_audit,
        },
    ]
    return pd.concat(development_parts, ignore_index=True), fixed_test, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/meta_stacked_fixed_test_leakage_safe")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .05, .10, .15, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=150)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be in (0,1)")
    if any(fraction < 0 or fraction > 1 - args.test_fraction for fraction in args.fractions):
        raise ValueError("fractions must not overlap the fixed test tail")

    config = MetaStackConfig(
        window=args.window, congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees, utility_trees=args.utility_trees,
        meta_trees=args.meta_trees, random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    rows, audits, expected_test_identity = [], {}, {}
    expected_evaluated_identity = {}
    for fraction in args.fractions:
        fraction_root = args.output / f"cal_{int(round(fraction * 100)):02d}"
        fraction_root.mkdir(parents=True, exist_ok=True)
        for scenario in args.scenarios:
            source_level, target_level = SCENARIOS[scenario]
            development, fixed_test, protocol_audit = (
                build_scenario_fixed_test_protocol(
                    data, source_level, target_level, fraction,
                    args.test_fraction,
                )
            )
            identity = frame_identity(fixed_test)
            previous = expected_test_identity.setdefault(target_level, identity)
            if identity != previous:
                raise RuntimeError(
                    f"fixed test identity changed for {target_level}: "
                    f"{previous} != {identity}"
                )

            model = fit_meta_stacker(development, config)
            eligible = development.loc[model["source_eligible_index"]]
            rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
            audit = {
                "adaptation_fraction": fraction,
                "fixed_test_fraction": args.test_fraction,
                "scenario": scenario,
                "source_level": source_level,
                "target_level": target_level,
                "development_rows_raw": len(development),
                "partition_rows": model["partition_rows"],
                "model_leakage_control": model["leakage_control"],
                "utility_correctness_counts": model["utility_correctness_counts"],
                "selected_meta_variant": model["selected_meta_variant"],
                "selected_meta_confidence": model["selected_meta_confidence"],
                "meta_selection_trials": model["meta_selection_trials"],
                "protocol": protocol_audit,
            }
            audits[f"{fraction}:{scenario}"] = audit
            scenario_dir = fraction_root / (
                f"scenario_{scenario}_{source_level.lower()}_to_"
                f"{target_level.lower()}"
            )
            scenario_dir.mkdir(parents=True, exist_ok=True)
            (scenario_dir / "leakage_safe_meta_audit.json").write_text(
                json.dumps(audit, indent=2) + "\n", encoding="utf-8"
            )

            result = predict_all(model, fixed_test)
            observed = result["observed"]
            evaluated_identity = frame_identity(observed)
            previous_evaluated = expected_evaluated_identity.setdefault(
                target_level, evaluated_identity
            )
            if evaluated_identity != previous_evaluated:
                raise RuntimeError(
                    f"evaluated test identity changed for {target_level}: "
                    f"{previous_evaluated} != {evaluated_identity}"
                )
            truth = observed.traffic_label.to_numpy()
            predictions = {
                key: value for key, value in result.items()
                if key.startswith("CDR_")
            }
            predictions["RF_expert_inputs"] = predict_rf(rf, observed)
            detail = {}
            for method, prediction in predictions.items():
                score = metrics(truth, prediction, APPLICATIONS)
                detail[method] = score
                rows.append({
                    "adaptation_fraction": fraction,
                    "fixed_test_fraction": args.test_fraction,
                    "scenario": scenario, "source": source_level,
                    "target": target_level, "method": method,
                    **{key: score[key] for key in (
                        "n", "accuracy", "balanced_accuracy",
                        "macro_f1", "weighted_f1",
                    )},
                })
            # Test identifiers are deliberately excluded from exported output.
            frame = observed[[
                "timestamp", "traffic_label", "congestion_level",
            ]].copy()
            for method, prediction in predictions.items():
                frame[f"prediction_{method}"] = prediction
            frame.join(result["routes"]).to_csv(
                scenario_dir / "predictions.csv", index=False
            )
            (scenario_dir / "metrics.json").write_text(
                json.dumps(detail, indent=2) + "\n", encoding="utf-8"
            )

    summary = pd.DataFrame(rows).sort_values(
        ["adaptation_fraction", "scenario", "method"]
    )
    for target_level, group in summary.groupby("target"):
        counts = group.groupby("adaptation_fraction").n.unique().map(tuple)
        if counts.map(len).max() != 1 or counts.map(lambda x: x[0]).nunique() != 1:
            raise RuntimeError(f"evaluated test rows are not fixed for {target_level}")
    summary.to_csv(args.output / "meta_stacked_fixed_test_summary.csv", index=False)
    manifest = {
        "config": asdict(config),
        "fractions": args.fractions,
        "fixed_test_fraction": args.test_fraction,
        "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Each scenario uses only its own target calibration prefix. "
            "The final target tail is immutable and excluded from all fitting and selection. "
            "Scaler and KMeans are fitted only on the earliest expert partition and frozen."
        ),
    }
    (args.output / "meta_stacked_fixed_test_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
