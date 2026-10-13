"""Build and evaluate four leakage-safe mixed-congestion protocols.

Protocols
---------
LM-H
    Train on all Low + Medium captures; test on all untouched High captures.
LH-M
    Train on all Low + High captures; test on all untouched Medium captures.
MH-L
    Train on all Medium + High captures; test on all untouched Low captures.
ALL-80-20
    Split every application/level capture chronologically: leading 80% is
    development and trailing 20% is test. Windows never cross the boundary.

The script materializes each combined development/test dataset for auditing and
then compares the strict leakage-safe S-Meta model with the original CDR-MLC
router, utility router, oracle diagnostic, and two pooled RF references.
Nothing is written into the Clean-Valid source directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, LEVELS, load_dataset
from compare_clean_valid import TIMING, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    fit_meta_stacker,
    predict_all,
)
from benchmarks.console_output import print_compact_results


PROTOCOLS = {
    "LM-H": {"train_levels": ("Low", "Medium"), "test_level": "High"},
    "LH-M": {"train_levels": ("Low", "High"), "test_level": "Medium"},
    "MH-L": {"train_levels": ("Medium", "High"), "test_level": "Low"},
    "ALL-80-20": {"train_levels": tuple(LEVELS), "test_level": "all"},
}


def expected_pairs(levels) -> set[tuple[str, str]]:
    return {(application, level) for application in APPLICATIONS for level in levels}


def observed_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame.traffic_label.astype(str), frame.congestion_level.astype(str)))


def validate_protocol_composition(
    name: str, development: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Fail loudly if an application/level capture is missing or misplaced."""
    if name == "ALL-80-20":
        expected_development = expected_pairs(LEVELS)
        expected_test = expected_pairs(LEVELS)
    else:
        specification = PROTOCOLS[name]
        expected_development = expected_pairs(specification["train_levels"])
        expected_test = expected_pairs((specification["test_level"],))

    for partition, frame, expected in (
        ("development", development, expected_development),
        ("test", test, expected_test),
    ):
        observed = observed_pairs(frame)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise ValueError(
                f"{name} {partition}: missing={missing}, unexpected={unexpected}"
            )


def frame_identity(frame: pd.DataFrame) -> str:
    """Hash ordered record identities without hashing predictions."""
    ordered = frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    )[["sequence_id", "source_file", "source_row"]]
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chronological_all_level_split(data: pd.DataFrame, train_fraction: float):
    """Split each of the 15 captures independently and chronologically."""
    development, test, captures = [], [], []
    for sequence_id, group in data.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * train_fraction)
        if not 0 < cut < len(group):
            raise ValueError(f"{sequence_id}: invalid chronological split")
        left = group.iloc[:cut].copy()
        right = group.iloc[cut:].copy()
        development.append(left)
        test.append(right)
        captures.append({
            "sequence_id": sequence_id,
            "application": str(group.iloc[0].traffic_label),
            "congestion_level": str(group.iloc[0].congestion_level),
            "total_rows": len(group),
            "development_rows": len(left),
            "test_rows": len(right),
            "development_last_source_row": int(left.iloc[-1].source_row),
            "test_first_source_row": int(right.iloc[0].source_row),
        })
    return (
        pd.concat(development, ignore_index=True),
        pd.concat(test, ignore_index=True),
        captures,
    )


def build_protocol(data: pd.DataFrame, name: str, train_fraction: float):
    specification = PROTOCOLS[name]
    if name == "ALL-80-20":
        development, test, captures = chronological_all_level_split(
            data, train_fraction
        )
        protocol = {
            "kind": "within_capture_chronological_split",
            "train_fraction": train_fraction,
            "test_fraction": 1.0 - train_fraction,
            "captures": captures,
        }
    else:
        train_levels = specification["train_levels"]
        test_level = specification["test_level"]
        development = data[data.congestion_level.isin(train_levels)].copy()
        test = data[data.congestion_level.eq(test_level)].copy()
        protocol = {
            "kind": "two_complete_levels_to_untouched_third_level",
            "train_levels": list(train_levels),
            "test_level": test_level,
            "target_level_used_in_training_or_selection": False,
        }
    development = development.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_ids = set(zip(development.source_file, development.source_row))
    test_ids = set(zip(test.source_file, test.source_row))
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"{name}: {len(overlap)} train/test record overlaps")
    if development.empty or test.empty:
        raise ValueError(f"{name}: empty development or test data")
    validate_protocol_composition(name, development, test)
    return development, test, protocol


def composition(frame: pd.DataFrame) -> list[dict]:
    table = (
        frame.groupby(["traffic_label", "congestion_level"], observed=True)
        .size().rename("rows").reset_index()
    )
    return table.to_dict(orient="records")


def materialize(frame: pd.DataFrame, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    ).to_csv(destination, index=False)
    hasher = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return {"path": str(destination), "rows": len(frame), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/mixed_level_protocols_leakage_safe",
    )
    parser.add_argument(
        "--protocols", nargs="+", choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
    )
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument(
        "--congestion-features", nargs="+",
        default=list(DEFAULT_CONGESTION_FEATURES),
    )
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-datasets", action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("train-fraction must be in (0,1)")

    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    rows, audits = [], {}
    for protocol_name in args.protocols:
        development, test, protocol = build_protocol(
            data, protocol_name, args.train_fraction
        )
        protocol_dir = args.output / protocol_name.lower()
        protocol_dir.mkdir(parents=True, exist_ok=True)
        exported = {}
        if args.export_datasets:
            exported = {
                "development": materialize(
                    development, protocol_dir / "datasets" / "train.csv"
                ),
                "test": materialize(
                    test, protocol_dir / "datasets" / "test.csv"
                ),
            }

        model = fit_meta_stacker(development, config)
        eligible = development.loc[model["source_eligible_index"]]
        rf_clean_valid = fit_rf(eligible, (), args.seed, args.rf_trees)
        rf_expert_inputs = fit_rf(
            eligible, TIMING, args.seed, args.rf_trees
        )
        result = predict_all(model, test)
        observed = result["observed"]
        truth = observed.traffic_label.to_numpy()
        predictions = {
            key: value for key, value in result.items()
            if key.startswith("CDR_")
        }
        predictions["RF_Clean_Valid"] = predict_rf(
            rf_clean_valid, observed
        )
        predictions["RF_Expert_Inputs"] = predict_rf(
            rf_expert_inputs, observed
        )

        details = {}
        for method, prediction in predictions.items():
            score = metrics(truth, prediction, APPLICATIONS)
            details[method] = score
            rows.append({
                "protocol": protocol_name,
                "method": method,
                **{key: score[key] for key in (
                    "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "weighted_f1",
                )},
            })
        (protocol_dir / "metrics.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )

        prediction_frame = observed[[
            "timestamp", "traffic_label", "congestion_level",
        ]].copy()
        for method, prediction in predictions.items():
            prediction_frame[f"prediction_{method}"] = prediction
        prediction_frame.join(result["routes"]).to_csv(
            protocol_dir / "predictions.csv", index=False
        )

        audit = {
            "protocol": protocol,
            "development_rows_raw": len(development),
            "test_rows_raw": len(test),
            "evaluated_test_rows": len(observed),
            "development_identity": frame_identity(development),
            "test_identity": frame_identity(test),
            "evaluated_test_identity": frame_identity(observed),
            "development_composition": composition(development),
            "test_composition": composition(test),
            "exported_datasets": exported,
            "partition_rows": model["partition_rows"],
            "leakage_control": model["leakage_control"],
            "utility_correctness_counts": model["utility_correctness_counts"],
            "selected_meta_variant": model["selected_meta_variant"],
            "selected_meta_confidence": model["selected_meta_confidence"],
            "meta_selection_trials": model["meta_selection_trials"],
        }
        audits[protocol_name] = audit
        (protocol_dir / "protocol_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    summary = pd.DataFrame(rows).sort_values(["protocol", "method"])
    summary.to_csv(args.output / "mixed_level_summary.csv", index=False)
    manifest = {
        "config": asdict(config),
        "protocols": args.protocols,
        "all_level_train_fraction": args.train_fraction,
        "rf_trees": args.rf_trees,
        "export_datasets": args.export_datasets,
        "oracle_is_diagnostic_not_deployable": True,
        "source_files_are_never_modified": True,
        "audits": audits,
    }
    (args.output / "mixed_level_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(summary)


if __name__ == "__main__":
    main()
