"""Compare fixed and congestion-oriented CDR-MLC routers in three scenarios."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import (
    SCENARIOS, TIMING, build_calibrated_protocol, fit_fixed_cdr, fit_rf,
    metrics, predict_fixed_cdr, predict_rf,
)
from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig,
    fit_congestion_cdr, predict_all,
)


LEVEL_ID = {"Low": 0, "Medium": 1, "High": 2}


def clustering_audit(observed, routes):
    levels = observed.congestion_level.map(LEVEL_ID).to_numpy()
    cluster = routes.congestion_route.to_numpy()
    table = pd.crosstab(
        pd.Series(observed.congestion_level.to_numpy(), name="level"),
        pd.Series(cluster, name="cluster"),
    )
    return {
        "ari_vs_level": float(adjusted_rand_score(levels, cluster)),
        "nmi_vs_level": float(normalized_mutual_info_score(levels, cluster)),
        "router_oracle_agreement": float(routes.router_matches_oracle.mean()),
        "cluster_by_level": {
            str(level): {str(key): int(value) for key, value in row.items()}
            for level, row in table.to_dict(orient="index").items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/congestion_feature_router")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = CongestionRouterConfig(
        window=args.window, features=tuple(args.features),
        expert_trees=args.expert_trees, random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(args.features))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    rows, audits = [], {}
    for fraction in args.fractions:
        grouped = {}
        for scenario in args.scenarios:
            source, target = SCENARIOS[scenario]
            grouped.setdefault(source, []).append((scenario, target))
        fraction_root = args.output / f"cal_{int(round(fraction * 100)):02d}"
        fraction_root.mkdir(parents=True, exist_ok=True)
        for source_level, targets in grouped.items():
            development, tails, calibration = build_calibrated_protocol(
                data, source_level, fraction
            )
            model = fit_congestion_cdr(development, config)
            fixed = fit_fixed_cdr(development, 3, args.seed, args.expert_trees)
            eligible = development.loc[model["source_eligible_index"]]
            # Identical exclusions to the fixed and congestion-router experts.
            rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
            source_key = f"{fraction}:{source_level}"
            audits[source_key] = {
                "calibration": calibration,
                "route_features": model["route_features"],
                "router_columns": model["router_columns"],
                "cluster_counts": model["cluster_counts"],
            }
            for scenario, target_level in targets:
                result = predict_all(model, tails[target_level])
                observed = result["observed"]
                truth = observed.traffic_label.to_numpy()
                predictions = {
                    "CDR_MLC_fixed_router": predict_fixed_cdr(fixed, tails[target_level]).loc[observed.index].to_numpy(),
                    "CDR_MLC_congestion_router": result["CDR_MLC_congestion_router"],
                    "CDR_MLC_oracle_router": result["CDR_MLC_oracle_router"],
                    "RF_expert_inputs": predict_rf(rf, observed),
                }
                scenario_dir = fraction_root / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
                scenario_dir.mkdir(parents=True, exist_ok=True)
                detail = {}
                for method, prediction in predictions.items():
                    score = metrics(truth, prediction, APPLICATIONS)
                    detail[method] = score
                    rows.append({
                        "adaptation_fraction": fraction, "scenario": scenario,
                        "source": source_level, "target": target_level, "method": method,
                        **{key: score[key] for key in ("n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
                    })
                audit = clustering_audit(observed, result["routes"])
                detail["clustering_audit"] = audit
                frame = observed[["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]].copy()
                for method, prediction in predictions.items():
                    frame[f"prediction_{method}"] = prediction
                frame = frame.join(result["routes"])
                frame.to_csv(scenario_dir / "predictions.csv", index=False)
                (scenario_dir / "metrics.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")

    summary = pd.DataFrame(rows).sort_values(["adaptation_fraction", "scenario", "method"])
    summary.to_csv(args.output / "congestion_feature_router_summary.csv", index=False)
    manifest = {
        "config": asdict(config), "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "All router descriptors use causal trailing windows. Neither traffic_label nor "
            "congestion_level is used to build, scale, fit, or invoke the KMeans router."
        ),
    }
    (args.output / "congestion_feature_router_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
