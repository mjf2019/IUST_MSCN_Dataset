"""Seven-protocol RF versus DBSCAN-CDR/MF-CDR confirmatory experiment.

The canonical MiniBatchKMeans implementation is not modified.  During this
ablation only, both CDR-MLC routers are replaced by the same development-only
DBSCAN adapter.  Results are printed and checkpointed after every protocol.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import compare_clean_valid as fixed
from adaptive_cdr_mlc import load_dataset
from benchmarks.console_output import print_compact_results
from confirmatory_rf_cdr_mlc import build_evaluations, evaluate_once
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from dbscan_router import make_dbscan_router, dbscan_router_audit
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig
from mixed_level_protocols_leakage_safe import PROTOCOLS


RENAME = {
    "RF_Clean_Valid": "RF_Clean_Valid",
    "Original_CDR_MLC": "DBSCAN_CDR_MLC",
    "MF_CDR_MLC": "DBSCAN_MF_CDR_MLC",
}


@contextmanager
def dbscan_router_patch(eps_grid, min_samples):
    original_factory = fixed.make_minibatch_kmeans
    original_audit = fixed.minibatch_kmeans_audit
    fixed.make_minibatch_kmeans = lambda **kwargs: make_dbscan_router(
        **kwargs, eps_grid=eps_grid, min_samples=min_samples
    )
    fixed.minibatch_kmeans_audit = dbscan_router_audit
    try:
        yield
    finally:
        fixed.make_minibatch_kmeans = original_factory
        fixed.minibatch_kmeans_audit = original_audit


def short_protocol(value):
    if str(value).startswith("Scenario-"):
        return "S" + str(value).split("-")[1]
    return {"LM-H": "S4", "LH-M": "S5", "MH-L": "S6",
            "ALL-80-20": "S7"}.get(str(value), str(value))


def print_protocol(rows):
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["protocol", "method"], sort=False).agg(
        n=("n", "first"),
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        weighted_f1=("weighted_f1", "mean"),
        fit_seconds=("fit_seconds", "mean"),
        predict_seconds=("predict_seconds", "mean"),
        inference_microseconds_per_input_row=(
            "inference_microseconds_per_input_row", "mean"
        ),
        throughput_input_rows_per_second=(
            "throughput_input_rows_per_second", "mean"
        ),
        peak_ram_mb=("peak_ram_mb", "mean"),
        peak_gpu_mb=("peak_gpu_mb", "mean"),
        ttfef_us_per_row=("ttfef_us_per_row", "mean"),
    ).reset_index()
    summary["protocol"] = summary.protocol.map(short_protocol)
    summary["method"] = summary.method.map({
        "RF_Clean_Valid": "RF",
        "DBSCAN_CDR_MLC": "DB-CDR",
        "DBSCAN_MF_CDR_MLC": "DB-MF",
    })
    print_compact_results(summary)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/confirmatory_dbscan_cdr_mlc")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"],
                        default=["1", "2", "3"])
    parser.add_argument("--protocols", nargs="*", choices=list(PROTOCOLS),
                        default=list(PROTOCOLS))
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--target-test-fraction", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--congestion-features", nargs="+",
                        default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    parser.add_argument(
        "--dbscan-eps-grid", nargs="+", type=float,
        default=[.50, .75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00],
    )
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct integers")
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = tuple(dict.fromkeys([
        "TcpRtt", "SynAck", "AckDat", *args.congestion_features
    ]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    evaluations = build_evaluations(
        data, args.scenarios, args.protocols, args.train_fraction,
        args.target_test_fraction,
    )

    all_rows, audits = [], {}
    for number, (development, test, definition) in enumerate(evaluations, 1):
        protocol_rows = []
        print(f"\n[{number}/{len(evaluations)}] START {definition['name']}",
              flush=True)
        for seed in args.seeds:
            config = MetaStackConfig(
                window=args.window,
                congestion_window=args.congestion_window,
                congestion_features=tuple(args.congestion_features),
                expert_trees=args.expert_trees,
                utility_trees=args.utility_trees,
                meta_trees=args.meta_trees,
                random_state=seed,
            ).validate()
            with dbscan_router_patch(
                tuple(args.dbscan_eps_grid), args.dbscan_min_samples
            ):
                rows, audit = evaluate_once(
                    development, test, definition, config, args.rf_trees
                )
            for row in rows:
                row["method"] = RENAME[row["method"]]
            protocol_rows.extend(rows)
            audits[f"{definition['name']}::seed={seed}"] = audit
        all_rows.extend(protocol_rows)
        safe_name = short_protocol(definition["name"])
        pd.DataFrame(protocol_rows).to_csv(
            args.output / f"{safe_name}_runs.csv", index=False
        )
        pd.DataFrame(all_rows).to_csv(
            args.output / "dbscan_confirmatory_runs.csv", index=False
        )
        print(f"[{number}/{len(evaluations)}] COMPLETED {safe_name}", flush=True)
        print_protocol(protocol_rows)

    manifest = {
        "methods": ["RF_Clean_Valid", "DBSCAN_CDR_MLC",
                    "DBSCAN_MF_CDR_MLC"],
        "router": "DBSCANRouter",
        "router_selection": (
            "label-free development-only silhouette among exactly "
            "three-cluster solutions"
        ),
        "out_of_sample_assignment": "nearest fitted DBSCAN core sample",
        "outside_eps_fallback": "nearest fitted DBSCAN core sample",
        "arguments": vars(args),
        "audits": audits,
    }
    manifest["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["arguments"].items()
    }
    (args.output / "dbscan_confirmatory_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nALL {len(evaluations)} PROTOCOLS COMPLETED -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
