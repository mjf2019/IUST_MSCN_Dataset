"""Whole-dataset Argus feature audit for the revised IUST MSCN captures.

This is a dataset-curation diagnostic, not a model-selection benchmark.  It
uses all 15 captures to find schema defects, identifiers, near duplicates,
invalid values, capture fingerprints, application-dominant fields, and
features that respond consistently to congestion within applications.

The script never deletes or rewrites the source .flow files.  Label-aware
scores are reported for review but do not trigger automatic removal.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import normalized_mutual_info_score

from adaptive_cdr_mlc import APPLICATIONS, LEVELS, load_dataset


HARD_EXCLUDE = {
    "StartTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Proto", "Dir",
    "Label", "Cause", "traffic_label", "congestion_level", "sequence_id",
    "source_file", "source_row", "timestamp", "partition", "level_id",
    "route_cluster",
}

# These Argus quantities cannot be negative in a valid decoded record.
NONNEGATIVE_PATTERNS = (
    r"^(Dur|Mean|StdDev|Sum|Min|Max|IdleTime)$",
    r"^(S|D)?(Tot)?Pkts$", r"^(S|D)?(Tot)?Bytes$",
    r"^(Src|Dst)?(Load|Rate|Loss)$", r"^(pLoss|pRetran)$",
    r"^(TcpRtt|SynAck|AckDat|SIntPkt|DIntPkt)$",
)


def epsilon_squared(groups: list[np.ndarray]) -> float | None:
    groups = [g[np.isfinite(g)] for g in groups]
    if len(groups) < 2 or min(map(len, groups)) < 2:
        return None
    if all(np.ptp(g) == 0 for g in groups) and len({float(g[0]) for g in groups}) == 1:
        return 0.0
    try:
        h = float(kruskal(*groups).statistic)
    except ValueError:
        return 0.0
    n, k = sum(map(len, groups)), len(groups)
    return float(max(0.0, (h - k + 1) / max(1, n - k)))


def quantile_codes(values: pd.Series, bins: int = 20) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(-1, index=values.index, dtype=int)
    valid = numeric.notna()
    if valid.sum() < 2 or numeric[valid].nunique() < 2:
        return result
    ranks = numeric[valid].rank(method="average", pct=True)
    result.loc[valid] = np.minimum((ranks * bins).astype(int), bins - 1)
    return result


def nmi_with_missing(codes: pd.Series, labels: pd.Series) -> float:
    return float(normalized_mutual_info_score(labels.astype(str), codes.astype(str)))


def is_expected_nonnegative(column: str) -> bool:
    return any(re.fullmatch(pattern, column, flags=re.IGNORECASE) for pattern in NONNEGATIVE_PATTERNS)


def numeric_audit(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    app = data.traffic_label
    level = data.congestion_level
    capture = data.sequence_id
    for column in columns:
        raw = data[column]
        values = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        missing = float(values.isna().mean())
        unique = int(finite.nunique())
        unique_fraction = float(unique / max(1, len(finite)))
        zero_fraction = float(values.eq(0).mean())
        negative_fraction = float(values.lt(0).mean())

        app_groups = [values[app.eq(label)].dropna().to_numpy() for label in APPLICATIONS]
        app_effect = epsilon_squared(app_groups)
        congestion_by_app = {}
        directions = []
        weights = []
        effects = []
        for label in APPLICATIONS:
            mask = app.eq(label)
            groups = [values[mask & level.eq(congestion)].dropna().to_numpy() for congestion in LEVELS]
            effect = epsilon_squared(groups)
            congestion_by_app[label] = effect
            if effect is not None:
                effects.append(effect)
                weights.append(sum(map(len, groups)))
            medians = [float(np.median(g)) if len(g) else np.nan for g in groups]
            delta = medians[-1] - medians[0]
            directions.append(0 if not np.isfinite(delta) or delta == 0 else int(np.sign(delta)))
        congestion_effect = float(np.average(effects, weights=weights)) if effects else None
        nonzero_directions = [d for d in directions if d]
        direction_consistency = (
            float(max(nonzero_directions.count(1), nonzero_directions.count(-1)) / len(nonzero_directions))
            if nonzero_directions else 0.0
        )
        codes = quantile_codes(values)
        rows.append({
            "feature": column,
            "dtype": str(raw.dtype),
            "rows": len(raw),
            "valid": int(values.notna().sum()),
            "missing_fraction": missing,
            "unique": unique,
            "unique_fraction": unique_fraction,
            "zero_fraction": zero_fraction,
            "negative_fraction": negative_fraction,
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "application_epsilon2": app_effect,
            "congestion_epsilon2_weighted": congestion_effect,
            "congestion_direction_consistency": direction_consistency,
            "application_nmi_binned": nmi_with_missing(codes, app),
            "congestion_nmi_binned": nmi_with_missing(codes, level),
            "capture_nmi_binned": nmi_with_missing(codes, capture),
            **{f"congestion_epsilon2_{label}": congestion_by_app[label] for label in APPLICATIONS},
        })
    return pd.DataFrame(rows)


def categorical_audit(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = data[column].fillna("<MISSING>").astype(str)
        rows.append({
            "feature": column,
            "dtype": str(data[column].dtype),
            "missing_fraction": float(data[column].isna().mean()),
            "unique": int(values.nunique()),
            "unique_fraction": float(values.nunique() / max(1, len(values))),
            "application_nmi": nmi_with_missing(values, data.traffic_label),
            "congestion_nmi": nmi_with_missing(values, data.congestion_level),
            "capture_nmi": nmi_with_missing(values, data.sequence_id),
        })
    return pd.DataFrame(rows)


def correlations(data: pd.DataFrame, columns: list[str], threshold: float, sample_rows: int) -> pd.DataFrame:
    usable = []
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) > 1:
            usable.append(column)
    sample = data[usable]
    if len(sample) > sample_rows:
        sample = sample.sample(sample_rows, random_state=42)
    matrix = sample.apply(pd.to_numeric, errors="coerce").corr(method="spearman", min_periods=20)
    rows = []
    for i, left in enumerate(usable):
        for right in usable[i + 1:]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_1": left, "feature_2": right, "spearman": float(value)})
    return pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "spearman"]
    )


def per_capture_quality(data: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    rows = []
    for sequence, group in data.groupby("sequence_id", sort=True):
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = values.dropna()
            rows.append({
                "capture": sequence,
                "application": group.traffic_label.iloc[0],
                "congestion": group.congestion_level.iloc[0],
                "feature": column,
                "rows": len(group),
                "missing_fraction": float(values.isna().mean()),
                "zero_fraction": float(values.eq(0).mean()),
                "unique": int(finite.nunique()),
                "median": float(finite.median()) if len(finite) else None,
                "q05": float(finite.quantile(.05)) if len(finite) else None,
                "q95": float(finite.quantile(.95)) if len(finite) else None,
            })
    return pd.DataFrame(rows)


def recommendations(data: pd.DataFrame, numeric_report: pd.DataFrame,
                    categorical_report: pd.DataFrame, duplicate_pairs: pd.DataFrame) -> dict:
    columns = list(data.columns)
    hard = sorted(c for c in columns if c in HARD_EXCLUDE or c.lower().startswith("unnamed"))
    quality_remove, quality_review, bias_review, congestion_candidates = [], [], [], []
    for row in numeric_report.itertuples():
        reasons = []
        if row.valid == 0 or row.unique <= 1:
            reasons.append("empty_or_constant")
        if row.missing_fraction >= .95:
            reasons.append("missing_at_least_95_percent")
        if row.unique_fraction >= .98:
            reasons.append("identifier_like_cardinality")
        if is_expected_nonnegative(row.feature) and row.negative_fraction > 0:
            reasons.append("negative_values_in_nonnegative_argus_field")
        if reasons:
            target = quality_remove if set(reasons) <= {"empty_or_constant", "missing_at_least_95_percent"} else quality_review
            target.append({"feature": row.feature, "reasons": reasons})
        app_effect = row.application_epsilon2 if pd.notna(row.application_epsilon2) else 0.0
        congestion_effect = row.congestion_epsilon2_weighted if pd.notna(row.congestion_epsilon2_weighted) else 0.0
        if app_effect >= .50 and congestion_effect < .02:
            bias_review.append({
                "feature": row.feature,
                "reason": "application_dominant_and_congestion_insensitive",
                "application_epsilon2": app_effect,
                "congestion_epsilon2": congestion_effect,
            })
        if congestion_effect >= .02 and row.congestion_direction_consistency >= .60:
            congestion_candidates.append({
                "feature": row.feature,
                "congestion_epsilon2": congestion_effect,
                "direction_consistency": row.congestion_direction_consistency,
            })
    categorical_review = []
    for row in categorical_report.itertuples():
        if row.unique_fraction >= .98 or row.capture_nmi >= .90:
            categorical_review.append({
                "feature": row.feature,
                "reasons": ["identifier_like_or_capture_fingerprint"],
                "unique_fraction": row.unique_fraction,
                "capture_nmi": row.capture_nmi,
            })
    return {
        "automatic_hard_exclude": hard,
        "automatic_quality_remove": quality_remove,
        "manual_argus_quality_review": quality_review,
        "manual_bias_review_not_automatic_removal": bias_review,
        "manual_categorical_review": categorical_review,
        "congestion_sensitive_candidates": sorted(
            congestion_candidates, key=lambda x: x["congestion_epsilon2"], reverse=True
        ),
        "near_duplicate_pairs_for_manual_representative_choice": duplicate_pairs.to_dict("records"),
        "policy": [
            "Labels, addresses, ports, timestamps, capture names and row IDs are always excluded from model inputs.",
            "Empty/constant and >=95% missing fields may be removed as data-quality defects.",
            "High application association alone is not leakage and never causes automatic removal.",
            "Label-aware full-dataset scores are diagnostic; they must not be used to tune a claimed untouched test result.",
            "One capture per application/level means capture artifacts and true level effects are not fully identifiable.",
        ],
    }


def write_report(output: Path, data: pd.DataFrame, audit: pd.DataFrame,
                 numeric: pd.DataFrame, categorical: pd.DataFrame,
                 duplicates: pd.DataFrame, policy: dict) -> None:
    top_bias = numeric.sort_values(
        ["application_epsilon2", "congestion_epsilon2_weighted"], ascending=[False, True]
    ).head(15)
    top_congestion = numeric.sort_values(
        ["congestion_epsilon2_weighted", "congestion_direction_consistency"], ascending=False
    ).head(15)
    lines = [
        "# Whole-dataset feature bias and Argus audit", "",
        f"Rows after service-direction filtering: **{len(data):,}**; captures: **{data.sequence_id.nunique()}**.",
        "", "This is a dataset-curation audit, not evidence that a reduced-feature CDR-MLC beats RF.",
        "No source capture was modified.", "", "## Input audit", "",
        audit.to_markdown(index=False), "", "## Most application-dominant numeric fields", "",
        top_bias[["feature", "application_epsilon2", "congestion_epsilon2_weighted",
                  "capture_nmi_binned", "missing_fraction"]].to_markdown(index=False),
        "", "## Most congestion-sensitive numeric fields", "",
        top_congestion[["feature", "congestion_epsilon2_weighted",
                        "congestion_direction_consistency", "application_epsilon2"]].to_markdown(index=False),
        "", "## Decisions requiring review", "",
        f"- Hard-excluded identifiers/metadata: {len(policy['automatic_hard_exclude'])}",
        f"- Data-quality removals: {len(policy['automatic_quality_remove'])}",
        f"- Argus-quality review: {len(policy['manual_argus_quality_review'])}",
        f"- Application-dominant review: {len(policy['manual_bias_review_not_automatic_removal'])}",
        f"- Near-duplicate pairs: {len(duplicates)}", "",
        "Exact rows and reasons are in the CSV/JSON companion files.", "",
        "## Identification limit", "",
        "There is only one capture for each application × congestion combination. Therefore a field that identifies a capture may reflect real congestion, an Argus/export artifact, or an uncontrolled run condition. A second independent capture is required to separate these explanations conclusively.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/New_Version")
    parser.add_argument("--output", type=Path, default=root / "outputs/feature_bias_argus_audit")
    parser.add_argument("--correlation-threshold", type=float, default=.995)
    parser.add_argument("--correlation-sample", type=int, default=30000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Empty candidates loads/filter all 15 captures without coercing a preselected feature list.
    data, input_audit = load_dataset(args.data_dir, tuple())
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    metadata = set(HARD_EXCLUDE) | {"traffic_label", "congestion_level", "sequence_id"}
    candidate_columns = [c for c in data.columns if c not in metadata]
    numeric_columns, categorical_columns = [], []
    for column in candidate_columns:
        converted = pd.to_numeric(data[column], errors="coerce")
        # Treat as numeric only if most non-missing source values parse as numbers.
        source_valid = data[column].notna().sum()
        if source_valid and converted.notna().sum() / source_valid >= .95:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    numeric_report = numeric_audit(data, numeric_columns)
    categorical_report = categorical_audit(data, categorical_columns)
    duplicate_pairs = correlations(
        data, numeric_columns, args.correlation_threshold, args.correlation_sample
    )
    capture_report = per_capture_quality(data, numeric_columns)
    policy = recommendations(data, numeric_report, categorical_report, duplicate_pairs)

    numeric_report.sort_values("feature").to_csv(args.output / "numeric_feature_audit.csv", index=False)
    categorical_report.sort_values("feature").to_csv(args.output / "categorical_feature_audit.csv", index=False)
    duplicate_pairs.to_csv(args.output / "near_duplicate_features.csv", index=False)
    capture_report.to_csv(args.output / "per_capture_feature_quality.csv", index=False)
    (args.output / "recommended_feature_policy.json").write_text(
        json.dumps(policy, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(
        args.output, data, input_audit, numeric_report, categorical_report,
        duplicate_pairs, policy,
    )
    print(json.dumps({
        "rows": len(data), "captures": int(data.sequence_id.nunique()),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
