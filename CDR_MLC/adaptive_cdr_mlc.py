"""Adaptive CDR-MLC with label-specific feature/window selection and soft gating.

Research extension of CDR-MLC; it is not the fixed-feature paper algorithm.
Selection uses train/validation only. The reserved test split is evaluated once.
At inference no true application or congestion label is read.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APPLICATIONS = ["HTTP", "SFTP", "SMTP", "SSH", "Video"]
LEVELS = ["Low", "Medium", "High"]
STATS = ["mean", "max", "median", "min", "std"]
SERVICES = {
    "HTTP": ("192.168.2.122", 8080),
    "SFTP": ("192.168.2.120", 22),
    "SMTP": ("192.168.2.120", 8025),
    "SSH": ("192.168.2.120", 22),
    "Video": ("192.168.2.121", 5000),
}
CLIENT = "192.168.1.111"

# Only observed, numeric flow measurements may be selected for congestion routing.
# Add/remove names here if the capture schema changes. Identifiers and labels are forbidden.
DEFAULT_CANDIDATES = [
    "TcpRtt", "SynAck", "AckDat", "Dur", "Mean", "StdDev", "Sum", "Min", "Max",
    "SIntPkt", "DIntPkt", "SrcLoad", "DstLoad", "Load", "SrcRate", "DstRate", "Rate",
    "pLoss", "pRetran", "SrcLoss", "DstLoss", "Loss", "PCRatio",
]
FORBIDDEN = {
    "StartTime", "SrcAddr", "DstAddr", "Proto", "Sport", "Dport", "Label", "Cause",
    "Dir", "traffic_label", "congestion_level", "capture_id", "sequence_id", "source_file", "source_row",
    "timestamp", "partition", "level_id", "route_cluster", "IdleTime",
}
CATEGORICAL = ["Flgs", "State", "TcpOpt"]
TEXT_COLUMNS = {
    "StartTime", "SrcAddr", "DstAddr", "Proto", "Flgs", "State", "TcpOpt",
    "Label", "Cause", "Dir",
}


@dataclass(frozen=True)
class Config:
    windows: tuple[int, ...] = (3, 5, 10, 20, 50, 100)
    candidates: tuple[str, ...] = tuple(DEFAULT_CANDIDATES)
    ranking_top_k: int = 6
    selection_seeds: tuple[int, ...] = (21, 42, 84)
    random_state: int = 42
    n_clusters: int = 3
    batch_size: int = 1024
    n_init: int = 10
    max_iter: int = 100
    gate_estimators: int = 100
    expert_estimators: int = 20
    min_confidence: float = 0.45
    gating: str = "soft"  # soft or hard
    train_fraction: float = 0.60
    validation_fraction: float = 0.20

    def validate(self):
        if not self.windows or any(isinstance(w, bool) or w < 1 for w in self.windows):
            raise ValueError("windows must contain positive integers")
        if len(set(self.candidates)) < 3 or self.ranking_top_k < 3:
            raise ValueError("at least three candidate features are required")
        if self.gating not in {"soft", "hard"}:
            raise ValueError("gating must be soft or hard")
        if not 0 < self.min_confidence <= 1:
            raise ValueError("min_confidence must be in (0, 1]")
        if self.train_fraction <= 0 or self.validation_fraction <= 0:
            raise ValueError("split fractions must be positive")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("a nonempty reserved test partition is required")
        return self


def load_dataset(data_dir: Path, candidates: tuple[str, ...]):
    """Load 15 captures, filter service direction, and retain capture chronology."""
    frames, audit = [], []
    for capture_number, path in enumerate(sorted(data_dir.glob("*.flow")), start=1):
        match = re.fullmatch(r"(HTTP|SFTP|SMTP|SSH|Video)_(Low|Medium|High)", path.stem)
        if not match:
            raise ValueError(f"unexpected capture filename: {path.name}")
        application, level = match.groups()
        server, port = SERVICES[application]
        frame = pd.read_csv(path, low_memory=False, on_bad_lines="error")
        frame.columns = frame.columns.str.strip()
        required = {"StartTime", "SrcAddr", "DstAddr", "Proto", "Sport", "Dport"}
        if required - set(frame):
            raise ValueError(f"{path.name}: missing {sorted(required-set(frame))}")
        # Restrict string cleanup to known Argus text fields. With recent
        # pandas versions, mixed numeric columns may have object dtype and a
        # blanket .str.strip() becomes a slow Python-element loop.
        for column in TEXT_COLUMNS & set(frame.columns):
            frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
        frame["source_row"] = np.arange(len(frame)) + 2
        dport = pd.to_numeric(frame.Dport, errors="coerce")
        sport = pd.to_numeric(frame.Sport, errors="coerce")
        tcp = frame.Proto.eq("tcp")
        forward = tcp & frame.SrcAddr.eq(CLIENT) & frame.DstAddr.eq(server) & dport.eq(port)
        reverse = tcp & frame.SrcAddr.eq(server) & frame.DstAddr.eq(CLIENT) & sport.eq(port)
        if reverse.any():
            raise ValueError(f"{path.name}: reverse rows require canonicalization")
        raw_rows = len(frame)
        frame = frame.loc[forward].copy()
        frame["timestamp"] = pd.to_datetime(
            frame.StartTime, format="%Y/%m/%d %H:%M:%S.%f", errors="raise"
        )
        for column in set(candidates) & set(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

        # Capture boundaries and flow boundaries have different roles.
        # Temporal partitions are created per capture_id, while rolling
        # statistics are computed independently inside each standard 5-tuple.
        # Neither identifier contains or is derived from the traffic label.
        frame["capture_id"] = f"capture_{capture_number:02d}"
        tuple_columns = ["SrcAddr", "DstAddr", "Sport", "Dport", "Proto"]
        tuple_view = frame[tuple_columns].copy()
        tuple_view["SrcAddr"] = tuple_view["SrcAddr"].astype("string").str.strip()
        tuple_view["DstAddr"] = tuple_view["DstAddr"].astype("string").str.strip()
        tuple_view["Proto"] = tuple_view["Proto"].astype("string").str.lower().str.strip()
        tuple_view["Sport"] = pd.to_numeric(
            tuple_view["Sport"], errors="coerce"
        ).astype("Int64").astype("string")
        tuple_view["Dport"] = pd.to_numeric(
            tuple_view["Dport"], errors="coerce"
        ).astype("Int64").astype("string")
        invalid_tuple = tuple_view.isna().any(axis=1)
        if invalid_tuple.any():
            raise ValueError(
                f"{path.name}: {int(invalid_tuple.sum())} retained rows have an incomplete 5-tuple"
            )
        frame["sequence_id"] = (
            frame["capture_id"].astype("string") + "|"
            + tuple_view["SrcAddr"] + "|" + tuple_view["DstAddr"] + "|"
            + tuple_view["Sport"] + "|" + tuple_view["Dport"] + "|"
            + tuple_view["Proto"]
        )
        frame["traffic_label"] = application
        frame["congestion_level"] = level
        frame["source_file"] = path.name

        sequence_sizes = frame.groupby("sequence_id", sort=False).size()
        audit.append({
            "capture_id": frame["capture_id"].iat[0],
            "file": path.name,
            "raw": raw_rows,
            "retained": len(frame),
            "five_tuple_sequences": int(sequence_sizes.size),
            "sequence_rows_min": int(sequence_sizes.min()),
            "sequence_rows_median": float(sequence_sizes.median()),
            "sequence_rows_max": int(sequence_sizes.max()),
            "sequences_shorter_than_window_3": int((sequence_sizes < 3).sum()),
        })
        frames.append(frame.sort_values(["timestamp", "source_row"], kind="stable"))
    if len(frames) != 15:
        raise ValueError(f"expected 15 captures, found {len(frames)}")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit)


def temporal_split(data: pd.DataFrame, config: Config):
    """Split each capture chronologically; 5-tuple windows reset in every partition."""
    parts = {"train": [], "validation": [], "test": []}
    for _, group in data.groupby("capture_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        first = int(config.train_fraction * len(group))
        second = int((config.train_fraction + config.validation_fraction) * len(group))
        if not 0 < first < second < len(group):
            raise ValueError(f"capture too short: {group.capture_id.iloc[0]}")
        parts["train"].append(group.iloc[:first].copy())
        parts["validation"].append(group.iloc[first:second].copy())
        parts["test"].append(group.iloc[second:].copy())
    return {name: pd.concat(groups, ignore_index=True) for name, groups in parts.items()}


def trend_frame(frame: pd.DataFrame, features, window: int):
    """Causal, complete trailing windows; invalid values break the window."""
    columns = [f"{feature}_{stat}" for feature in features for stat in STATS]
    chunks = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable").copy()
        values = group[list(features)].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(values).all(axis=1) & values.ge(0).all(axis=1)
        segments = (~valid).cumsum()
        for _, segment in group.loc[valid].groupby(segments[valid], sort=False):
            values = segment[list(features)].apply(pd.to_numeric, errors="coerce")
            output = {}
            for feature in features:
                rolling = values[feature].rolling(window, min_periods=window)
                for stat in STATS:
                    output[f"{feature}_{stat}"] = (
                        rolling.std(ddof=0) if stat == "std" else getattr(rolling, stat)()
                    )
            matrix = pd.DataFrame(output, index=segment.index).dropna()
            if len(matrix):
                joined = segment.loc[matrix.index].copy()
                joined[columns] = matrix
                chunks.append(joined)
    if not chunks:
        return frame.iloc[:0].assign(**{c: pd.Series(dtype=float) for c in columns})
    return pd.concat(chunks).sort_index(kind="stable")


def rank_candidates(training: pd.DataFrame, application: str, candidates):
    """Marginal training-only congestion ranking; selection is finalized on validation."""
    subset = training[training.traffic_label.eq(application)]
    scores = []
    for feature in candidates:
        if feature not in subset or feature in FORBIDDEN:
            continue
        groups = [
            pd.to_numeric(subset.loc[subset.congestion_level.eq(level), feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            for level in LEVELS
        ]
        if min(map(len, groups)) < 10:
            continue
        try:
            statistic = float(kruskal(*groups).statistic)
        except ValueError:
            continue
        n = sum(map(len, groups))
        epsilon2 = max(0.0, (statistic - len(LEVELS) + 1) / (n - len(LEVELS)))
        scores.append({"application": application, "feature": feature, "epsilon2": epsilon2})
    ranking = pd.DataFrame(scores).sort_values(["epsilon2", "feature"], ascending=[False, True])
    if len(ranking) < 3:
        raise ValueError(f"{application}: fewer than three usable candidate features")
    return ranking.reset_index(drop=True)


def fit_router(matrix: pd.DataFrame, config: Config, seed: int):
    columns = [c for c in matrix if any(c.startswith(f + "_") for f in config.candidates)]
    x = matrix[columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    router = MiniBatchKMeans(
        n_clusters=config.n_clusters, batch_size=config.batch_size, n_init=config.n_init,
        max_iter=config.max_iter, random_state=seed,
    ).fit(z)
    return scaler, router, router.predict(z), columns


def select_configurations(parts, config: Config, output: Path):
    """Jointly choose three fields and W separately for each known training label."""
    level_id = {level: i for i, level in enumerate(LEVELS)}
    selections, trials, rankings = {}, [], []
    for application in APPLICATIONS:
        ranking = rank_candidates(parts["train"], application, config.candidates)
        rankings.append(ranking)
        top = ranking.feature.head(min(config.ranking_top_k, len(ranking))).tolist()
        best = None
        for features in itertools.combinations(top, 3):
            for window in config.windows:
                train = trend_frame(
                    parts["train"][parts["train"].traffic_label.eq(application)], features, window
                )
                validation = trend_frame(
                    parts["validation"][parts["validation"].traffic_label.eq(application)],
                    features, window,
                )
                if len(train) < config.n_clusters or validation.empty:
                    trials.append({"application": application, "features": "|".join(features),
                                   "window": window, "status": "insufficient_windows"})
                    continue
                y_validation = validation.congestion_level.map(level_id).to_numpy()
                seed_scores = []
                for seed in config.selection_seeds:
                    scaler, router, _, columns = fit_router(train, config, seed)
                    clusters = router.predict(scaler.transform(validation[columns]))
                    score = float(adjusted_rand_score(y_validation, clusters))
                    seed_scores.append(score)
                    trials.append({"application": application, "features": "|".join(features),
                                   "window": window, "seed": seed, "validation_ari": score,
                                   "train_n": len(train), "validation_n": len(validation), "status": "ok"})
                candidate = (float(np.mean(seed_scores)), -float(np.std(seed_scores)), -window, features)
                if best is None or candidate > best[0]:
                    best = (candidate, {"features": list(features), "window": int(window),
                                       "validation_ari_mean": float(np.mean(seed_scores)),
                                       "validation_ari_std": float(np.std(seed_scores))})
        if best is None:
            raise ValueError(f"{application}: no valid feature/window configuration")
        selections[application] = best[1]
    pd.concat(rankings).to_csv(output / "training_feature_ranking.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "selection_trials.csv", index=False)
    (output / "selected_configurations.json").write_text(json.dumps(selections, indent=2) + "\n")
    return selections


def select_classifier_columns(frame, route_features):
    """Build label-free gate/expert inputs and exclude all adaptive router fields."""
    numeric, categorical = [], []
    excluded = FORBIDDEN | set(route_features)
    for column in frame.columns:
        if column in excluded or re.search(r"\.\d+$", column):
            continue
        if column in CATEGORICAL:
            if frame[column].nunique(dropna=True) > 1:
                categorical.append(column)
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            numeric.append(column)
    return numeric, categorical


def make_preprocessor(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("no usable classifier features")
    return ColumnTransformer(transformers, remainder="drop")


def build_route_views(frame, selections):
    return {
        application: trend_frame(frame, spec["features"], spec["window"])
        for application, spec in selections.items()
    }


def common_rows(views):
    common = None
    for view in views.values():
        common = set(view.index) if common is None else common & set(view.index)
    return sorted(common or [])


def fit_final(parts, selections, config: Config):
    """Refit gate, label-specific routers, and 3 multiclass experts per router."""
    # Fresh unique row IDs are required because train/validation were indexed independently.
    development = pd.concat(
        [parts["train"], parts["validation"]], ignore_index=True
    )
    views = build_route_views(development, selections)
    rows = common_rows(views)
    if not rows:
        raise ValueError("no common development rows across selected windows")
    development = development.loc[rows]
    route_features = sorted({f for spec in selections.values() for f in spec["features"]})
    numeric, categorical = select_classifier_columns(development, route_features)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(development[numeric + categorical])
    labels = development.traffic_label.to_numpy()
    gate = RandomForestClassifier(
        n_estimators=config.gate_estimators, class_weight="balanced",
        random_state=config.random_state, n_jobs=-1,
    ).fit(x, labels)
    banks = {}
    for application in APPLICATIONS:
        view = views[application].loc[rows]
        feature_columns = [
            f"{feature}_{stat}" for feature in selections[application]["features"] for stat in STATS
        ]
        # The router configuration and its geometry are learned only from this
        # application's observed development rows across the three levels.
        own_label = development.traffic_label.eq(application).to_numpy()
        scaler = StandardScaler().fit(view.loc[own_label, feature_columns])
        own_z = scaler.transform(view.loc[own_label, feature_columns])
        router = MiniBatchKMeans(
            n_clusters=config.n_clusters, batch_size=config.batch_size, n_init=config.n_init,
            max_iter=config.max_iter, random_state=config.random_state,
        ).fit(own_z)
        # Apply the frozen label-specific router to every development row so
        # each routed expert remains a multiclass application classifier.
        z = scaler.transform(view[feature_columns])
        routes = router.predict(z)
        experts = {}
        for cluster in range(config.n_clusters):
            mask = routes == cluster
            if not mask.any():
                raise ValueError(f"{application}: empty final cluster {cluster}")
            experts[cluster] = RandomForestClassifier(
                n_estimators=config.expert_estimators, class_weight="balanced",
                random_state=config.random_state, n_jobs=-1,
            ).fit(x[mask], labels[mask])
        banks[application] = {
            "features": selections[application]["features"],
            "window": selections[application]["window"],
            "columns": feature_columns,
            "scaler": scaler, "router": router, "experts": experts,
        }
    return {
        "config": asdict(config), "selections": selections, "preprocessor": preprocessor,
        "numeric": numeric, "categorical": categorical, "gate": gate, "banks": banks,
        "classes": gate.classes_.tolist(), "development_rows": len(rows),
    }


def aligned_probabilities(estimator, x, classes):
    raw = estimator.predict_proba(x)
    result = np.zeros((x.shape[0], len(classes)))
    positions = {label: i for i, label in enumerate(classes)}
    for source, label in enumerate(estimator.classes_):
        result[:, positions[label]] = raw[:, source]
    return result


def predict(model, frame):
    """Predict without consulting true traffic or congestion labels."""
    views = {
        application: trend_frame(frame, bank["features"], bank["window"])
        for application, bank in model["banks"].items()
    }
    rows = common_rows(views)
    if not rows:
        raise ValueError("no complete common inference windows")
    observed = frame.loc[rows]
    x = model["preprocessor"].transform(observed[model["numeric"] + model["categorical"]])
    classes = model["classes"]
    gate = aligned_probabilities(model["gate"], x, classes)
    if model["config"]["gating"] == "hard":
        hard = np.zeros_like(gate)
        hard[np.arange(len(gate)), gate.argmax(axis=1)] = 1.0
        gate = hard
    # Optional uncertainty fallback: preserve soft weights below confidence threshold.
    confidence = gate.max(axis=1)
    mixture = np.zeros((len(rows), len(classes)))
    route_report = pd.DataFrame(index=rows)
    for app_index, application in enumerate(classes):
        bank = model["banks"][application]
        view = views[application].loc[rows]
        z = bank["scaler"].transform(view[bank["columns"]])
        routes = bank["router"].predict(z)
        probabilities = np.zeros_like(mixture)
        for cluster, expert in bank["experts"].items():
            mask = routes == cluster
            if mask.any():
                probabilities[mask] = aligned_probabilities(expert, x[mask], classes)
        weights = gate[:, app_index]
        mixture += weights[:, None] * probabilities
        route_report[f"gate_{application}"] = weights
        route_report[f"cluster_{application}"] = routes
    zero = mixture.sum(axis=1) == 0
    if zero.any():
        mixture[zero] = gate[zero]
    mixture /= mixture.sum(axis=1, keepdims=True)
    prediction = np.asarray(classes)[mixture.argmax(axis=1)]
    route_report["gate_confidence"] = confidence
    route_report["low_confidence"] = confidence < model["config"]["min_confidence"]
    route_report["prediction"] = prediction
    route_report["source_file"] = observed.source_file.to_numpy()
    route_report["source_row"] = observed.source_row.to_numpy()
    return prediction, mixture, route_report, observed


def evaluate_reserved(model, test, output):
    prediction, probabilities, routes, observed = predict(model, test)
    truth = observed.traffic_label.to_numpy()
    metrics = {
        "n": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "classification_report": classification_report(truth, prediction, output_dict=True, zero_division=0),
        "confusion_matrix_labels": model["classes"],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=model["classes"]).tolist(),
    }
    routes["true_label"] = truth
    for i, label in enumerate(model["classes"]):
        routes[f"probability_{label}"] = probabilities[:, i]
    routes.to_csv(output / "reserved_predictions.csv", index=False)
    (output / "reserved_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/New_Version")
    parser.add_argument("--output", type=Path, default=root / "outputs/adaptive_cdr_mlc")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 10, 20, 50, 100])
    parser.add_argument("--ranking-top-k", type=int, default=6)
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[21, 42, 84])
    parser.add_argument("--gating", choices=["soft", "hard"], default="soft")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()
    config = Config(
        windows=tuple(args.windows), candidates=tuple(args.candidates),
        ranking_top_k=args.ranking_top_k, selection_seeds=tuple(args.selection_seeds),
        gating=args.gating,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_dataset(args.data_dir, config.candidates)
    audit.to_csv(args.output / "input_audit.csv", index=False)
    parts = temporal_split(data, config)
    selections = select_configurations(parts, config, args.output)
    model = fit_final(parts, selections, config)
    joblib.dump(model, args.output / "adaptive_cdr_mlc.joblib")
    metrics = evaluate_reserved(model, parts["test"], args.output)
    (args.output / "run_manifest.json").write_text(json.dumps({
        "config": asdict(config), "selected": selections,
        "development_rows": model["development_rows"],
        "reserved_rows_scored": metrics["n"],
        "warning": "Research extension; true labels are not inference inputs.",
    }, indent=2) + "\n")
    print(json.dumps({"selected": selections, "reserved_accuracy": metrics["accuracy"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
