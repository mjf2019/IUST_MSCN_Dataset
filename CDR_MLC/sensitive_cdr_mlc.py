"""Source-only sensitivity-selected and feature-modulated CDR-MLC.

This experimental extension never reads target congestion labels. It selects a
router feature triplet and window on a chronological source validation split,
weights source samples by unsupervised temporal severity, and blends soft-routed
experts with a global RF fallback.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from adaptive_cdr_mlc import (
    APPLICATIONS, aligned_probabilities, make_preprocessor,
    select_classifier_columns, trend_frame,
)


@dataclass(frozen=True)
class SensitiveConfig:
    candidates: tuple[str, ...] = (
        "TcpRtt", "SynAck", "AckDat", "SrcRate", "DstRate",
        "SrcLoad", "DstLoad", "Load", "Rate", "Dur",
    )
    windows: tuple[int, ...] = (3, 10, 20)
    ranking_top_k: int = 5
    random_state: int = 42
    n_clusters: int = 3
    n_init: int = 10
    max_iter: int = 100
    batch_size: int = 1024
    expert_trees: int = 20
    global_trees: int = 100
    modulation_strength: float = 1.0
    global_blend: float = 0.25
    soft_temperature: float = 1.0
    min_cluster_fraction: float = 0.10

    def validate(self):
        if len(set(self.candidates)) < 3 or self.ranking_top_k < 3:
            raise ValueError("at least three sensitivity candidates are required")
        if not self.windows or min(self.windows) < 1:
            raise ValueError("windows must be positive")
        if not 0 <= self.global_blend <= 1:
            raise ValueError("global_blend must be in [0,1]")
        if self.soft_temperature <= 0 or self.modulation_strength < 0:
            raise ValueError("temperature must be positive and modulation nonnegative")
        if not 0 < self.min_cluster_fraction < 1 / self.n_clusters:
            raise ValueError("invalid min_cluster_fraction")
        return self


def chronological_split(source: pd.DataFrame, fraction: float = .75):
    train, validation = [], []
    for _, group in source.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * fraction)
        if not 0 < cut < len(group):
            raise ValueError("source capture is too short")
        train.append(group.iloc[:cut].copy())
        validation.append(group.iloc[cut:].copy())
    return pd.concat(train), pd.concat(validation)


def temporal_sensitivity_ranking(train: pd.DataFrame, config: SensitiveConfig):
    """Rank source fields by robust within-sequence movement, without labels."""
    rows = []
    for feature in config.candidates:
        scores = []
        for _, group in train.groupby("sequence_id", sort=False):
            values = pd.to_numeric(group[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(values) < 20 or values.nunique() < 3:
                continue
            q25, q75 = values.quantile([.25, .75])
            scale = float(q75 - q25)
            if scale <= 0:
                scale = float(values.std(ddof=0))
            if scale <= 0:
                continue
            movement = float(values.diff().abs().median() / scale)
            excursion = float((values.quantile(.95) - values.quantile(.05)) / scale)
            scores.append(np.log1p(movement) + np.log1p(excursion))
        if scores:
            rows.append({
                "feature": feature,
                "source_temporal_sensitivity": float(np.median(scores)),
                "sequences": len(scores),
            })
    ranking = pd.DataFrame(rows).sort_values(
        ["source_temporal_sensitivity", "feature"], ascending=[False, True]
    )
    if len(ranking) < 3:
        raise ValueError("fewer than three usable sensitivity features")
    return ranking.reset_index(drop=True)


def _rf(trees, seed):
    return RandomForestClassifier(
        n_estimators=trees, criterion="gini", max_features="sqrt",
        min_samples_leaf=1, bootstrap=True, n_jobs=-1, random_state=seed,
    )


def _soft_routes(z, centers, temperature):
    distance = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    scale = np.median(distance[np.isfinite(distance)])
    scale = max(float(scale), 1e-12) * temperature
    logits = -distance / scale
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _severity(z, strength):
    raw = np.sqrt(np.mean(np.square(z), axis=1))
    cap = max(float(np.quantile(raw, .95)), 1e-12)
    return 1.0 + strength * np.clip(raw / cap, 0, 1)


def fit_model(train: pd.DataFrame, features, window: int, config: SensitiveConfig):
    view = trend_frame(train, features, window)
    columns = [f"{feature}_{stat}" for feature in features
               for stat in ("mean", "max", "median", "min", "std")]
    if len(view) < 30:
        raise ValueError("insufficient complete trend windows")
    scaler = StandardScaler().fit(view[columns])
    z = scaler.transform(view[columns])
    router = MiniBatchKMeans(
        n_clusters=config.n_clusters, batch_size=config.batch_size,
        n_init=config.n_init, max_iter=config.max_iter,
        random_state=config.random_state,
    ).fit(z)
    hard = router.predict(z)
    counts = np.bincount(hard, minlength=config.n_clusters)
    fractions = counts / counts.sum()
    if fractions.min() < config.min_cluster_fraction:
        raise ValueError("unbalanced router clusters")
    soft = _soft_routes(z, router.cluster_centers_, config.soft_temperature)
    modulation = _severity(z, config.modulation_strength)
    raw = train.loc[view.index]
    numeric, categorical = select_classifier_columns(raw, features)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    y = raw.traffic_label.to_numpy()
    global_model = _rf(config.global_trees, config.random_state).fit(
        x, y, sample_weight=modulation
    )
    experts = {}
    for cluster in range(config.n_clusters):
        # Soft membership preserves minority applications near regime borders.
        weights = modulation * soft[:, cluster]
        experts[cluster] = _rf(config.expert_trees, config.random_state + cluster).fit(
            x, y, sample_weight=weights
        )
    return {
        "features": list(features), "window": int(window), "columns": columns,
        "scaler": scaler, "router": router, "preprocessor": preprocessor,
        "numeric": numeric, "categorical": categorical, "global": global_model,
        "experts": experts, "classes": APPLICATIONS,
        "cluster_fractions": fractions.tolist(), "train_rows": len(raw),
        "config": config,
    }


def predict_proba(model, frame: pd.DataFrame):
    view = trend_frame(frame, model["features"], model["window"])
    raw = frame.loc[view.index]
    z = model["scaler"].transform(view[model["columns"]])
    route_weights = _soft_routes(
        z, model["router"].cluster_centers_, model["config"].soft_temperature
    )
    x = model["preprocessor"].transform(
        raw[model["numeric"] + model["categorical"]]
    )
    classes = model["classes"]
    expert_mixture = np.zeros((len(raw), len(classes)))
    for cluster, expert in model["experts"].items():
        expert_mixture += route_weights[:, [cluster]] * aligned_probabilities(
            expert, x, classes
        )
    global_probability = aligned_probabilities(model["global"], x, classes)
    blend = model["config"].global_blend
    probability = (1 - blend) * expert_mixture + blend * global_probability
    return probability, raw, route_weights


def predict(model, frame: pd.DataFrame):
    probability, raw, routes = predict_proba(model, frame)
    labels = np.asarray(model["classes"])[probability.argmax(axis=1)]
    return pd.Series(labels, index=raw.index, name="Sensitive_CDR_MLC"), routes


def select_and_fit(source: pd.DataFrame, output, config: SensitiveConfig):
    """Select configuration on source validation, then refit full source."""
    config.validate()
    train, validation = chronological_split(source)
    ranking = temporal_sensitivity_ranking(train, config)
    top = ranking.feature.head(min(config.ranking_top_k, len(ranking))).tolist()
    trials, candidates = [], []
    for features in itertools.combinations(top, 3):
        for window in config.windows:
            base = {"features": "|".join(features), "window": window}
            try:
                candidate = fit_model(train, features, window, config)
                probability, observed, _ = predict_proba(candidate, validation)
                prediction = np.asarray(APPLICATIONS)[probability.argmax(axis=1)]
                score = float(f1_score(
                    observed.traffic_label, prediction, average="macro", zero_division=0
                ))
                balance = min(candidate["cluster_fractions"])
                trials.append({**base, "status": "ok", "source_validation_macro_f1": score,
                               "smallest_cluster_fraction": balance,
                               "validation_rows": len(observed)})
                key = (score, balance, -window, features)
                candidates.append((key, features, window))
            except ValueError as error:
                trials.append({**base, "status": "rejected", "reason": str(error)})
    if not candidates:
        raise ValueError("no valid sensitivity-selected CDR configuration")
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / "sensitive_feature_ranking.csv", index=False)
    # A configuration can be balanced on the chronological training prefix but
    # become unbalanced after refitting on the complete source capture. Preserve
    # validation ranking and take the highest-ranked candidate that is also
    # feasible on the full source. The target level is never consulted.
    final = None
    selected_candidate = None
    for key, features, window in sorted(candidates, reverse=True):
        try:
            final = fit_model(source, features, window, config)
            selected_candidate = (key, features, window)
            break
        except ValueError as error:
            trials.append({
                "features": "|".join(features), "window": window,
                "status": "rejected_at_full_source_refit", "reason": str(error),
            })
    pd.DataFrame(trials).to_csv(output / "sensitive_selection_trials.csv", index=False)
    if final is None or selected_candidate is None:
        raise ValueError(
            "no validation-ranked configuration remains balanced after full-source refit; "
            "reduce --sensitive-min-cluster-fraction explicitly or expand candidates"
        )
    best = selected_candidate
    selected = {
        "features": list(best[1]), "window": int(best[2]),
        "source_validation_macro_f1": float(best[0][0]),
        "source_validation_smallest_cluster_fraction": float(best[0][1]),
        "full_source_smallest_cluster_fraction": float(min(final["cluster_fractions"])),
        "modulation": "1 + strength * clipped robust temporal severity",
        "uses_true_congestion_level": False,
    }
    (output / "sensitive_selected.json").write_text(
        __import__("json").dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    return final, selected
