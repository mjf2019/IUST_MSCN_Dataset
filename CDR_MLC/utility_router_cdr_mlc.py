"""Expected-correctness routing for the unchanged fixed CDR-MLC experts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import fit_fixed_cdr
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig
from congestion_selective_router_cdr_mlc import _enhanced_outputs
from learned_router_cdr_mlc import LearnedRouterConfig, _oracle_route, _refit_experts_with_frozen_router
from selective_router_cdr_mlc import three_way_split


@dataclass(frozen=True)
class UtilityRouterConfig:
    window: int = 3
    congestion_window: int = 10
    congestion_features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    expert_trees: int = 20
    utility_trees: int = 150
    max_depth: int = 12
    min_samples_leaf: int = 12
    expert_train_fraction: float = .60
    utility_train_fraction: float = .20
    utility_gain_candidates: tuple[float, ...] = (.00, .05, .10, .15, .20, .30, 1.01)
    soft_temperatures: tuple[float, ...] = (.25, .50, 1.0, 2.0)
    soft_kmeans_priors: tuple[float, ...] = (.00, .25, .50, 1.0, 2.0)
    expert_probability_powers: tuple[float, ...] = (.50, 1.0, 2.0)
    random_state: int = 42

    def validate(self):
        if self.window < 1 or self.congestion_window < 3:
            raise ValueError("base window must be positive and congestion window >= 3")
        if len(set(self.congestion_features)) < 3:
            raise ValueError("at least three congestion features are required")
        if not 0 < self.expert_train_fraction < 1 or not 0 < self.utility_train_fraction < 1:
            raise ValueError("training fractions must be in (0,1)")
        if self.expert_train_fraction + self.utility_train_fraction >= 1:
            raise ValueError("a nonempty threshold-selection partition is required")
        return self


def _congestion_config(config):
    return CongestionRouterConfig(
        window=config.congestion_window,
        features=tuple(config.congestion_features),
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    ).validate()


def _balanced_weights(levels, target):
    """Balance both observed development levels and binary correctness."""
    levels = np.asarray(levels)
    target = np.asarray(target)
    weights = np.ones(len(target), dtype=float)
    for values in (levels, target):
        unique, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique, counts):
            weights[values == value] *= len(values) / (len(unique) * count)
    return weights / weights.mean()


def _fit_utility_models(features, predictions, truth, levels, config):
    models, constants, counts = {}, {}, {}
    for expert in range(3):
        target = (predictions[:, expert] == truth).astype(int)
        counts[str(expert)] = {
            "correct": int(target.sum()), "incorrect": int((1 - target).sum())
        }
        unique = np.unique(target)
        if len(unique) == 1:
            models[expert], constants[expert] = None, int(unique[0])
            continue
        models[expert] = RandomForestClassifier(
            n_estimators=config.utility_trees,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=config.random_state + 100 + expert,
        ).fit(features, target, sample_weight=_balanced_weights(levels, target))
        constants[expert] = None
    return models, constants, counts


def _utility_matrix(models, constants, features):
    result = np.zeros((len(features), 3), dtype=float)
    for expert in range(3):
        model = models[expert]
        if model is None:
            result[:, expert] = float(constants[expert] == 1)
            continue
        classes = list(model.classes_)
        if 1 in classes:
            result[:, expert] = model.predict_proba(features)[:, classes.index(1)]
    return result


def _utility_routes(kroute, utility, minimum_gain):
    row = np.arange(len(kroute))
    alternative = utility.argmax(axis=1)
    gain = utility[row, alternative] - utility[row, kroute]
    override = (alternative != kroute) & (gain >= minimum_gain)
    route = kroute.copy()
    route[override] = alternative[override]
    return route, alternative, gain, override


def _level_scores(truth, prediction, levels):
    scores = {}
    for level in pd.unique(levels):
        mask = np.asarray(levels) == level
        scores[str(level)] = float(f1_score(
            truth[mask], prediction[mask], average="macro", zero_division=0
        ))
    return scores


def _soft_mixture(probabilities, utility, kroute, temperature,
                  kmeans_prior, probability_power):
    """Blend all expert posteriors using calibrated expected correctness."""
    safe_utility = np.clip(utility, 1e-6, 1.0)
    weights = safe_utility ** (1.0 / temperature)
    weights[np.arange(len(kroute)), kroute] += kmeans_prior
    weights /= weights.sum(axis=1, keepdims=True)

    calibrated = np.clip(probabilities, 1e-12, 1.0) ** probability_power
    calibrated /= calibrated.sum(axis=2, keepdims=True)
    mixture = np.einsum("ne,nec->nc", weights, calibrated)
    mixture /= mixture.sum(axis=1, keepdims=True)
    return mixture, weights


def fit_utility_router(source, config: UtilityRouterConfig):
    config.validate()
    congestion_config = _congestion_config(config)
    split = three_way_split(
        source, config.expert_train_fraction, config.utility_train_fraction
    )
    # Frozen paper router and final expert bank.
    full_model = fit_fixed_cdr(source, config.window, config.random_state, config.expert_trees)
    helper = LearnedRouterConfig(
        window=config.window, expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    preliminary = _refit_experts_with_frozen_router(full_model, split["expert"], helper)

    raw, _, kroute, probabilities, predictions, features, feature_columns = _enhanced_outputs(
        preliminary, split["correction"], congestion_config
    )
    truth = raw.traffic_label.to_numpy()
    models, constants, correctness_counts = _fit_utility_models(
        features, predictions, truth, raw.congestion_level.to_numpy(), config
    )

    validation_raw, _, validation_kroute, validation_prob, validation_pred, validation_features, _ = _enhanced_outputs(
        preliminary, split["threshold"], congestion_config
    )
    validation_truth = validation_raw.traffic_label.to_numpy()
    validation_levels = validation_raw.congestion_level.to_numpy()
    utilities = _utility_matrix(models, constants, validation_features)
    row = np.arange(len(validation_raw))
    base_prediction = validation_pred[row, validation_kroute]
    base_level_scores = _level_scores(validation_truth, base_prediction, validation_levels)
    trials = []
    for minimum_gain in config.utility_gain_candidates:
        route, _, _, override = _utility_routes(validation_kroute, utilities, minimum_gain)
        prediction = validation_pred[row, route]
        level_scores = _level_scores(validation_truth, prediction, validation_levels)
        level_deltas = {
            level: level_scores[level] - base_level_scores[level]
            for level in level_scores
        }
        trials.append({
            "minimum_utility_gain": float(minimum_gain),
            "macro_f1": float(f1_score(
                validation_truth, prediction, average="macro", zero_division=0
            )),
            "overrides": int(override.sum()),
            "override_fraction": float(override.mean()),
            "level_macro_f1": level_scores,
            "level_macro_f1_delta": level_deltas,
            "worst_level_delta": float(min(level_deltas.values())),
        })
    # The 1.01 no-override candidate guarantees a safe feasible fallback.
    feasible = [trial for trial in trials if trial["worst_level_delta"] >= -1e-12]
    selected = max(feasible, key=lambda trial: (
        trial["worst_level_delta"], trial["macro_f1"],
        trial["minimum_utility_gain"], -trial["overrides"],
    ))

    selected_hard_route, _, _, _ = _utility_routes(
        validation_kroute, utilities, selected["minimum_utility_gain"]
    )
    hard_prediction = validation_pred[row, selected_hard_route]
    hard_level_scores = _level_scores(
        validation_truth, hard_prediction, validation_levels
    )
    soft_trials = [{
        "mode": "hard_fallback",
        "temperature": None,
        "kmeans_prior": None,
        "probability_power": None,
        "macro_f1": float(f1_score(
            validation_truth, hard_prediction, average="macro", zero_division=0
        )),
        "level_macro_f1": hard_level_scores,
        "level_macro_f1_delta_vs_hard": {
            level: 0.0 for level in hard_level_scores
        },
        "worst_level_delta_vs_hard": 0.0,
    }]
    for temperature in config.soft_temperatures:
        for kmeans_prior in config.soft_kmeans_priors:
            for probability_power in config.expert_probability_powers:
                mixture, weights = _soft_mixture(
                    validation_prob, utilities, validation_kroute,
                    temperature, kmeans_prior, probability_power,
                )
                # Expert posteriors always follow the global APPLICATIONS order.
                prediction = np.asarray(APPLICATIONS)[mixture.argmax(axis=1)]
                level_scores = _level_scores(
                    validation_truth, prediction, validation_levels
                )
                deltas = {
                    level: level_scores[level] - hard_level_scores[level]
                    for level in level_scores
                }
                soft_trials.append({
                    "mode": "soft",
                    "temperature": float(temperature),
                    "kmeans_prior": float(kmeans_prior),
                    "probability_power": float(probability_power),
                    "macro_f1": float(f1_score(
                        validation_truth, prediction, average="macro", zero_division=0
                    )),
                    "level_macro_f1": level_scores,
                    "level_macro_f1_delta_vs_hard": deltas,
                    "worst_level_delta_vs_hard": float(min(deltas.values())),
                    "mean_max_weight": float(weights.max(axis=1).mean()),
                })
    soft_feasible = [
        trial for trial in soft_trials
        if trial["worst_level_delta_vs_hard"] >= -1e-12
    ]
    selected_soft = max(soft_feasible, key=lambda trial: (
        trial["worst_level_delta_vs_hard"], trial["macro_f1"],
        trial["mode"] == "hard_fallback",
    ))
    full_model.update({
        "utility_router_config": config,
        "congestion_config": congestion_config,
        "congestion_columns": feature_columns,
        "utility_models": models,
        "utility_constants": constants,
        "selected_utility_gain": selected["minimum_utility_gain"],
        "utility_gain_trials": trials,
        "soft_mixture_trials": soft_trials,
        "selected_soft_mixture": selected_soft,
        "utility_training_rows": len(raw),
        "utility_correctness_counts": correctness_counts,
        "utility_kmeans_error_rate": float(np.mean(
            predictions[np.arange(len(raw)), kroute] != truth
        )),
    })
    return full_model


def predict_all(model, frame):
    raw, _, kroute, probabilities, predictions, features, _ = _enhanced_outputs(
        model, frame, model["congestion_config"]
    )
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], features
    )
    route, alternative, gain, override = _utility_routes(
        kroute, utility, model["selected_utility_gain"]
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    row = np.arange(len(raw))
    selected_soft = model["selected_soft_mixture"]
    if selected_soft["mode"] == "hard_fallback":
        soft_prediction = predictions[row, route]
        soft_weights = np.eye(3)[route]
    else:
        mixture, soft_weights = _soft_mixture(
            probabilities, utility, kroute,
            selected_soft["temperature"], selected_soft["kmeans_prior"],
            selected_soft["probability_power"],
        )
        soft_prediction = np.asarray(APPLICATIONS)[mixture.argmax(axis=1)]
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, kroute],
        "CDR_MLC_utility_router": predictions[row, route],
        "CDR_MLC_soft_utility_router": soft_prediction,
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "minibatch_kmeans_route": kroute, "utility_alternative": alternative,
            "utility_route": route, "oracle_route": oracle,
            "utility_gain": gain, "route_overridden": override,
            "utility_matches_oracle": route == oracle,
            "minibatch_kmeans_matches_oracle": kroute == oracle,
            **{f"expert_{expert}_utility": utility[:, expert] for expert in range(3)},
            **{f"expert_{expert}_soft_weight": soft_weights[:, expert] for expert in range(3)},
        }, index=raw.index),
    }
