from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CDRMLCConfig:
    target_column: str = "label"
    congestion_features: tuple[str, ...] = ("SynAck", "AckDat", "TcpRtt")
    window_statistics: tuple[str, ...] = ("mean", "median", "std", "min", "max")
    window_size: int = 3
    n_clusters: int = 3
    n_estimators: int = 20
    random_state: int = 42
    drop_columns: tuple[str, ...] = ("IdleTime",)


class CausalWindowTransformer:
    """Compute statistics from the current and previous rows only."""

    SUPPORTED_STATISTICS = {"mean", "median", "std", "min", "max"}

    def __init__(
        self,
        features: Sequence[str],
        statistics: Sequence[str],
        window_size: int,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least one")
        unsupported = set(statistics) - self.SUPPORTED_STATISTICS
        if unsupported:
            raise ValueError(f"Unsupported window statistics: {sorted(unsupported)}")
        self.features = tuple(features)
        self.statistics = tuple(statistics)
        self.window_size = window_size

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing congestion features: {sorted(missing)}")

        # Pandas rolling is vectorised and, with center=False, contains only the
        # current and previous rows. This is substantially faster than a Python
        # row loop on the full 1.43 GB experiment dataset.
        columns: dict[str, pd.Series] = {}
        for feature in self.features:
            values = frame[feature].astype(np.float32, copy=False)
            rolling = values.rolling(window=self.window_size, min_periods=1)
            statistic_series = {
                "mean": rolling.mean(),
                "median": rolling.median(),
                "std": rolling.std(ddof=1).fillna(0.0),
                "min": rolling.min(),
                "max": rolling.max(),
            }
            for statistic in self.statistics:
                columns[f"{feature}_{statistic}"] = statistic_series[statistic].astype(
                    np.float32
                )
        return pd.DataFrame(columns, index=frame.index)


class CDRMLC:
    """Hard-gated mixture of multiclass Random Forest experts.

    The gate is trained and evaluated exclusively on causal statistics derived from
    congestion-sensitive features. Traffic-class targets are never accepted as model
    features and never participate in expert routing.
    """

    def __init__(self, config: CDRMLCConfig | None = None) -> None:
        self.config = config or CDRMLCConfig()
        self.window_transformer = CausalWindowTransformer(
            self.config.congestion_features,
            self.config.window_statistics,
            self.config.window_size,
        )
        self.routing_scaler = StandardScaler()
        self.router = MiniBatchKMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            batch_size=1024,
            n_init=10,
        )
        self.experts: dict[int, RandomForestClassifier] = {}
        self.classification_features_: list[str] = []
        self.training_route_counts_: dict[int, int] = {}
        self.classes_: np.ndarray | None = None
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "CDRMLC":
        prepared = self._prepare_frame(frame, require_target=True)
        target = prepared[self.config.target_column].copy()
        self.classes_ = np.sort(target.unique())
        self.classification_features_ = self._select_classification_features(prepared)
        if not self.classification_features_:
            raise ValueError("No classification features remain after leakage checks")

        routing_features = self.window_transformer.transform(prepared)
        scaled_routing = self.routing_scaler.fit_transform(routing_features)
        self.router.fit(scaled_routing)
        train_clusters = self.router.predict(scaled_routing)
        self.training_route_counts_ = {
            int(cluster): int(count)
            for cluster, count in zip(*np.unique(train_clusters, return_counts=True))
        }

        predictors = prepared.loc[:, self.classification_features_]
        self.experts = {}
        for cluster_id in range(self.config.n_clusters):
            mask = train_clusters == cluster_id
            if not np.any(mask):
                raise RuntimeError(f"Router produced empty training cluster {cluster_id}")
            expert = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
            expert.fit(predictors.loc[mask], target.loc[mask])
            self.experts[cluster_id] = expert

        self.is_fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        prepared = self._prepare_frame(frame, require_target=False)
        self._validate_inference_schema(prepared)
        routing_features = self.window_transformer.transform(prepared)
        clusters = self.router.predict(self.routing_scaler.transform(routing_features))
        predictors = prepared.loc[:, self.classification_features_]

        predictions = np.empty(len(prepared), dtype=object)
        for cluster_id, expert in self.experts.items():
            mask = clusters == cluster_id
            if np.any(mask):
                predictions[mask] = expert.predict(predictors.loc[mask])
        return predictions

    def predict_with_routes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        prepared = self._prepare_frame(frame, require_target=False)
        self._validate_inference_schema(prepared)
        routing_features = self.window_transformer.transform(prepared)
        clusters = self.router.predict(self.routing_scaler.transform(routing_features))
        predictors = prepared.loc[:, self.classification_features_]
        predictions = np.empty(len(prepared), dtype=object)
        for cluster_id, expert in self.experts.items():
            mask = clusters == cluster_id
            if np.any(mask):
                predictions[mask] = expert.predict(predictors.loc[mask])
        return predictions, clusters

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "config": asdict(self.config),
            "classification_feature_count": len(self.classification_features_),
            "classification_features": list(self.classification_features_),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "expert_training_classes": {
                str(cluster): expert.classes_.tolist()
                for cluster, expert in self.experts.items()
            },
            "training_route_counts": self.training_route_counts_,
        }

    def _prepare_frame(self, frame: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        if require_target and self.config.target_column not in frame.columns:
            raise ValueError(f"Target column {self.config.target_column!r} is missing")
        prepared = frame.drop(columns=list(self.config.drop_columns), errors="ignore").copy()
        if len(prepared) == 0:
            raise ValueError("Input frame is empty")
        return prepared

    def _select_classification_features(self, frame: pd.DataFrame) -> list[str]:
        forbidden = {
            self.config.target_column,
            *self.config.congestion_features,
        }
        candidates = []
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column in forbidden or self._looks_like_target_metadata(column):
                continue
            candidates.append(column)
        return candidates

    @staticmethod
    def _looks_like_target_metadata(column: str) -> bool:
        normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"label", "class", "target", "y"}:
            return True
        return (
            "label" in normalized
            or normalized.startswith("class_")
            or normalized.startswith("target_")
            or normalized.startswith("index_in_")
            or normalized.startswith("unnamed:")
        )

    def _validate_inference_schema(self, frame: pd.DataFrame) -> None:
        required = set(self.classification_features_) | set(self.config.congestion_features)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required inference features: {sorted(missing)}")

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("CDRMLC must be fitted before inference")
