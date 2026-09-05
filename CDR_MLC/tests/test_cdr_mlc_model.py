import numpy as np
import pandas as pd

from cdr_mlc import CDRMLC, CDRMLCConfig
from cdr_mlc.model import CausalWindowTransformer


def make_dataset(seed: int, size: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    level = np.repeat(np.arange(3), size // 3)
    labels = np.array(["HTTP", "SSH", "VIDEO"])[np.arange(size) % 3]
    return pd.DataFrame(
        {
            "SynAck": level + rng.normal(0, 0.1, size),
            "AckDat": 2 * level + rng.normal(0, 0.1, size),
            "TcpRtt": 5 * level + rng.normal(0, 0.2, size),
            "packet_size": rng.normal(500, 50, size),
            "packet_rate": rng.normal(100, 10, size),
            "label": labels,
            "label_encoded": np.arange(size) % 3,
            "index_in_label": np.arange(size),
            "level_label": level,
        }
    )


def test_target_like_columns_never_enter_an_expert():
    model = CDRMLC(CDRMLCConfig(n_estimators=5)).fit(make_dataset(1))
    assert model.classification_features_ == ["packet_size", "packet_rate"]
    for expert in model.experts.values():
        assert list(expert.feature_names_in_) == ["packet_size", "packet_rate"]


def test_predictions_do_not_require_or_change_with_a_label():
    model = CDRMLC(CDRMLCConfig(n_estimators=5)).fit(make_dataset(2))
    test = make_dataset(3)
    without_target = test.drop(columns=["label"])
    changed_target = test.assign(label="WRONG")
    np.testing.assert_array_equal(model.predict(without_target), model.predict(changed_target))


def test_training_routes_are_exactly_router_predictions():
    train = make_dataset(4)
    model = CDRMLC(CDRMLCConfig(n_estimators=5)).fit(train)
    prepared = train.drop(columns=["IdleTime"], errors="ignore")
    routing = model.window_transformer.transform(prepared)
    routes = model.router.predict(model.routing_scaler.transform(routing))
    for cluster_id, expert in model.experts.items():
        assert expert.n_features_in_ == 2
        assert np.any(routes == cluster_id)


def test_window_transformer_is_causal():
    original = make_dataset(5)
    changed = original.copy()
    changed.loc[10:, ["SynAck", "AckDat", "TcpRtt"]] = 1_000_000
    transformer = CausalWindowTransformer(
        ("SynAck", "AckDat", "TcpRtt"),
        ("mean", "median", "std", "min", "max"),
        3,
    )
    pd.testing.assert_frame_equal(
        transformer.transform(original).iloc[:10],
        transformer.transform(changed).iloc[:10],
    )
