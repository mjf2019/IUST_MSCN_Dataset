import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).parents[1] / "CDR-MLC.ipynb"


def load_pipeline_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    pipeline_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "class SlidingWindowStats" in "".join(cell["source"])
    )
    namespace = {"__name__": "cdr_mlc_notebook"}
    exec("".join(pipeline_cell["source"]), namespace)
    return namespace


def make_dataset(seed, size=240):
    rng = np.random.default_rng(seed)
    congestion = np.repeat(np.arange(3), size // 3)
    labels = np.array(["HTTP", "SSH", "VIDEO"])[np.arange(size) % 3]
    return pd.DataFrame(
        {
            "SynAck": congestion + rng.normal(0, 0.15, size),
            "AckDat": 2 * congestion + rng.normal(0, 0.15, size),
            "TcpRtt": 5 * congestion + rng.normal(0, 0.25, size),
            "packet_size": rng.normal(500, 50, size),
            "packet_rate": rng.normal(100, 10, size),
            "label": labels,
        }
    )


def test_classifier_features_exclude_every_label_representation():
    ns = load_pipeline_namespace()
    result = ns["universal_clustering_pipeline"](
        make_dataset(1),
        external_test_df=make_dataset(2),
        n_clusters=3,
        window_size=3,
        ensure_label_coverage=False,
        use_at_least_one_rule=False,
    )

    forbidden = {"label", "label_encoded", "class", "cluster"}
    assert forbidden.isdisjoint(result["classification_features"])


def test_sliding_window_features_do_not_use_future_rows():
    ns = load_pipeline_namespace()
    original = make_dataset(8, size=6)
    changed_future = original.copy()
    changed_future.loc[3:, ["SynAck", "AckDat", "TcpRtt"]] = 1_000_000

    original_stats, _ = ns["compute_sliding_window_stats"](
        original, ["SynAck", "AckDat", "TcpRtt"], 3,
        ["mean", "median", "std", "min", "max"]
    )
    changed_stats, _ = ns["compute_sliding_window_stats"](
        changed_future, ["SynAck", "AckDat", "TcpRtt"], 3,
        ["mean", "median", "std", "min", "max"]
    )

    pd.testing.assert_frame_equal(original_stats.iloc[:3], changed_stats.iloc[:3])


def test_deployment_prediction_is_independent_of_supplied_label():
    ns = load_pipeline_namespace()
    result = ns["universal_clustering_pipeline"](
        make_dataset(3),
        external_test_df=make_dataset(4),
        n_clusters=3,
        window_size=3,
        ensure_label_coverage=False,
        use_at_least_one_rule=False,
    )
    sample = make_dataset(5, size=6).iloc[0].to_dict()

    predictions = []
    for label_value in (None, "HTTP", "VIDEO"):
        predictor = ns["create_deployment_pipeline"](result)
        candidate = dict(sample)
        if label_value is None:
            candidate.pop("label", None)
        else:
            candidate["label"] = label_value
        predictions.append(predictor(candidate))

    assert predictions[0] == predictions[1] == predictions[2]


def test_removed_oracle_rule_fails_closed():
    ns = load_pipeline_namespace()
    try:
        ns["universal_clustering_pipeline"](
            make_dataset(6),
            external_test_df=make_dataset(7),
            n_clusters=3,
            window_size=3,
            ensure_label_coverage=False,
            use_at_least_one_rule=True,
        )
    except ValueError as error:
        assert "leaked test labels" in str(error)
    else:
        raise AssertionError("The removed test-label oracle was unexpectedly enabled")
