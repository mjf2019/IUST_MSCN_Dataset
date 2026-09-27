import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_cdr_mlc import STATS, trend_frame, trend_values
from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES,
    CongestionRouterConfig,
    congestion_feature_frame,
    congestion_feature_values,
)


def fixture():
    rng = np.random.default_rng(2026)
    rows = []
    for sequence, count in (("capture-b", 31), ("capture-a", 37)):
        values = rng.lognormal(mean=2.0, sigma=.7, size=(count, 13))
        for row in range(count):
            rows.append({
                "sequence_id": sequence,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(row, "s"),
                "source_row": row,
                **dict(zip(DEFAULT_CONGESTION_FEATURES, values[row])),
            })
    frame = pd.DataFrame(rows)
    # An invalid observation must split, rather than compress, causal history.
    frame.loc[12, "TcpRtt"] = np.nan
    # Ensure both engines restore source-index order after per-sequence sorting.
    return frame.sample(frac=1.0, random_state=7)


class CongestionContextEngineTests(unittest.TestCase):
    def test_vectorized_engine_matches_reference_pandas_engine(self):
        frame = fixture()
        base = CongestionRouterConfig(window=5)
        vectorized = congestion_feature_frame(frame, base)[0]
        reference = congestion_feature_frame(
            frame, replace(base, context_engine="pandas")
        )[0]
        router_columns = [
            column for column in reference if column.startswith("router_")
        ]
        self.assertTrue(vectorized.index.equals(reference.index))
        self.assertEqual(router_columns, [
            column for column in vectorized if column.startswith("router_")
        ])
        np.testing.assert_allclose(
            vectorized[router_columns].to_numpy(),
            reference[router_columns].to_numpy(),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_inference_values_avoid_raw_frame_copy_without_changing_values(self):
        frame = fixture()
        config = CongestionRouterConfig(window=5)
        view, _ = congestion_feature_frame(frame, config)
        index, columns, values = congestion_feature_values(frame, config)
        self.assertTrue(index.equals(view.index))
        np.testing.assert_array_equal(values, view[columns].to_numpy())


class TrendEngineTests(unittest.TestCase):
    def test_vectorized_ttfef_matches_reference_pandas_engine(self):
        frame = fixture()
        features = ("TcpRtt", "AckDat", "SynAck")
        vectorized = trend_frame(frame, features, 5)
        reference = trend_frame(frame, features, 5, engine="pandas")
        columns = [
            f"{feature}_{stat}" for feature in features for stat in STATS
        ]
        self.assertTrue(vectorized.index.equals(reference.index))
        np.testing.assert_allclose(
            vectorized[columns].to_numpy(),
            reference[columns].to_numpy(),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_ttfef_inference_api_returns_same_matrix(self):
        frame = fixture()
        features = ("TcpRtt", "AckDat", "SynAck")
        view = trend_frame(frame, features, 5)
        index, columns, values = trend_values(frame, features, 5)
        self.assertTrue(index.equals(view.index))
        np.testing.assert_array_equal(values, view[columns].to_numpy())


if __name__ == "__main__":
    unittest.main()
