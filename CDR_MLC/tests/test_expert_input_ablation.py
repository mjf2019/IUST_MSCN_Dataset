"""Integration checks on synthetic flows; these are not paper results."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import TIMING
from expert_input_ablation import evaluate_pair, summarize
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all


def synthetic_data():
    rng = np.random.default_rng(101)
    frames = []
    for level_id, level in enumerate(("Low", "Medium", "High")):
        for label_id, label in enumerate(APPLICATIONS):
            count = 150
            values = {f"feature_{i}": rng.uniform(.1, 2, count) + label_id
                      for i in range(32)}
            # All early partitions contain three distinct, repeating regimes.
            regime = np.arange(count) % 3
            for i, name in enumerate(TIMING):
                values[name] = 1 + 10 * regime + level_id + i + rng.uniform(0, .1, count)
            frame = pd.DataFrame(values)
            frame["timestamp"] = pd.date_range("2026-01-01", periods=count, freq="s")
            frame["source_row"] = np.arange(count) + 2
            frame["source_file"] = f"{label}_{level}.flow"
            frame["sequence_id"] = f"{label}_{level}"
            frame["traffic_label"] = label
            frame["congestion_level"] = level
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class ExpertInputAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = synthetic_data()
        cls.development = data[data.congestion_level.isin(["Low", "Medium"])].reset_index(drop=True)
        cls.test = data[data.congestion_level.eq("High")].reset_index(drop=True)
        cls.config = MetaStackConfig(
            congestion_window=3, expert_trees=3, utility_trees=3, meta_trees=3,
            congestion_features=tuple(TIMING), random_state=42,
        )

    def test_pair_with_and_without_final_refit(self):
        for refit in (True, False):
            with self.subTest(refit=refit):
                rows, predictions, audit = evaluate_pair(
                    self.development, self.test, replace(self.config, refit_experts=refit)
                )
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["n"], rows[1]["n"])
                self.assertEqual(len(predictions), rows[0]["n"])
                for method, count in (("MF_Separated_Expert_Inputs", 32), ("MF_All_Expert_Inputs", 35)):
                    self.assertEqual(len(audit[method]["expert_input_columns"]), count)
                    self.assertEqual(audit[method]["encoded_expert_dimension"], count)
                self.assertTrue(audit["same_router_scaler_and_congestion_values_verified"])

    def test_pair_with_24_and_27_usable_features(self):
        dropped = [f"feature_{i}" for i in range(24, 32)]
        # Match columns rejected as constant by the production selector.
        development = self.development.copy()
        test = self.test.copy()
        development[dropped] = 0.0
        test[dropped] = 0.0
        rows, _, audit = evaluate_pair(development, test, self.config)
        self.assertEqual([r["raw_expert_features"] for r in rows], [24, 27])
        self.assertTrue(audit["same_router_scaler_and_congestion_values_verified"])

    def test_default_model_and_hidden_test_labels(self):
        default = fit_meta_stacker(self.development, self.config)
        explicit = fit_meta_stacker(self.development, replace(self.config, include_timing_in_experts=False))
        a = predict_all(default, self.test)["CDR_MLC_meta_stacker"]
        b = predict_all(explicit, self.test)["CDR_MLC_meta_stacker"]
        np.testing.assert_array_equal(a, b)
        hidden = self.test.copy()
        hidden["traffic_label"] = "HTTP"
        hidden["congestion_level"] = "Low"
        np.testing.assert_array_equal(a, predict_all(default, hidden)["CDR_MLC_meta_stacker"])

    def test_delta_direction(self):
        rows = [{"protocol": "test", "seed": 42, "method": method,
                 **{key: score for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}}
                for method, score in (("MF_Separated_Expert_Inputs", .8), ("MF_All_Expert_Inputs", .9))]
        _, _, delta = summarize(rows)
        self.assertAlmostEqual(delta.iloc[0].accuracy_delta_all_minus_separated, .1)


if __name__ == "__main__":
    unittest.main()
