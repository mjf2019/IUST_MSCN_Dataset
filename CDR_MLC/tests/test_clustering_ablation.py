"""Synthetic integration checks; not experimental results for the paper."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clustering_ablation import METHODS, SCORES, evaluate_pair, summarize
from compare_clean_valid import TIMING
from expert_partitioning import random_expert_assignments
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all
from test_expert_input_ablation import synthetic_data


class ClusteringAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = synthetic_data()
        cls.development = data[data.congestion_level.isin(["Low", "Medium"])].reset_index(drop=True)
        cls.test = data[data.congestion_level.eq("High")].reset_index(drop=True)
        cls.config = MetaStackConfig(congestion_window=5, expert_trees=3,
            utility_trees=3, meta_trees=3, congestion_features=tuple(TIMING), random_state=42)

    def test_partitions_ignore_labels_values_order_and_future_rows(self):
        frame = self.development
        a = random_expert_assignments(frame, 42)
        altered = frame.copy()
        altered["traffic_label"] = "hidden"
        altered["congestion_level"] = "hidden"
        altered[TIMING] = -999
        np.testing.assert_array_equal(a, random_expert_assignments(altered, 42))
        subset = frame.iloc[::3]
        np.testing.assert_array_equal(a[::3], random_expert_assignments(subset, 42))
        np.testing.assert_array_equal(a[::-1], random_expert_assignments(frame.iloc[::-1], 42))
        self.assertEqual(set(a), {0, 1, 2})
        self.assertFalse(np.array_equal(a, random_expert_assignments(frame, 52)))

    def test_no_clustering_calls_and_hidden_test_labels(self):
        config = replace(self.config, use_clustering=False)
        with patch("compare_clean_valid.make_minibatch_kmeans", side_effect=AssertionError("MBK called")), \
             patch("compare_clean_valid.StandardScaler", side_effect=AssertionError("scaler called")):
            model = fit_meta_stacker(self.development, config)
            result = predict_all(model, self.test)
            hidden = self.test.copy()
            hidden["traffic_label"] = "HTTP"
            hidden["congestion_level"] = "Low"
            hidden["source_file"] = "opaque_test_identity.flow"
            np.testing.assert_array_equal(result["CDR_MLC_meta_stacker"],
                predict_all(model, hidden)["CDR_MLC_meta_stacker"])
        self.assertIsNone(model["router"])
        self.assertIsNone(model["scaler"])
        self.assertIsNone(result["CDR_MLC_actual_router"])
        self.assertTrue(result["routes"].minibatch_kmeans_route.isna().all())

    def test_pair_preserves_context_rows_and_budgets_with_both_refit_modes(self):
        for refit in (True, False):
            with self.subTest(refit=refit):
                rows, predictions, audit = evaluate_pair(self.development, self.test,
                    replace(self.config, refit_experts=refit))
                self.assertEqual(rows[0]["n"], rows[1]["n"])
                self.assertEqual(len(predictions), rows[0]["n"])
                self.assertEqual(rows[0]["meta_input_dimension"] - rows[1]["meta_input_dimension"], 6)
                self.assertTrue(audit["same_scored_rows_and_context_values_verified"])
                self.assertTrue(audit["no_MBK_scaler_distances_or_route_features_verified"])

    def test_mbk_parameters_propagation_and_control_invariance(self):
        for refit in (True, False):
            with self.subTest(refit=refit):
                config = replace(self.config, mbk_n_init=50, mbk_batch_size=2048,
                    mbk_max_iter=300, refit_experts=refit)
                rows, predictions, audit = evaluate_pair(self.development, self.test, config)
                self.assertEqual(audit[METHODS[0]]["router_audit"]["n_init"], 50)
                self.assertEqual(audit[METHODS[0]]["router_audit"]["batch_size"], 2048)
                self.assertEqual(audit[METHODS[0]]["router_audit"]["max_iter"], 300)
                control10 = fit_meta_stacker(self.development,
                    replace(config, use_clustering=False, mbk_n_init=10,
                        mbk_batch_size=1024, mbk_max_iter=100))
                control50 = fit_meta_stacker(self.development,
                    replace(config, use_clustering=False))
                np.testing.assert_array_equal(
                    predict_all(control10, self.test)["CDR_MLC_meta_stacker"],
                    predict_all(control50, self.test)["CDR_MLC_meta_stacker"])
                self.assertEqual(rows[0]["n"], rows[1]["n"])
        for parameter in ("mbk_n_init", "mbk_batch_size", "mbk_max_iter"):
            for value in (0, -1):
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{parameter: value}).validate()

    def test_summary_delta_and_equal_protocol_weighting(self):
        rows = [{"protocol": protocol, "seed": 42, "method": method,
                 **{key: score for key in SCORES}}
                for protocol, scores in (("s1", (.8, .7)), ("s2", (1., .8)))
                for method, score in zip(METHODS, scores)]
        _, _, delta, means, mean_delta = summarize(rows)
        self.assertAlmostEqual(delta.iloc[0].accuracy_delta_no_clustering_minus_full, -.1)
        self.assertAlmostEqual(mean_delta.iloc[0].accuracy_delta_no_clustering_minus_full, -.15)
        self.assertAlmostEqual(means.set_index("method").loc[METHODS[0], "accuracy"], .9)
        self.assertTrue((means.protocols == 2).all())


if __name__ == "__main__":
    unittest.main()
