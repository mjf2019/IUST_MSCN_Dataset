"""Synthetic integration checks; not experimental results for the paper."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clustering_ablation import METHODS, SCORES, evaluate_pair, summarize
from compare_clean_valid import TIMING, class_covered_mbk_routes
from expert_partitioning import random_expert_assignments, random_expert_partition
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all
from test_expert_input_ablation import synthetic_data
from dynamic_drift_patterns import ordered_development_tail, build_stream


class ClusteringAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = synthetic_data()
        cls.development = data[data.congestion_level.isin(["Low", "Medium"])].reset_index(drop=True)
        cls.test = data[data.congestion_level.eq("High")].reset_index(drop=True)
        cls.config = MetaStackConfig(congestion_window=5, expert_trees=3,
            utility_trees=3, meta_trees=3, congestion_features=tuple(TIMING), random_state=42)

    def test_rng_uses_only_count_seed_and_no_balancing(self):
        frame = self.development
        a = random_expert_assignments(frame, 42)
        expected = np.random.default_rng(42).integers(0, 3, size=len(frame), dtype=np.int64)
        np.testing.assert_array_equal(a, expected)
        altered = frame.copy()
        altered["traffic_label"] = "hidden"
        altered["congestion_level"] = "hidden"
        altered["source_file"] = "renamed.flow"
        altered["source_row"] = -999
        altered[TIMING] = -999
        np.testing.assert_array_equal(a, random_expert_assignments(altered, 42))
        np.testing.assert_array_equal(a, random_expert_assignments(frame.iloc[::-1], 42))
        np.testing.assert_array_equal(a[:20], random_expert_assignments(frame.iloc[:20], 42))
        self.assertFalse(np.array_equal(a, random_expert_assignments(frame, 52)))
        # Tiny samples retain raw draws; no equal-size quotas or redraws.
        np.testing.assert_array_equal(random_expert_assignments(frame.iloc[:7], 42),
            np.random.default_rng(42).integers(0, 3, size=7, dtype=np.int64))

    def test_saved_draws_survive_reordering_and_development_extension(self):
        frame = self.development
        preliminary = frame.iloc[::3]
        a, state = random_expert_partition(preliminary, 42)
        b, final = random_expert_partition(frame, 42, previous=state)
        np.testing.assert_array_equal(a, b[::3])
        reordered, again = random_expert_partition(frame.iloc[::-1], 42, previous=final)
        np.testing.assert_array_equal(reordered, b[::-1])
        renamed = preliminary.copy()
        renamed.index = renamed.index + 100000
        same, _ = random_expert_partition(renamed, 42)
        np.testing.assert_array_equal(a, same)
        self.assertEqual(final["rng_state"], again["rng_state"])

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
                saved = audit[METHODS[1]]["random_partition"]
                self.assertEqual(len(saved["row_indices"]), len(saved["assignments"]))
                self.assertEqual(saved["seed"], self.config.random_state)
                if refit:
                    preliminary = fit_meta_stacker(self.development,
                        replace(self.config, use_clustering=False, refit_experts=False))
                    initial = preliminary["random_partition"]
                    final_map = dict(zip(saved["row_indices"], saved["assignments"]))
                    self.assertEqual(initial["assignments"],
                        [final_map[key] for key in initial["row_indices"]])

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

    def test_constrained_assignment_is_minimum_cost_without_duplicates(self):
        from itertools import product
        distances = np.array([[0., 4., 8.], [1., 3., 7.], [2., 2., 6.],
                              [3., 1., 5.], [1., 5., 9.]])
        routes, audit = class_covered_mbk_routes(distances, np.array(["HTTP"] * 5))
        self.assertEqual(set(routes), {0, 1, 2})
        cost = (distances[np.arange(5), routes] ** 2).sum()
        optimum = min(sum(distances[i, c] ** 2 for i, c in enumerate(candidate))
            for candidate in product(range(3), repeat=5) if len(set(candidate)) == 3)
        self.assertAlmostEqual(cost, optimum)
        self.assertFalse(audit["duplicates_or_oversampling"])
        with self.assertRaises(ValueError):
            class_covered_mbk_routes(distances[:2], np.array(["HTTP"] * 2))

    def test_label_aware_class_coverage_and_inference_label_independence(self):
        for refit in (False, True):
            config = replace(self.config, mbk_label_aware=True, refit_experts=refit)
            rows, _, audit = evaluate_pair(self.development, self.test, config)
            full = audit[METHODS[0]]
            self.assertTrue(full["mbk_label_aware"])
            for item in full["initial_class_coverage_audit"]["class_counts"]:
                self.assertTrue(all(count >= 1 for count in item["constrained"]))
            if refit:
                for item in full["refit_class_coverage_audit"]["class_counts"]:
                    self.assertTrue(all(count >= 1 for count in item["constrained"]))
            self.assertFalse(audit[METHODS[1]]["mbk_label_aware"])
        model = fit_meta_stacker(self.development, config)
        hidden = self.test.copy()
        hidden["traffic_label"] = "HTTP"
        hidden["congestion_level"] = "Low"
        np.testing.assert_array_equal(
            predict_all(model, self.test)["CDR_MLC_meta_stacker"],
            predict_all(model, hidden)["CDR_MLC_meta_stacker"])
        plain = fit_meta_stacker(self.development, replace(self.config, use_clustering=False))
        aware = fit_meta_stacker(self.development, replace(config, use_clustering=False))
        self.assertEqual(plain["random_partition"], aware["random_partition"])
        np.testing.assert_array_equal(
            predict_all(plain, self.test)["CDR_MLC_meta_stacker"],
            predict_all(aware, self.test)["CDR_MLC_meta_stacker"])

    def test_dynamic_stream_pairs_use_disjoint_identical_rows(self):
        development, tails = ordered_development_tail(synthetic_data(), .80)
        dev_ids = set(zip(development.source_file, development.source_row))
        for scenario in ("D1", "D2", "D3"):
            stream, segments = build_stream(tails, scenario, block_rows=20, seed=42)
            test_ids = set(zip(stream.source_file, stream.source_row))
            self.assertFalse(dev_ids & test_ids)
            self.assertEqual(len(test_ids), len(stream))
            repeat, _ = build_stream(tails, scenario, block_rows=20, seed=42)
            self.assertEqual(stream.to_dict("records"), repeat.to_dict("records"))
            rows, predictions, audit = evaluate_pair(development, stream,
                replace(self.config, mbk_label_aware=True))
            self.assertEqual(rows[0]["n"], rows[1]["n"])
            self.assertEqual(len(predictions), rows[0]["n"])
            self.assertIn("drift_segment", predictions)
            self.assertTrue(audit["same_scored_rows_and_context_values_verified"])

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
