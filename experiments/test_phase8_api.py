"""Phase 8 integration checks against the frozen local Phase 6/7 artifacts."""

import unittest

from fastapi import HTTPException

from experiments.phase8_api import (METHODS, PriceRequest, _root, method_overview,
                                    price_compare, price_options, price_recommend)


@unittest.skipUnless((_root() / "phase7_policy_robustness" / "manifest.json").is_file(),
                     "Phase 6/7 artifacts are not present")
class Phase8InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = price_options()

    def test_options_and_evidence_use_the_same_frozen_cohort(self):
        overview = method_overview()
        weeks = sum(len(item["weeks"]) for item in self.options["items"])
        self.assertEqual(weeks, self.options["eligible_test_weeks"])
        self.assertEqual(weeks, overview["pricing_test_eligible_rows"])
        self.assertEqual(len(self.options["items"]), 9)
        self.assertEqual(overview["global_policy"]["method"], "hold_current")
        self.assertTrue(all(paper["comparison_level"] == "method_only"
                            for paper in overview["papers"]))
        demand = overview["demand_holdout"]
        selected = next(row for row in demand if row["split"] == "test"
                        and row["segment"] == "in_support"
                        and row["model"] == overview["demand_selected_model"])
        naive = next(row for row in demand if row["split"] == "test"
                     and row["segment"] == "in_support"
                     and row["model"] == "naive_last")
        self.assertEqual(selected["n"], naive["n"])
        self.assertGreater(selected["rmse"], naive["rmse"])

    def test_validation_rule_reproduces_all_four_selected_method_types(self):
        for method in ("hold_current", "heuristic", "ppo_frozen",
                       "forecast_grid_ensemble"):
            item = next(item for item in self.options["items"]
                        if item["selected_method"] == method)
            result = price_recommend(PriceRequest(
                sku=item["sku"], week=item["weeks"][0]["week"]))
            with self.subTest(method=method):
                self.assertEqual(result["method_used"], method)
                self.assertTrue(result["historical_reproduction"])
                self.assertIn(len(result["curve"]), (81, 82))
                self.assertEqual(len(result["context"]["history"]), 12)
                self.assertGreaterEqual(result["recommended_price"],
                                        result["price_limits"]["local_min"] - 1e-8)
                self.assertLessEqual(result["recommended_price"],
                                     result["price_limits"]["local_max"] + 1e-8)

    def test_entered_current_price_does_not_replace_observed_history(self):
        item = self.options["items"][0]
        week = item["weeks"][0]["week"]
        current = (item["train_min"] + item["train_max"]) / 2
        result = price_recommend(PriceRequest(
            sku=item["sku"], week=week, current_price=current))
        self.assertAlmostEqual(result["reference_price"], current)
        self.assertAlmostEqual(result["recommended_price"], current)
        self.assertAlmostEqual(result["context"]["previous_week_price"],
                               item["weeks"][0]["reference_price"])
        self.assertFalse(result["historical_reproduction"])

    def test_comparison_reproduces_each_method_on_one_shared_decision(self):
        item = next(row for row in self.options["items"]
                    if row["selected_method"] == "ppo_frozen")
        result = price_compare(PriceRequest(sku=item["sku"], week=item["weeks"][0]["week"]))
        comparison = result["comparison"]
        rows = comparison["rows"]
        self.assertEqual({row["method"] for row in rows}, set(METHODS))
        self.assertEqual(sum(row["is_active"] for row in rows), 1)
        self.assertTrue(comparison["validation_scores_use_historical_reference"])
        active = next(row for row in rows if row["is_active"])
        hold = next(row for row in rows if row["method"] == "hold_current")
        self.assertEqual(active["method"], result["method_used"])
        self.assertEqual(active["seed"], result["seed"])
        self.assertAlmostEqual(active["recommended_price"], result["recommended_price"])
        self.assertAlmostEqual(active["forecast_revenue"], result["forecast"]["revenue_recommended"])
        self.assertAlmostEqual(active["simulator_revenue"], result["simulator"]["revenue_recommended"])
        self.assertAlmostEqual(hold["recommended_price"], result["reference_price"])
        self.assertAlmostEqual(hold["simulator_uplift_pct"], 0)
        self.assertAlmostEqual(active["validation_mean_uplift_pct"],
                               item["validation_uplift_pct"])
        self.assertTrue(all(result["price_limits"]["local_min"] - 1e-8
                            <= row["recommended_price"]
                            <= result["price_limits"]["local_max"] + 1e-8
                            for row in rows))

    def test_comparison_custom_price_and_direct_seed_keep_observed_history(self):
        item = next(row for row in self.options["items"]
                    if row["selected_method"] == "ppo_frozen")
        current = (item["train_min"] + item["train_max"]) / 2
        result = price_compare(PriceRequest(sku=item["sku"],
            week=item["weeks"][0]["week"], current_price=current,
            method="ppo_frozen", seed=42))
        active = next(row for row in result["comparison"]["rows"] if row["is_active"])
        self.assertEqual(active["seed"], 42)
        self.assertEqual(active["seed_source"], "user_choice")
        self.assertAlmostEqual(active["recommended_price"], result["recommended_price"])
        self.assertAlmostEqual(result["reference_price"], current)
        self.assertAlmostEqual(result["context"]["previous_week_price"],
                               item["weeks"][0]["reference_price"])
        self.assertFalse(result["comparison"]["validation_scores_use_historical_reference"])

    def test_rejects_unsupported_week_price_and_seed(self):
        item = self.options["items"][0]
        week = item["weeks"][0]["week"]
        cases = [
            PriceRequest(sku=item["sku"], week="2010-01-04"),
            PriceRequest(sku=item["sku"], week=week,
                         current_price=item["train_max"] + 100),
            PriceRequest(sku=item["sku"], week=week,
                         method="validation_rule", seed=42),
        ]
        for request in cases:
            with self.subTest(request=request.model_dump()):
                with self.assertRaises(HTTPException) as caught:
                    price_recommend(request)
                self.assertEqual(caught.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
