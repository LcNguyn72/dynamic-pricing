"""Regression checks for normalization by the matching seed and simulator."""

import unittest

import pandas as pd


class PricingNormalizedReportTests(unittest.TestCase):
    def test_mean_of_seed_indices_uses_the_matched_reference(self):
        from experiments.pricing_normalized_report import normalize_runs, summarize

        rows = pd.DataFrame([
            ("a", 42, "base", 10.0), ("a", 42, "policy", 20.0),
            ("a", 123, "base", 100.0), ("a", 123, "policy", 100.0),
            ("b", 42, "base", 40.0), ("b", 42, "policy", 20.0),
            ("b", 123, "base", 50.0), ("b", 123, "policy", 50.0),
        ], columns=["simulator", "seed", "policy", "raw_revenue"])
        result = normalize_runs(rows, "base", ["base", "policy"], [42, 123], ["a", "b"])
        self.assertTrue(result[result["policy"].eq("base")]["revenue_index"].eq(100).all())
        table = summarize(result).set_index(["simulator", "policy"])
        self.assertEqual(table.loc[("a", "policy"), "index_mean"], 150.0)
        self.assertEqual(table.loc[("a", "policy"), "index_min"], 100.0)
        self.assertEqual(table.loc[("b", "policy"), "index_mean"], 75.0)
        self.assertNotEqual(table.loc[("a", "policy"), "index_mean"], 100 * 60.0 / 55.0)

    def test_missing_rows_and_zero_baseline_are_rejected(self):
        from experiments.pricing_normalized_report import normalize_runs

        columns = ["simulator", "seed", "policy", "raw_revenue"]
        base = pd.DataFrame([("a", 42, "base", 0.0), ("a", 42, "policy", 100.0)], columns=columns)
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_runs(base, "base", ["base", "policy"], [42], ["a"])
        base.loc[0, "raw_revenue"] = 50.0
        with self.assertRaisesRegex(ValueError, "missing|duplicate|unexpected"):
            normalize_runs(base.iloc[:1], "base", ["base", "policy"], [42], ["a"])
