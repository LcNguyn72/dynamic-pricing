"""Verification rejects mislabeled test rows and keeps zero sales in the metrics."""

import unittest

import pandas as pd


class Phase5ForecastAuditTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({
            "dataset": ["m5"] * 3, "sku": ["A", "A", "B"],
            "week": ["2015-08-03", "2015-08-10", "2015-08-03"],
            "qty": [0.0, 4.0, 8.0], "price": [1.0, 2.0, 3.0],
        })
        self.predictions = self.reference.assign(predicted_qty=[2.0, 4.0, 6.0])

    def test_metrics_keep_zero_sales_but_do_not_use_zero_in_mape(self):
        from experiments.phase5_verify import align_predictions, summarize, sufficient

        rows = align_predictions(self.predictions.iloc[::-1], self.reference)
        metrics = summarize(sufficient(rows).sum(), train_mean=2.0, train_std=1.0)
        self.assertAlmostEqual(metrics["rmse"], (8 / 3) ** 0.5)
        self.assertAlmostEqual(metrics["r2"], 0.75)
        self.assertAlmostEqual(metrics["mape_pct_positive_targets"], 12.5)
        self.assertEqual(metrics["mape_positive_target_count"], 2)
        self.assertEqual(metrics["n"], 3)

    def test_duplicate_or_changed_label_fails(self):
        from experiments.phase5_verify import align_predictions

        bad_keys = self.predictions.iloc[[0, 0, 2]]
        with self.assertRaisesRegex(ValueError, "keys"):
            align_predictions(bad_keys, self.reference)
        bad_labels = self.predictions.copy()
        bad_labels.loc[0, "qty"] = 1
        with self.assertRaisesRegex(ValueError, "qty"):
            align_predictions(bad_labels, self.reference)

    def test_constant_zero_actuals_make_r2_and_percentage_metrics_undefined(self):
        from experiments.phase5_verify import summarize, sufficient

        rows = self.predictions.assign(qty=0.0)
        metrics = summarize(sufficient(rows).sum(), train_mean=0.0, train_std=0.0)
        self.assertIsNone(metrics["r2"])
        self.assertIsNone(metrics["wape_pct"])
        self.assertIsNone(metrics["mape_pct_positive_targets"])
        self.assertIsNone(metrics["rmse_train_mean_relative"])


class Phase5PricingAuditTests(unittest.TestCase):
    def test_matched_unit_multiplier_detects_different_simulated_shocks(self):
        from experiments.phase5_verify import check_pricing_noise

        rows = pd.DataFrame({
            "simulator": ["gaussian_song"] * 2, "seed": [42, 42],
            "sku": ["A", "A"], "week": ["2015-08-03"] * 2,
            "chosen_price": [1.0, 2.0], "expected_revenue": [10.0, 16.0],
            "simulated_units": [15.0, 12.0],
        })
        check_pricing_noise(rows)
        rows.loc[1, "simulated_units"] = 11
        with self.assertRaisesRegex(ValueError, "noise"):
            check_pricing_noise(rows)
