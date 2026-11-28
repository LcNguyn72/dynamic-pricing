import unittest

import numpy as np
import pandas as pd

from experiments.forecast_feature_ablation import build_features, metric_totals, scores


class FeatureAblationTests(unittest.TestCase):
    def setUp(self):
        weeks = pd.date_range("2020-01-06", periods=5, freq="W-MON")
        self.panel = pd.DataFrame({
            "sku": np.repeat(["A", "B"], len(weeks)),
            "week": weeks.tolist() * 2,
            "qty": [2, 0, 6, 8, 10, 11, 12, 13, 14, 15],
            "price": [2.0, 2.0, 4.0, 4.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "observed_sales": [True, True, False, True, True] + [True] * 5,
            "event_days": [0, 0, 1, 1, 2] * 2,
            "snap_days": [0, 1, 2, 2, 3] * 2,
        })

    def test_sales_features_use_only_earlier_weeks_within_sku(self):
        features = build_features(self.panel)
        self.assertTrue(np.isnan(features.loc[5, "qty_lag_1"]))
        self.assertEqual(features.loc[2, "qty_lag_1"], 0)
        self.assertTrue(np.isnan(features.loc[3, "qty_lag_1"]))
        self.assertEqual(features.loc[3, "qty_last"], 0)
        self.assertAlmostEqual(features.loc[4, "qty_mean_4"], (2 + 0 + 8) / 3, places=6)
        changed = self.panel.copy()
        changed.loc[4, "qty"] = 100_000
        pd.testing.assert_series_equal(features.loc[4], build_features(changed).loc[4])

    def test_known_calendar_and_target_price_can_change_without_using_target_qty(self):
        features = build_features(self.panel)
        self.assertEqual(features.loc[4, "event_days"], 2)
        self.assertEqual(features.loc[4, "price"], 3.0)
        self.assertEqual(features.loc[4, "price_lag_1"], 4.0)
        self.assertAlmostEqual(features.loc[4, "price_change"], -0.25)
        self.assertTrue(np.isnan(features.loc[3, "price_lag_1"]))

    def test_pooled_metrics_equal_direct_metrics(self):
        actual = np.array([1., 2., 10., 50.])
        prediction = np.array([3., 2., 12., 45.])
        pooled = metric_totals(actual[:2], prediction[:2]) + metric_totals(actual[2:], prediction[2:])
        result = scores(pooled)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mae"], 2.25)
        self.assertAlmostEqual(result["rmse"], np.sqrt(33 / 4))
        self.assertEqual(result, scores(metric_totals(actual, prediction)))
