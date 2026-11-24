"""Core invariants for the paper-aligned forecaster."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.song_forecast import SongDemandForecaster, SongTransformer, feature_arrays


class SongForecastTests(unittest.TestCase):
    def setUp(self):
        row = {"sku": "A", "week": pd.Timestamp("2011-10-10"),
               "qty": 9999.0, "price": 2.0, "event_days": 1, "snap_days": 2}
        for lag in range(1, 13):
            row[f"qty_lag_{lag}"] = 10 + lag
            row[f"price_lag_{lag}"] = 2 + lag / 100
            row[f"event_days_lag_{lag}"] = lag % 3
            row[f"snap_days_lag_{lag}"] = lag % 2
        self.row = row
        self.scalers = {"A": {"qty_mean": 20.0, "qty_std": 5.0,
                               "price_mean": 2.0, "price_std": 0.5}}

    def test_current_target_is_not_a_model_feature(self):
        original = feature_arrays(pd.DataFrame([self.row]), self.scalers)
        altered = feature_arrays(pd.DataFrame([{**self.row, "qty": 1e9}]), self.scalers)
        np.testing.assert_array_equal(original[0], altered[0])
        np.testing.assert_array_equal(original[1], altered[1])
        self.assertEqual(original[0].shape, (1, 12, 9))

    def test_shorter_history_uses_the_most_recent_weeks_in_training_and_pricing(self):
        frame = pd.DataFrame([self.row])
        full, full_current = feature_arrays(frame, self.scalers)
        history, current = feature_arrays(frame, self.scalers, lookback=8)
        np.testing.assert_array_equal(history, full[:, -8:])
        np.testing.assert_array_equal(current, full_current)
        forecaster = SongDemandForecaster(SongTransformer(lookback=8), self.scalers)
        price = 2.0
        prediction = forecaster.predict_q(self.row, price)
        gradient = forecaster.price_gradient(self.row, price)
        self.assertTrue(np.isfinite(prediction))
        self.assertTrue(np.isfinite(gradient))
        with self.assertRaises(ValueError):
            feature_arrays(frame, self.scalers, lookback=13)

    def test_price_gradient_matches_finite_difference(self):
        model = SongDemandForecaster(SongTransformer(), self.scalers)
        price = 2.0
        epsilon = 1e-2
        automatic = model.price_gradient(self.row, price)
        numerical = (model.predict_q(self.row, price + epsilon)
                     - model.predict_q(self.row, price - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(automatic, numerical, delta=max(0.1, abs(numerical) * 0.08))


if __name__ == "__main__":
    unittest.main()
