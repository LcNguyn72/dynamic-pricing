"""Check validation boundaries and the ARIMA fallback used by Phase 4."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


class Phase4ForecastTests(unittest.TestCase):
    def test_sparse_arima_uses_only_latest_observation_before_each_decision(self):
        from experiments.phase4_forecast import arima_validation

        weeks = pd.date_range("2020-01-06", periods=5, freq="7D")
        weekly = pd.DataFrame({
            "sku": "A", "week": weeks, "split": ["train"] * 3 + ["validation"] * 2,
            "qty": [10, 12, 0, 17, 9999], "observed_sales": [True, True, False, True, True],
        })
        predicted, fallback = arima_validation(weekly, weekly.iloc[3:])
        self.assertEqual(list(predicted), [12, 17])
        self.assertEqual(fallback, {"rows": 2, "skus": 1})

    def test_forecast_refuses_any_test_input(self):
        from experiments.phase4_forecast import arima_validation

        weekly = pd.DataFrame({"split": ["train", "validation", "test"]})
        with self.assertRaisesRegex(ValueError, "only train and validation"):
            arima_validation(weekly, weekly.iloc[1:2])

    def test_failed_arima_fit_or_nonfinite_forecast_uses_fallback(self):
        from experiments.phase4_forecast import arima_validation

        weekly = pd.DataFrame({
            "sku": "A", "week": pd.date_range("2020-01-06", periods=13, freq="7D"),
            "split": ["train"] * 12 + ["validation"], "qty": list(range(1, 13)) + [9999],
            "observed_sales": True,
        })
        for converged, yhat in ((False, 41.0), (True, np.nan)):
            with self.subTest(converged=converged), patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as fit:
                fitted = Mock(mle_retvals={"converged": converged})
                fitted.forecast.return_value = [yhat]
                fitted.append.return_value = fitted
                fit.return_value.fit.return_value = fitted
                prediction, fallback = arima_validation(weekly, weekly.iloc[-1:])
                self.assertEqual(list(prediction), [12])
                self.assertEqual(fallback, {"rows": 1, "skus": 1})


class Phase4PricingTests(unittest.TestCase):
    def test_grid_uses_predicted_revenue_and_keeps_reference_on_ties(self):
        from experiments.phase4_pricing import grid_price

        prices = np.array([1.0, 1.2, 1.5])
        self.assertEqual(grid_price(prices, np.array([18.0, 16.0, 10.0]), 1.0), 1.2)
        self.assertEqual(grid_price(prices, np.array([30.0, 25.0, 20.0]), 1.2), 1.2)
