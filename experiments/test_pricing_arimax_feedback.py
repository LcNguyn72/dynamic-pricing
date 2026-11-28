"""Regression checks for continuous-price ARIMAX and its simulation feedback."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ArimaxFeedbackTests(unittest.TestCase):
    def test_order_zero_uses_closed_form_normal_regression(self):
        from experiments.pricing_arimax_feedback import fit_model

        rng = np.random.default_rng(5)
        price = rng.uniform(1, 5, 130)
        qty = 25 + 3 * price + rng.normal(0, 1, len(price))
        qty[10:17] = np.nan
        with patch("experiments.pricing_arimax_feedback.SARIMAX", side_effect=AssertionError("unneeded optimizer")):
            state = fit_model(qty, price, 0, "level")
        mask = np.isfinite(qty)
        slope = np.polyfit(price[mask], qty[mask], 1)[0]
        self.assertAlmostEqual(state["params"][1] * state["qty_sd"] / state["price_sd"], slope)

    def test_candidate_forecast_and_gradient_match_state_space_for_log_price(self):
        from experiments.pricing_arimax_feedback import ArimaxAdapter, fit_model

        rng = np.random.default_rng(19)
        price = rng.uniform(1, 5, 130)
        qty = 130 - 22 * np.log(price) + rng.normal(0, 4, len(price))
        state = fit_model(qty, price, 2, "log")
        self.assertIsNotNone(state)
        x = (np.log(price) - state["price_mean"]) / state["price_sd"]
        y = (qty - state["qty_mean"]) / state["qty_sd"]
        result = SARIMAX(y, exog=x[:, None], order=(2, 0, 0), trend="c").filter(state["params"])
        row = {"sku": "A", "qty_lag_1": qty[-1], "qty_lag_2": qty[-2],
               "price_lag_1": price[-1], "price_lag_2": price[-2]}
        adapter = ArimaxAdapter({"A": state})
        for p in (2.0, 4.0):
            future = np.array([[(np.log(p) - state["price_mean"]) / state["price_sd"]]])
            actual = result.forecast(1, exog=future)[0] * state["qty_sd"] + state["qty_mean"]
            self.assertAlmostEqual(adapter.predict_q(row, p), actual, places=8)
            epsilon = 1e-4
            numerical = (adapter.predict_q(row, p + epsilon) - adapter.predict_q(row, p - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(adapter.price_gradient(row, p), numerical, places=5)

    def test_feedback_changes_only_the_observed_skus_coefficient(self):
        from experiments.pricing_arimax_feedback import ArimaxAdapter

        state = dict(order=1, transform="level", params=[0.0, 0.0, 0.1, 1.0],
                     price_min=1.0, price_max=5.0, price_mean=3.0, price_sd=1.0,
                     qty_mean=50.0, qty_sd=10.0)
        adapter = ArimaxAdapter({"A": state, "B": state})
        row = {"sku": "A", "qty_lag_1": 50.0, "price_lag_1": 3.0}
        adapter.fine_tune([(row, 3.0, 50.0)] * 8)
        self.assertEqual(adapter.counts, {})
        adapter.fine_tune([(row, float(p), float(50 - 10 * (p - 3))) for p in (1, 2, 3, 4, 5)] * 4)
        self.assertLess(adapter.price_gradient(row, 3.0), 0)
        self.assertEqual(adapter.price_gradient({**row, "sku": "B"}, 3.0), 0)
        self.assertEqual(adapter.counts, {"A": 1})
        self.assertEqual(state["params"][1], 0)

    def test_rollout_inputs_withhold_current_prices_and_quantities(self):
        from experiments.pricing_arimax_feedback import simulation_inputs

        weeks = pd.date_range("2015-01-05", periods=16, freq="7D")
        panel = pd.DataFrame({"sku": "A", "week": weeks, "qty": np.arange(16),
                              "price": 2.0, "event_days": 0, "snap_days": 0,
                              "observed_sales": True, "split": ["train"] * 12 + ["validation"] * 4})
        left = simulation_inputs(panel, ["A"], weeks[12], 4, "validation")
        panel.loc[panel["split"].eq("validation"), ["qty", "price"]] = 999
        right = simulation_inputs(panel, ["A"], weeks[12], 4, "validation")
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(left.loc[0, "qty_lag_1"], 11.0)
        self.assertTrue(left.loc[1:, "qty_lag_1"].isna().all())
