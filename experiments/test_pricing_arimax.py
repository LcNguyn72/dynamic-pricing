import unittest

import numpy as np


class PricingArimaxTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        price = rng.uniform(3, 6, 100)
        qty = 200 - 18 * price + rng.normal(0, 2, len(price))
        self.price = price
        self.qty = qty
        self.variant = "arimax_price"

    def test_current_sales_cannot_influence_same_week_forecast(self):
        from experiments.pricing_arimax import fit_series, forecast_future

        model, status = fit_series(self.qty[:70], self.price[:70], self.variant)
        self.assertEqual(status, "ok")
        future = self.qty[70:].copy()
        baseline = forecast_future(model, self.variant, self.qty[:70], self.price[:70], future, self.price[70:])
        future[8:] *= 10
        modified = forecast_future(model, self.variant, self.qty[:70], self.price[:70], future, self.price[70:])
        np.testing.assert_allclose(modified[:9], baseline[:9])

    def test_candidate_at_same_history_changes_the_predicted_quantity(self):
        from experiments.pricing_arimax import candidate_quantities, fit_series, forecast_future

        model, status = fit_series(self.qty[:70], self.price[:70], self.variant)
        self.assertEqual(status, "ok")
        history = self.qty[:70]
        inputs = self.price[:70]
        for price in (3.1, 5.8):
            actual = forecast_future(model, self.variant, history, inputs, self.qty[70:71],
                                     np.array([price]))[0]
            expected = forecast_future(model, self.variant, history, inputs, self.qty[70:71],
                                       self.price[70:71])[0]
            candidate = candidate_quantities(expected, self.price[70], np.array([price]), model)[0]
            self.assertAlmostEqual(candidate, actual, places=9)
        self.assertLess(model["price_slope"], 0)

    def test_constant_train_price_is_ineligible_for_a_price_model(self):
        from experiments.pricing_arimax import fit_series

        model, status = fit_series(self.qty, np.full(100, 5.0), self.variant)
        self.assertEqual(status, "constant_price")
        self.assertIsNone(model)
