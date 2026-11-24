import unittest

import numpy as np
import pandas as pd
import torch

from experiments.pricing_arimax import fit_series, forecast_future
from experiments.song_forecast import SongDemandForecaster, SongTransformer


class PricingPolicyCompareTests(unittest.TestCase):
    def test_arimax_adapter_matches_state_space_one_step_with_candidate_price(self):
        from experiments.pricing_policy_compare import ArimaxDemand

        rng = np.random.default_rng(9)
        prices = rng.uniform(1, 5, 100)
        quantities = 80 - 9 * prices + rng.normal(0, 3, 100)
        variant = "arimax_price"
        state, status = fit_series(quantities[:90], prices[:90], variant)
        self.assertEqual(status, "ok")
        candidates = np.array([2.0, 3.0, 4.0])
        expected = [forecast_future(state, variant, quantities[:90], prices[:90],
                                    np.array([np.nan]), np.array([p]))[0] for p in candidates]
        row = {"sku": "A", "qty_lag_1": quantities[89], "price_lag_1": prices[89],
               "qty": 100000, "price": 400}
        adapter = ArimaxDemand({"A": state})
        np.testing.assert_allclose(adapter.predict_prices(row, candidates), expected)
        with self.assertRaises(ValueError):
            adapter.predict_prices(row, np.array([np.nan]))

    def test_short_checkpoint_uses_same_features_for_training_and_pricing(self):
        from experiments.pricing_policy_compare import TunedForecaster

        model = SongTransformer(lookback=8)
        scalers = {"A": dict(qty_mean=10.0, qty_std=4.0, price_mean=2.0, price_std=1.0)}
        row = {"sku": "A", "week": pd.Timestamp("2015-08-03"), "event_days": 1, "snap_days": 0}
        for lag in range(1, 13):
            row[f"qty_lag_{lag}"] = 3 + lag
            row[f"price_lag_{lag}"] = 2.0 + lag / 20
        regular = SongDemandForecaster(model, scalers)
        fast = TunedForecaster(model, scalers, seed=42)
        tested = np.array([2.0, 2.5, 3.0])
        np.testing.assert_allclose(fast.predict_prices(row, tested),
                                   [regular.predict_q(row, p) for p in tested], rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(fast.price_gradient(row, 2.5), regular.price_gradient(row, 2.5), places=5)
        before = [p.detach().clone() for p in model.parameters()]
        fast.fine_tune([(row, 2.5, 30.0)])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)))

    def test_expansion_picks_one_series_per_training_volume_band(self):
        from experiments.pricing_policy_compare import volume_band_skus

        volumes = pd.Series({f"S{i}": i for i in range(20)}, dtype=float)
        selected = volume_band_skus(volumes, 5)
        self.assertEqual(len(set(selected)), 5)
        ranks = {int(sku[1:]) // 4 for sku in selected}
        self.assertEqual(ranks, {0, 1, 2, 3, 4})
