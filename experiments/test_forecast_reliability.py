from collections import defaultdict
import unittest

import numpy as np
import pandas as pd

from experiments.m5_demand_coverage import _store_scores, borrow_price_state, price_effect
from experiments.uci_xgb_reliability import build_inputs


class ForecastReliabilityTests(unittest.TestCase):
    def test_uci_features_use_only_own_prior_weeks(self):
        weeks = list(pd.date_range("2010-12-06", periods=4, freq="7D"))
        panel = pd.DataFrame({"sku": ["A"] * 4 + ["B"] * 4, "week": weeks * 2,
                              "qty": [1, 0, 5, 10, 201, 202, 203, 204]})
        features, baseline = build_inputs(panel)
        self.assertAlmostEqual(baseline.iloc[3], 2.0)
        self.assertTrue(np.isnan(features.at[4, "qty_lag_1"]))
        self.assertEqual(features.at[2, "christmas_week"], 1)
        panel.loc[3, "qty"] = 200
        pd.testing.assert_series_equal(build_inputs(panel)[0].loc[3], features.loc[3])

    def test_m5_transfer_requires_two_donors_at_the_anchor(self):
        donor = {"price_min": 8, "price_max": 12, "price_mean": 10,
                 "qty_mean": 10, "price_slope": -1}
        self.assertIsNone(borrow_price_state(10, 20, [donor]))
        self.assertIsNone(borrow_price_state(15, 20, [donor, donor]))
        state = borrow_price_state(10, 20, [donor, donor])
        self.assertAlmostEqual(state["slope"], -2)
        self.assertAlmostEqual(state["low"], 9.5)
        self.assertAlmostEqual(state["high"], 10.5)

    def test_m5_price_effect_conditions_on_previous_and_candidate_price(self):
        self.assertEqual(price_effect(10, 10, 10, 0.5, -2), 0)
        self.assertEqual(price_effect(11, 10, 10, 0.5, -2), -2)
        self.assertEqual(price_effect(11, 11, 10, 0.5, -2), -1)

    def test_m5_fixed_price_evidence_excludes_weeks_at_another_price(self):
        panel = pd.DataFrame({"sku": ["A"] * 3,
                              "week": pd.date_range("2020-01-06", periods=3, freq="7D"),
                              "qty": [20, 22, 100], "price": [10, 10, 11],
                              "split": ["train", "test", "test"], "observed_sales": [True] * 3})
        state = pd.DataFrame([{"sku": "A", "status": "fixed_price_only",
                               "train_price_reference": 10, "phi": 0.5, "qty_mean": 20, "center": 0}])
        totals = defaultdict(lambda: np.zeros(5))
        keys = defaultdict(set)
        _store_scores(panel, state, totals, keys)
        scope = ("test", "fixed_price_at_train_reference")
        self.assertEqual(keys[scope], {"A"})
        self.assertEqual(totals[(*scope, 0.0)][:3].tolist(), [1, 2, 4])
