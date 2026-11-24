import unittest

import numpy as np
import pandas as pd

from experiments.price_feature_ablation import candidate_features


class CandidateFeaturesTests(unittest.TestCase):
    def test_candidate_replaces_target_price_and_its_relative_change(self):
        row = pd.Series({"qty_lag_1": 20.0, "price": 4.0, "price_lag_1": 2.0, "price_change": 1.0})
        rows = candidate_features(row, np.array([1.0, 3.0]), 2.0)
        np.testing.assert_allclose(rows["price"], [1, 3])
        np.testing.assert_allclose(rows["price_change"], [-0.5, 0.5])
        np.testing.assert_allclose(rows["qty_lag_1"], [20, 20])
        self.assertEqual(row["price"], 4.0)
        self.assertEqual(row["price_change"], 1.0)

    def test_price_change_rounds_after_division_as_during_training(self):
        prior = 1.23456789
        prices = np.array([1.00000013, prior, 1.67])
        row = pd.Series({"price": 4.0, "price_lag_1": np.float32(prior), "price_change": 1.0})
        rows = candidate_features(row, prices, prior)
        np.testing.assert_array_equal(rows["price_change"], (prices / prior - 1).astype(np.float32))
