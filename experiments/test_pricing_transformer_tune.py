"""Protect the locked fit split and matched validation budget."""

import unittest

import pandas as pd


class PricingTransformerTuneTests(unittest.TestCase):
    def test_train_rows_exclude_validation_test_and_unobserved_targets(self):
        from experiments.pricing_transformer_tune import train_rows

        panel = pd.DataFrame({
            "split": ["train", "train", "validation", "test"],
            "observed_sales": [True, False, True, True],
            "qty": [0., 10., 20., 30.],
        })
        self.assertEqual(train_rows(panel)["qty"].tolist(), [0.])

    def test_choose_config_by_mean_validation_rmse_with_all_required_seeds(self):
        from experiments.pricing_transformer_tune import choose_config

        def trial(lookback, seed, rmse, valid=True):
            return dict(lookback=lookback, seed=seed, epoch=2, learning_rate=0.001,
                        rmse=rmse, valid_curve=valid)

        rows = [trial(8, 42, 1), trial(8, 123, 21), trial(12, 42, 10), trial(12, 123, 10),
                trial(7, 42, 1), trial(6, 42, 1), trial(6, 123, 1, valid=False)]
        self.assertEqual(choose_config(pd.DataFrame(rows), (42, 123))["lookback"], 12)
