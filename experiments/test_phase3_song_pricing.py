"""Check that Phase 3 policy episodes cannot see held-out sales or prices."""

import unittest

import pandas as pd
import torch

from experiments.data import make_windows


class Phase3ValidationTests(unittest.TestCase):
    def setUp(self):
        weeks = pd.date_range("2020-01-06", periods=18, freq="7D")
        self.validation_start = weeks[14]
        self.weekly = pd.DataFrame({
            "sku": "A", "week": weeks,
            "split": ["train"] * 14 + ["validation"] * 3 + ["test"],
            "observed_sales": True, "qty": range(18), "price": 2.0, "qty_z": 0.0,
            "price_z": 0.0, "event_days": [i % 4 for i in range(18)], "snap_days": 0,
        })

    def test_policy_inputs_are_invariant_to_held_out_prices_and_quantities(self):
        from experiments.phase3_song_pricing import validation_inputs

        original = validation_inputs(make_windows(self.weekly), ["A"], self.validation_start, 3)
        altered = self.weekly.copy()
        heldout = altered["split"].ne("train")
        altered.loc[heldout, ["price", "qty"]] = 9999
        actual = validation_inputs(make_windows(altered), ["A"], self.validation_start, 3)
        pd.testing.assert_frame_equal(actual, original)
        self.assertNotIn("qty", actual)
        self.assertNotIn("price", actual)
        self.assertEqual(actual.iloc[0]["qty_lag_1"], 13)
        self.assertEqual(actual.iloc[0]["price_lag_1"], 2)
        self.assertTrue(actual.iloc[1:]["qty_lag_1"].isna().all())
        self.assertEqual(actual.iloc[1]["event_days_lag_1"], 2)

    def test_missing_week_fails_without_replacing_it_with_a_later_week(self):
        from experiments.phase3_song_pricing import validation_inputs

        self.weekly.loc[self.weekly["week"].eq(self.validation_start + pd.Timedelta(weeks=1)),
                        "observed_sales"] = False
        with self.assertRaisesRegex(ValueError, "A.*validation"):
            validation_inputs(make_windows(self.weekly), ["A"], self.validation_start, 2)

    def test_forecast_feedback_does_not_advance_ppo_torch_rng(self):
        from experiments.phase3_song_pricing import FeedbackForecaster
        from experiments.song_forecast import SongTransformer

        windows = make_windows(self.weekly)
        samples = [(windows.iloc[0], 2.0, 20.0)]
        scalers = {"A": {"qty_mean": 12.0, "qty_std": 2.0, "price_mean": 2.0, "price_std": 1.0}}
        with torch.random.fork_rng(devices=[]):
            forecaster = FeedbackForecaster(SongTransformer(), scalers, seed=42)
            before = torch.get_rng_state()
            feedback_before = forecaster.feedback_rng_state.clone()
            forecaster.fine_tune(samples)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            self.assertFalse(torch.equal(forecaster.feedback_rng_state, feedback_before))
