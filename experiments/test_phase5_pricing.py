"""Phase 5 pricing boundaries and equivalence of its faster inference paths."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator
from experiments.song_forecast import LOOKBACK, SongTransformer, feature_arrays


class PricingTestInputsTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range("2015-05-11", periods=LOOKBACK + 3, freq="7D")
        start = dates[LOOKBACK]
        frame = pd.DataFrame({
            "sku": "X@CA_1", "week": dates,
            "split": np.where(dates < start, "validation", "test"),
            "observed_sales": True, "price": np.linspace(1.3, 3.4, len(dates)),
            "qty": np.arange(len(dates), dtype=float),
            "event_days": np.arange(len(dates)) % 3, "snap_days": np.arange(len(dates)) % 7,
        })
        return frame, start

    def test_heldout_prices_and_units_do_not_enter_episode(self):
        from experiments.phase5_pricing import test_inputs

        frame, start = self.fixture()
        inputs = test_inputs(frame, ["X@CA_1"], start, 3)
        heldout = frame["week"].ge(start)
        modified = frame.copy()
        modified.loc[heldout, ["qty", "price"]] = np.nan
        pd.testing.assert_frame_equal(inputs, test_inputs(modified, ["X@CA_1"], start, 3))
        self.assertEqual(inputs.iloc[0]["qty_lag_1"], 11)
        self.assertTrue(inputs.iloc[1:]["qty_lag_1"].isna().all())
        self.assertEqual(inputs.iloc[1]["event_days_lag_1"], frame.iloc[LOOKBACK]["event_days"])

    def test_never_replaces_an_unobserved_week(self):
        from experiments.phase5_pricing import test_inputs

        frame, start = self.fixture()
        frame.loc[frame["week"].eq(start), "observed_sales"] = False
        with self.assertRaisesRegex(ValueError, "locked test"):
            test_inputs(frame, ["X@CA_1"], start, 3)

    def test_fast_song_features_match_shared_function(self):
        from experiments.phase5_pricing import FastFeedbackForecaster, test_inputs

        frame, start = self.fixture()
        row = test_inputs(frame, ["X@CA_1"], start, 3).iloc[0]
        scalers = {"X@CA_1": {"qty_mean": 3.6, "qty_std": 2.12,
                               "price_mean": 2.37, "price_std": .82}}
        fast = FastFeedbackForecaster(SongTransformer(), scalers, seed=42)
        h, c = fast._tensor(row, 1.88)
        expected_h, expected_c = feature_arrays(pd.DataFrame([row.to_dict()]), scalers, np.array([1.88]))
        self.assertTrue(torch.equal(h, torch.from_numpy(expected_h)))
        self.assertTrue(torch.equal(c, torch.from_numpy(expected_c)))

    def test_song_prices_share_history_without_changing_predictions(self):
        from experiments.phase4_pricing import _song_quantities
        from experiments.phase5_pricing import FastFeedbackForecaster, test_inputs

        frame, start = self.fixture()
        row = test_inputs(frame, ["X@CA_1"], start, 3).iloc[0]
        scalers = {"X@CA_1": {"qty_mean": 34.6, "qty_std": 2.12,
                               "price_mean": 2.37, "price_std": .82}}
        fast = FastFeedbackForecaster(SongTransformer(), scalers, seed=42)
        candidates = np.linspace(1.3, 3.4, 42)
        np.testing.assert_allclose(fast.predict_prices(row, candidates),
                                   _song_quantities(fast, row, candidates), atol=1e-4, rtol=1e-6)


class PricingOracleTests(unittest.TestCase):
    def test_cached_oracle_matches_original_with_rolling_context(self):
        from experiments.phase5_pricing import CachedGaussian, CachedLag

        frame = pd.DataFrame({"sku": "X", "week": pd.date_range("2020-12-28", periods=24, freq="7D"),
                              "split": "train", "observed_sales": True,
                              "price": np.linspace(1, 2, 24), "qty": np.arange(24) + 1,
                              "event_days": np.arange(24) % 3, "snap_days": np.arange(24) % 4})
        week = pd.Timestamp("2021-12-27")
        for expected, actual in ((KernelDemandSimulator(), CachedGaussian()),
                                 (LagAdjustedKernelSimulator(.75), CachedLag(.75))):
            expected.fit(frame)
            actual.fit(frame)
            for last_qty in (0, 15, 43):
                context = {"event_days": 2, "snap_days": 1, "qty_lag_1": last_qty, "price_lag_1": 1.42}
                np.testing.assert_allclose(
                    actual.oracle_revenue("X", week, context=context),
                    expected.oracle_revenue("X", week, context=context), rtol=1e-13)


class PricingResumeTests(unittest.TestCase):
    def test_reuses_completed_unit_and_rejects_a_corrupted_file(self):
        from experiments.phase5_pricing import _unit

        seen = []

        def train(stage):
            seen.append("fit")
            (stage / "policy.zip").write_bytes(b"synthetic policy data")
            return {"training_steps": 512}

        with TemporaryDirectory() as work:
            unit = Path(work) / "seed_42" / "ddpg"
            original = _unit(unit, {"seed": 42}, train)
            self.assertEqual(original, _unit(unit, {"seed": 42}, train))
            self.assertEqual(seen, ["fit"])
            (unit / "policy.zip").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _unit(unit, {"seed": 42}, train)
