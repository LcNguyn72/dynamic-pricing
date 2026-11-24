"""Focused invariants for fair forecast scoring and simulator gradients."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.forecast_benchmark import _score, _validate_windows


class ForecastBenchmarkTests(unittest.TestCase):
    def test_mape_excludes_zero_targets_but_other_metrics_include_them(self):
        rows = pd.DataFrame({"qty": [0.0, 10.0]})
        score = _score(rows, np.array([2.0, 8.0]))
        self.assertEqual(score["n"], 2)
        self.assertEqual(score["mape_n"], 1)
        self.assertAlmostEqual(score["mape_pct"], 20.0)
        self.assertAlmostEqual(score["wape_pct"], 40.0)
        self.assertAlmostEqual(score["rmse"], 2.0)

    def test_future_training_week_is_rejected(self):
        rows = []
        for week, split in zip(pd.date_range("2020-01-06", periods=4, freq="7D"),
                               ("train", "test", "train", "test")):
            rows.append({
                "sku": "A", "week": week, "qty": 2.0, "price": 1.0,
                "split": split, "observed_sales": True, "month": week.month,
                "week_of_year": week.isocalendar().week,
                "qty_lag_1": 1.0, "price_lag_1": 1.0,
            })
        with self.assertRaisesRegex(ValueError, "Test must begin"):
            _validate_windows(pd.DataFrame(rows), lookback=1)

    def test_price_gradient_matches_finite_difference(self):
        try:
            import torch
            from experiments.forecast_torch import PriceForecaster, SequenceForecaster
        except ImportError:
            self.skipTest("PyTorch is optional")
        torch.manual_seed(7)
        model = SequenceForecaster(
            architecture="transformer", lookback=3,
            history_dim=2, auxiliary_dim=6,
            transformer_dim=16, transformer_heads=4, transformer_layers=1,
        )
        meta = {
            "config": {"lookback": 3}, "with_price": True,
            "sku_to_idx": {"A": 0},
            "scalers": {"A": {"qty_mean": 10.0, "qty_std": 3.0,
                               "price_mean": 5.0, "price_std": 2.0}},
        }
        forecaster = PriceForecaster(model, meta)
        row = {
            "sku": "A", "month": 2, "week_of_year": 7,
            "qty_lag_1": 9.0, "qty_lag_2": 12.0, "qty_lag_3": 8.0,
            "price_lag_1": 5.1, "price_lag_2": 5.0, "price_lag_3": 4.8,
        }
        price = 5.2
        epsilon = 0.001
        automatic = forecaster.price_gradient(row, price)
        numerical = (forecaster.predict_q(row, price + epsilon)
                     - forecaster.predict_q(row, price - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(automatic, numerical, delta=0.02)

    def test_m5_adapter_features_and_simulated_window(self):
        try:
            import torch
            from experiments.forecast_torch import (
                PriceForecaster, SequenceForecaster, feature_arrays,
            )
        except ImportError:
            self.skipTest("PyTorch is optional")
        from experiments.data import make_windows
        from experiments.forecast_benchmark import _tabular_features
        from experiments.m5 import load_m5_weekly

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = pd.date_range("2015-01-05", periods=24 * 7, freq="D")
            days = [f"d_{i + 1}" for i in range(len(dates))]
            calendar = []
            for i, date in enumerate(dates):
                calendar.append({
                    "date": date.date().isoformat(), "wm_yr_wk": str(100 + i // 7),
                    "d": days[i],
                    "event_name_1": "Holiday" if i == 5 * 7 + 1 else None,
                    "event_type_1": "National" if i == 5 * 7 + 1 else None,
                    "event_name_2": None, "event_type_2": None,
                    "snap_CA": int(i % 7 == 2), "snap_TX": 0, "snap_WI": 0,
                })
            pd.DataFrame(calendar).to_csv(root / "calendar.csv", index=False)
            pd.DataFrame([
                {"item_id": "FOODS_1_001", "store_id": "CA_1",
                 "wm_yr_wk": str(100 + week), "sell_price": 10.0 + 0.05 * week}
                for week in range(24)
            ]).to_csv(root / "sell_prices.csv", index=False)
            sales = {
                "id": "FOODS_1_001_CA_1_validation", "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": "CA_1",
                "state_id": "CA",
                **{day: 1 + (i // 7) % 3 for i, day in enumerate(days)},
            }
            pd.DataFrame([sales]).to_csv(root / "sales_train_validation.csv", index=False)
            weekly, manifest = load_m5_weekly(root)
            windows = make_windows(weekly, lookback=3)
            self.assertIn("event_days", windows)
            self.assertIn("snap_days", windows)
            event = windows.loc[windows["event_days"].gt(0)].iloc[0]
            sku = str(event["sku"])
            skus = {sku: 0}
            covariates = ("event_days", "snap_days")
            history, auxiliary = feature_arrays(
                pd.DataFrame([event]), skus, manifest["scalers"], 3, True, covariates
            )
            tabular = _tabular_features(
                pd.DataFrame([event]), [sku], manifest["scalers"], 3, True, covariates
            )
            self.assertEqual(history.shape, (1, 3, 2))
            self.assertEqual(auxiliary.shape, (1, 8))
            self.assertAlmostEqual(float(auxiliary[0, 4]), 1 / 7)
            self.assertAlmostEqual(float(auxiliary[0, 5]), 1 / 7)
            self.assertAlmostEqual(float(tabular[0, 10]), 1 / 7)
            self.assertAlmostEqual(float(tabular[0, 11]), 1 / 7)
            model = SequenceForecaster(
                "transformer", lookback=3, history_dim=2,
                auxiliary_dim=8, transformer_dim=16,
                transformer_heads=4, transformer_layers=1,
            )
            forecaster = PriceForecaster(model, {
                "config": {"lookback": 3}, "with_price": True,
                "sku_to_idx": skus, "scalers": manifest["scalers"],
                "calendar_covariates": list(covariates),
            })
            simulated = event.copy()
            simulated["qty_lag_1"] = 100.0
            simulated["price_lag_1"] = 11.0
            self.assertTrue(np.isfinite(forecaster.predict_q(simulated, 10.5)))
            self.assertTrue(np.isfinite(forecaster.price_gradient(simulated, 10.5)))
            from experiments.forecast_benchmark import DEFAULT_MODELS, run_forecast_benchmark
            from experiments.forecast_torch import load_price_forecaster
            manifest["lookback_weeks"] = 3
            metrics, predictions = run_forecast_benchmark(
                windows, root / "forecast", manifest,
                models=DEFAULT_MODELS, torch_epochs=1, strict=True,
            )
            self.assertEqual(set(metrics.loc[metrics["split"].eq("test"), "model"]), set(DEFAULT_MODELS))
            self.assertEqual(predictions.loc[predictions["split"].eq("test"), "week"].nunique(), 4)
            saved = load_price_forecaster(root / "forecast" / "models")
            self.assertEqual(saved.calendar_covariates, covariates)
            self.assertTrue(np.isfinite(saved.predict_q(simulated, 10.5)))
            train_only = load_price_forecaster(root / "forecast" / "models", train_only=True)
            self.assertEqual(train_only.metadata["checkpoint_scope"], "train_only")
            self.assertTrue(np.isfinite(train_only.price_gradient(simulated, 10.5)))


if __name__ == "__main__":
    unittest.main()
