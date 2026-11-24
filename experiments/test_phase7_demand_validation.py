"""Focused checks for train-only folds and Phase 7 demand diagnostics."""

import unittest

import pandas as pd

from experiments.phase7_demand_validation import (
    _fit_models, _metrics, _predict_rows, _rolling_folds,
)


class DemandValidationTests(unittest.TestCase):
    def test_rolling_origins_do_not_use_or_overlap_future_blocks(self):
        weeks = list(pd.date_range("2020-01-06", periods=191, freq="7D"))
        folds = _rolling_folds(weeks)
        self.assertEqual(len(folds), 3)
        self.assertEqual([len(fold["eval_weeks"]) for fold in folds], [20, 20, 20])
        for fold in folds:
            self.assertLess(fold["fit_weeks"][-1], fold["eval_weeks"][0])
            self.assertLessEqual(fold["eval_weeks"][-1], weeks[-1])
        self.assertEqual(len(set(week for fold in folds for week in fold["eval_weeks"])), 60)

    def test_target_quantity_does_not_fit_or_enter_prediction(self):
        weeks = pd.date_range("2020-01-06", periods=25, freq="7D")
        panel = pd.DataFrame({
            "sku": "A", "week": weeks,
            "qty": [20.0 + (i % 3) for i in range(24)] + [999.0],
            "price": [1.0 if i % 2 else 1.1 for i in range(24)] + [1.0],
            "split": ["train"] * 24 + ["test"],
            "observed_sales": True,
            "event_days": 0, "snap_days": 0,
        })
        panel["qty_lag_1"] = panel["qty"].shift(1)
        panel["price_lag_1"] = panel["price"].shift(1)
        models = _fit_models(panel)
        first = _predict_rows(models, panel.iloc[-1:], "test")
        changed = panel.copy()
        changed.loc[24, "qty"] = 1.0
        second = _predict_rows(models, changed.iloc[-1:], "test")
        self.assertEqual(first["predicted_qty"].tolist(), second["predicted_qty"].tolist())
        self.assertEqual(len(models["song_kernel"].surfaces["A"].quantities), 24)

        flat = models["flat_response_0.75"]
        context = {"qty_lag_1": 20.0, "price_lag_1": 1.0,
                   "event_days": 0, "snap_days": 0}
        self.assertAlmostEqual(flat.predict_mean("A", 1.0, weeks[-1], context),
                               flat.predict_mean("A", 1.1, weeks[-1], context))

    def test_phase6_reference_and_target_support_are_separate(self):
        rows = pd.DataFrame([
            {"split": "test", "fold": None, "sku": "A", "week": "2020-01-06",
             "observed_qty": 10.0, "predicted_qty": 9.0, "model": "naive_last",
             "price_in_train_support": True},
            {"split": "test", "fold": None, "sku": "A", "week": "2020-01-13",
             "observed_qty": 10.0, "predicted_qty": 9.0, "model": "naive_last",
             "price_in_train_support": False},
        ])
        keys = {("test", "A", "2020-01-06"), ("test", "A", "2020-01-13")}
        metrics = _metrics(rows, phase6_keys=keys).set_index("segment")
        self.assertEqual(metrics.loc["phase6_pricing_eligible", "n"], 2)
        self.assertEqual(metrics.loc["phase6_pricing_and_target_in_support", "n"], 1)


if __name__ == "__main__":
    unittest.main()
