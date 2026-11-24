"""Behavioral checks for Phase 1 held-out keys and train-only pricing selection."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.phase1_finalize import (
    _check_scalers, _require_full_week_grid, _require_panel_matches_source,
    _require_past_lags, _split_counts,
    select_pricing_series, write_test_keys,
)


class Phase1FinalizeTests(unittest.TestCase):
    def test_full_grid_audit_rejects_one_missing_sku_week(self) -> None:
        weeks = pd.date_range("2020-01-06", periods=3, freq="7D")
        rows = pd.DataFrame([
            {"sku": sku, "week": week}
            for sku in ("A", "B") for week in weeks
            if not (sku == "B" and week == weeks[1])
        ])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _require_full_week_grid(rows, series=2, weeks=3)

    def test_uci_audit_rejects_stale_panel_quantities(self) -> None:
        source = pd.DataFrame([
            {"sku": "A", "week": pd.Timestamp("2020-01-06"), "qty": 5, "price": 2.0},
        ])
        panel = pd.DataFrame([
            {"sku": "A", "week": pd.Timestamp("2020-01-06"), "qty": 4, "price": 2.0,
             "observed_sales": True},
        ])
        with self.assertRaisesRegex(ValueError, "upstream"):
            _require_panel_matches_source(panel, source)

    def test_lag_audit_catches_corrupted_week_two_lag(self) -> None:
        weekly = pd.DataFrame([
            {"sku": "A", "week": pd.Timestamp("2020-01-06"), "qty": 1, "price": 2},
            {"sku": "A", "week": pd.Timestamp("2020-01-13"), "qty": 3, "price": 4},
            {"sku": "A", "week": pd.Timestamp("2020-01-20"), "qty": 5, "price": 6},
        ])
        windows = pd.DataFrame([{
            "sku": "A", "week": pd.Timestamp("2020-01-20"),
            "qty_lag_1": 3, "price_lag_1": 4,
            "qty_lag_2": 999, "price_lag_2": 2,
        }])
        with self.assertRaisesRegex(ValueError, "qty_lag_2"):
            _require_past_lags(windows, weekly, lookback=2)

    def test_split_audit_rejects_missing_calendar_week(self) -> None:
        weekly = pd.DataFrame({
            "week": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20",
                                    "2020-02-03", "2020-02-10", "2020-02-17"]),
            "split": ["train", "train", "validation", "validation", "test", "test"],
        })
        expected = {"train_weeks": 2, "validation_weeks": 2, "test_weeks": 2,
                    "train_end": "2020-01-13", "test_start": "2020-02-10"}
        with self.assertRaisesRegex(ValueError, "calendar gap"):
            _split_counts(weekly, expected)

    def test_panel_scaling_uses_train_even_without_saved_scaler_manifest(self) -> None:
        weekly = pd.DataFrame([
            {"sku": "A", "week": pd.Timestamp("2020-01-06"), "split": "train",
             "observed_sales": True, "qty": 1, "price": 2, "qty_z": -1, "price_z": -1},
            {"sku": "A", "week": pd.Timestamp("2020-01-13"), "split": "train",
             "observed_sales": True, "qty": 3, "price": 4, "qty_z": 1, "price_z": 1},
            {"sku": "A", "week": pd.Timestamp("2020-01-20"), "split": "test",
             "observed_sales": True, "qty": 101, "price": 100,
             "qty_z": 99, "price_z": 97},
        ])
        _check_scalers(weekly, None)
        weekly.loc[2, "price_z"] = 0
        with self.assertRaisesRegex(ValueError, "price_z"):
            _check_scalers(weekly, None)

    def test_test_key_bytes_are_sorted_and_only_observed_targets_are_scored(self) -> None:
        rows = pd.DataFrame([
            {"sku": "B", "week": pd.Timestamp("2020-01-13"), "split": "test",
             "observed_sales": True, "qty": 0},
            {"sku": "A", "week": pd.Timestamp("2020-01-13"), "split": "test",
             "observed_sales": False, "qty": 0},
            {"sku": "A", "week": pd.Timestamp("2020-01-06"), "split": "test",
             "observed_sales": True, "qty": 5},
            {"sku": "B", "week": pd.Timestamp("2019-12-30"), "split": "train",
             "observed_sales": True, "qty": 3},
        ])
        expected = b"dataset,sku,week\nm5,A,2020-01-06\nm5,B,2020-01-13\n"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "keys.csv"
            result = write_test_keys([rows], "m5", path)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())

    def test_pricing_series_uses_train_only_and_sku_tie_break(self) -> None:
        weeks = pd.date_range("2020-01-06", periods=16, freq="7D")
        rows = []
        for sku in ("A@CA_1", "B@CA_1", "C@CA_1"):
            for i, week in enumerate(weeks):
                rows.append({
                    "sku": sku, "week": week,
                    "split": "train" if i < 14 else "test",
                    "observed_sales": True,
                    "qty": (1000 if sku == "B@CA_1" and i >= 14 else
                            100 if sku == "C@CA_1" else 2),
                    "price": 1.0 if sku == "C@CA_1" or i % 2 == 0 else 2.0,
                })
        weekly = pd.DataFrame(rows)
        chosen = select_pricing_series(weekly, "CA_1", test_weeks=2, lookback=12)
        self.assertEqual(chosen["sku"], "A@CA_1")
        self.assertEqual(chosen["train_units"], 28)
        self.assertEqual(chosen["test_decisions"], 2)

    def test_pricing_series_fails_if_train_winner_lacks_test_week(self) -> None:
        weeks = pd.date_range("2020-01-06", periods=16, freq="7D")
        rows = []
        for sku in ("A@CA_1", "B@CA_1"):
            for i, week in enumerate(weeks):
                rows.append({
                    "sku": sku, "week": week,
                    "split": "train" if i < 14 else "test",
                    "observed_sales": not (sku == "A@CA_1" and i == 15),
                    "qty": 3 if sku == "A@CA_1" else 2,
                    "price": 1.0 if i % 2 == 0 else 2.0,
                })
        with self.assertRaisesRegex(ValueError, "A@CA_1.*test"):
            select_pricing_series(pd.DataFrame(rows), "CA_1", test_weeks=2, lookback=12)


if __name__ == "__main__":
    unittest.main()
