"""Focused checks for time splits and past-only feature construction."""

from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.data import _positive_std, load_experiment_data, make_rolling_validation_folds, make_windows
from experiments.song_online_retail import select_train_eligible


class ExperimentDataTests(unittest.TestCase):
    def test_constant_noninteger_price_has_safe_scaler(self) -> None:
        self.assertEqual(_positive_std(pd.Series([2.2] * 7)), 1.0)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        weeks = pd.date_range("2020-01-06", periods=20, freq="7D")
        records = []
        for sku in ("A", "B"):
            for i, week in enumerate(weeks):
                if sku == "A" and i == 4:
                    continue  # One unobserved SKU-week in the middle of train.
                records.append({
                    "sku": sku,
                    "week": week,
                    "qty": 10000 if i == 19 else i + (2 if sku == "B" else 1),
                    "price": 1000 if i == 19 else i + 10,
                })
        pd.DataFrame(records).to_csv(self.root / "weekly.csv", index=False)
        pd.DataFrame({"sku": ["A", "B"], "tier": ["low", "high"]}).to_csv(
            self.root / "selected.csv", index=False)

    def test_chronological_split_scaling_and_unobserved_week(self) -> None:
        weekly, manifest = load_experiment_data(self.root / "weekly.csv", self.root / "selected.csv", 3)
        self.assertEqual(manifest["counts"]["train"]["weeks"], 14)
        self.assertEqual(manifest["counts"]["validation"]["weeks"], 3)
        self.assertEqual(manifest["counts"]["test"]["weeks"], 3)
        self.assertEqual(manifest["missing_sku_weeks"], 1)
        missing = weekly.loc[(weekly["sku"] == "A") & (weekly["week"] == pd.Timestamp("2020-02-03"))].iloc[0]
        self.assertFalse(missing["observed_sales"])
        self.assertTrue(missing["price_imputed"])
        self.assertEqual(missing["qty"], 0)
        self.assertEqual(missing["price"], 13)
        # The extreme test-only observation must not affect fit-time scaling.
        self.assertLess(manifest["scalers"]["A"]["qty_mean"], 20)
        self.assertLess(manifest["scalers"]["A"]["price_mean"], 30)

    def test_lags_are_past_only_and_do_not_cross_sku(self) -> None:
        weekly, _ = load_experiment_data(self.root / "weekly.csv", self.root / "selected.csv", 3)
        windows = make_windows(weekly, 3)
        first_a = windows.loc[windows["sku"] == "A"].iloc[0]
        first_b = windows.loc[windows["sku"] == "B"].iloc[0]
        self.assertEqual(first_a["qty"], 4)
        self.assertEqual(first_a["qty_lag_1"], 3)
        self.assertEqual(first_a["qty_lag_3"], 1)
        self.assertEqual(first_b["qty"], 5)
        self.assertEqual(first_b["qty_lag_1"], 4)
        self.assertEqual(first_b["qty_lag_3"], 2)
        self.assertTrue((windows["week"] > pd.Timestamp("2020-01-06")).all())

    def test_literature_split_profiles_reserve_future_test(self) -> None:
        for profile, expected_test_weeks in (("retail2_80_20", 4), ("retail2_last12", 12)):
            weekly, manifest = load_experiment_data(
                self.root / "weekly.csv", self.root / "selected.csv", 3, profile)
            self.assertEqual(manifest["counts"]["test"]["weeks"], expected_test_weeks)
            self.assertEqual(manifest["counts"]["validation"]["weeks"], 0)
            self.assertLess(weekly.loc[weekly["split"] == "train", "week"].max(),
                            weekly.loc[weekly["split"] == "test", "week"].min())
        with self.assertRaises(ValueError):
            make_rolling_validation_folds(weekly, n_folds=3, horizon_weeks=12)

    def test_split_calendar_uses_full_weekly_source_before_sku_selection(self) -> None:
        weeks = pd.date_range("2020-01-06", periods=20, freq="7D")
        rows = [
            {"sku": "A", "week": week, "qty": 2, "price": 3.0}
            for week in weeks[1:-1]
        ] + [
            {"sku": "B", "week": weeks[0], "qty": 1, "price": 1.0},
            {"sku": "B", "week": weeks[-1], "qty": 1, "price": 1.0},
        ]
        pd.DataFrame(rows).to_csv(self.root / "boundary_weekly.csv", index=False)
        pd.DataFrame({"sku": ["A"]}).to_csv(self.root / "boundary_selected.csv", index=False)
        weekly, manifest = load_experiment_data(
            self.root / "boundary_weekly.csv", self.root / "boundary_selected.csv", 3)
        self.assertEqual(manifest["first_week"], "2020-01-06")
        self.assertEqual(manifest["last_week"], "2020-05-18")
        self.assertEqual(manifest["counts"]["train"]["weeks"], 14)
        self.assertEqual(manifest["counts"]["validation"]["weeks"], 3)
        self.assertEqual(manifest["counts"]["test"]["weeks"], 3)
        self.assertEqual(weekly["week"].min(), weeks[0])
        self.assertEqual(weekly["week"].max(), weeks[-1])

    def test_uci_cohort_ranking_uses_train_only(self) -> None:
        weekly = pd.DataFrame([
            {"sku": "A", "week": week, "qty": qty}
            for week, qty in zip(pd.date_range("2020-01-06", periods=4, freq="7D"), [1, 1, 1, 10000])
        ] + [
            {"sku": "B", "week": week, "qty": qty}
            for week, qty in zip(pd.date_range("2020-01-06", periods=4, freq="7D"), [2, 2, 2, 1])
        ])
        descriptions = pd.DataFrame({"sku": ["A", "B"], "description": ["a", "b"]})
        selected = select_train_eligible(
            weekly, descriptions, pd.Timestamp("2020-01-20"), min_train_weeks=2, max_skus=1
        )
        self.assertEqual(selected["sku"].tolist(), ["B"])
        self.assertEqual(selected["train_qty"].tolist(), [6])

    def test_preprocessing_manifest_provenance_and_mismatch(self) -> None:
        weekly_path = self.root / "weekly.csv"
        selected_path = self.root / "selected_products.csv"
        selected_path.write_bytes((self.root / "selected.csv").read_bytes())
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        upstream_path = self.root / "preprocessing_manifest.json"
        upstream = {
            "dataset": "UCI Online Retail ID 352",
            "selected_skus": ["A", "B"],
            "selection_note": "train-only synthetic cohort",
            "weekly_sha256": digest(weekly_path),
            "selected_sha256": digest(selected_path),
        }
        upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
        _, manifest = load_experiment_data(weekly_path, selected_path, 3)
        self.assertEqual(manifest["dataset"], "UCI Online Retail ID 352")
        self.assertEqual(manifest["selection_scope_note"], "train-only synthetic cohort")
        upstream["selected_skus"] = ["B", "A"]
        upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Selected SKUs do not match"):
            load_experiment_data(weekly_path, selected_path, 3)


if __name__ == "__main__":
    unittest.main()
