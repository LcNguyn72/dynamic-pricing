"""Test that the Phase 2 M5 adapter preserves past context and train scope."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from experiments.song_forecast import _sha256


class Phase2M5Tests(unittest.TestCase):
    def test_trial_selection_uses_train_and_keeps_calendar_lags(self):
        from experiments.phase2_song import m5_store_windows

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, weekly = root / "manifest_CA_1.json", root / "weekly_CA_1.csv"
            manifest.write_text(json.dumps({"scalers": {sku: {"price_mean": 2}
                                                         for sku in "ABC"}}), encoding="utf-8")
            rows = []
            for sku in "ABC":
                for i, week in enumerate(pd.date_range("2014-07-07", periods=16, freq="W-MON")):
                    split = "train" if i < 14 else "validation" if i == 14 else "test"
                    observed = (sku, i) != ("A", 13)
                    qty = {"A": 3, "B": 9, "C": 2}[sku] if split == "train" else 1
                    if sku == "C" and split != "train":
                        qty = 1000
                    rows.append({"sku": sku, "week": week, "split": split, "qty": qty,
                                 "price": 2 + i / 100, "qty_z": 0, "price_z": 0,
                                 "event_days": int(i == 2), "snap_days": i % 3,
                                 "promotion_proxy": True, "observed_sales": observed})
            pd.DataFrame(rows).to_csv(weekly, index=False)
            report = {"store": "CA_1", "manifest_sha256": _sha256(manifest),
                      "weekly_sha256": _sha256(weekly)}
            windows, scalers, _ = m5_store_windows(root, report, "A", 2)
            self.assertEqual(set(scalers), {"A", "B"})
            self.assertEqual(set(windows["split"]), {"train", "validation"})
            a_train = windows.loc[windows["sku"].eq("A") & windows["split"].eq("train")]
            self.assertEqual(len(a_train), 1)
            row = windows.loc[windows["sku"].eq("A") & windows["split"].eq("validation")].iloc[0]
            self.assertEqual(row["qty_lag_1"], 3)
            self.assertAlmostEqual(row["price_lag_1"], 2.13)
            self.assertEqual(row["event_days_lag_12"], 1)
            self.assertEqual(row["snap_days_lag_12"], 2)
            self.assertNotIn("promotion_flag", windows)

            with weekly.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "weekly_CA_1.csv.*SHA256"):
                m5_store_windows(root, report, "A", 2)
