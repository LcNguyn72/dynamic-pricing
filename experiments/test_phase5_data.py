"""Phase 5 window boundaries and held-out key integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


class Phase5DataTests(unittest.TestCase):
    def test_windows_use_only_prior_calendar_weeks_and_preserve_observed_zero(self):
        from experiments.phase5_data import windows_from_weekly

        weeks = pd.date_range("2020-01-06", periods=14, freq="7D")
        panel = pd.concat([
            pd.DataFrame({"sku": sku, "week": weeks,
                          "split": ["train"] * 12 + ["test"] * 2,
                          "observed_sales": True, "qty": history + targets,
                          "price": [2.0] * 12 + [3.0, 4.0],
                          "event_days": [1] * 12 + [6, 2], "snap_days": 0})
            for sku, history, targets in (("A", list(range(1, 13)), [32000, 0]),
                                          ("B", [7] * 12, [5, 6]))
        ]).sample(frac=1, random_state=42)
        windows = windows_from_weekly(panel).set_index(["sku", "week"])
        first = windows.loc[("A", weeks[12])]
        second = windows.loc[("A", weeks[13])]
        self.assertEqual(first["qty_lag_1"], 12)
        self.assertEqual(first["price_lag_1"], 2.0)
        self.assertEqual(first["event_days_lag_1"], 1)
        self.assertEqual(second["qty_lag_1"], 32000)
        self.assertEqual(second["price_lag_1"], 3.0)
        self.assertEqual(second["event_days_lag_1"], 6)
        self.assertEqual(second["qty"], 0)
        self.assertTrue(second["observed_sales"])
        self.assertEqual(windows.loc[("B", weeks[12]), "qty_lag_12"], 7)

    def test_frozen_shard_keys_reject_missing_and_same_count_substitution(self):
        from experiments.phase5_data import verify_shard_test_keys

        digest = hashlib.sha256(b"dataset,sku,week\nm5,A@CA_1,2020-01-06\nm5,A@CA_1,2020-01-13\n").hexdigest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {"root": root.resolve(), "test_key_shards": {"m5": {"CA_1": {"n": 2, "sha256": digest}}}}
            frame = pd.DataFrame({"sku": ["A@CA_1"] * 2, "week": pd.to_datetime(["2020-01-13", "2020-01-06"]),
                                  "split": ["test"] * 2, "observed_sales": [True] * 2})
            self.assertEqual(verify_shard_test_keys(root, "m5", contract, "CA_1", frame)["n"], 2)
            with self.assertRaisesRegex(ValueError, "frozen test keys"):
                verify_shard_test_keys(root, "m5", contract, "CA_1", frame.iloc[:1])
            frame.loc[0, "sku"] = "B@CA_1"
            with self.assertRaisesRegex(ValueError, "frozen test keys"):
                verify_shard_test_keys(root, "m5", contract, "CA_1", frame)

    def test_cached_windows_preserve_types_and_rebuild_after_tampering(self):
        from experiments.phase5_data import _cache_frame

        frame = pd.DataFrame({"sku": ["A", "B"], "split": ["train", "test"],
                              "week": pd.Series(pd.to_datetime(["2020-01-06", "2020-01-13"]), dtype="datetime64[us]"),
                              "qty": [0.0, 1.0], "observed_sales": [True, False]})
        calls = []

        def create():
            calls.append(True)
            return frame

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"source_sha256": "frozen"}
            for _ in range(2):
                actual = _cache_frame(root, "m5", "CA_1", "windows", identity, ["A", "B"], create)
                pd.testing.assert_frame_equal(actual, frame)
            self.assertEqual(len(calls), 1)
            cached = next(root.rglob("*.npz"))
            with cached.open("ab") as stream:
                stream.write(b"corrupt")
            _cache_frame(root, "m5", "CA_1", "windows", identity, ["A", "B"], create)
            self.assertEqual(len(calls), 2)
