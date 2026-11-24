"""Focused checks for the Phase 6 price rule and validation-only selection."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.phase6_optimize import ForecastGridOptimizer, _policy_features, _select_per_sku


class PriceSelectionTests(unittest.TestCase):
    def test_policy_input_excludes_target_week_labels_and_price_proxies(self):
        row = {"sku": "A", "week": "2020-01-06", "split": "test",
               "observed_sales": True, "event_days": 1, "snap_days": 2,
               "qty": 999, "price": 999, "revenue": 999,
               "qty_z": 999, "price_z": 999,
               "price_change_pct": 999, "promotion_proxy": True}
        for lag in range(1, 13):
            row[f"qty_lag_{lag}"] = lag
            row[f"price_lag_{lag}"] = 10.0
        features = _policy_features(pd.Series(row))
        self.assertEqual(features["qty_lag_1"], 1)
        self.assertEqual(features["price_lag_1"], 10.0)
        self.assertTrue({"qty", "price", "revenue", "qty_z", "price_z",
                         "price_change_pct", "promotion_proxy"}.isdisjoint(features.index))

    def test_grid_respects_support_and_prefers_hold_on_ties(self):
        class Simulator:
            surfaces = {"A": SimpleNamespace(price_min=8.0, price_max=12.0)}

            @staticmethod
            def price_bounds(sku, reference, max_change):
                return max(8.0, reference * (1 - max_change)), min(12.0, reference * (1 + max_change))

        optimizer = ForecastGridOptimizer({42: object()}, Simulator())
        row = pd.Series({"sku": "A", "price_lag_1": 10.0})
        candidates, lo, hi = optimizer.prices("A", 10.0)
        self.assertAlmostEqual(lo, 9.0)
        self.assertAlmostEqual(hi, 11.0)
        self.assertTrue(np.all((candidates >= lo) & (candidates <= hi)))
        self.assertIn(10.0, candidates)

        optimizer.forecast_revenue = lambda _, prices: np.asarray([100 - (prices - 10.5) ** 2])
        self.assertAlmostEqual(optimizer.recommend(row)["recommended_price"], 10.5)
        entered = optimizer.recommend(row, reference_price=11.0)
        self.assertAlmostEqual(entered["reference_price"], 11.0)
        self.assertAlmostEqual(row["price_lag_1"], 10.0)
        optimizer.forecast_revenue = lambda _, prices: np.asarray([100 - (prices - 10.8) ** 2])
        capped = optimizer.recommend(row)
        self.assertAlmostEqual(capped["recommended_price"], 10.8)
        self.assertLessEqual(capped["cap5_recommended_price"], 10.5)
        optimizer.forecast_revenue = lambda _, prices: np.full((1, len(prices)), 100.0)
        self.assertAlmostEqual(optimizer.recommend(row)["recommended_price"], 10.0)
        with self.assertRaises(ValueError):
            optimizer.prices("A", 13.0)

    def test_sku_policy_is_chosen_on_validation_even_when_test_disagrees(self):
        rows = []
        for split in ("validation", "test"):
            for sku in ("A", "B"):
                for seed in (42, 123, 456):
                    for method in ("hold_current", "heuristic", "ppo_frozen"):
                        revenue = 100.0
                        if sku == "A" and method == "ppo_frozen":
                            revenue = ({42: 106, 123: 104, 456: 108}[seed]
                                       if split == "validation" else 90)
                        elif sku == "A" and method == "heuristic":
                            revenue = 105.0 if split == "validation" else 120.0
                        elif sku == "B" and method == "ppo_frozen":
                            revenue = 99.0 if split == "validation" else 130.0
                        rows.append({
                            "split": split, "sku": sku, "week": "2020-01-06",
                            "seed": seed, "method": method,
                            "simulator_expected_revenue": revenue,
                            "hold_expected_revenue": 100.0,
                            "chosen_price": 11.0 if method != "hold_current" else 10.0,
                            "reference_price": 10.0,
                        })
        decisions = pd.DataFrame(rows)
        chosen, selected, _ = _select_per_sku(decisions)
        by_sku = chosen.set_index("sku")
        self.assertEqual((by_sku.loc["A", "method"], by_sku.loc["A", "seed"]),
                         ("ppo_frozen", 42))
        self.assertEqual(by_sku.loc["B", "method"], "hold_current")
        self.assertEqual(len(selected), 4)

        changed_test = decisions.copy()
        changed_test.loc[changed_test["split"].eq("test"), "simulator_expected_revenue"] = 1.0
        changed, _, _ = _select_per_sku(changed_test)
        self.assertEqual(chosen[["sku", "method", "seed"]].to_dict("records"),
                         changed[["sku", "method", "seed"]].to_dict("records"))


if __name__ == "__main__":
    unittest.main()
