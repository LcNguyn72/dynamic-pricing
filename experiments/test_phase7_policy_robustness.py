"""Check that evaluator changes never reselect the Phase 6 policy."""

import unittest

import pandas as pd

from experiments.phase7_policy_robustness import VARIANTS, _summarize


class FrozenPolicyRobustnessTests(unittest.TestCase):
    def test_frozen_selected_price_can_lose_under_another_demand_assumption(self):
        records = []
        selected = []
        for split, week in (("validation", "2020-01-06"), ("test", "2020-01-13")):
            selected.append({"split": split, "sku": "A", "week": week, "seed": 42,
                             "method": "ppo_frozen", "chosen_price": 11.0})
            for variant in VARIANTS:
                for method, price in (("hold_current", 10.0), ("ppo_frozen", 11.0)):
                    revenue = 100.0 if method == "hold_current" else (
                        90.0 if variant == "flat_price_response" else 110.0)
                    records.append({
                        "variant": variant, "split": split, "sku": "A", "week": week,
                        "seed": 42, "method": method, "chosen_price": price,
                        "expected_revenue": revenue, "hold_expected_revenue": 100.0,
                    })
        _, _, frozen, by_sku = _summarize(pd.DataFrame(records), pd.DataFrame(selected))
        test = frozen.loc[frozen["split"].eq("test")].set_index("variant")
        self.assertAlmostEqual(test.loc["lag_selected", "uplift_vs_hold_pct"], 10.0)
        self.assertAlmostEqual(test.loc["flat_price_response", "uplift_vs_hold_pct"], -10.0)
        self.assertEqual(len(by_sku), len(VARIANTS) * 2)


if __name__ == "__main__":
    unittest.main()
