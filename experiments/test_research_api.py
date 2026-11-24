"""Guard the read-only dashboard against split and comparison mistakes."""

from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import pandas as pd

from experiments.research_api import forecast_series, pricing_replay


def _forecast_files(*, wrong_target: bool = False):
    aggregate = pd.DataFrame({"model": ["song_transformer_price", "naive_last"]})
    per_sku = pd.DataFrame({"model": ["song_transformer_price", "naive_last"],
                            "sku": ["A", "A"], "n": [1, 1],
                            "rmse_mean": [2.0, 3.0], "rmse_sd": [0.2, 0.0],
                            "mae_mean": [1.0, 2.0]})
    all_runs = pd.DataFrame({"model": ["song_transformer_price", "naive_last"],
                             "seed": [42, 42]})

    def read(path: Path):
        if path.name == "aggregate.csv":
            return aggregate
        if path.name == "per_sku.csv":
            return per_sku
        if path.name == "all_runs.csv":
            return all_runs
        if path.name == "comparison_predictions.csv":
            naive = "multiseed" in str(path)
            return pd.DataFrame({
                "model": ["naive_last" if naive else "song_transformer_price"] * 2,
                "split": ["validation", "test"], "sku": ["A", "A"],
                "week": ["2020-01-06", "2020-01-13"], "price": [2.0, 2.0],
                "y_true": [90.0, 11.0 if naive and wrong_target else 10.0],
                "y_pred": [80.0, 8.0 if naive else 9.0],
            })
        raise AssertionError(path)

    return read


class ResearchApiTests(unittest.TestCase):
    def test_forecast_series_only_uses_test_and_matches_same_target(self):
        with patch("experiments.research_api._data_root", return_value=Path("unused")), \
             patch("experiments.research_api._csv", side_effect=_forecast_files()):
            result = forecast_series("uci", "A", 42, "song_transformer_price,naive_last")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["week"], "2020-01-13")
        self.assertEqual(result["rows"][0]["actual"], 10.0)
        self.assertEqual(result["rows"][0]["naive_last"], 8.0)

    def test_forecast_series_rejects_mismatched_test_target(self):
        with patch("experiments.research_api._data_root", return_value=Path("unused")), \
             patch("experiments.research_api._csv", side_effect=_forecast_files(wrong_target=True)):
            with self.assertRaises(HTTPException) as caught:
                forecast_series("uci", "A", 42, "song_transformer_price,naive_last")
        self.assertEqual(caught.exception.status_code, 503)

    def test_replay_pairs_fixed_price_by_sku_and_week(self):
        manifest = {"seeds": [42], "policies": ["fixed_mean", "ddpg"], "evaluation_cycles": 2}
        episodes = pd.DataFrame([
            {"policy": "ddpg", "seed": 42, "sku": "A", "week": "2020-01-13",
             "chosen_price": 3.0, "simulated_units": 20.0, "simulated_revenue": 60.0,
             "reference_price": 2.0, "forecast_update_count": 0},
            {"policy": "fixed_mean", "seed": 42, "sku": "A", "week": "2020-01-06",
             "chosen_price": 2.0, "simulated_units": 20.0, "simulated_revenue": 40.0,
             "reference_price": 2.0, "forecast_update_count": 0},
            {"policy": "ddpg", "seed": 42, "sku": "A", "week": "2020-01-06",
             "chosen_price": 2.5, "simulated_units": 20.0, "simulated_revenue": 50.0,
             "reference_price": 2.0, "forecast_update_count": 0},
            {"policy": "fixed_mean", "seed": 42, "sku": "A", "week": "2020-01-13",
             "chosen_price": 2.0, "simulated_units": 25.0, "simulated_revenue": 50.0,
             "reference_price": 2.0, "forecast_update_count": 0},
        ])
        with patch("experiments.research_api._pricing_root", return_value=Path("unused")), \
             patch("experiments.research_api._json", return_value=manifest), \
             patch("experiments.research_api._csv", return_value=episodes):
            result = pricing_replay("lag_adjusted_kernel", "ddpg", 42)
        self.assertEqual([row["week"] for row in result["rows"]],
                         ["2020-01-06", "2020-01-13"])
        self.assertEqual([row["fixed_revenue"] for row in result["rows"]], [40.0, 50.0])
        self.assertAlmostEqual(result["totals"]["uplift_pct"], 100 * (110 / 90 - 1))


if __name__ == "__main__":
    unittest.main()
