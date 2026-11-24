"""Verify the one-step UI scenario against the saved Phase 4 decision."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np
import pandas as pd
import torch

from experiments.pricing_inference_api import (
    ScenarioRequest, _bundle, _evaluation_rows, _root, scenario_infer, scenario_options,
)


class PricingInferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (_root() / "phase5c_online_state" / "manifest.json").is_file():
            raise unittest.SkipTest("Phase 5C M5 artifacts are not available")
        cls.options = scenario_options()
        cls.sku = cls.options["items"][0]["sku"]
        cls.week = date.fromisoformat(cls.options["items"][0]["weeks"][0]["week"])
        _, simulator, _ = _bundle()
        cls.starting_price = float(np.mean(np.exp(simulator.surfaces[cls.sku].log_prices)))

    def request(self, **changes):
        values = {"sku": self.sku, "week": self.week,
                  "current_price": self.starting_price, "seed": 42}
        values.update(changes)
        return ScenarioRequest(**values)

    def test_first_decision_matches_saved_frozen_ppo_episode(self):
        result = scenario_infer(self.request())
        episodes = pd.read_csv(_root() / "pricing_lag_512_5seeds" / "policy_episodes.csv")
        saved = episodes.loc[(episodes["policy"] == "ppo_frozen")
                             & (episodes["seed"] == 42)
                             & (episodes["sku"] == self.sku)
                             & (episodes["week"] == self.week.isoformat())].iloc[0]
        self.assertAlmostEqual(result["recommended_price"], float(saved["chosen_price"]), places=6)
        self.assertEqual(len(result["curve"]), 41)
        self.assertTrue(result["price_min"] <= result["recommended_price"] <= result["price_max"])

    def test_current_week_sales_target_does_not_affect_inference(self):
        baseline = scenario_infer(self.request())
        manifest, _, _ = _bundle()
        changed = _evaluation_rows(self.sku, manifest)
        changed.loc[changed["week"].dt.date.eq(self.week), ["qty", "revenue"]] = 1_000_000_000
        with patch("experiments.pricing_inference_api._evaluation_rows", return_value=changed):
            altered = scenario_infer(self.request())
        self.assertAlmostEqual(altered["recommended_price"], baseline["recommended_price"], places=8)
        self.assertAlmostEqual(altered["forecast"]["units_recommended"],
                               baseline["forecast"]["units_recommended"], places=6)

    def test_rejects_price_outside_train_support(self):
        lower = self.options["items"][0]["price_min"]
        with self.assertRaises(HTTPException) as caught:
            scenario_infer(self.request(current_price=lower * 0.5))
        self.assertEqual(caught.exception.status_code, 422)

    def test_rejects_unselected_week_and_seed(self):
        for request in (self.request(week=date(2020, 1, 1)), self.request(seed=999)):
            with self.assertRaises(HTTPException) as caught:
                scenario_infer(request)
            self.assertEqual(caught.exception.status_code, 422)

    def test_online_replay_matches_saved_prices_before_and_after_feedback(self):
        episodes = pd.read_csv(_root() / "pricing_lag_512_5seeds" / "policy_episodes.csv")
        for seed, week_index in ((42, 0), (42, 10), (123, 19)):
            week = date.fromisoformat(self.options["items"][0]["weeks"][week_index]["week"])
            result = scenario_infer(self.request(seed=seed, week=week,
                                                 current_price=None,
                                                 policy_variant="ppo_feedback_gradient"))
            saved = episodes.loc[(episodes["policy"] == "ppo_feedback_gradient")
                                 & (episodes["seed"] == seed)
                                 & (episodes["sku"] == self.sku)
                                 & (episodes["week"] == week.isoformat())].iloc[0]
            self.assertAlmostEqual(result["recommended_price"], float(saved["chosen_price"]), places=6)
            self.assertEqual(result["prior_simulated_decisions"], week_index)
            self.assertEqual(result["forecast_updates_before_week"], week_index // 10)

    def test_online_inference_restores_torch_rng_and_hides_current_sales(self):
        week = date.fromisoformat(self.options["items"][0]["weeks"][10]["week"])
        request = self.request(week=week, current_price=None,
                               policy_variant="ppo_feedback_gradient")
        rng_before = torch.get_rng_state().clone()
        baseline = scenario_infer(request)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        original = _evaluation_rows

        def changed_rows(sku, pricing_manifest):
            rows = original(sku, pricing_manifest)
            if sku == self.sku:
                rows.loc[rows["week"].dt.date.eq(week), ["qty", "revenue"]] = 1_000_000_000
            return rows

        with patch("experiments.pricing_inference_api._evaluation_rows", side_effect=changed_rows):
            altered = scenario_infer(request)
        self.assertAlmostEqual(altered["recommended_price"], baseline["recommended_price"], places=8)
        self.assertEqual(baseline["context"]["simulated_history_weeks"], 10)


if __name__ == "__main__":
    unittest.main()
