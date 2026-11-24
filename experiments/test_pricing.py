"""Focused checks for causal boundaries in the simulated pricing benchmark."""

import unittest

import numpy as np
import pandas as pd

from experiments.pricing_benchmark import PricingEnvironment, evaluate_policy
from experiments.pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator
from experiments.pricing_benchmark import validate_simulator


class PricingSimulatorTests(unittest.TestCase):
    def setUp(self):
        weeks = pd.date_range("2020-01-06", periods=30, freq="7D")
        rows = []
        for i, week in enumerate(weeks):
            rows.append({
                "sku": "A",
                "week": week,
                "qty": 30.0 if i == 29 else 100.0 - i,
                "price": 2.0 if i == 29 else (1.0 if i % 2 == 0 else 1.1),
                "split": "test" if i == 29 else "train",
                "observed_sales": True,
            })
        self.weekly = pd.DataFrame(rows)

    def test_fit_ignores_test_price_and_demand(self):
        simulator = KernelDemandSimulator().fit(self.weekly)
        surface = simulator.surfaces["A"]
        self.assertEqual(surface.price_max, 1.1)
        self.assertEqual(len(surface.quantities), 29)
        self.assertNotIn(30.0, surface.quantities)

    def test_policy_action_stays_in_common_training_support(self):
        simulator = KernelDemandSimulator().fit(self.weekly)
        windows = self.weekly.copy()
        windows["month"] = windows["week"].dt.month
        windows["week_of_year"] = windows["week"].dt.isocalendar().week.astype(int)
        for lag in range(1, 13):
            windows[f"qty_lag_{lag}"] = windows["qty"].shift(lag)
            windows[f"price_lag_{lag}"] = windows["price"].shift(lag)
        windows = windows.dropna()
        env = PricingEnvironment(windows, simulator, "test")
        obs, _ = env.reset(seed=42, options={"sku": "A"})
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertLessEqual(env.price_from_action(1.0), 1.1)
        self.assertGreaterEqual(env.price_from_action(-1.0), 1.0)
        frame, summary = evaluate_policy(env, "fixed_mean", seed=42)
        self.assertEqual(len(frame), 1)
        self.assertEqual(summary["n_sku_weeks"], 1)
        self.assertGreaterEqual(frame.iloc[0]["expected_regret"], 0)
        env.reset(seed=42, options={"sku": "A"})
        _, reward, _, _, info = env.step(np.array([0.0]))
        self.assertAlmostEqual(reward * env.reward_scale, info["simulated_revenue"])

    def test_paper_absolute_action_reaches_full_training_support(self):
        simulator = KernelDemandSimulator().fit(self.weekly)
        windows = self.weekly.copy()
        windows["month"] = windows["week"].dt.month
        windows["week_of_year"] = windows["week"].dt.isocalendar().week.astype(int)
        for lag in range(1, 13):
            windows[f"qty_lag_{lag}"] = windows["qty"].shift(lag)
            windows[f"price_lag_{lag}"] = windows["price"].shift(lag)
        env = PricingEnvironment(windows.dropna(), simulator, "test",
                                 action_mode="absolute", include_sku_one_hot=False)
        observation, _ = env.reset(seed=42, options={"sku": "A"})
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertAlmostEqual(env.price_from_action(-1.0), 1.0)
        self.assertAlmostEqual(env.price_from_action(1.0), 1.1)
        self.assertAlmostEqual(env.price_from_action(float(env.action_for_price(1.07)[0])), 1.07,
                               places=5)

    def test_noise_is_reproducible(self):
        simulator = KernelDemandSimulator().fit(self.weekly)
        a = simulator.sample("A", 1.05, self.weekly.iloc[-1]["week"], np.random.default_rng(7))
        b = simulator.sample("A", 1.05, self.weekly.iloc[-1]["week"], np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_known_calendar_context_changes_m5_surface(self):
        weekly = self.weekly.copy()
        weekly["event_days"] = [3 if i % 2 else 0 for i in range(len(weekly))]
        weekly["snap_days"] = 0
        weekly.loc[weekly["event_days"].eq(3), "qty"] += 100
        simulator = KernelDemandSimulator().fit(weekly)
        week = weekly.iloc[-1]["week"]
        event = simulator.predict_mean("A", 1.05, week, {"event_days": 3, "snap_days": 0})
        normal = simulator.predict_mean("A", 1.05, week, {"event_days": 0, "snap_days": 0})
        self.assertGreater(event, normal)

    def test_lag_adjustment_uses_only_past_quantity(self):
        base = KernelDemandSimulator().fit(self.weekly)
        adaptive = LagAdjustedKernelSimulator(recent_weight=1.0).fit(self.weekly)
        week = self.weekly.iloc[-1]["week"]
        price = 1.05
        context = {"qty_lag_1": 37.0, "price_lag_1": price}
        self.assertAlmostEqual(adaptive.predict_mean("A", price, week, context), 37.0)
        self.assertNotAlmostEqual(base.predict_mean("A", price, week, context), 37.0)
        self.assertAlmostEqual(adaptive.predict_mean("A", price, week, None),
                               base.predict_mean("A", price, week, None))

    def test_simulator_validation_split_never_reads_test_targets(self):
        adaptive = LagAdjustedKernelSimulator(recent_weight=0.5).fit(self.weekly)
        frame = validate_simulator(adaptive, self.weekly, splits=("test",))
        self.assertEqual(set(frame["split"]), {"test"})
        self.assertEqual(len(frame), 1)

    def test_short_training_episodes_cover_multiple_series(self):
        weekly = pd.concat([self.weekly.assign(sku="A"), self.weekly.assign(sku="B")],
                           ignore_index=True)
        simulator = KernelDemandSimulator().fit(weekly)
        windows = weekly.sort_values(["sku", "week"]).copy()
        for lag in range(1, 13):
            windows[f"qty_lag_{lag}"] = windows.groupby("sku")["qty"].shift(lag)
            windows[f"price_lag_{lag}"] = windows.groupby("sku")["price"].shift(lag)
        env = PricingEnvironment(windows.dropna(), simulator, "train",
                                 max_episode_steps=3, random_start=True)
        seen_starts = set()
        for seed in range(10):
            env.reset(seed=seed)
            seen_starts.add((env.sku, env.index))
            for step in range(3):
                _, _, terminated, _, _ = env.step(np.array([0.0]))
                self.assertEqual(terminated, step == 2)
        self.assertGreater(len(seen_starts), 2)
        self.assertEqual(env.episodes_started, 10)

    def test_feedback_sampling_does_not_change_demand_noise(self):
        class FixedForecast:
            def predict_q(self, row, price):
                return 50.0

            def price_gradient(self, row, price):
                return -1.0

            def fine_tune(self, samples, epochs=1, lr=1e-4):
                return None

        weekly = self.weekly.copy()
        weekly.loc[weekly.index >= 27, "split"] = "test"
        simulator = KernelDemandSimulator().fit(weekly)
        windows = weekly.copy()
        for lag in range(1, 13):
            windows[f"qty_lag_{lag}"] = windows["qty"].shift(lag)
            windows[f"price_lag_{lag}"] = windows["price"].shift(lag)
        windows = windows.dropna()
        plain = PricingEnvironment(windows, simulator, "test", FixedForecast())
        updating = PricingEnvironment(
            windows, simulator, "test", FixedForecast(), online_forecast_update=True,
            update_every=1,
        )
        plain.reset(seed=77, options={"sku": "A"})
        updating.reset(seed=77, options={"sku": "A"})
        for _ in range(3):
            _, _, _, _, a = plain.step(np.array([0.0]))
            _, _, _, _, b = updating.step(np.array([0.0]))
            self.assertAlmostEqual(a["simulated_units"], b["simulated_units"])

    def test_online_forecast_resets_between_skus(self):
        calls = []

        class UpdatingForecast:
            def __init__(self):
                self.updates = 0

            def predict_q(self, row, price):
                calls.append((str(row["sku"]), self.updates))
                return 50.0 + self.updates

            def price_gradient(self, row, price):
                return -1.0

            def fine_tune(self, samples, epochs=1, lr=1e-4):
                self.updates += 1

        weekly = pd.concat([self.weekly.assign(sku="A"), self.weekly.assign(sku="B")], ignore_index=True)
        weekly.loc[weekly["week"].ge(weekly["week"].sort_values().unique()[-3]), "split"] = "test"
        simulator = KernelDemandSimulator().fit(weekly)
        windows = weekly.sort_values(["sku", "week"]).copy()
        for lag in range(1, 13):
            windows[f"qty_lag_{lag}"] = windows.groupby("sku")["qty"].shift(lag)
            windows[f"price_lag_{lag}"] = windows.groupby("sku")["price"].shift(lag)
        windows = windows.dropna()
        env = PricingEnvironment(
            windows, simulator, "test", UpdatingForecast(),
            online_forecast_update=True, update_every=1,
        )
        evaluate_policy(env, "fixed_mean", seed=42)
        first_b = next(updates for sku, updates in calls if sku == "B")
        self.assertEqual(first_b, 0)


if __name__ == "__main__":
    unittest.main()
