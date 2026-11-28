"""Paper-inspired pricing-policy benchmark on a common train-only simulator.

All reported revenue from this module is simulated. The Online Retail II data
contains observed prices only, so policy scores cannot be treated as measured
counterfactual business revenue.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Callable, Protocol

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .pricing_simulator import KernelDemandSimulator


class ForecastAdapter(Protocol):
    def predict_q(self, window_row: pd.Series, candidate_price: float) -> float: ...
    def price_gradient(self, window_row: pd.Series, current_price: float) -> float: ...
    def fine_tune(self, samples: list[tuple[pd.Series, float, float]], epochs: int = 1,
                  lr: float = 1e-4) -> None: ...


class PricingEnvironment(gym.Env):
    """One product per episode; simulated sales update the 12-week history."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        windows: pd.DataFrame,
        simulator: KernelDemandSimulator,
        split: str,
        forecaster: ForecastAdapter | None = None,
        with_gradient: bool = True,
        online_forecast_update: bool = False,
        max_change: float = 0.10,
        lookback: int = 12,
        update_every: int = 10,
        feedback_capacity: int = 200,
        update_batch: int = 32,
        initial_price_mode: str = "train_mean",
        action_mode: str = "relative",
        include_sku_one_hot: bool = True,
        max_episode_steps: int | None = None,
        random_start: bool = False,
    ):
        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if lookback < 1 or update_every < 1 or feedback_capacity < 1 or update_batch < 1:
            raise ValueError("lookback and update settings must be positive")
        if online_forecast_update and forecaster is None:
            raise ValueError("Online forecast updates require a forecaster")
        if initial_price_mode not in {"train_mean", "last_observed"}:
            raise ValueError("initial_price_mode must be train_mean or last_observed")
        if action_mode not in {"relative", "absolute"}:
            raise ValueError("action_mode must be relative or absolute")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if random_start and max_episode_steps is None:
            raise ValueError("random_start requires max_episode_steps")
        self.simulator = simulator
        self.split = split
        self.forecaster = forecaster
        self.with_gradient = bool(with_gradient)
        self.online_forecast_update = bool(online_forecast_update)
        self.max_change = float(max_change)
        self.lookback = int(lookback)
        self.update_every = int(update_every)
        self.update_batch = int(update_batch)
        self.initial_price_mode = initial_price_mode
        self.action_mode = action_mode
        self.include_sku_one_hot = bool(include_sku_one_hot)
        self.max_episode_steps = max_episode_steps
        self.random_start = bool(random_start)
        self.feedback: deque[tuple[pd.Series, float, float]] = deque(maxlen=feedback_capacity)
        self.feedback_rng = np.random.default_rng(1729)
        self.interactions = 0
        self.feedback_updates = 0
        self.skus = sorted(simulator.surfaces)
        selected = windows.loc[windows["split"].eq(split) & windows["sku"].astype(str).isin(self.skus)].copy()
        selected["week"] = pd.to_datetime(selected["week"])
        selected = selected.loc[selected["observed_sales"].astype(bool)]
        self.rows_by_sku = {
            str(sku): group.sort_values("week").reset_index(drop=True)
            for sku, group in selected.groupby("sku")
        }
        self.available_skus = sorted(self.rows_by_sku)
        if not self.available_skus:
            raise ValueError(f"No observed window rows for split={split}")
        self._sku_index = {sku: i for i, sku in enumerate(self.skus)}
        all_train_revenue = np.concatenate([
            np.exp(surface.log_prices) * surface.quantities
            for surface in simulator.surfaces.values()
        ])
        # A single constant preserves the paper's revenue objective while
        # keeping numerical rewards near unit scale for pooled RL training.
        self.reward_scale = max(float(np.mean(all_train_revenue)), 1.0)
        # The forecast module passes both its demand estimate and price
        # gradient to the policy (the paper's two forecast outputs).
        self._state_size = 2 * self.lookback + 2 + 2 + 1 + 1 + 1 + (len(self.skus) if self.include_sku_one_hot else 0)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(self._state_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.sku: str | None = None
        self.rows: pd.DataFrame | None = None
        self.index = 0
        self.episode_steps = 0
        self.episodes_started = 0
        self.seen_skus: set[str] = set()
        self.history_qty = np.empty(0)
        self.history_price = np.empty(0)
        self.current_price = 0.0

    def _current_row(self) -> pd.Series:
        assert self.rows is not None
        row = self.rows.iloc[self.index].copy()
        for lag in range(1, self.lookback + 1):
            row[f"qty_lag_{lag}"] = self.history_qty[lag - 1]
            row[f"price_lag_{lag}"] = self.history_price[lag - 1]
        row["price"] = self.current_price
        return row

    def _observation(self) -> np.ndarray:
        assert self.sku is not None
        surface = self.simulator.surfaces[self.sku]
        qty_mean = float(np.mean(np.log1p(surface.quantities)))
        qty_sd = max(float(np.std(np.log1p(surface.quantities))), 0.1)
        price_mean = float(np.mean(surface.log_prices))
        price_sd = max(float(np.std(surface.log_prices)), 0.05)
        history_q = np.clip((np.log1p(self.history_qty) - qty_mean) / qty_sd, -10, 10)
        history_p = np.clip((np.log(self.history_price) - price_mean) / price_sd, -10, 10)
        row = self._current_row()
        week_no = int(pd.Timestamp(row["week"]).isocalendar().week)
        season = np.array([np.sin(2 * np.pi * week_no / 52.0),
                           np.cos(2 * np.pi * week_no / 52.0)])
        calendar_context = np.array([
            float(row["event_days"]) / 7.0 if "event_days" in row.index else 0.0,
            float(row["snap_days"]) / 7.0 if "snap_days" in row.index else 0.0,
        ])
        reference = np.clip((np.log(self.current_price) - price_mean) / price_sd, -10, 10)
        predicted_demand = float(np.mean(surface.quantities))
        gradient = 0.0
        if self.forecaster is not None:
            predicted_demand = float(self.forecaster.predict_q(row, self.current_price))
            if self.with_gradient:
                raw_gradient = float(self.forecaster.price_gradient(row, self.current_price))
                gradient = float(np.clip(raw_gradient * self.current_price / max(np.mean(surface.quantities), 1.0), -10, 10))
        demand_feature = float(np.clip(
            (np.log1p(predicted_demand) - qty_mean) / qty_sd, -10, 10
        ))
        sku_one_hot = np.zeros(len(self.skus), dtype=float) if self.include_sku_one_hot else np.empty(0)
        if self.include_sku_one_hot:
            sku_one_hot[self._sku_index[self.sku]] = 1.0
        observation = np.concatenate([
            history_q, history_p, season, calendar_context,
            [reference, demand_feature, gradient], sku_one_hot,
        ])
        return observation.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # Feedback sampling must not consume the simulator's noise stream;
            # policies are compared with common random demand shocks.
            self.feedback_rng = np.random.default_rng(seed + 1_000_003)
        wanted = (options or {}).get("sku")
        if wanted is None:
            self.sku = str(self.np_random.choice(self.available_skus))
        else:
            self.sku = str(wanted)
            if self.sku not in self.rows_by_sku:
                raise KeyError(f"SKU {self.sku} is not available in {self.split}")
        self.rows = self.rows_by_sku[self.sku]
        max_start = max(0, len(self.rows) - (self.max_episode_steps or len(self.rows)))
        self.index = int(self.np_random.integers(max_start + 1)) if self.random_start else 0
        self.episode_steps = 0
        self.episodes_started += 1
        self.seen_skus.add(self.sku)
        first = self.rows.iloc[self.index]
        self.history_qty = np.array([max(0.0, float(first[f"qty_lag_{lag}"]))
                                     for lag in range(1, self.lookback + 1)], dtype=float)
        self.history_price = np.array([float(first[f"price_lag_{lag}"])
                                       for lag in range(1, self.lookback + 1)], dtype=float)
        surface = self.simulator.surfaces[self.sku]
        starting_price = (float(np.mean(np.exp(surface.log_prices)))
                          if self.initial_price_mode == "train_mean" else float(self.history_price[0]))
        self.current_price = float(np.clip(starting_price, surface.price_min, surface.price_max))
        return self._observation(), {"sku": self.sku}

    def price_from_action(self, action: float) -> float:
        assert self.sku is not None
        if self.action_mode == "absolute":
            surface = self.simulator.surfaces[self.sku]
            lo, hi = surface.price_min, surface.price_max
            return float(lo + (float(np.clip(action, -1.0, 1.0)) + 1.0) * (hi - lo) / 2.0)
        lo, hi = self.simulator.price_bounds(self.sku, self.current_price, self.max_change)
        a = float(np.clip(action, -1.0, 1.0))
        return float(self.current_price + (a * (self.current_price - lo) if a < 0 else a * (hi - self.current_price)))

    def action_for_price(self, desired_price: float) -> np.ndarray:
        assert self.sku is not None
        if self.action_mode == "absolute":
            surface = self.simulator.surfaces[self.sku]
            span = surface.price_max - surface.price_min
            action = 0.0 if span <= 0 else 2.0 * (float(desired_price) - surface.price_min) / span - 1.0
            return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)
        lo, hi = self.simulator.price_bounds(self.sku, self.current_price, self.max_change)
        price = float(np.clip(desired_price, lo, hi))
        span = self.current_price - lo if price < self.current_price else hi - self.current_price
        action = 0.0 if span <= 0 else (price - self.current_price) / span
        return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)

    def step(self, action: np.ndarray):
        assert self.sku is not None and self.rows is not None
        row = self._current_row()
        price = self.price_from_action(float(np.asarray(action).reshape(-1)[0]))
        week = row["week"]
        context = {key: row[key] for key in ("event_days", "snap_days", "qty_lag_1", "price_lag_1")
                   if key in row.index}
        units = self.simulator.sample(self.sku, price, week, self.np_random, context)
        revenue = price * units
        expected_revenue = price * self.simulator.predict_mean(self.sku, price, week, context)
        oracle = self.simulator.oracle_revenue(self.sku, week, context=context)
        reward = revenue / self.reward_scale
        if self.online_forecast_update:
            self.feedback.append((row, price, units))
            self.interactions += 1
            if self.interactions % self.update_every == 0:
                sample_n = min(self.update_batch, len(self.feedback))
                indices = self.feedback_rng.choice(len(self.feedback), size=sample_n, replace=False)
                batch = [self.feedback[int(i)] for i in indices]
                assert self.forecaster is not None
                self.forecaster.fine_tune(batch, epochs=1, lr=1e-4)
                self.feedback_updates += 1
        self.history_qty = np.concatenate(([units], self.history_qty[:-1]))
        self.history_price = np.concatenate(([price], self.history_price[:-1]))
        self.current_price = price
        self.index += 1
        self.episode_steps += 1
        terminated = (self.index >= len(self.rows)
                      or (self.max_episode_steps is not None
                          and self.episode_steps >= self.max_episode_steps))
        if not terminated:
            next_week = pd.Timestamp(self.rows.iloc[self.index]["week"])
            skipped_weeks = max(0, int((next_week - pd.Timestamp(week)).days // 7) - 1)
            for _ in range(skipped_weeks):
                # A missing transaction week is contextual history, not a
                # simulated sale or a decision cycle in this benchmark.
                self.history_qty = np.concatenate(([0.0], self.history_qty[:-1]))
                self.history_price = np.concatenate(([price], self.history_price[:-1]))
        next_observation = np.zeros(self._state_size, dtype=np.float32) if terminated else self._observation()
        info = {
            "sku": self.sku,
            "week": pd.Timestamp(week).date().isoformat(),
            "chosen_price": price,
            "simulated_units": units,
            "simulated_revenue": revenue,
            "expected_revenue": expected_revenue,
            "oracle_revenue": oracle,
            "expected_regret": max(0.0, oracle - expected_revenue),
            "reference_price": float(row["price_lag_1"]),
            "forecast_update_count": self.feedback_updates,
        }
        return next_observation, float(reward), terminated, False, info


def _baseline_action(name: str, env: PricingEnvironment) -> np.ndarray:
    assert env.sku is not None
    if name == "fixed_mean":
        surface = env.simulator.surfaces[env.sku]
        return env.action_for_price(float(np.mean(np.exp(surface.log_prices))))
    if name == "heuristic":
        rolling = float(np.mean(env.history_qty[: min(4, env.lookback)]))
        direction = 1.0 if env.history_qty[0] > rolling else -1.0
        return env.action_for_price(env.current_price * (1 + 0.05 * direction))
    if name == "kernel_grid":
        lo, hi = env.simulator.price_bounds(env.sku, env.current_price, env.max_change)
        candidates = np.linspace(lo, hi, 81)
        row = env._current_row()
        context = {key: row[key] for key in ("event_days", "snap_days", "qty_lag_1", "price_lag_1")
                   if key in row.index}
        best = max(candidates, key=lambda p: p * env.simulator.predict_mean(
            env.sku, float(p), row["week"], context
        ))
        return env.action_for_price(float(best))
    raise ValueError(f"Unknown baseline {name}")


def evaluate_policy(
    env: PricingEnvironment,
    policy_name: str,
    policy: Callable[[np.ndarray, PricingEnvironment], np.ndarray] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate independent SKU episodes with matched demand noise.

    Online feedback is allowed *within* each SKU's chronological test weeks.
    The forecaster is restored before the next SKU so a later test week from
    one product cannot influence an earlier test week from another product.
    """
    episodes = []
    initial_forecaster = deepcopy(env.forecaster) if env.online_forecast_update else None
    for sku_index, sku in enumerate(env.available_skus):
        if initial_forecaster is not None:
            env.forecaster = deepcopy(initial_forecaster)
            env.feedback.clear()
            env.interactions = 0
            env.feedback_updates = 0
        observation, _ = env.reset(seed=seed + sku_index, options={"sku": sku})
        done = False
        while not done:
            action = _baseline_action(policy_name, env) if policy is None else policy(observation, env)
            observation, _, done, _, info = env.step(action)
            episodes.append({"policy": policy_name, "seed": seed, **info})
    frame = pd.DataFrame(episodes)
    total = float(frame["simulated_revenue"].sum())
    expected_total = float(frame["expected_revenue"].sum())
    return frame, {
        "policy": policy_name,
        "seed": seed,
        "n_sku_weeks": int(len(frame)),
        "simulated_revenue": total,
        "expected_revenue": expected_total,
        "expected_regret": float(frame["expected_regret"].sum()),
    }


def train_ppo(
    windows: pd.DataFrame,
    simulator: KernelDemandSimulator,
    forecaster: ForecastAdapter | None,
    *,
    seed: int = 42,
    total_timesteps: int = 2048,
    with_gradient: bool = True,
    online_forecast_update: bool = False,
    output_path: str | Path | None = None,
):
    """Train PPO with the paper's principal actor/critic hyperparameters."""
    from stable_baselines3 import PPO

    env = PricingEnvironment(windows, simulator, "train", forecaster,
                             with_gradient=with_gradient,
                             online_forecast_update=online_forecast_update)
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0, device="cpu",
        policy_kwargs={"net_arch": {"pi": [256, 256, 256], "vf": [256, 256, 256]}},
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        batch_size=64, n_steps=256,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target))
    return model


def validate_simulator(simulator: KernelDemandSimulator, weekly: pd.DataFrame,
                       splits: tuple[str, ...] = ("validation", "test")) -> pd.DataFrame:
    """Observed-price predictive check; this does not validate counterfactuals."""
    rows = weekly.sort_values(["sku", "week"]).copy()
    rows["qty_lag_1"] = rows.groupby("sku")["qty"].shift(1)
    rows["price_lag_1"] = rows.groupby("sku")["price"].shift(1)
    rows = rows.loc[rows["split"].isin(splits)].copy()
    if "observed_sales" in rows:
        rows = rows.loc[rows["observed_sales"].astype(bool)]
    rows = rows.loc[rows["sku"].astype(str).isin(simulator.surfaces)]
    rows["simulator_pred_qty"] = [
        simulator.predict_mean(
            str(r.sku), float(r.price), r.week,
            {key: getattr(r, key) for key in ("event_days", "snap_days", "qty_lag_1", "price_lag_1")
             if hasattr(r, key)},
        )
        for r in rows.itertuples(index=False)
    ]
    rows["absolute_error"] = np.abs(rows["qty"] - rows["simulator_pred_qty"])
    return rows[["sku", "week", "split", "qty", "price", "simulator_pred_qty", "absolute_error"]]
