"""Train and export the M5 shop PPO policy on the frozen V2 pricing cohort."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, __version__ as sb3_version
import torch

from final_system.bundle import DEFAULT_MODEL_DIR, DEFAULT_PREPARED_ROOT, atomic_write, load_bundle
from final_system.engine import _price_point
from final_system.feedback import update_arimax
from final_system.ppo_policy import (
    DEFAULT_PPO_PATH, HIDDEN_WIDTH, SCHEMA_VERSION, STATE_FEATURE_COUNT,
    load_ppo_policy, policy_observation, policy_version, price_from_action,
)
from final_system.schemas import FeedbackObservation, WeekObservation

from .phase5_data import iter_weekly, load_contract
from .pricing_simulator import KernelDemandSimulator


ROLLOUT_STEPS = 256
TRAIN_EPISODE_WEEKS = 20


@dataclass(frozen=True)
class PricingWeek:
    week: date
    price: float
    qty: float
    event_days: int
    snap_days: int


def pricing_inputs(root, contract, bundle):
    pricing = contract["protocol"]["pricing"]
    series_ids = tuple(sorted(pricing["cohort_skus"]))
    states = {key: bundle.data["m5"][key]["arimax"] for key in series_ids}
    if any(state is None for state in states.values()):
        raise ValueError("Serving bundle has no ARIMAX fit for a frozen pricing-cohort series")

    weekly_parts = []
    for store, weekly, _ in iter_weekly(root, "m5", contract):
        weekly_parts.append(weekly.loc[weekly["sku"].isin(series_ids) & weekly["observed_sales"]])
        print(f"[PPO] Verified cohort data for {store}", flush=True)
    weekly = pd.concat(weekly_parts, ignore_index=True)
    train = weekly.loc[weekly["split"].eq("train")].copy()
    config = bundle.manifest["config"]
    m5_dates = config["datasets"]["m5"]
    train_end = date.fromisoformat(m5_dates["train_end"])
    test_start = date.fromisoformat(m5_dates["test_start"])
    if train["week"].max().date() != train_end or weekly.loc[weekly["split"].eq("test"), "week"].min().date() != test_start:
        raise ValueError("Serving dates and frozen V2 cohort dates differ")

    simulator = KernelDemandSimulator(config["pricing"]["gaussian_seasonal_bandwidth_weeks"]).fit(train)
    if set(simulator.surfaces) != set(series_ids):
        raise ValueError("Train-fitted simulator does not cover the frozen cohort")

    horizon = pricing["test_weeks_per_series"]
    train_windows, test_windows, fixed_prices = {}, {}, {}
    for key, rows in weekly.groupby("sku", sort=True):
        by_week = {row.week.date(): PricingWeek(row.week.date(), float(row.price), float(row.qty),
                                                int(row.event_days), int(row.snap_days))
                   for row in rows.itertuples(index=False)}
        train_rows = rows.loc[rows["split"].eq("train")]
        bounds = states[key]["price_min"], states[key]["price_max"]
        if not np.allclose(bounds, (train_rows["price"].min(), train_rows["price"].max()), rtol=0.0, atol=1e-9):
            raise ValueError(f"Train simulator and ARIMAX disagree on price bounds: {key}")
        fixed_prices[key] = float(train_rows["price"].median())
        starts = [week for week in by_week if week + timedelta(weeks=TRAIN_EPISODE_WEEKS) <= train_end]
        train_windows[key] = [window for start in starts
                              if (window := complete_window(by_week, start, TRAIN_EPISODE_WEEKS)) is not None]
        test_windows[key] = complete_window(by_week, test_start - timedelta(weeks=1), horizon)
        if not train_windows[key] or test_windows[key] is None:
            raise ValueError(f"Unavailable contiguous train or frozen test window: {key}")

    return series_ids, states, simulator, train_windows, test_windows, fixed_prices


def complete_window(by_week, previous_week, horizon):
    weeks = tuple(by_week.get(previous_week + timedelta(weeks=offset)) for offset in range(horizon + 1))
    return weeks if all(entry is not None for entry in weeks) else None


def replay_key_digest(prepared, test_windows):
    targets = {(key, row.week.isoformat()) for key, window in test_windows.items() for row in window[1:]}
    with (prepared / "test_keys_m5.csv").open(encoding="utf-8", newline="") as stream:
        matched = {(row["sku"], row["week"]) for row in csv.DictReader(stream)
                   if (row["sku"], row["week"]) in targets}
    if matched != targets:
        raise ValueError("A PPO replay target is missing from the frozen M5 test keys")
    canonical = "dataset,sku,week\n" + "".join(f"m5,{sku},{week}\n" for sku, week in sorted(matched))
    return sha256(canonical.encode()).hexdigest()


class M5PricingEnvironment(gym.Env):
    def __init__(self, series_ids, states, simulator, train_windows):
        self.series_ids = series_ids
        self.states = states
        self.simulator = simulator
        self.train_windows = train_windows
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-8.0, 8.0, shape=(STATE_FEATURE_COUNT + len(series_ids),),
                                              dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            key = self.series_ids[self.np_random.integers(len(self.series_ids))]
            windows = self.train_windows[key]
            window = windows[self.np_random.integers(len(windows))]
        else:
            key, window = options["series_id"], options["window"]
        self.key, self.window = key, window
        self.state = self.states[key]
        self.position = 1
        self.previous = WeekObservation(week=window[0].week, qty=window[0].qty, price=window[0].price)
        self.samples = []
        self.update_count = 0
        self.reward_scale = max(1.0, self.state["price_mean"] * self.state["qty_mean"])
        return self.observation(), {}

    def observation(self):
        values = policy_observation(self.series_ids, self.key, self.previous.qty, self.previous.price,
                                    self.previous.price, self.window[self.position].week, self.state)
        return np.asarray(values, dtype=np.float32)

    def step(self, action):
        price = price_from_action(self.state, float(action[0]))
        return self.step_price(price)

    def step_price(self, price):
        current = self.window[self.position]
        known_calendar = {"event_days": current.event_days, "snap_days": current.snap_days}
        qty = self.simulator.sample(self.key, price, current.week, self.np_random, known_calendar)
        revenue = price * qty
        self.samples.append(FeedbackObservation(week=current.week, qty=qty, price=price,
                                                previous_qty=self.previous.qty, previous_price=self.previous.price))
        self.state, used = update_arimax(self.states[self.key], self.samples)
        self.update_count += bool(used)
        self.previous = WeekObservation(week=current.week, qty=qty, price=price)
        self.position += 1
        done = self.position == len(self.window)
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if done else self.observation()
        return observation, revenue / self.reward_scale, done, False, {"revenue": revenue}


def evaluate(policy, model, env, test_windows, fixed, seeds, grid_points):
    runs = []
    for seed in seeds:
        for method in ("fixed_train_median", "arimax_grid_feedback", "ppo_feedback"):
            revenue, update_count = 0.0, 0
            for position, (key, window) in enumerate(sorted(test_windows.items())):
                obs, _ = env.reset(seed=seed + 10_000 * position, options={"series_id": key, "window": window})
                done = False
                while not done:
                    if method == "ppo_feedback":
                        price = policy.recommend(sku=key, store_id=key.rsplit("@", 1)[1],
                                                 previous_qty=env.previous.qty, previous_price=env.previous.price,
                                                 current_price=env.previous.price, week=env.window[env.position].week,
                                                 state=env.state)
                        sb3_action, _ = model.predict(obs, deterministic=True)
                        if abs(price - price_from_action(env.state, float(sb3_action[0]))) > 2e-6:
                            raise ValueError("Exported actor differs from training policy")
                    elif method == "arimax_grid_feedback":
                        price = grid_price(env.state, env.previous, grid_points)
                    else:
                        price = fixed[key]
                    obs, _, done, _, info = env.step_price(price)
                    revenue += info["revenue"]
                update_count += env.update_count
            runs.append({"seed": seed, "method": method, "revenue": revenue, "feedback_updates": update_count})
    reference = {row["seed"]: row["revenue"] for row in runs if row["method"] == "fixed_train_median"}
    for row in runs:
        row["revenue_index"] = 100.0 * row["revenue"] / reference[row["seed"]]
    summary = {}
    for method in ("fixed_train_median", "arimax_grid_feedback", "ppo_feedback"):
        totals = [row["revenue"] for row in runs if row["method"] == method]
        indices = [row["revenue_index"] for row in runs if row["method"] == method]
        summary[method] = {"revenue_mean": mean(totals), "revenue_std": stdev(totals),
                           "revenue_index_mean": mean(indices), "revenue_index_std": stdev(indices)}
    return {"runs": runs, "summary": summary}


def grid_price(state, previous, points):
    low, high = state["price_min"], state["price_max"]
    prices = np.linspace(low, high, points).tolist()
    if low <= previous.price <= high:
        prices.append(previous.price)
    candidates = [_price_point(state, previous, price) for price in prices]
    best_revenue = max(point["expected_revenue"] for point in candidates)
    tied = [point for point in candidates if best_revenue - point["expected_revenue"] <= 1e-8]
    return min(tied, key=lambda point: (abs(point["price"] - previous.price), point["price"]))["price"]


def run(prepared, model_dir, output, steps, seed):
    if steps < ROLLOUT_STEPS or steps % ROLLOUT_STEPS:
        raise ValueError(f"Use a positive multiple of {ROLLOUT_STEPS} PPO steps")
    torch.set_num_threads(1)
    bundle = load_bundle(model_dir)
    contract = load_contract(prepared)
    keys, states, simulator, train_windows, test_windows, fixed = pricing_inputs(prepared, contract, bundle)
    replay_sha256 = replay_key_digest(prepared, test_windows)
    env = M5PricingEnvironment(keys, states, simulator, train_windows)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [HIDDEN_WIDTH, HIDDEN_WIDTH],
                                                               "vf": [HIDDEN_WIDTH, HIDDEN_WIDTH]},
                                                "activation_fn": torch.nn.Tanh},
                n_steps=ROLLOUT_STEPS, batch_size=64, n_epochs=6, learning_rate=3e-4, ent_coef=0.005,
                seed=seed, device="cpu")
    print(f"[PPO] Training {steps} continuous actions on {len(keys)} train-only M5 series", flush=True)
    model.learn(total_timesteps=steps)
    layers = [module for module in model.policy.mlp_extractor.policy_net if isinstance(module, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    payload = {"schema_version": SCHEMA_VERSION,
               "demand_bundle_sha256": bundle.manifest["file_sha256"], "series_ids": keys,
               "training": {"algorithm": "stable_baselines3.PPO", "sb3_version": sb3_version,
                            "torch_version": str(torch.__version__), "seed": seed, "steps": model.num_timesteps,
                            "fit_scope": "train_only", "episode_weeks": TRAIN_EPISODE_WEEKS,
                            "simulator": "train_fitted_gaussian_kernel", "feedback": "shared_arimax_beta_ridge",
                            "train_end": bundle.manifest["config"]["datasets"]["m5"]["train_end"],
                            "phase1_sha256": contract["phase1_sha256"]},
               "layers": [{"weights": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
                          for layer in layers]}
    payload["version"] = policy_version(payload)
    encoded = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        actor_file = Path(staging) / "actor.json"
        actor_file.write_bytes(encoded)
        policy = load_ppo_policy(bundle, actor_file)
        report = evaluate(policy, model, env, test_windows, fixed, contract["protocol"]["pricing"]["seeds"],
                          bundle.manifest["config"]["pricing"]["grid_points"])
    report.update({"policy_version": policy.version, "pricing_scope": policy.scope,
                   "test_start": str(next(iter(test_windows.values()))[1].week),
                   "test_end": str(next(iter(test_windows.values()))[-1].week),
                   "test_decisions_per_strategy_per_seed": sum(len(window) - 1 for window in test_windows.values()),
                   "test_keys_sha256": replay_sha256,
                   "revenue_index_reference": "100 = fixed train-median price in same simulator and seed"})
    atomic_write(output, encoded)
    atomic_write(output.with_suffix(".evaluation.json"), (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[PPO] Wrote {policy.version} to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PPO_PATH)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.prepared, args.model_dir, args.output, args.steps, args.seed)


if __name__ == "__main__":
    main()
