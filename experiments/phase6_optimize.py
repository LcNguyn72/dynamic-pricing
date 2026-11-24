"""Choose and audit one-step M5 prices under a declared revenue objective.

Every method sees the same observed 12-week history and the same price range:
the intersection of train price support and a ±10% move from last week's
observed price. Expected revenue is calculated in the selected simulator for
comparison only; it is not observed or causal business revenue.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, PPO, SAC
import torch

from .pricing_simulator import LagAdjustedKernelSimulator
from .song_forecast import feature_arrays
from .song_pricing_pilot import _environment, load_forecaster


DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"
MAX_CHANGE = 0.10
GRID_POINTS = 81
METHODS = ("hold_current", "fixed_mean", "heuristic", "forecast_grid_ensemble",
           "ppo_frozen", "ddpg", "sac", "simulator_oracle")
LEARNED = {"ppo_frozen": PPO, "ddpg": DDPG, "sac": SAC}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} does not match the saved experiment")


def _cohort(windows: pd.DataFrame, skus: list[str], split: str,
            simulator: LagAdjustedKernelSimulator) -> tuple[pd.DataFrame, dict]:
    rows = windows.loc[windows["split"].eq(split)
                       & windows["observed_sales"].astype(bool)
                       & windows["sku"].isin(skus)].copy()
    groups = rows.sort_values(["sku", "week"]).groupby("sku", group_keys=False)
    rows = groups.tail(20) if split == "validation" else groups.head(20)
    counts = rows.groupby("sku").size().to_dict()
    if counts != {sku: 20 for sku in skus}:
        raise ValueError(f"Expected 20 observed {split} weeks per SKU: {counts}")
    rows["reference_in_support"] = rows.apply(
        lambda row: (simulator.surfaces[row["sku"]].price_min
                     <= float(row["price_lag_1"])
                     <= simulator.surfaces[row["sku"]].price_max), axis=1)
    excluded = rows.loc[~rows["reference_in_support"], ["sku", "week"]].copy()
    eligible = rows.loc[rows["reference_in_support"]].copy()
    if eligible.empty:
        raise ValueError(f"No {split} rows have an in-support reference price")
    return eligible, {
        "candidate_rows": len(rows), "eligible_rows": len(eligible),
        "excluded_outside_train_support": len(excluded),
        "eligible_skus": int(eligible["sku"].nunique()),
        "first_week": eligible["week"].min().date().isoformat(),
        "last_week": eligible["week"].max().date().isoformat(),
    }


class ForecastGridOptimizer:
    """Maximize mean Transformer revenue on 81 prices plus the hold price."""

    def __init__(self, forecasters: dict[int, object], simulator: LagAdjustedKernelSimulator,
                 max_change: float = MAX_CHANGE, grid_points: int = GRID_POINTS):
        if not forecasters or grid_points < 2 or not 0 < max_change < 1:
            raise ValueError("Forecasters, at least two grid points and a valid move cap are required")
        self.forecasters = forecasters
        self.simulator = simulator
        self.max_change = float(max_change)
        self.grid_points = int(grid_points)

    def prices(self, sku: str, reference: float) -> tuple[np.ndarray, float, float]:
        surface = self.simulator.surfaces[sku]
        if not surface.price_min <= reference <= surface.price_max:
            raise ValueError("Reference price must be inside training support")
        lo, hi = self.simulator.price_bounds(sku, reference, self.max_change)
        candidates = np.unique(np.append(np.linspace(lo, hi, self.grid_points), reference))
        return candidates, lo, hi

    def forecast_revenue(self, row: pd.Series, prices: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame([row.to_dict()] * len(prices))
        estimates = []
        for forecaster in self.forecasters.values():
            history, current = feature_arrays(
                frame, forecaster.scalers, prices.astype(np.float32))
            with torch.no_grad():
                z = forecaster.model(torch.from_numpy(history),
                                     torch.from_numpy(current)).numpy()
            scaler = forecaster.scalers[str(row["sku"])]
            qty = np.maximum(0.0, z * float(scaler["qty_std"]) + float(scaler["qty_mean"]))
            estimates.append(prices * qty)
        return np.asarray(estimates, dtype=float)

    def recommend(self, row: pd.Series, reference_price: float | None = None) -> dict:
        # A scenario may use an entered current price while all twelve
        # historical lag prices remain the prices actually observed.
        reference = float(row["price_lag_1"] if reference_price is None else reference_price)
        prices, lo, hi = self.prices(str(row["sku"]), reference)
        revenue_by_seed = self.forecast_revenue(row, prices)
        means = revenue_by_seed.mean(axis=0)
        # The reference price is included, so predicted mean gain cannot be negative.
        maximum = float(means.max())
        ties = np.flatnonzero(np.isclose(means, maximum, rtol=0, atol=1e-8))
        best = min(ties, key=lambda index: (abs(prices[index] - reference), prices[index]))
        hold = int(np.argmin(np.abs(prices - reference)))
        cap5 = np.flatnonzero((prices >= reference * .95 - 1e-8)
                              & (prices <= reference * 1.05 + 1e-8))
        cap5_max = float(means[cap5].max())
        cap5_ties = [int(index) for index in cap5
                     if np.isclose(means[index], cap5_max, rtol=0, atol=1e-8)]
        cap5_best = min(cap5_ties, key=lambda index: (abs(prices[index] - reference), prices[index]))
        seed_best_prices = []
        for seed_revenue in revenue_by_seed:
            seed_ties = np.flatnonzero(np.isclose(
                seed_revenue, seed_revenue.max(), rtol=0, atol=1e-8))
            seed_best = min(seed_ties,
                            key=lambda index: (abs(prices[index] - reference), prices[index]))
            seed_best_prices.append(float(prices[seed_best]))
        return {
            "reference_price": reference, "recommended_price": float(prices[best]),
            "forecast_revenue_at_reference": float(means[hold]),
            "forecast_revenue_at_recommendation": maximum,
            "forecast_gain_pct": (100 * (maximum / means[hold] - 1)
                                  if means[hold] > 0 else None),
            "seed_revenue_at_recommendation": revenue_by_seed[:, best].tolist(),
            "seed_best_prices": seed_best_prices,
            "cap5_recommended_price": float(prices[cap5_best]),
            "cap5_forecast_revenue_at_recommendation": cap5_max,
            "local_min": lo, "local_max": hi,
            "candidate_prices": prices,
        }


def _policy_observations(row: pd.Series, simulator, forecaster,
                         reference_price: float | None = None):
    single = pd.DataFrame([row.to_dict()])
    demand_env = _environment(single, simulator, str(row["split"]), forecaster,
                              gradient=True, feedback=False)
    demand_env.reset(seed=42, options={"sku": str(row["sku"])})
    demand_env.current_price = float(row["price_lag_1"] if reference_price is None
                                     else reference_price)
    demand_obs = demand_env._observation()
    bare_env = _environment(single, simulator, str(row["split"]))
    bare_env.reset(seed=42, options={"sku": str(row["sku"])})
    bare_env.current_price = float(row["price_lag_1"] if reference_price is None
                                   else reference_price)
    bare_obs = bare_env._observation()
    return (demand_env, demand_obs), (bare_env, bare_obs)


def _candidate_prices(row: pd.Series, recommendation: dict, simulator,
                      policy_models: dict[int, dict], forecaster) -> dict[int, dict[str, float]]:
    reference = recommendation["reference_price"]
    lo, hi = recommendation["local_min"], recommendation["local_max"]
    surface = simulator.surfaces[str(row["sku"])]
    fixed = float(np.mean(np.exp(surface.log_prices)))
    recent = np.asarray([float(row[f"qty_lag_{lag}"]) for lag in range(1, 5)])
    direction = 1.0 if recent[0] > recent.mean() else -1.0
    context = {name: row[name] for name in ("event_days", "snap_days",
                                           "qty_lag_1", "price_lag_1") if name in row.index}
    grid = recommendation["candidate_prices"]
    simulated = np.asarray([price * simulator.predict_mean(str(row["sku"]), float(price),
                                                            row["week"], context)
                            for price in grid])
    oracle = float(grid[int(np.argmax(simulated))])
    base = {
        "hold_current": reference,
        "fixed_mean": float(np.clip(fixed, lo, hi)),
        "heuristic": float(np.clip(reference * (1 + .05 * direction), lo, hi)),
        "forecast_grid_ensemble": recommendation["recommended_price"],
        "simulator_oracle": oracle,
    }
    (demand_env, demand_obs), (bare_env, bare_obs) = _policy_observations(
        row, simulator, forecaster)
    choices = {}
    for seed, models in policy_models.items():
        result = dict(base)
        for method, model in models.items():
            env, observation = ((demand_env, demand_obs) if method == "ppo_frozen"
                                else (bare_env, bare_obs))
            if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
                raise ValueError(f"{method} observation shape changed")
            action = float(np.asarray(model.predict(observation, deterministic=True)[0]).reshape(-1)[0])
            result[method] = float(np.clip(env.price_from_action(action), lo, hi))
        choices[seed] = result
    return choices


def _policy_features(row: pd.Series) -> pd.Series:
    """Keep only information available before the target week's sale."""
    required = {"sku", "week", "split", "observed_sales"}
    required.update(f"{name}_lag_{lag}" for lag in range(1, 13)
                    for name in ("qty", "price"))
    missing = required.difference(row.index)
    if missing:
        raise ValueError(f"Missing pre-decision features: {sorted(missing)}")
    known = required | {"event_days", "snap_days"}
    known.update(f"{name}_lag_{lag}" for lag in range(1, 13)
                 for name in ("event_days", "snap_days", "promotion_flag"))
    return row.loc[[name for name in row.index if name in known]]


def _score_rows(split: str, rows: pd.DataFrame, optimizer: ForecastGridOptimizer,
                simulator, policy_models: dict[int, dict], forecaster) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []
    grid_details = []
    for row in rows.itertuples(index=False):
        original = pd.Series(row._asdict())
        features = _policy_features(original)
        recommendation = optimizer.recommend(features)
        choices = _candidate_prices(features, recommendation, simulator,
                                    policy_models, forecaster)
        sku = str(features["sku"])
        context = {name: features[name] for name in ("event_days", "snap_days",
                                                      "qty_lag_1", "price_lag_1")
                   if name in features.index}
        reference = recommendation["reference_price"]
        hold_revenue = reference * simulator.predict_mean(sku, reference, features["week"], context)
        oracle_price = next(iter(choices.values()))["simulator_oracle"]
        oracle_revenue = oracle_price * simulator.predict_mean(
            sku, oracle_price, features["week"], context)
        week = pd.Timestamp(features["week"]).date().isoformat()
        grid_details.append({
            "split": split, "sku": sku, "week": week,
            "reference_price": reference,
            "recommended_price": recommendation["recommended_price"],
            "forecast_revenue_at_reference": recommendation["forecast_revenue_at_reference"],
            "forecast_revenue_at_recommendation": recommendation["forecast_revenue_at_recommendation"],
            "forecast_gain_pct": recommendation["forecast_gain_pct"],
            "seed_revenue_sd_at_recommendation": float(np.std(
                recommendation["seed_revenue_at_recommendation"], ddof=1)),
            "seed_best_price_range_pct": float(100 * (
                max(recommendation["seed_best_prices"])
                - min(recommendation["seed_best_prices"])) / reference),
            "cap5_recommended_price": recommendation["cap5_recommended_price"],
            "cap5_changes_recommendation": bool(abs(
                recommendation["cap5_recommended_price"]
                - recommendation["recommended_price"]) > 1e-8),
            "cap5_forecast_revenue_at_recommendation": recommendation["cap5_forecast_revenue_at_recommendation"],
            "local_min": recommendation["local_min"],
            "local_max": recommendation["local_max"],
        })
        for seed, methods in choices.items():
            if set(methods) != set(METHODS):
                raise AssertionError("A method is missing from the common decision set")
            for method, price in methods.items():
                expected_units = float(simulator.predict_mean(sku, price,
                                                             features["week"], context))
                expected_revenue = price * expected_units
                decisions.append({
                    "split": split, "sku": sku, "week": week,
                    "seed": seed, "method": method,
                    "reference_price": reference, "chosen_price": price,
                    "local_min": recommendation["local_min"],
                    "local_max": recommendation["local_max"],
                    "price_change_pct": 100 * (price / reference - 1),
                    "simulator_expected_units": expected_units,
                    "simulator_expected_revenue": expected_revenue,
                    "hold_expected_revenue": hold_revenue,
                    "oracle_expected_revenue": oracle_revenue,
                    "expected_regret": max(0.0, oracle_revenue - expected_revenue),
                    "below_hold": expected_revenue < hold_revenue - 1e-8,
                    "at_local_bound": (price <= recommendation["local_min"] + 1e-6
                                       or price >= recommendation["local_max"] - 1e-6),
                })
    return pd.DataFrame(decisions), pd.DataFrame(grid_details)


def _summaries(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.duplicated(["split", "sku", "week", "seed", "method"]).any():
        raise AssertionError("Duplicate decision row")
    expected = set(METHODS)
    for _, frame in decisions.groupby(["split", "sku", "week", "seed"]):
        if set(frame["method"]) != expected:
            raise AssertionError("Methods were not evaluated on the same SKU-week")
    runs = decisions.groupby(["split", "method", "seed"], as_index=False).agg(
        decisions=("sku", "size"),
        expected_revenue=("simulator_expected_revenue", "sum"),
        hold_revenue=("hold_expected_revenue", "sum"),
        expected_regret=("expected_regret", "sum"),
        below_hold_pct=("below_hold", "mean"),
        at_local_bound_pct=("at_local_bound", "mean"),
        mean_abs_price_change_pct=("price_change_pct", lambda values: float(np.mean(np.abs(values)))),
    )
    runs["uplift_vs_hold_pct"] = 100 * (runs["expected_revenue"] / runs["hold_revenue"] - 1)
    runs[["below_hold_pct", "at_local_bound_pct"]] *= 100
    summary = runs.groupby(["split", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), decisions=("decisions", "first"),
        uplift_mean_pct=("uplift_vs_hold_pct", "mean"),
        uplift_sd_pct=("uplift_vs_hold_pct", "std"),
        expected_revenue_mean=("expected_revenue", "mean"),
        below_hold_mean_pct=("below_hold_pct", "mean"),
        at_local_bound_mean_pct=("at_local_bound_pct", "mean"),
        mean_abs_price_change_pct=("mean_abs_price_change_pct", "mean"),
    )
    return runs, summary


def _validation_selection(runs: pd.DataFrame, summary: pd.DataFrame) -> dict:
    valid = summary.loc[summary["split"].eq("validation")
                        & ~summary["method"].eq("simulator_oracle")]
    winner = valid.sort_values(["uplift_mean_pct", "method"],
                               ascending=[False, True]).iloc[0]
    method = str(winner["method"])
    selected_seed = None
    if method in LEARNED:
        seeded = runs.loc[runs["split"].eq("validation")
                          & runs["method"].eq(method)].sort_values("uplift_vs_hold_pct")
        selected_seed = int(seeded.iloc[len(seeded) // 2]["seed"])
    return {
        "criterion": "Highest mean validation expected simulator revenue, excluding simulator oracle",
        "method": method, "seed_if_learned": selected_seed,
        "validation_uplift_vs_hold_pct": float(winner["uplift_mean_pct"]),
        "status": "exploratory simulator winner, not a verified business-optimal price",
    }


def _select_per_sku(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use validation alone to choose a method and, if needed, one seed per SKU."""
    validation = decisions.loc[decisions["split"].eq("validation")
                               & ~decisions["method"].eq("simulator_oracle")]
    runs = validation.groupby(["sku", "method", "seed"], as_index=False).agg(
        expected_revenue=("simulator_expected_revenue", "sum"),
        hold_revenue=("hold_expected_revenue", "sum"))
    runs["uplift_pct"] = 100 * (runs["expected_revenue"] / runs["hold_revenue"] - 1)
    methods = runs.groupby(["sku", "method"], as_index=False).agg(
        mean_validation_uplift_pct=("uplift_pct", "mean"),
        sd_validation_uplift_pct=("uplift_pct", "std"))
    # Hold wins exact ties. This avoids a price change without measured
    # simulator benefit on validation.
    methods["tie_priority"] = methods["method"].ne("hold_current").astype(int)
    best = (methods.sort_values(
        ["sku", "mean_validation_uplift_pct", "tie_priority", "method"],
        ascending=[True, False, True, True]).groupby("sku", as_index=False).head(1))
    selected = []
    for row in best.itertuples(index=False):
        if row.method in LEARNED:
            choices = runs.loc[runs["sku"].eq(row.sku)
                               & runs["method"].eq(row.method)].sort_values(
                                   ["uplift_pct", "seed"])
            seed = int(choices.iloc[len(choices) // 2]["seed"])
        else:
            seed = 42
        selected.append({
            "sku": row.sku, "method": row.method, "seed": seed,
            "validation_mean_uplift_pct": float(row.mean_validation_uplift_pct),
            "validation_seed_sd_pct": float(row.sd_validation_uplift_pct),
        })
    selected_frame = pd.DataFrame(selected)
    selected_decisions = decisions.merge(
        selected_frame[["sku", "method", "seed"]], on=["sku", "method", "seed"],
        how="inner", validate="many_to_one")
    for split in ("validation", "test"):
        selected_keys = set(zip(selected_decisions.loc[selected_decisions["split"].eq(split), "sku"],
                                selected_decisions.loc[selected_decisions["split"].eq(split), "week"]))
        expected_keys = set(zip(decisions.loc[decisions["split"].eq(split), "sku"],
                                decisions.loc[decisions["split"].eq(split), "week"]))
        if selected_keys != expected_keys or len(selected_decisions.loc[selected_decisions["split"].eq(split)]) != len(expected_keys):
            raise AssertionError(f"Per-SKU selection did not cover every {split} decision once")
    test_per_sku = selected_decisions.loc[selected_decisions["split"].eq("test")].groupby("sku").agg(
        eligible_test_weeks=("week", "size"),
        test_expected_revenue=("simulator_expected_revenue", "sum"),
        test_hold_revenue=("hold_expected_revenue", "sum")).reset_index()
    test_per_sku["exploratory_test_uplift_pct"] = 100 * (
        test_per_sku["test_expected_revenue"] / test_per_sku["test_hold_revenue"] - 1)
    selected_frame = selected_frame.merge(
        test_per_sku[["sku", "eligible_test_weeks", "exploratory_test_uplift_pct"]],
        on="sku", how="left", validate="one_to_one")
    result = {}
    for split, frame in selected_decisions.groupby("split"):
        result[split] = {
            "decisions": len(frame),
            "simulator_expected_uplift_vs_hold_pct": float(
                100 * (frame["simulator_expected_revenue"].sum()
                       / frame["hold_expected_revenue"].sum() - 1)),
            "changed_price_decisions": int(np.sum(
                np.abs(frame["chosen_price"] - frame["reference_price"]) > 1e-8)),
        }
    return selected_frame, selected_decisions, result


def run(pilot_dir: Path = DEFAULT_ROOT, pricing_dir: Path | None = None,
        output_dir: Path | None = None) -> dict:
    pilot_dir = Path(pilot_dir)
    pricing_dir = Path(pricing_dir) if pricing_dir else pilot_dir / "pricing_lag_512_5seeds"
    output_dir = Path(output_dir) if output_dir else pilot_dir / "phase6_price_selection"
    manifest_path = pilot_dir / "manifest.json"
    windows_path, weekly_path = pilot_dir / "windows.csv", pilot_dir / "weekly.csv"
    pricing_manifest_path = pricing_dir / "pricing_manifest.json"
    pricing_manifest = json.loads(pricing_manifest_path.read_text(encoding="utf-8"))
    if pricing_manifest["simulator_variant"] != "lag_adjusted_kernel":
        raise ValueError("Phase 6 uses the validation-selected simulator")
    _check_hash(manifest_path, pricing_manifest["source_manifest_sha256"], "Prepared manifest")
    _check_hash(Path(pricing_manifest["forecast_checkpoint"]),
                pricing_manifest["forecast_checkpoint_sha256"], "Base forecaster")
    simulator_config_path = Path(pricing_manifest["simulator_config"])
    _check_hash(simulator_config_path, pricing_manifest["simulator_config_sha256"],
                "Simulator configuration")
    simulator_config = json.loads(simulator_config_path.read_text(encoding="utf-8"))
    _check_hash(weekly_path, simulator_config["prepared_weekly_sha256"], "Weekly data")
    prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_hash(windows_path, prepared["windows_sha256"], "Forecast windows")
    if len(pricing_manifest["evaluation_skus"]) != 10 or len(pricing_manifest["seeds"]) != 5:
        raise ValueError("Phase 6 expects the locked 10-SKU, five-seed M5 pilot")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weekly = pd.read_csv(weekly_path, dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(windows_path, dtype={"sku": str}, parse_dates=["week"])
    simulator = LagAdjustedKernelSimulator(
        float(simulator_config["recent_weight"]),
        ratio_limit=float(simulator_config["ratio_limit"])).fit(weekly)
    forecasters = {}
    forecaster_hashes = {}
    policy_models = {}
    policy_hashes = {}
    for seed in pricing_manifest["seeds"]:
        song_dir = pilot_dir / "multiseed" / f"seed{seed}" / "song"
        song_run = json.loads((song_dir / f"run_seed{seed}.json").read_text(encoding="utf-8"))
        if (song_run["source_manifest_sha256"] != sha256(manifest_path)
                or song_run["pricing_checkpoint_fit_scope"] != "train"):
            raise ValueError(f"Forecast seed {seed} was not fitted on this training panel")
        forecast_path = song_dir / f"song_transformer_seed{seed}.pt"
        forecasters[seed] = load_forecaster(forecast_path)
        forecaster_hashes[str(seed)] = sha256(forecast_path)
        policy_models[seed] = {}
        for method, cls in LEARNED.items():
            model_path = pricing_dir / "models" / f"{method}_seed{seed}.zip"
            policy_models[seed][method] = cls.load(str(model_path), device="cpu")
            policy_hashes[f"{method}_seed{seed}"] = sha256(model_path)
    optimizer = ForecastGridOptimizer(forecasters, simulator)
    decisions = []
    grid_details = []
    cohort_stats = {}
    for split in ("validation", "test"):
        rows, cohort_stats[split] = _cohort(
            windows, pricing_manifest["evaluation_skus"], split, simulator)
        frame, grid = _score_rows(split, rows, optimizer, simulator,
                                  policy_models, forecasters[42])
        decisions.append(frame)
        grid_details.append(grid)
        print(f"[Phase 6] {split}: {len(rows)} SKU-weeks × {len(METHODS)} methods "
              f"× {len(policy_models)} seeds", flush=True)
    all_decisions = pd.concat(decisions, ignore_index=True)
    all_grid = pd.concat(grid_details, ignore_index=True)
    runs, summary = _summaries(all_decisions)
    selection = _validation_selection(runs, summary)
    per_sku, selected_decisions, per_sku_result = _select_per_sku(all_decisions)
    test_row = summary.loc[summary["split"].eq("test")
                           & summary["method"].eq(selection["method"])].iloc[0]
    selection["exploratory_test_uplift_vs_hold_pct"] = float(test_row["uplift_mean_pct"])
    observed_check_path = pilot_dir / "simulator_selection" / "observed_price_metrics.csv"
    observed = pd.read_csv(observed_check_path)
    test_check = observed.loc[observed["split"].eq("test")
                              & observed["segment"].eq("in_support")]
    sim_rmse = float(test_check.loc[test_check["model"].eq("selected_lag_kernel"), "rmse"].iloc[0])
    naive_rmse = float(test_check.loc[test_check["model"].eq("naive_last"), "rmse"].iloc[0])
    simulator_gate_passed = sim_rmse <= naive_rmse
    output_dir.mkdir(parents=True, exist_ok=True)
    all_decisions.to_csv(output_dir / "one_step_decisions.csv", index=False)
    all_grid.to_csv(output_dir / "forecast_grid_prices.csv", index=False)
    runs.to_csv(output_dir / "policy_runs.csv", index=False)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    per_sku.to_csv(output_dir / "selected_policy_by_sku.csv", index=False)
    selected_decisions.to_csv(output_dir / "selected_price_decisions.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": prepared["dataset"],
        "scope": "One-step observed-history snapshots; not Phase 4's sequential online rollout",
        "prepared_manifest_sha256": sha256(manifest_path),
        "weekly_sha256": sha256(weekly_path),
        "windows_sha256": sha256(windows_path),
        "pricing_manifest_sha256": sha256(pricing_manifest_path),
        "simulator_config_sha256": sha256(simulator_config_path),
        "simulator_observed_check_sha256": sha256(observed_check_path),
        "forecaster_checkpoint_sha256_by_seed": forecaster_hashes,
        "policy_checkpoint_sha256_by_method_seed": policy_hashes,
        "seeds": pricing_manifest["seeds"],
        "methods": METHODS,
        "evaluation_skus": pricing_manifest["evaluation_skus"],
        "cohort": cohort_stats,
        "price_rule": {
            "objective": "maximize mean forecast price * units across five train-only Transformer seeds",
            "grid_points": GRID_POINTS, "max_change_fraction": MAX_CHANGE,
            "feasible_range": "train support intersect reference price ±10%, with hold price included",
            "reference": "previous observed week's price",
        },
        "sensitivity": {
            "cap5_price": "best forecast price within ±5% using the same 81-price grid plus hold",
            "seed_best_price_range_pct": "range of five seed-specific forecast-optimal prices as percent of reference",
        },
        "global_validation_selection": selection,
        "per_sku_validation_selection": {
            "criterion": "Highest mean validation simulator uplift per SKU; hold wins ties; learned-policy seed is the median validation run",
            "methods_chosen": per_sku["method"].value_counts().to_dict(),
            "validation_and_exploratory_test": per_sku_result,
            "status": "exploratory simulator-specific rule, not verified business optimality",
        },
        "simulator_test_observed_price_gate": {
            "passed": simulator_gate_passed,
            "selected_simulator_rmse": sim_rmse, "naive_last_rmse": naive_rmse,
        },
        "business_deployment_supported": False,
        "limits": [
            "M5 has no observed demand at unchosen prices; expected revenue is simulator-based",
            "The simulator did not beat naive demand prediction on test observed prices",
            "The existing test period has already been inspected in Phase 4 and is exploratory",
            "The same validation period was used to select the simulator; validation policy ranking is exploratory",
            "Phase 6 snapshots use observed history and do not reproduce Phase 4 online feedback rollouts",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pricing-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.pilot_dir, args.pricing_dir, args.output_dir)
    print(json.dumps({"global_selection": result["global_validation_selection"],
                      "per_sku_selection": result["per_sku_validation_selection"],
                      "simulator_gate": result["simulator_test_observed_price_gate"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
