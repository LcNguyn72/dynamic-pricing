"""Compare tuned ARIMAX grid, PPO and feedback against a validation-fixed price."""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import PPO
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels

from .phase1_finalize import _observed
from .phase5_data import LOCK_PATH, PHASE1_ROOT, iter_weekly, load_contract, windows_from_weekly
from .phase5_pricing import FIXED_MAX, _TestEnvironment, _atomic_json, _grid_action, _unit, test_inputs
from .pricing_arimax import _write_csv
from .pricing_benchmark import evaluate_policy
from .pricing_policy_compare import CACHE, KEYS, STEP3, audit_decisions, samplers, verified
from .song_forecast import LOOKBACK, _sha256
from .song_pricing_pilot import _ppo


OUTPUT = PHASE1_ROOT.with_name("pricing_arimax_feedback_v2")
STEP4 = PHASE1_ROOT.with_name("pricing_policy_compare_v2")
GRID_BASE = "grid_arimax_step3"
GRID_TUNED = "grid_arimax_tuned"
PPO_FIXED = "ppo_arimax_frozen"
PPO_FEEDBACK = "ppo_arimax_feedback"
FIXED_SKU = "fixed_validation_sku"
METHODS = (FIXED_MAX, FIXED_SKU, GRID_BASE, GRID_TUNED, PPO_FIXED, PPO_FEEDBACK)
RIDGE = 32.0
MIN_FEEDBACK = 4
ORDERS = (0, 1, 2)
TRANSFORMS = ("level", "log")


def _scaled_price(price, state):
    values = np.asarray(price, dtype=float)
    if state["transform"] == "log":
        values = np.log(values)
    return (values - state["price_mean"]) / state["price_sd"]


def fit_model(qty, price, order, transform):
    """Fit a stationary AR error model and a contemporaneous price regressor on train."""
    observed = np.isfinite(qty)
    p = price[observed]
    if p.size < 13 or not np.isfinite(p).all() or np.any(p <= 0):
        return None
    x = np.log(p) if transform == "log" else p
    state = {
        "order": order, "transform": transform, "price_min": float(p.min()), "price_max": float(p.max()),
        "price_mean": float(x.mean()), "price_sd": float(x.std()),
        "qty_mean": float(qty[observed].mean()), "qty_sd": float(qty[observed].std()),
    }
    if min(state["price_sd"], state["qty_sd"]) <= 1e-8:
        return None
    if order == 0:
        # With independent Gaussian errors the exact ML solution is OLS on
        # observed pairs, with innovation variance estimated as RSS / n.
        design = np.column_stack((np.ones(len(x)), (x - state["price_mean"]) / state["price_sd"]))
        target = (qty[observed] - state["qty_mean"]) / state["qty_sd"]
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        state["params"] = [*coefficients.tolist(), float(np.mean(residual ** 2))]
        return state
    # A missing observed sale supplies no likelihood contribution. Supply a neutral
    # finite price design on rows where the Kalman filter has no observation.
    clean_price = np.where(np.isfinite(price) & (price > 0), price, float(p.mean()))
    y = (qty - state["qty_mean"]) / state["qty_sd"]
    exog = _scaled_price(clean_price, state)[:, None]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, exog=exog, order=(order, 0, 0), trend="c")
            result = model.fit(disp=False, maxiter=80)
            if not result.mle_retvals.get("converged", False):
                result = model.fit(disp=False, method="powell", maxiter=80)
    except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if not result.mle_retvals.get("converged", False) or not np.isfinite(result.params).all():
        return None
    state["params"] = result.params.tolist()
    return state


class ArimaxAdapter:
    """Use the last known residuals and optionally update one coefficient per SKU."""

    def __init__(self, states, coefficients=None):
        self.states = states
        self.coefficients = {} if coefficients is None else dict(coefficients)
        self.counts = {}

    @staticmethod
    def _features(row, state, prices):
        params = state["params"]
        rho = np.asarray(params[2:-1])
        lags = range(1, state["order"] + 1)
        previous_qty = np.array([row[f"qty_lag_{lag}"] for lag in lags], dtype=float)
        previous_price = np.array([row[f"price_lag_{lag}"] for lag in lags], dtype=float)
        if (not np.isfinite(previous_qty).all() or not np.isfinite(previous_price).all()
                or np.any(previous_price <= 0)):
            raise ValueError("ARIMAX requires contiguous observed or simulated residual history")
        scaled_qty = (previous_qty - state["qty_mean"]) / state["qty_sd"]
        z_without_price = params[0] + float(rho @ scaled_qty)
        x_filtered = _scaled_price(prices, state) - float(rho @ _scaled_price(previous_price, state))
        return z_without_price, x_filtered

    def _raw(self, row, prices):
        sku = str(row["sku"])
        state = self.states[sku]
        prices = np.asarray(prices, dtype=float)
        if (not np.isfinite(prices).all() or np.any(prices <= 0)
                or np.any(prices < state["price_min"] - 1e-6)
                or np.any(prices > state["price_max"] + 1e-6)):
            raise ValueError("Candidate prices must be finite and in train support")
        intercept, x = self._features(row, state, prices)
        beta = self.coefficients.get(sku, state["params"][1])
        return state["qty_mean"] + (intercept + beta * x) * state["qty_sd"], beta, state

    def predict_prices(self, row, prices):
        raw, _, _ = self._raw(row, prices)
        return np.maximum(raw, 0.0)

    def predict_q(self, row, price):
        return float(self.predict_prices(row, [price])[0])

    def price_gradient(self, row, price):
        raw, beta, state = self._raw(row, [price])
        slope = beta * state["qty_sd"] / state["price_sd"]
        if state["transform"] == "log":
            slope /= price
        return float(slope) if raw[0] > 0 else 0.0

    def fine_tune(self, samples, epochs=1, lr=1e-4):
        """Ridge update of beta toward its train estimate; leave AR terms fixed."""
        grouped = defaultdict(list)
        for row, price, quantity in samples:
            grouped[str(row["sku"])].append((row, float(price), float(quantity)))
        for sku, examples in grouped.items():
            if len(examples) < MIN_FEEDBACK or np.ptp([price for _, price, _ in examples]) <= 1e-6:
                continue
            state = self.states[sku]
            pairs = []
            for row, price, observed in examples:
                intercept, x = self._features(row, state, [price])
                target = (observed - state["qty_mean"]) / state["qty_sd"] - intercept
                pairs.append((x[0], target))
            x, y = np.asarray(pairs, dtype=float).T
            beta = (float(x @ y) + RIDGE * state["params"][1]) / (float(x @ x) + RIDGE)
            if not np.isfinite(beta):
                raise ValueError("Non-finite ARIMAX price coefficient after feedback")
            self.coefficients[sku] = beta
            self.counts[sku] = self.counts.get(sku, 0) + 1


def simulation_inputs(weekly, skus, first, n_weeks, split):
    """Reuse the frozen simulator input shape for any chronological pricing window."""
    last = first + pd.Timedelta(weeks=n_weeks - 1)
    start = first - pd.Timedelta(weeks=LOOKBACK)
    part = weekly.loc[weekly["sku"].isin(skus) & weekly["week"].between(start, last)].copy()
    target = part["week"].ge(first)
    if not part.loc[target, "split"].eq(split).all() or len(part) != len(skus) * (LOOKBACK + n_weeks):
        raise ValueError(f"Cannot build the requested {split} calendar")
    if not _observed(part).all():
        raise ValueError("Pricing windows require observed prices and an observed warm start")
    part["split"] = np.where(target, "test", "validation")
    return test_inputs(part, skus, first, n_weeks)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_weekly_and_states(root, cache, contract, step3, step4):
    previous = read_json(step4 / "run.json")
    if previous["status"] != "complete" or previous["expanded"]["series"] != 50:
        raise ValueError("Run both Step 4 cohorts before benchmarking feedback")
    core = list(contract["lock"]["pricing"]["cohort_skus"])
    expanded = pd.read_csv(step4 / "expanded_cohort.csv")["sku"].tolist()
    ref_path = verified(step4 / "expanded" / "test_keys.csv", previous["expanded"]["files"]["test_keys.csv"])
    if set(expanded) != set(pd.read_csv(ref_path)["sku"]):
        raise ValueError("Expanded SKU list differs from audited Step 4 keys")
    if len(set(core + expanded)) != 60:
        raise ValueError("Expected 10 original and 50 distinct expanded SKU–store series")
    model_run = read_json(step3 / "run.json")
    if (model_run["selected"] != "arimax_price"
            or model_run["identity"]["phase1_sha256"] != contract["phase1_sha256"]
            or model_run["identity"]["fit_scope"] != "train_only"):
        raise ValueError("Unexpected Step 3 model provenance")
    parts, baseline = [], {}
    for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
        used = set(core + expanded)
        parts.append(weekly.loc[weekly["sku"].isin(used)])
        meta = model_run["models_by_store"][store]
        path = verified(step3 / meta["models"], meta["models_sha256"])
        params = pd.read_csv(path, keep_default_na=False)
        params = params.loc[params["sku"].isin(used) & params["model"].eq("arimax_price") & params["status"].eq("ok")]
        baseline.update({row.sku: {**json.loads(row.parameters), "order": 1, "transform": "level"}
                         for row in params.itertuples(index=False)})
    if set(baseline) != set(core + expanded):
        raise ValueError("An original ARIMAX model is missing from the matched cohort")
    return pd.concat(parts, ignore_index=True), baseline, {"core": core, "expanded": expanded}


def validation_fixed_prices(inputs, available, n_grid):
    """Select one price per SKU by deterministic validation simulator expectation."""
    rows = []
    for sku, episode in inputs.groupby("sku", sort=True):
        first = episode.iloc[0]
        surface = next(iter(available.values())).surfaces[sku]
        grid = np.linspace(surface.price_min, surface.price_max, n_grid)
        scores = []
        for price in grid:
            total = 0.0
            for simulator in available.values():
                prior_qty, prior_price = float(first["qty_lag_1"]), float(first["price_lag_1"])
                for row in episode.itertuples(index=False):
                    context = {"event_days": row.event_days, "snap_days": row.snap_days,
                               "qty_lag_1": prior_qty, "price_lag_1": prior_price}
                    quantity = simulator.predict_mean(sku, float(price), row.week, context)
                    total += float(price) * quantity
                    prior_qty, prior_price = quantity, float(price)
            scores.append(total / len(available))
        index = int(np.argmax(scores))
        rows.append({"sku": sku, "price": float(grid[index]), "validation_mean_expected_revenue": scores[index]})
    return pd.DataFrame(rows)


class LoggedEnvironment(_TestEnvironment):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["coefficient_updates"] = int(self.diagnostic.counts.get(info["sku"], 0)
                                          if self.forecaster is None else
                                          self.forecaster.counts.get(info["sku"], 0))
        return obs, reward, done, truncated, info


def evaluate(name, seed, inputs, available, states, fixed, n_grid, policy=None, coefficients=None):
    frames = []
    for sim, simulator in available.items():
        demand = ArimaxAdapter(states, coefficients)
        feedback = name == PPO_FEEDBACK
        env = LoggedEnvironment(inputs, simulator, demand, demand if policy is not None else None,
                                gradient=policy is not None, feedback=feedback)
        if policy is not None:
            action = lambda obs, env: policy.predict(obs, deterministic=True)[0]
        elif name == FIXED_SKU:
            action = lambda obs, env: env.action_for_price(fixed[env.sku])
        elif name == FIXED_MAX:
            action = lambda obs, env: np.array([1.0], dtype=np.float32)
        else:
            action = _grid_action(demand.predict_prices, n_grid)
        frame, _ = evaluate_policy(env, name, action, seed)
        frame["train_price_min"] = frame["sku"].map({sku: s.price_min for sku, s in simulator.surfaces.items()})
        frame["train_price_max"] = frame["sku"].map({sku: s.price_max for sku, s in simulator.surfaces.items()})
        frame["at_upper_bound"] = np.isclose(frame["chosen_price"], frame["train_price_max"], rtol=0, atol=1e-8)
        frame["at_lower_bound"] = np.isclose(frame["chosen_price"], frame["train_price_min"], rtol=0, atol=1e-8)
        frames.append(frame.assign(simulator=sim))
    return pd.concat(frames, ignore_index=True)


def forecast_quality(states, validation):
    adapter = ArimaxAdapter(states)
    predictions = np.array([adapter.predict_q(row, row["price"]) for row in validation])
    errors = predictions - np.array([row["qty"] for row in validation])
    return float(np.sqrt(np.mean(errors**2))), float(np.mean(np.abs(errors)))


def _candidate_name(order, transform):
    return f"ar{order}_{transform}"


def tune(output, weekly, baseline, groups, val_inputs, val_start, available, config, seeds):
    rows = []
    fixed_frame = validation_fixed_prices(val_inputs, available, config["grid_points"])
    _write_csv(output / "validation_fixed_prices.csv", fixed_frame)
    fixed = fixed_frame.set_index("sku")["price"].to_dict()
    scopes = {sku: group for group, skus in groups.items() for sku in skus}
    normalizer = ["cohort", "simulator", "seed"]
    fixed_pass = pd.concat([evaluate(FIXED_SKU, seed, val_inputs, available, baseline, fixed, config["grid_points"])
                            for seed in seeds], ignore_index=True)
    fixed_pass["cohort"] = fixed_pass["sku"].map(scopes)
    fixed_total = fixed_pass.groupby(normalizer)["simulated_revenue"].sum()

    train = weekly.loc[weekly["split"].eq("train")]
    train_by_sku = {sku: (group["qty"].where(_observed(group)).to_numpy(dtype=float),
                          group["price"].to_numpy(dtype=float))
                    for sku, group in train.groupby("sku", sort=True)}
    reference = windows_from_weekly(weekly.loc[weekly["split"].ne("test")])
    reference = reference.loc[reference["split"].eq("validation") & reference["week"].ge(val_start) &
                              _observed(reference)]
    supported = reference["sku"].map({sku: state["price_min"] for sku, state in baseline.items()})
    ceiling = reference["sku"].map({sku: state["price_max"] for sku, state in baseline.items()})
    validation = reference.loc[reference["price"].between(supported - 1e-7, ceiling + 1e-7)].to_dict("records")
    if not validation:
        raise ValueError("No validation labels at prices in train support")
    candidates, per_seed = {}, []
    for order, transform in itertools.product(ORDERS, TRANSFORMS):
        label = _candidate_name(order, transform)
        states = baseline if (order, transform) == (1, "level") else {
            sku: fit_model(qty, price, order, transform) for sku, (qty, price) in train_by_sku.items()
        }
        fitted = sum(state is not None for state in states.values())
        if fitted != len(baseline):
            rows.append({"variant": label, "order": order, "transform": transform,
                         "native_series": fitted, "eligible": False})
            print(f"[ARIMAX tune] {label}: {fitted}/{len(baseline)} fits", flush=True)
            continue
        candidates[label] = states
        rmse, mae = forecast_quality(states, validation)
        trials = pd.concat([evaluate("grid", seed, val_inputs, available, states, fixed, config["grid_points"])
                            for seed in seeds], ignore_index=True)
        trials["cohort"] = trials["sku"].map(scopes)
        metric = trials.groupby(normalizer)["simulated_revenue"].sum()
        paired = (100.0 * (metric - fixed_total) / fixed_total).rename("uplift_pct")
        per_seed.append(paired.reset_index().assign(variant=label))
        rows.append({"variant": label, "order": order, "transform": transform, "native_series": fitted,
                     "eligible": True, "validation_n": len(validation), "validation_rmse": rmse,
                     "validation_mae": mae, "validation_revenue_vs_fixed_pct": float(paired.mean())})
        print(f"[ARIMAX tune] {label}: RMSE {rmse:.3f}, validation revenue uplift {paired.mean():.3f}%",
              flush=True)
    result = pd.DataFrame(rows)
    ranked = result.loc[result["eligible"]].sort_values(
        ["validation_revenue_vs_fixed_pct", "validation_rmse", "order", "transform"],
        ascending=[False, True, True, True])
    winner = str(ranked.iloc[0]["variant"])
    _write_csv(output / "validation_trials.csv", result)
    _write_csv(output / "validation_paired.csv", pd.concat(per_seed, ignore_index=True))
    _atomic_json(output / "tuned_states.json", candidates[winner])
    _atomic_json(output / "selection.json", {"selected": winner, "validation_seeds": list(seeds),
                "rule": "highest mean paired validation simulator revenue vs SKU-specific fixed price, tie by RMSE",
                "state_sha256": _sha256(output / "tuned_states.json"),
                "fixed_sha256": _sha256(output / "validation_fixed_prices.csv")})
    return candidates[winner], fixed, winner


def ppo_pair(output, history, simulator, states, seed, budget, name, identity):
    feedback = name == PPO_FEEDBACK

    def build(stage):
        estimator = ArimaxAdapter(states)
        policy, stats = _ppo(history, simulator, estimator, seed, budget, True, feedback, stage / "policy")
        callbacks = int(policy.get_env().get_attr("feedback_updates")[0])
        if policy.num_timesteps != budget or callbacks != (budget // 10 if feedback else 0):
            raise ValueError("Unexpected PPO training interactions")
        _atomic_json(stage / "feedback_state.json", {"coefficients": estimator.coefficients,
                     "counts": estimator.counts, "policy_sha256": _sha256(stage / "policy.zip"),
                     "seed": seed, "method": name})
        return {"training_budget": budget, "feedback_callback_count": callbacks,
                "coefficient_update_count": sum(estimator.counts.values()), **stats}

    _unit(output, {**identity, "method": name, "seed": seed, "budget": budget}, build)
    meta = read_json(output / "feedback_state.json")
    return PPO.load(verified(output / "policy.zip", meta["policy_sha256"]), device="cpu"), meta["coefficients"]


def summarize(frame, output):
    cols = ["simulator", "seed", "policy"]
    groups = frame.groupby(cols)
    runs = groups.agg(revenue=("simulated_revenue", "sum"), regret=("expected_regret", "sum"),
                       upper=("at_upper_bound", "mean"), lower=("at_lower_bound", "mean"))
    per_sku = frame.groupby([*cols, "sku"]).agg(
        price_span=("chosen_price", lambda s: s.max() - s.min()),
        updates=("coefficient_updates", "max"))
    runs["coefficient_updates"] = per_sku["updates"].groupby(level=cols).sum()
    runs["share_moving"] = per_sku["price_span"].gt(0.01).groupby(level=cols).mean()
    runs["boundary_pct"] = (runs["upper"] + runs["lower"]) * 100
    fixed = runs["revenue"].xs(FIXED_SKU, level="policy").rename("fixed_revenue")
    runs = runs.reset_index().merge(fixed, on=["simulator", "seed"], validate="many_to_one")
    runs["delta_vs_fixed"] = runs["revenue"] - runs["fixed_revenue"]
    runs["uplift_vs_fixed_pct"] = 100 * runs["delta_vs_fixed"] / runs["fixed_revenue"]
    scores = runs.groupby(["simulator", "policy"], as_index=False).agg(
        mean_revenue=("revenue", "mean"), std_revenue=("revenue", "std"), mean_regret=("regret", "mean"),
        paired_uplift_vs_fixed_pct=("uplift_vs_fixed_pct", "mean"),
        seeds_beating_fixed=("delta_vs_fixed", lambda s: int(s.gt(.01).sum())),
        boundary_pct=("boundary_pct", "mean"), coefficient_updates=("coefficient_updates", "sum"),
        sku_seed_with_move_pct=("share_moving", lambda s: 100 * s.mean()))
    _write_csv(output / "runs.csv", runs)
    _write_csv(output / "summary.csv", scores.sort_values(["simulator", "mean_revenue"], ascending=[True, False]))
    chosen_pairs = [(GRID_TUNED, GRID_BASE), (PPO_FIXED, GRID_TUNED), (PPO_FEEDBACK, PPO_FIXED),
                    (PPO_FEEDBACK, GRID_TUNED)]
    comparisons = []
    for sim, subset in runs.groupby("simulator"):
        revenue = subset.pivot(index="seed", columns="policy", values="revenue")
        for candidate, base in chosen_pairs:
            delta = revenue[candidate] - revenue[base]
            comparisons.append({"simulator": sim, "candidate": candidate, "reference": base,
                                "mean_revenue_delta": float(delta.mean()),
                                "mean_paired_uplift_pct": float((100 * delta / revenue[base]).mean()),
                                "seed_wins": int(delta.gt(0.01).sum())})
    _write_csv(output / "paired.csv", pd.DataFrame(comparisons))


def test_scope(output, inputs, available, baseline, tuned, fixed, pairs, config, seeds, identity):
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "test_keys.csv", inputs[KEYS])
    blocks = []
    for seed in seeds:
        for name in METHODS:
            policy, coefficients = pairs.get((name, seed), (None, None))
            states = baseline if name == GRID_BASE else tuned

            def build(stage):
                frame = evaluate(name, seed, inputs, available, states, fixed, config["grid_points"],
                                 policy=policy, coefficients=coefficients)
                _write_csv(stage / "decisions.csv", frame)
                return {"decisions": len(frame)}

            episode = output / "episodes" / str(seed) / name
            _unit(episode, {**identity, "seed": seed, "method": name, "cohort": output.name}, build)
            blocks.append(pd.read_csv(episode / "decisions.csv", dtype={"sku": str, "week": str}))
            print(f"[ARIMAX test {output.name}] {seed} {name}", flush=True)
    decisions = pd.concat(blocks, ignore_index=True)
    alias = {PPO_FEEDBACK: "ppo_feedback_gradient"}
    audit_decisions(decisions.assign(policy=decisions["policy"].replace(alias)), inputs,
                    [alias.get(name, name) for name in METHODS], seeds, list(available))
    fixed_rows = decisions.loc[decisions["policy"].eq(FIXED_SKU)]
    if not np.allclose(fixed_rows["chosen_price"], fixed_rows["sku"].map(fixed), rtol=0, atol=1e-6):
        raise ValueError("SKU fixed-price baseline deviated from the validation choice")
    feedback = decisions["policy"].eq(PPO_FEEDBACK)
    if (decisions.loc[~feedback, "coefficient_updates"].ne(0).any()
            or (decisions["coefficient_updates"] > decisions["forecast_update_count"]).any()):
        raise ValueError("Unexpected ARIMAX update during the test episode")
    _write_csv(output / "decisions.csv", decisions)
    summarize(decisions, output)
    return {"n_sku": inputs["sku"].nunique(), "n_keys": len(inputs), "decisions": len(decisions),
            "files": {file.name: _sha256(file) for file in output.glob("*.csv")}}


def run(root, cache, output, step3, step4):
    started = perf_counter()
    torch.set_num_threads(1)
    contract = load_contract(root, LOCK_PATH)
    root = contract["root"]
    lock = contract["lock"]
    config = lock["pricing"]
    seeds = lock["phase5_seeds"]
    spec = contract["protocol"]["datasets"]["m5"]
    weeks = contract["protocol"]["pricing"]["test_weeks_per_series"]
    val_start = pd.Timestamp(spec["validation_end"]) - pd.Timedelta(weeks=weeks - 1)
    test_start = pd.Timestamp(spec["test_start"]) + pd.Timedelta(weeks=weeks)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": _sha256(Path(__file__)), "phase1_sha256": contract["phase1_sha256"],
                "lock_sha256": contract["lock_sha256"], "step3_sha256": _sha256(step3 / "run.json"),
                "step4_sha256": _sha256(step4 / "run.json"),
                "expanded_cohort_sha256": _sha256(step4 / "expanded_cohort.csv"),
                "torch": str(torch.__version__), "stable_baselines3": stable_baselines3.__version__,
                "statsmodels": statsmodels.__version__, "validation_start": str(val_start.date()),
                "test_start": str(test_start.date()), "weeks": weeks, "feedback_ridge": RIDGE,
                "minimum_feedback_samples": MIN_FEEDBACK}
    marker = output / "identity.json"
    if marker.is_file() and read_json(marker) != identity:
        raise ValueError("Use a new output directory for different artifacts or runtime")
    _atomic_json(marker, identity)
    weekly, baseline, groups = load_weekly_and_states(root, cache, contract, step3, step4)
    train = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    validate = simulation_inputs(weekly, groups["core"] + groups["expanded"], val_start, weeks, "validation")
    selection_path = output / "selection.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        tuned = read_json(verified(output / "tuned_states.json", selection["state_sha256"]))
        prices = pd.read_csv(verified(output / "validation_fixed_prices.csv", selection["fixed_sha256"]))
        fixed = prices.set_index("sku")["price"].to_dict()
        winner = selection["selected"]
    else:
        tuned, fixed, winner = tune(output, weekly, baseline, groups, validate, val_start,
                                     samplers(train, config), config, seeds[:3])
    print(f"[ARIMAX selection] {winner}; testing {test_start.date()} onward", flush=True)

    core_train = train.loc[train["sku"].isin(groups["core"])]
    core_samplers = samplers(core_train, config)
    history = windows_from_weekly(weekly.loc[weekly["sku"].isin(groups["core"]) & weekly["split"].eq("train")])
    history = history.loc[_observed(history)]
    policy_identity = {**identity, "selected_state_sha256": _sha256(output / "tuned_states.json"),
                       "fixed_price_sha256": _sha256(output / "validation_fixed_prices.csv")}
    budget = config["training_timesteps_per_policy"]
    pairs = {}
    for seed in seeds:
        for name in (PPO_FIXED, PPO_FEEDBACK):
            pairs[name, seed] = ppo_pair(output / "checkpoints" / str(seed) / name, history,
                                         core_samplers["gaussian_song"], tuned, seed, budget, name, policy_identity)
            print(f"[ARIMAX PPO] {seed} {name}", flush=True)
    scopes = {}
    for group, skus in groups.items():
        panel = train.loc[train["sku"].isin(skus)]
        available = core_samplers if group == "core" else samplers(panel, config)
        inputs = simulation_inputs(weekly, skus, test_start, weeks, "test")
        scopes[group] = test_scope(output / group, inputs, available, baseline, tuned, fixed, pairs,
                                   config, seeds, policy_identity)
    _atomic_json(output / "run.json", {"status": "complete", "configuration": winner, "methods": list(METHODS),
                 "seeds": seeds, "test_end": str((test_start + pd.Timedelta(weeks=weeks-1)).date()),
                 "ppo_train_timesteps": budget, "scopes": scopes, "identity": policy_identity,
                 "seconds": perf_counter()-started})
    print(f"[ARIMAX complete] {sum(x['decisions'] for x in scopes.values())} audited decisions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--step3-dir", type=Path, default=STEP3)
    parser.add_argument("--step4-dir", type=Path, default=STEP4)
    args = parser.parse_args()
    run(args.phase1_root, args.cache_dir, args.output_dir, args.step3_dir, args.step4_dir)


if __name__ == "__main__":
    main()
