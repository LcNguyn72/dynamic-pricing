"""Run reproducible Phase 5 pricing on the locked M5 calendar test cohort."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import DDPG, PPO, SAC
from torch.nn import functional as F
from xgboost import XGBRegressor

from .data import make_windows
from .forecast_benchmark import KNOWN_CALENDAR_COVARIATES, _tabular_features
from .phase1_finalize import PROJECT_ROOT, _observed
from .phase2_song import M5_COLUMNS, PHASE1_ROOT, _read_json, _verify, _write_json
from .phase3_song_pricing import (
    CALENDAR, FEEDBACK, FEEDBACK_TORCH_SEED_OFFSET, KEYS, PPO_HYPERPARAMETERS,
    PPO_SETTINGS, TRAIN_HISTORY, FeedbackForecaster,
)
from .phase4_forecast import calendar_features, checked_protocol
from .phase4_pricing import COMPARISONS, XGBDemand, grid_price
from .pricing_benchmark import PricingEnvironment, evaluate_policy
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator, _week_number
from .song_forecast import CURRENT_FEATURES, HISTORY_FEATURES, LOOKBACK, SongTransformer, _scalers, _season, _sha256, feature_arrays
from .song_pricing_pilot import _offpolicy, _ppo

OUTPUT_ROOT = PHASE1_ROOT.with_name("phase5_v2") / "pricing"
LOCK_PATH = PROJECT_ROOT / "experiments" / "phase5_settings_v2.json"
FIXED_MAX = "fixed_max_observed_train_price"
ALGORITHMS = {"ddpg": DDPG, "sac": SAC, **{name: PPO for name in PPO_SETTINGS}}


def test_inputs(weekly: pd.DataFrame, skus: list[str], start: pd.Timestamp, weeks: int) -> pd.DataFrame:
    """Expose only the observed pre-test starting history and known calendar."""
    if weekly.duplicated(KEYS).any():
        raise ValueError("Duplicate SKU-week in pricing source")
    dates = pd.date_range(start, periods=weeks, freq="7D")
    past = pd.date_range(start - pd.Timedelta(weeks=LOOKBACK), periods=LOOKBACK, freq="7D")
    parts = []
    for sku in sorted(skus):
        group = weekly.loc[weekly["sku"].eq(sku)].set_index("week")
        full = group.reindex(past.append(dates))
        actual = full.loc[dates]
        if not actual["split"].eq("test").all() or not _observed(actual).all():
            raise ValueError(f"{sku}: missing priced week in locked test calendar")
        lookback = full.loc[past]
        actual_history = lookback[["qty", "price"]].to_numpy(dtype=float)
        if (not lookback["split"].isin(("train", "validation")).all()
                or not np.isfinite(actual_history).all()
                or (actual_history[:, 0] < 0).any() or (actual_history[:, 1] <= 0).any()):
            raise ValueError(f"{sku}: invalid 12-week history before locked test calendar")
        calendar = full[["event_days", "snap_days"]].copy()
        if not np.isfinite(calendar.to_numpy()).all():
            raise ValueError(f"{sku}: missing event or SNAP context")
        for lag in range(1, LOOKBACK + 1):
            for column in ("event_days", "snap_days"):
                calendar[f"{column}_lag_{lag}"] = calendar[column].shift(lag)
        out = calendar.loc[dates].reset_index(names="week")
        out.insert(0, "sku", sku)
        out["split"] = "test"
        out["observed_sales"] = True
        for lag in range(1, LOOKBACK + 1):
            for index, field in enumerate(("qty", "price")):
                out[f"{field}_lag_{lag}"] = np.nan
                out.loc[0, f"{field}_lag_{lag}"] = actual_history[-lag, index]
        parts.append(out[[*KEYS, "split", "observed_sales", *CALENDAR, *TRAIN_HISTORY]])
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=512)
def _date_features(week: pd.Timestamp) -> np.ndarray:
    dates = pd.Series(pd.date_range(end=week, periods=LOOKBACK + 1, freq="7D"))
    return np.column_stack(_season(dates))


class FastFeedbackForecaster(FeedbackForecaster):
    """Use the same features as Song while caching the fixed calendar transform."""

    def _arrays(self, row, price):
        scale = self.scalers[str(row["sku"])]
        qm, qs, pm, ps = (np.float32(scale[key]) for key in ("qty_mean", "qty_std", "price_mean", "price_std"))
        dates = _date_features(pd.Timestamp(row["week"]))
        lags = range(LOOKBACK, 0, -1)
        q = np.asarray([row[f"qty_lag_{lag}"] for lag in lags], dtype=np.float32)
        p = np.asarray([row[f"price_lag_{lag}"] for lag in lags], dtype=np.float32)
        h = np.empty((1, LOOKBACK, len(HISTORY_FEATURES)), dtype=np.float32)
        c = np.empty((1, len(CURRENT_FEATURES)), dtype=np.float32)
        h[0, :, 0] = (q - qm) / qs
        h[0, :, 1] = (p - pm) / ps
        h[0, :, 2:6] = dates[:-1]
        for i, column in enumerate(("event_days", "snap_days", "promotion_flag")):
            divisor = 7.0 if i < 2 else 1.0
            h[0, :, 6 + i] = np.asarray([row.get(f"{column}_lag_{lag}", 0) for lag in lags],
                                        dtype=np.float32) / divisor
            c[0, 5 + i] = np.float32(row.get(column, 0)) / divisor
        c[0, 0] = (np.float32(price) - pm) / ps
        c[0, 1:5] = dates[-1]
        return h, c

    def _tensor(self, row, price):
        h, c = self._arrays(row, price)
        return torch.from_numpy(h), torch.from_numpy(c).requires_grad_(True)

    def predict_prices(self, row, prices: np.ndarray) -> np.ndarray:
        h, c = self._arrays(row, float(prices[0]))
        scale = self.scalers[str(row["sku"])]
        current = np.repeat(c, len(prices), axis=0)
        current[:, 0] = (prices.astype(np.float32) - np.float32(scale["price_mean"])) / np.float32(scale["price_std"])
        model = self.model
        with torch.no_grad():
            encoded = model.encoder(model.embedding(torch.from_numpy(h)) + model.position)[:, -1, :]
            context = model.norm(encoded).expand(len(prices), -1)
            z = model.head(torch.cat((context, torch.from_numpy(current)), dim=1)).squeeze(-1).numpy()
        return np.maximum(0, z * scale["qty_std"] + scale["qty_mean"])


class _OracleGrid:
    @lru_cache(maxsize=5000)
    def _grid(self, sku, week, event, snap, with_context, grid_size):
        surface = self.surfaces[sku]
        prices = np.linspace(surface.price_min, surface.price_max, grid_size)
        calendar = {"event_days": event, "snap_days": snap} if with_context else None
        base = np.array([KernelDemandSimulator.predict_mean(self, sku, price, week, calendar) for price in prices])
        return prices, base

    def _full_grid(self, sku, week, context, grid_size):
        values = context or {}
        return self._grid(sku, _week_number(week), float(values.get("event_days", 0)),
                          float(values.get("snap_days", 0)), context is not None, grid_size)


class CachedGaussian(_OracleGrid, KernelDemandSimulator):
    def oracle_revenue(self, sku, week, current_price=None, max_change=.10, grid_size=100, context=None):
        if current_price is not None:
            return super().oracle_revenue(sku, week, current_price, max_change, grid_size, context)
        prices, base = self._full_grid(sku, week, context, grid_size)
        return float(np.max(prices * base))


class CachedLag(_OracleGrid, LagAdjustedKernelSimulator):
    def oracle_revenue(self, sku, week, current_price=None, max_change=.10, grid_size=100, context=None):
        if current_price is not None:
            return super().oracle_revenue(sku, week, current_price, max_change, grid_size, context)
        prices, base = self._full_grid(sku, week, context, grid_size)
        if self.recent_weight and context is not None:
            last_qty, last_price = float(context.get("qty_lag_1", np.nan)), float(context.get("price_lag_1", np.nan))
            if np.isfinite(last_qty) and last_qty >= 0 and np.isfinite(last_price) and last_price > 0:
                ref = KernelDemandSimulator.predict_mean(self, sku, last_price, week, context)
                if ref > 1e-8:
                    anchored = last_qty * np.clip(base / ref, 1 / self.ratio_limit, self.ratio_limit)
                    base = (1 - self.recent_weight) * base + self.recent_weight * anchored
        return float(np.max(prices * base))


def _contract(root: Path, lock_path: Path):
    phase1, protocol = checked_protocol(root)
    lock = _read_json(lock_path)
    config, spec = lock["pricing"], protocol["pricing"]
    cohort = config["cohort_skus"]
    demand_skus = config["pricing_demand_training_skus"]
    fit = lock["forecast"]["m5"]["settings_by_model"]
    if (lock["protocol_sha256"] != phase1["protocol_sha256"] or lock["protocol_id"] != protocol["protocol_id"]
            or lock["selection_split"] != "validation"
            or lock["test_pricing_forecaster_fit_scope"] != "train_only_validation_selects_epoch"
            or cohort != spec["cohort_skus"]
            or len(cohort) != spec["cohort_store_count"] or not set(cohort).issubset(demand_skus)
            or len(set(demand_skus)) != len(demand_skus) or len(demand_skus) != 120
            or config["phase5_methods"] != list(COMPARISONS) or config["additional_phase5_control"] != FIXED_MAX
            or config["primary_song_simulator"] != "gaussian_song"
            or config["test_decisions_per_seed"] != spec["evaluation_cycles"]
            or config["feedback_config"] != FEEDBACK or config["feedback_torch_seed_offset"] != FEEDBACK_TORCH_SEED_OFFSET
            or config["ppo_hyperparameters"] != PPO_HYPERPARAMETERS
            or lock["phase5_seeds"] != spec["seeds"]
            or config["training_timesteps_per_policy"] <= 0
            or config["training_timesteps_per_policy"] % PPO_HYPERPARAMETERS["n_steps"]
            or [fit["song_transformer"][key] for key in ("layers", "heads", "width")] != [4, 8, 128]):
        raise ValueError("Pricing runner does not match the frozen Phase 5 settings")
    frozen = lock["frozen_test_keys"]["m5"]
    if (frozen["count"] != protocol["datasets"]["m5"]["primary_test_key_count"]
            or frozen["sha256"] != protocol["datasets"]["m5"]["primary_test_key_sha256"]):
        raise ValueError("Phase 5 M5 keys differ from V2")
    _verify(root / "test_keys_m5.csv", frozen["sha256"])
    return phase1, protocol, lock


def _data(root: Path, phase1: dict, protocol: dict, lock: dict):
    sku_order = lock["pricing"]["cohort_skus"]
    demand_skus = set(lock["pricing"]["pricing_demand_training_skus"])
    phase1_m5 = phase1["m5"]
    cohort_path = root / "m5_pricing_cohort.csv"
    _verify(cohort_path, phase1_m5["pricing_cohort_sha256"])
    cohort = pd.read_csv(cohort_path, dtype={"sku": str})
    cfg, spec = protocol["datasets"]["m5"], protocol["pricing"]
    start, weeks = pd.Timestamp(cfg["test_start"]), spec["test_weeks_per_series"]
    end = start + pd.Timedelta(weeks=weeks - 1)
    if (list(cohort["sku"]) != sku_order or list(cohort["store"]) != cfg["store_ids"]
            or not cohort["test_decisions"].eq(weeks).all()
            or not cohort["first_test_week"].eq(start.strftime("%Y-%m-%d")).all()
            or not cohort["last_test_week"].eq(end.strftime("%Y-%m-%d")).all()):
        raise ValueError("Phase 1 pricing cohort differs from the locked calendar")
    test_keys = pd.read_csv(root / "test_keys_m5.csv", usecols=KEYS, dtype={"sku": str}, parse_dates=["week"])
    expected_keys = test_keys.loc[test_keys["sku"].isin(sku_order) & test_keys["week"].between(start, end)]
    del test_keys
    reports = {row["store"]: row for row in phase1_m5["stores"]}
    train_frames, cohort_frames, scalers = [], [], {}
    for store in cfg["store_ids"]:
        source = reports[store]
        panel = root / "m5" / f"weekly_{store}.csv"
        manifest = root / "m5" / f"manifest_{store}.json"
        _verify(panel, source["weekly_sha256"])
        _verify(manifest, source["manifest_sha256"])
        names = {sku for sku in demand_skus if sku.endswith("@" + store)}
        store_scalers = _read_json(manifest)["scalers"]
        scalers.update({sku: store_scalers[sku] for sku in names})
        train_parts, cohort_parts = [], []
        cohort_sku = cohort.loc[cohort["store"].eq(store), "sku"].iloc[0]
        for chunk in pd.read_csv(panel, dtype={"sku": str}, parse_dates=["week"],
                                 usecols=M5_COLUMNS, chunksize=100_000):
            train = chunk["sku"].isin(names) & chunk["split"].eq("train")
            train_parts.append(chunk.loc[train])
            cohort_parts.append(chunk.loc[chunk["sku"].eq(cohort_sku) & chunk["week"].le(end)])
        train_frames.append(pd.concat(train_parts, ignore_index=True))
        cohort_frames.append(pd.concat(cohort_parts, ignore_index=True))
    weekly = pd.concat(cohort_frames, ignore_index=True)
    if set(scalers) != demand_skus or any(
        not frame["week"].le(pd.Timestamp(cfg["train_end"])).all() or frame.duplicated(KEYS).any()
        for frame in train_frames
    ):
        raise ValueError("Invalid frozen train-only demand cohort")
    inputs = test_inputs(weekly, sku_order, start, weeks)
    if (len(inputs) != lock["pricing"]["test_decisions_per_seed"]
            or not inputs[KEYS].equals(expected_keys.sort_values(KEYS).reset_index(drop=True))):
        raise ValueError("Pricing decisions differ from Phase 1's frozen test keys")
    pricing_train = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    windows = []
    for part in train_frames:
        frame = make_windows(part, LOOKBACK)
        windows.append(calendar_features(frame.loc[_observed(frame)]).reset_index(drop=True))
    if {sku for frame in windows for sku in frame["sku"]} != demand_skus:
        raise ValueError("A frozen demand training SKU lacks forecast windows")
    history = pd.concat([part.loc[part["sku"].isin(sku_order)] for part in windows], ignore_index=True)
    return pricing_train, inputs, windows, scalers, history


def _training_blocks(windows, scalers):
    blocks = []
    for part in windows:
        h, c = feature_arrays(part, scalers)
        mean, std, _, _ = _scalers(part, scalers)
        target = (part["qty"].to_numpy(dtype=np.float32) - mean) / std
        blocks.append((h, c, target))
    return blocks


def _train_demand(out: Path, seed: int, windows, blocks, scalers, settings) -> dict:
    start = perf_counter()
    song = settings["song_transformer"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SongTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=song["learning_rate"])
    n_rows = sum(len(block[2]) for block in blocks)
    batch = song["batch_size"]
    losses = []
    for epoch in range(1, song["best_epoch"] + 1):
        model.train()
        squared_error = 0.0
        for shard in rng.permutation(len(blocks)):
            h, c, y = blocks[shard]
            indexes = rng.permutation(len(y))
            for start_row in range(0, len(y), batch):
                ix = indexes[start_row : start_row + batch]
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(torch.from_numpy(h[ix]), torch.from_numpy(c[ix])), torch.from_numpy(y[ix]))
                loss.backward()
                optimizer.step()
                squared_error += float(loss.detach()) * len(ix)
        mse = squared_error / n_rows
        if not np.isfinite(mse):
            raise ValueError(f"Non-finite Song training MSE at epoch {epoch}")
        losses.append(mse)
        print(f"[pricing] seed {seed} Song epoch {epoch}: train MSE(z) {mse:.6f}", flush=True)
    torch.save({
        "state_dict": model.state_dict(), "scalers": scalers, "seed": seed, "dataset": "m5",
        "fit_scope": "train_only", "history_features": HISTORY_FEATURES,
        "current_features": CURRENT_FEATURES, "lookback": LOOKBACK, "parameters": song,
    }, out / "song.pt")
    train = pd.concat(windows, ignore_index=True)
    if set(train["split"]) != {"train"}:
        raise ValueError("Pricing forecasters must learn only from train")
    skus = sorted(scalers)
    covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in train.columns)
    features = _tabular_features(train, skus, scalers, LOOKBACK, True, covariates)
    params = dict(settings["xgboost_price"]["parameters"], random_state=seed)
    xgb = XGBRegressor(**params).fit(features, np.log1p(train["qty"].to_numpy(dtype=np.float32)))
    xgb.save_model(out / "xgboost_price.json")
    _write_json(out / "xgboost_price_meta.json", {
        "parameters": params, "calendar_covariates": list(covariates),
        "with_price": True, "skus": skus, "scalers": scalers, "fit_scope": "train_only", "seed": seed,
    })
    return {"train_rows": n_rows, "train_skus": len(skus), "fit_scope": "train_only",
            "song_epoch_mse_z": losses, "seconds": perf_counter() - start}


def _load_song(saved: dict, sku_order: list[str], seed: int) -> FastFeedbackForecaster:
    if (saved["history_features"] != HISTORY_FEATURES or saved["current_features"] != CURRENT_FEATURES
            or saved["lookback"] != LOOKBACK or saved["seed"] != seed):
        raise ValueError("Wrong Song checkpoint layout or seed")
    model = SongTransformer()
    model.load_state_dict(saved["state_dict"])
    return FastFeedbackForecaster(model, {sku: saved["scalers"][sku] for sku in sku_order},
                                  seed=seed, feedback_rng_state=saved.get("feedback_rng_state"))


class _TestEnvironment(PricingEnvironment):
    def __init__(self, inputs, simulator, diagnostic, forecaster, *, gradient, feedback):
        super().__init__(inputs, simulator, "test", forecaster, with_gradient=gradient,
                         online_forecast_update=feedback, action_mode="absolute",
                         include_sku_one_hot=False, **FEEDBACK)
        self.diagnostic = diagnostic

    def step(self, action):
        agent = self.forecaster if self.forecaster is not None else self.diagnostic
        price = self.price_from_action(float(np.asarray(action).reshape(-1)[0]))
        prediction = agent.predict_q(self._current_row(), price)
        obs, reward, ended, truncated, info = super().step(action)
        info["predicted_qty"] = prediction
        return obs, reward, ended, truncated, info


def _fixed_max_action(_, env):
    return np.array([1.0], dtype=np.float32)


def _grid_action(predict, grid_points):
    def choose(_, env):
        row = env._current_row()
        surface = env.simulator.surfaces[env.sku]
        prices = np.unique(np.append(np.linspace(surface.price_min, surface.price_max, grid_points),
                                     np.clip(env.current_price, surface.price_min, surface.price_max)))
        chosen = grid_price(prices, predict(row, prices), env.current_price)
        return env.action_for_price(chosen)
    return choose


def _evaluate_test(name, seed, model, simulator, inputs, diagnostic, forecaster, *, grid_points):
    gradient, feedback = PPO_SETTINGS.get(name, (False, False))
    env = _TestEnvironment(inputs, simulator, diagnostic, forecaster, gradient=gradient, feedback=feedback)
    action = None
    if model is not None:
        def action(observation, _):
            return model.predict(observation, deterministic=True)[0]
    elif name == FIXED_MAX:
        action = _fixed_max_action
    elif name.startswith("grid_"):
        action = _grid_action(diagnostic.predict_prices, grid_points)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        rows, summary = evaluate_policy(env, name, action, seed)
    expected = inputs[KEYS].copy()
    expected["week"] = expected["week"].dt.strftime("%Y-%m-%d")
    if not rows[KEYS].equals(expected):
        raise ValueError(f"{name}: test decision keys differ from locked schedule")
    rows["train_price_min"] = rows["sku"].map({sku: item.price_min for sku, item in simulator.surfaces.items()})
    rows["train_price_max"] = rows["sku"].map({sku: item.price_max for sku, item in simulator.surfaces.items()})
    p = rows["chosen_price"].to_numpy()
    low = rows["train_price_min"].to_numpy()
    high = rows["train_price_max"].to_numpy()
    upper = np.isclose(p, high, rtol=0, atol=1e-10)
    lower = np.isclose(p, low, rtol=0, atol=1e-10)
    if (np.any(p < low - 1e-10) or np.any(p > high + 1e-10)
            or (name == FIXED_MAX and not upper.all())
            or not np.isfinite(rows.select_dtypes(include="number").to_numpy()).all()):
        raise ValueError(f"{name}: non-finite output or wrong price bounds")
    position = rows.groupby("sku").cumcount() + 1
    expected_updates = position // FEEDBACK["update_every"] if feedback else position * 0
    if not rows["forecast_update_count"].eq(expected_updates).all():
        raise ValueError(f"{name}: online test update schedule differs from frozen config")
    rows["at_upper_bound"] = upper
    rows["at_lower_bound"] = lower
    rows["cumulative_simulated_revenue"] = rows["simulated_revenue"].cumsum()
    summary.update({
        "upper_price_pct": float(upper.mean() * 100), "lower_price_pct": float(lower.mean() * 100),
        "boundary_pct": float((upper | lower).mean() * 100),
        "test_forecast_updates": int(rows.groupby("sku")["forecast_update_count"].max().sum()),
    })
    return rows, summary


def _atomic_json(path: Path, data: dict) -> None:
    with NamedTemporaryFile(suffix=".json", prefix=f".{path.stem}-", dir=path.parent, delete=False) as tmp:
        stage = Path(tmp.name)
    try:
        _write_json(stage, data)
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _unit(path: Path, identity: dict, build) -> dict:
    """Commit an entire demand fit or policy train/test unit after successful checks."""
    if path.exists():
        record = _read_json(path / "manifest.json")
        if record["identity"] != identity:
            raise ValueError(f"Saved {path.name} differs from the locked run")
        for name, digest in record["files"].items():
            _verify(path / name, digest)
        print(f"[pricing] reuse {path.parent.name}/{path.name}", flush=True)
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        record = build(stage)
        record["identity"] = identity
        record["files"] = {file.name: _sha256(file) for file in stage.iterdir()}
        _write_json(stage / "manifest.json", record)
        stage.rename(path)
    return record


def _policy_unit(out: Path, name: str, seed: int, samplers, inputs, history,
                 saved, demand_hash: str, diagnostic, config) -> dict:
    started = perf_counter()
    model = None
    forecaster = _load_song(saved, list(samplers["gaussian_song"].surfaces), seed) if name in PPO_SETTINGS else None
    stats = {"training_timesteps": 0, "training_forecast_updates": 0}
    if name in ALGORITHMS:
        timesteps = config["training_timesteps_per_policy"]
        gaussian = samplers["gaussian_song"]
        if forecaster is not None:
            gradient, feedback = PPO_SETTINGS[name]
            model, training = _ppo(history, gaussian, forecaster, seed, timesteps, gradient, feedback, out / "policy")
        else:
            model, training = _offpolicy(name, history, gaussian, seed, timesteps, out / "policy")
            feedback = False
        updates = int(model.get_env().get_attr("feedback_updates")[0])
        if model.num_timesteps != timesteps or updates != (timesteps // FEEDBACK["update_every"] if feedback else 0):
            raise ValueError(f"Unexpected budget or updates for {name}, seed {seed}")
        stats.update(training_timesteps=timesteps, training_forecast_updates=updates, **training)
        if forecaster is not None:
            state = forecaster.model.state_dict()
            max_change = max(float((p - saved["state_dict"][key]).abs().max()) for key, p in state.items())
            if (max_change > 0) != feedback:
                raise ValueError(f"Unexpected online weight changes for {name}, seed {seed}")
            torch.save({
                "state_dict": state, "scalers": forecaster.scalers, "seed": seed, "policy": name,
                "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES,
                "lookback": LOOKBACK, "fit_scope": "train_only_simulated_feedback",
                "source_demand_sha256": demand_hash, "ppo_sha256": _sha256(out / "policy.zip"),
                "feedback_rng_state": forecaster.feedback_rng_state,
            }, out / "transformer.pt")
            stats["transformer_max_weight_change"] = max_change
        # Score the exact persisted checkpoint, including when the unit runs uninterrupted.
        model = ALGORITHMS[name].load(out / "policy.zip", device="cpu")
        if forecaster is not None:
            pair = torch.load(out / "transformer.pt", map_location="cpu", weights_only=True)
            forecaster = _load_song(pair, list(gaussian.surfaces), seed)
    frames, summaries = [], []
    for simulator_name, simulator in samplers.items():
        frame, summary = _evaluate_test(name, seed, model, simulator, inputs, diagnostic, forecaster,
                                        grid_points=config["grid_points"])
        frames.append(frame.assign(simulator=simulator_name))
        summaries.append({**summary, "simulator": simulator_name, **stats})
    pd.concat(frames, ignore_index=True).to_csv(out / "decisions.csv", index=False)
    return {"summaries": summaries, "seconds": perf_counter() - started}


def _run_status(output: Path, seeds: list[int], policies: list[str], keys_count: int, simulators: list[str],
                identity: dict) -> dict:
    units = {seed: [name for name in policies if (output / f"seed_{seed}" / name / "manifest.json").is_file()]
             for seed in seeds}
    complete = sum(map(len, units.values()))
    progress = {"status": "complete" if complete == len(seeds) * len(policies) else "incomplete",
                "completed_units": complete, "planned_units": len(seeds) * len(policies),
                "completed_decision_rows": complete * keys_count * len(simulators),
                "planned_decision_rows": len(seeds) * len(policies) * keys_count * len(simulators),
                "policies_by_seed": {str(seed): units[seed] for seed in seeds}, **identity}
    _atomic_json(output / "pricing_progress.json", progress)
    return progress


def _aggregate(output: Path, inputs: pd.DataFrame, seeds: list[int], policies: list[str], simulator_names: list[str],
               identity: dict, config: dict, wall_seconds: float) -> dict:
    cases, files = [], []
    for seed in seeds:
        for name in policies:
            folder = output / f"seed_{seed}" / name
            record = _read_json(folder / "manifest.json")
            if not all(record["identity"].get(key) == value for key, value in identity.items()):
                raise ValueError(f"The {seed}/{name} manifest belongs to another run")
            if record["identity"]["policy"] != name or record["identity"]["seed"] != seed:
                raise ValueError("Incorrect policy checkpoint identity")
            for file, digest in record["files"].items():
                _verify(folder / file, digest)
            cases.extend(record["summaries"])
            files.append(pd.read_csv(folder / "decisions.csv", dtype={"sku": str, "week": str}))
    rows = pd.concat(files, ignore_index=True)
    key_columns = ["simulator", "policy", "seed", *KEYS]
    n_rows = len(inputs) * len(seeds) * len(policies) * len(simulator_names)
    expected = inputs[KEYS].copy()
    expected["week"] = expected["week"].dt.strftime("%Y-%m-%d")
    groups = rows.groupby(["simulator", "policy", "seed"], sort=False)
    if (len(rows) != n_rows or rows.duplicated(key_columns).any()
            or set(rows["simulator"]) != set(simulator_names) or set(rows["seed"]) != set(seeds)
            or set(rows["policy"]) != set(policies)
            or len(groups) != len(simulator_names) * len(policies) * len(seeds)):
        raise ValueError("Pricing has missing, unexpected or duplicate decisions")
    for (sim, name, seed), part in groups:
        if not part[KEYS].reset_index(drop=True).equals(expected):
            raise ValueError(f"{sim}/{seed}/{name} differs from the locked keys")
    if not np.allclose(rows["chosen_price"] * rows["simulated_units"], rows["simulated_revenue"], rtol=1e-12):
        raise ValueError("Invalid raw simulated revenue")
    positive = rows["expected_revenue"].gt(0)
    noise = (rows.loc[positive, "simulated_revenue"] / rows.loc[positive, "expected_revenue"])
    noise_group = rows.loc[positive, ["simulator", "seed", *KEYS]].assign(multiplier=noise)
    spread = noise_group.groupby(["simulator", "seed", *KEYS])["multiplier"].agg(["min", "max"])
    if not np.allclose(spread["min"], spread["max"], rtol=1e-12):
        raise ValueError("Simulator noise varies across policies on the same seed and key")
    expectation = pd.DataFrame(cases)
    keyed = ["simulator", "policy", "seed"]
    for field in ("simulated_revenue", "expected_revenue", "expected_regret"):
        calculated = rows.groupby(keyed)[field].sum().sort_index()
        known = expectation.set_index(keyed)[field].sort_index()
        if not calculated.index.equals(known.index) or not np.allclose(calculated, known, rtol=1e-12):
            raise ValueError(f"{field} summary differs from decisions")
    runs = expectation
    for control, label in (("fixed_mean", "fixed_mean"), (FIXED_MAX, "fixed_max"), ("ppo_frozen", "ppo_frozen")):
        pairs = runs[["simulator", "seed", "policy", "simulated_revenue"]]
        ref = (pairs.loc[pairs["policy"].eq(control)].drop(columns="policy")
               .rename(columns={"simulated_revenue": f"revenue_{label}"}))
        runs = runs.merge(ref, on=["simulator", "seed"], validate="many_to_one")
        difference = runs["simulated_revenue"] - runs[f"revenue_{label}"]
        runs[f"delta_vs_{label}"] = difference
        runs[f"uplift_vs_{label}_pct"] = 100 * difference / runs[f"revenue_{label}"].where(runs[f"revenue_{label}"].gt(0))
    board = runs.groupby(["simulator", "policy"], as_index=False).agg(
        seeds=("seed", "size"), mean_revenue=("simulated_revenue", "mean"),
        revenue_std=("simulated_revenue", "std"), revenue_min=("simulated_revenue", "min"),
        revenue_max=("simulated_revenue", "max"), mean_regret=("expected_regret", "mean"),
        uplift_vs_fixed_mean_pct=("uplift_vs_fixed_mean_pct", "mean"),
        uplift_vs_fixed_max_pct=("uplift_vs_fixed_max_pct", "mean"),
        uplift_vs_ppo_frozen_pct=("uplift_vs_ppo_frozen_pct", "mean"),
        upper_price_pct=("upper_price_pct", "mean"), lower_price_pct=("lower_price_pct", "mean"),
        price_boundary_pct=("boundary_pct", "mean"),
    ).sort_values(["simulator", "mean_revenue", "policy"], ascending=[True, False, True])
    by_sku = rows.groupby([*keyed, "sku"], as_index=False).agg(
        decisions=("chosen_price", "size"), simulated_revenue=("simulated_revenue", "sum"),
        distinct_prices=("chosen_price", "nunique"), upper_price_share=("at_upper_bound", "mean"),
        expected_regret=("expected_regret", "sum"),
    )
    pair_keys = ["simulator", "seed", *KEYS]
    ddpg = rows.loc[rows["policy"].eq("ddpg"), [*pair_keys, "chosen_price", "simulated_revenue"]]
    fixed = rows.loc[rows["policy"].eq(FIXED_MAX), [*pair_keys, "chosen_price", "simulated_revenue"]]
    paired = ddpg.merge(fixed, on=pair_keys, validate="one_to_one", suffixes=("_ddpg", "_fixed_max"))
    paired["same_price"] = np.isclose(paired["chosen_price_ddpg"], paired["chosen_price_fixed_max"], rtol=0, atol=1e-10)
    paired["delta_revenue"] = paired["simulated_revenue_ddpg"] - paired["simulated_revenue_fixed_max"]
    paired = paired.groupby(["simulator", "seed"], as_index=False).agg(
        decisions=("same_price", "size"), same_price_as_fixed_max=("same_price", "sum"),
        revenue_ddpg=("simulated_revenue_ddpg", "sum"), revenue_fixed_max=("simulated_revenue_fixed_max", "sum"),
        delta_revenue=("delta_revenue", "sum"),
    )
    paired["uplift_vs_fixed_max_pct"] = 100 * paired["delta_revenue"] / paired["revenue_fixed_max"].where(paired["revenue_fixed_max"].gt(0))
    outputs = {
        "pricing_test_decisions.csv": rows, "pricing_test.csv": board, "pricing_test_runs.csv": runs,
        "pricing_test_by_sku.csv": by_sku, "ddpg_vs_fixed_max.csv": paired,
    }
    for filename, frame in outputs.items():
        path = output / filename
        with NamedTemporaryFile(suffix=".csv", prefix=f".{path.stem}-", dir=output, delete=False) as tmp:
            stage = Path(tmp.name)
        try:
            frame.to_csv(stage, index=False)
            os.replace(stage, path)
        finally:
            stage.unlink(missing_ok=True)
    result = {"status": "complete", "split": "test", "seeds": seeds, "simulators": simulator_names,
              "policies": policies, "n_rows": n_rows, "decisions_per_policy_per_seed": len(inputs),
              "pricing_training_config": config, "decision_first_week": expected["week"].min(),
              "decision_last_week": expected["week"].max(), "match_noise_by_key": True,
              "decisions_sha256": _sha256(output / "pricing_test_decisions.csv"),
              "files": {file: _sha256(output / file) for file in outputs},
              "trained_once_per_seed_in": config["primary_song_simulator"],
              "wall_seconds_current_invocation": wall_seconds, **identity}
    _atomic_json(output / "pricing_run.json", result)
    return result


def run(root: Path, lock_path: Path, output: Path, selected_seeds: list[int] | None = None) -> dict:
    started = perf_counter()
    torch.set_num_threads(1)
    phase1, protocol, lock = _contract(root, lock_path)
    seeds, config = lock["phase5_seeds"], lock["pricing"]
    requested = seeds if selected_seeds is None else selected_seeds
    if not requested or not set(requested).issubset(seeds) or len(set(requested)) != len(requested):
        raise ValueError(f"Allowed distinct seeds: {seeds}")
    methods = [*config["phase5_methods"], config["additional_phase5_control"]]
    primary, secondary = config["primary_song_simulator"], config["sensitivity_simulator"]
    samplers = {
        primary: CachedGaussian(config["gaussian_seasonal_bandwidth_weeks"]),
        secondary: CachedLag(config["sensitivity_weight"], config["gaussian_seasonal_bandwidth_weeks"],
                             config["ratio_limit"]),
    }
    print("[pricing] verifying and loading frozen Phase 1 panels", flush=True)
    train, inputs, windows, scalers, history = _data(root, phase1, protocol, lock)
    for simulator in samplers.values():
        simulator.fit(train)
        if set(simulator.surfaces) != set(config["cohort_skus"]):
            raise ValueError("A Phase 1 pricing SKU is absent from the simulator")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"phase1_sha256": _sha256(root / "phase1_manifest.json"), "protocol_sha256": lock["protocol_sha256"],
                "lock_sha256": _sha256(lock_path), "m5_test_keys_sha256": lock["frozen_test_keys"]["m5"]["sha256"],
                "pricing_cohort_sha256": phase1["m5"]["pricing_cohort_sha256"],
                "runner_sha256": _sha256(Path(__file__)),
                "torch_version": str(torch.__version__), "sb3_version": stable_baselines3.__version__}
    progress = _run_status(output, seeds, methods, len(inputs), list(samplers), identity)
    if not (output / "pricing_test_keys.csv").exists():
        inputs[KEYS].to_csv(output / "pricing_test_keys.csv", index=False)
    actual_keys = pd.read_csv(output / "pricing_test_keys.csv", dtype={"sku": str}, parse_dates=["week"])
    if not actual_keys.equals(inputs[KEYS]):
        raise ValueError("Saved pricing test calendar differs from Phase 1")
    blocks = _training_blocks(windows, scalers)
    for seed in requested:
        folder = output / f"seed_{seed}"
        print(f"[pricing] seed {seed}: demand fit and {len(methods)} policies", flush=True)
        demand_dir = folder / "demand"
        demand = _unit(demand_dir, {**identity, "seed": seed}, lambda out: _train_demand(
            out, seed, windows, blocks, scalers, lock["forecast"]["m5"]["settings_by_model"]))
        saved = torch.load(demand_dir / "song.pt", map_location="cpu", weights_only=True)
        xgb = XGBDemand(demand_dir / "xgboost_price.json", _read_json(demand_dir / "xgboost_price_meta.json"))
        song = _load_song(saved, config["cohort_skus"], seed)
        for method in methods:
            diagnostic = xgb if method == "grid_xgboost_price" else song
            record = _unit(folder / method, {**identity, "seed": seed, "policy": method,
                                           "demand_files": demand["files"]},
                           lambda out: _policy_unit(out, method, seed, samplers, inputs, history,
                                                    saved, demand["files"]["song.pt"], diagnostic, config))
            progress = _run_status(output, seeds, methods, len(inputs), list(samplers), identity)
            print(f"[pricing] {seed}/{method}: {record['seconds']:.2f}s "
                  f"({progress['completed_units']}/{progress['planned_units']} units)", flush=True)
    if progress["status"] == "complete":
        return _aggregate(output, inputs, seeds, methods, list(samplers), identity, config, perf_counter() - started)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, action="append",
                        help="Run only specified locked seed(s); default runs or reuses all five")
    args = parser.parse_args()
    result = run(args.phase1_root, args.lock, args.output_dir, args.seed)
    print(f"[pricing] {result['status']}: {result.get('n_rows', result.get('completed_decision_rows'))} decisions",
          flush=True)


if __name__ == "__main__":
    main()
