"""Tune a pooled UCI XGBoost correction to the four-week quantity forecast."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .forecast_feature_ablation import ARTIFACTS, KEYS, _json, _sha, metric_totals, scores
from .phase1_finalize import _observed
from .phase5_data import PHASE1_ROOT, iter_weekly, load_contract, verify_shard_test_keys, windows_from_weekly


CALENDAR = ("month_sin", "month_cos", "week_sin", "week_cos", "nov_dec", "christmas_week")
SETTINGS = tuple(dict(max_depth=d, min_child_weight=c, reg_lambda=l)
                 for d, c, l in product((2, 4, 6), (12, 32), (5, 25)))
BASE_SETTINGS = dict(n_estimators=350, early_stopping_rounds=25, learning_rate=0.05,
                     reg_alpha=0.25, subsample=0.85, colsample_bytree=0.9,
                     tree_method="hist", objective="reg:squarederror", eval_metric="rmse")
WEIGHTS = (0.25, 0.5, 1.0)
SEEDS = (42, 123, 456)


def build_inputs(panel):
    """Use prior sales and calendar information known at the forecast origin."""
    if not panel.sort_values(KEYS).index.equals(panel.index) or panel.duplicated(KEYS).any():
        raise ValueError("UCI panel must be unique and ordered by SKU and week")
    qty = panel["qty"].groupby(panel["sku"], sort=False)
    history = pd.DataFrame({f"qty_lag_{lag}": qty.shift(lag) for lag in range(1, 13)})
    features = history[["qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_4", "qty_lag_8", "qty_lag_12"]].copy()
    for width in (4, 8, 12):
        past = history.iloc[:, :width]
        features[f"qty_mean_{width}"] = past.mean(axis=1)
        features[f"qty_median_positive_{width}"] = past.where(past.gt(0)).median(axis=1)
        features[f"observed_weeks_{width}"] = past.gt(0).sum(axis=1)
    features["qty_max_12"] = history.max(axis=1)
    dates = panel["week"]
    month = dates.dt.month
    week = dates.dt.isocalendar().week.astype(float)
    features["month_sin"], features["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    features["week_sin"], features["week_cos"] = np.sin(2 * np.pi * week / 52.1775), np.cos(2 * np.pi * week / 52.1775)
    features["nov_dec"] = month.ge(11)
    christmas = pd.to_datetime({"year": dates.dt.year, "month": 12, "day": 25})
    features["christmas_week"] = (christmas - dates).dt.days.between(0, 6)
    return features.astype(np.float32), features["qty_mean_4"].astype(float)


def _fit(parts, parameters, seed, threads):
    train, valid = parts["train"], parts["validation"]
    model = XGBRegressor(**BASE_SETTINGS, **parameters, random_state=seed, n_jobs=threads)
    model.fit(train[0], train[1] - train[2], eval_set=[(valid[0], valid[1] - valid[2])], verbose=False)
    return model


def _measure(part, correction, weight):
    _, actual, base = part
    return scores(metric_totals(actual, np.maximum(0, base + weight * correction)))


def run(root, output, cache, threads):
    contract = load_contract(root)
    _, panel, _ = next(iter_weekly(root, "uci", contract, cache_root=cache))
    windows = windows_from_weekly(panel)
    frozen = verify_shard_test_keys(root, "uci", contract, "uci", windows)
    indexes = pd.MultiIndex.from_frame(panel[KEYS])
    evaluation = {
        name: indexes.isin(pd.MultiIndex.from_frame(windows.loc[windows["split"].eq(name) & _observed(windows), KEYS]))
        for name in ("validation", "test")
    }
    features, base = build_inputs(panel)
    train = panel["split"].eq("train") & _observed(panel) & base.notna()
    masks = {"train": train, **evaluation}
    volume_features = [name for name in features if name not in CALENDAR]
    groups = {"history": volume_features, "history_calendar": list(features)}
    output.mkdir(parents=True, exist_ok=True)
    trials, results, choices = [], [], {}
    for split in ("validation", "test"):
        baseline = (None, panel.loc[masks[split], "qty"].to_numpy(), base[masks[split]].to_numpy())
        results.append({"feature_group": "mean_4", "split": split, "seed": None, **_measure(baseline, 0, 0)})
    for group, names in groups.items():
        parts = {split: (features.loc[mask, names], panel.loc[mask, "qty"].to_numpy(),
                         base[mask].to_numpy()) for split, mask in masks.items()}
        best = None
        for i, parameters in enumerate(SETTINGS):
            model = _fit(parts, parameters, SEEDS[0], threads)
            correction = model.predict(parts["validation"][0])
            for weight in WEIGHTS:
                result = _measure(parts["validation"], correction, weight)
                trial = {"feature_group": group, "config_index": i, "weight": weight,
                         "trees": int(model.best_iteration + 1), **parameters, **result}
                trials.append(trial)
                if best is None or trial["rmse"] < best[0]["rmse"]:
                    best = trial, model
        choice, first = best
        choices[group] = {**choice, "features": names}
        print(f"{group}: validation RMSE {choice['rmse']:.3f}, config {choice['config_index']}, "
              f"weight {choice['weight']}, trees {choice['trees']}", flush=True)
        for seed in SEEDS:
            model = first if seed == SEEDS[0] else _fit(parts, SETTINGS[choice["config_index"]], seed, threads)
            for split in ("validation", "test"):
                result = _measure(parts[split], model.predict(parts[split][0]), choice["weight"])
                results.append({"feature_group": group, "split": split, "seed": seed, **result})
            model_path = output / f"{group}_seed{seed}.ubj"
            model.save_model(model_path)

    pd.DataFrame(trials).to_csv(output / "validation_trials.csv", index=False)
    pd.DataFrame(results).to_csv(output / "scores.csv", index=False)
    _json(output / "run.json", {
        "test_keys": frozen, "train_rows": int(train.sum()), "validation_rows": int(evaluation["validation"].sum()),
        "phase1_sha256": contract["phase1_sha256"], "source_sha256": _sha(Path(__file__)),
        "selection_seed": SEEDS[0], "replication_seeds": list(SEEDS[1:]),
        "selected_feature_group": min(choices, key=lambda group: choices[group]["rmse"]),
        "selected_configurations": choices, "settings": BASE_SETTINGS,
    })
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "phase5_v2" / "data_cache")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "uci_xgb_reliability_v2")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.prepared_root, args.output_dir, args.cache_dir, args.threads)


if __name__ == "__main__":
    main()
