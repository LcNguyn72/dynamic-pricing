"""Compare cumulative feature groups on the frozen weekly M5 split."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from .phase1_finalize import _observed
from .phase5_data import PHASE1_ROOT, iter_weekly, load_contract, verify_shard_test_keys


ARTIFACTS = PHASE1_ROOT.parent
KEYS = ["sku", "week"]
LAG_ONE = ("qty_lag_1", "qty_last")
SALES = LAG_ONE + ("qty_lag_2", "qty_lag_4", "qty_lag_52", "qty_mean_4", "qty_mean_12")
CALENDAR = SALES + ("week_sin", "week_cos", "event_days", "snap_days")
FEATURES = {
    "xgb_lag1": LAG_ONE,
    "xgb_history": SALES,
    "xgb_calendar": CALENDAR,
    "xgb_price": CALENDAR + ("price", "price_lag_1", "price_change"),
}
BASELINES = ("last_observed", "ar1_no_price", "arimax_price")
FIT = dict(n_estimators=120, max_depth=5, learning_rate=0.07, min_child_weight=15,
           objective="reg:squarederror", tree_method="hist", random_state=42)
TOTALS = ("n", "abs_error", "sq_error", "sum_qty", "sum_qty_sq")
SPLITS = ("validation", "test")


def build_features(weekly):
    """Target-week calendar and scheduled price are known inputs. Quantity lags are strictly causal."""
    if weekly.duplicated(KEYS).any() or not weekly.sort_values(KEYS).index.equals(weekly.index):
        raise ValueError("Weekly rows must be unique and sorted by SKU and calendar week")
    sku = weekly["sku"]
    observed = _observed(weekly)
    qty = weekly["qty"].where(observed)
    price = weekly["price"].where(observed)
    qty_groups = qty.groupby(sku, sort=False, observed=True)
    features = pd.DataFrame({f"qty_lag_{lag}": qty_groups.shift(lag) for lag in (1, 2, 4, 52)})
    past = features["qty_lag_1"].groupby(sku, sort=False, observed=True)
    features["qty_last"] = past.ffill()
    for width in (4, 12):
        mean = past.rolling(width, min_periods=1).mean()
        features[f"qty_mean_{width}"] = mean.droplevel(0).reindex(weekly.index)
    angle = 2 * np.pi * (weekly["week"].dt.isocalendar().week.astype(float) - 1) / 52.1775
    features["week_sin"] = np.sin(angle)
    features["week_cos"] = np.cos(angle)
    features[["event_days", "snap_days"]] = weekly[["event_days", "snap_days"]]
    features["price"] = price
    features["price_lag_1"] = price.groupby(sku, sort=False, observed=True).shift(1)
    features["price_change"] = price / features["price_lag_1"].where(features["price_lag_1"].gt(0)) - 1
    return features.astype(np.float32)


def metric_totals(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (actual.shape != predicted.shape or actual.ndim != 1 or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError("Scores require identical nonempty, finite target and prediction vectors")
    error = actual - predicted
    return np.array([len(actual), np.abs(error).sum(), error @ error, actual.sum(), actual @ actual])


def scores(totals):
    n, absolute, squared, qty, qty_sq = totals
    variation = qty_sq - qty * qty / n
    return {"n": int(n), "rmse": float(np.sqrt(squared / n)), "mae": float(absolute / n),
            "r2": float(1 - squared / variation) if variation > 0 else None}


def _sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _baseline_meta(directory, run, store, split):
    if split == "test":
        entry = run["test_predictions"][store]
        return directory / entry["file"], entry["sha256"]
    entry = run["models_by_store"][store]
    return directory / entry["validation"], entry["validation_sha256"]


def _baseline_totals(path, digest, target):
    if _sha(path) != digest:
        raise ValueError(f"{path.name}: baseline file hash changed")
    columns = KEYS + ["qty", "reference_last", "raw_ar1_no_price", "raw_arimax_price"]
    base = pd.read_csv(path, dtype={"sku": str}, parse_dates=["week"], usecols=columns)
    base = base.sort_values(KEYS).reset_index(drop=True)
    target = target.reset_index(drop=True)
    for key in KEYS + ["qty"]:
        if not np.array_equal(target[key].to_numpy(), base[key].to_numpy()):
            raise ValueError(f"{path.name}: baseline rows disagree with the panel on {key}")
    y = target["qty"].to_numpy()
    last = base["reference_last"]
    ar1 = base["raw_ar1_no_price"].fillna(last).clip(lower=0)
    arimax = base["raw_arimax_price"].fillna(ar1).clip(lower=0)
    return {name: metric_totals(y, pred).tolist()
            for name, pred in zip(BASELINES, (last, ar1, arimax), strict=True)}


def _fit_store(store, weekly, output, baseline_directory, reference, threads):
    observed = _observed(weekly)
    train = observed & weekly["split"].eq("train")
    heldout = {name: observed & weekly["split"].eq(name) for name in SPLITS}
    all_features = build_features(weekly)
    stats = {}
    for split, mask in heldout.items():
        path, checksum = _baseline_meta(baseline_directory, reference, store, split)
        stats[split] = _baseline_totals(path, checksum, weekly.loc[mask, KEYS + ["qty"]])
    y_train = weekly.loc[train, "qty"].to_numpy(dtype=np.float32)
    y = {split: weekly.loc[mask, "qty"].to_numpy() for split, mask in heldout.items()}
    hashes = {}
    for name, columns in FEATURES.items():
        started = perf_counter()
        model = xgb.XGBRegressor(**FIT, device="cpu", n_jobs=threads)
        model.fit(all_features.loc[train, list(columns)], y_train)
        for split, mask in heldout.items():
            predicted = model.predict(all_features.loc[mask, list(columns)]).clip(min=0)
            stats[split][name] = metric_totals(y[split], predicted).tolist()
        model_path = output / f"{name}.ubj"
        temp_model = model_path.with_name(f".{name}.tmp.ubj")
        model.save_model(temp_model)
        temp_model.replace(model_path)
        hashes[model_path.name] = _sha(model_path)
        print(f"{store} {name}: {perf_counter() - started:.1f}s, "
              f"validation RMSE {scores(stats['validation'][name])['rmse']:.3f}", flush=True)
    return {"training_rows": int(train.sum()), "stats": stats, "models": hashes}


def _write_results(output, records, identity):
    detail = []
    pooled = []
    for split in SPLITS:
        for model in (*BASELINES, *FEATURES):
            parts = []
            for store, result in records.items():
                values = result["stats"][split][model]
                parts.append(values)
                detail.append({"split": split, "model": model, "store": store, **scores(values)})
            pooled.append({"split": split, "model": model, **scores(np.array(parts).sum(axis=0))})
    validation = {row["model"]: row for row in pooled if row["split"] == "validation"}
    best = min(FEATURES, key=lambda name: validation[name]["rmse"])
    pd.DataFrame(pooled).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(detail).to_csv(output / "by_store.csv", index=False)
    _json(output / "run.json", {"status": "complete", "stores": list(records),
                               "identity": identity, "validation_selected_xgb": best,
                               "selection_metric": "minimum pooled validation RMSE",
                               "store_records": {store: {"training_rows": rec["training_rows"],
                                                         "test_keys": rec["test_keys"]}
                                                 for store, rec in records.items()}})
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)
    print(f"Validation-selected XGBoost: {best}", flush=True)


def run(root, output, baseline_directory, cache, stores, threads):
    contract = load_contract(root)
    store_ids = contract["protocol"]["datasets"]["m5"]["store_ids"]
    selected = store_ids if stores is None else stores
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(store_ids):
        raise ValueError("Provide unique M5 store IDs")
    run_path = baseline_directory / "run.json"
    reference = json.loads(run_path.read_text(encoding="utf-8"))
    identity = {"phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"],
                "baseline_run_sha256": _sha(run_path), "source_sha256": _sha(Path(__file__)),
                "xgboost_version": xgb.__version__, "features": {name: list(value) for name, value in FEATURES.items()},
                "fit": FIT, "threads": threads,
                "m5_frozen_test_keys_sha256": contract["phase1"]["m5"]["test_keys"]["sha256"]}
    if (reference["status"] != "complete" or reference["identity"]["fit_scope"] != "train_only"
            or reference["identity"]["phase1_sha256"] != identity["phase1_sha256"]
            or reference["test_keys_sha256"] != identity["m5_frozen_test_keys_sha256"]):
        raise ValueError("Step 3 baselines must match the frozen M5 panel and use train-only fitting")
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text(encoding="utf-8")) != identity:
        raise ValueError("Choose a new output directory for different benchmark settings")
    _json(identity_file, identity)
    records, remaining = {}, []
    for store in selected:
        directory = output / store
        marker = directory / "scores.json"
        if marker.is_file():
            rec = json.loads(marker.read_text(encoding="utf-8"))
            if (rec["test_keys"] == contract["test_key_shards"]["m5"][store]
                    and set(rec["models"]) == {f"{name}.ubj" for name in FEATURES}
                    and all((directory / name).is_file() and _sha(directory / name) == digest
                            for name, digest in rec["models"].items())):
                records[store] = rec
                print(f"{store}: reused checked results", flush=True)
                continue
        remaining.append(store)
    if remaining:
        for store, weekly, _ in iter_weekly(root, "m5", contract, shards=remaining, cache_root=cache):
            verify = verify_shard_test_keys(root, "m5", contract, store, weekly)
            directory = output / store
            directory.mkdir(exist_ok=True)
            print(f"{store}: fitting feature groups on the training weeks", flush=True)
            record = _fit_store(store, weekly, directory, baseline_directory, reference, threads)
            records[store] = {**record, "test_keys": verify}
            _json(directory / "scores.json", records[store])
    _write_results(output, {store: records[store] for store in selected}, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--baseline-dir", type=Path, default=ARTIFACTS / "pricing_arimax_v2")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "forecast_feature_ablation_v2")
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "phase5_v2" / "data_cache")
    parser.add_argument("--stores", nargs="+", help="Optional subset for a smoke run. Default is all ten stores.")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.prepared_root, args.output_dir, args.baseline_dir, args.cache_dir, args.stores, args.threads)


if __name__ == "__main__":
    main()
