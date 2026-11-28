"""Step 3: benchmark ARIMAX price response and train-only M5 forecasts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

from .phase1_finalize import _observed, _sha256
from .phase5_data import LOCK_PATH, PHASE1_ROOT, iter_weekly, load_contract


PHASE5 = PHASE1_ROOT.with_name("phase5_v2")
STEP2 = PHASE1_ROOT.with_name("pricing_transformer_tune_v2")
OUTPUT = PHASE1_ROOT.with_name("pricing_arimax_v2")
BASE = "ar1_no_price"
TRANSFORMER = "transformer_tuned"
VARIANTS = ("arimax_price", "arimax_price_fourier")
FIT_MODELS = (BASE, *VARIANTS)
KEYS = ["sku", "week"]
MIN_PAIRS = 12
MAXITER = 50
YEAR_WEEKS = 52.18


def _design(price, state, variant, start=0):
    if variant == BASE:
        return None
    standard_price = (np.where(np.isfinite(price) & (price > 0), price, state["price_mean"])
                      - state["price_mean"]) / state["price_sd"]
    if variant == VARIANTS[0]:
        return standard_price[:, None]
    phase = (2.0 * np.pi / YEAR_WEEKS) * np.arange(start, start + len(price))
    return np.column_stack((standard_price, np.sin(phase), np.cos(phase)))


def _model(qty, price, state, variant):
    y = (qty - state["qty_mean"]) / state["qty_sd"]
    return SARIMAX(y, exog=_design(price, state, variant), order=(1, 0, 0), trend="c",
                   enforce_stationarity=True)


def fit_series(qty: np.ndarray, price: np.ndarray, variant: str):
    observed = np.isfinite(qty)
    adjacent = observed[:-1] & observed[1:]
    if adjacent.sum() < MIN_PAIRS:
        return None, "sparse_train"
    state = {"qty_mean": float(qty[observed].mean()), "qty_sd": float(qty[observed].std())}
    if state["qty_sd"] <= 1e-8:
        return None, "constant_demand"
    if variant != BASE:
        observed_price = price[observed]
        if not np.isfinite(observed_price).all() or np.any(observed_price <= 0):
            return None, "invalid_price"
        paired = np.r_[adjacent, False] | np.r_[False, adjacent]
        lo, hi = float(observed_price.min()), float(observed_price.max())
        if np.ptp(price[paired]) <= 1e-8:
            return None, "constant_price"
        state.update(price_min=lo, price_max=hi, price_mean=float(observed_price.mean()),
                     price_sd=float(observed_price.std()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _model(qty, price, state, variant).fit(disp=False, maxiter=MAXITER)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None, "fit_failed"
    if not result.mle_retvals.get("converged", False) or not np.isfinite(result.params).all():
        return None, "not_converged"
    state["params"] = result.params.tolist()
    if variant != BASE:
        coefficient = state["params"][result.param_names.index("x1")]
        state["price_slope"] = coefficient * state["qty_sd"] / state["price_sd"]
    return state, "ok"


def forecast_future(state, variant, train_qty, train_price, future_qty, future_price):
    """Update the fixed model with past observed sales only when forecasting each week."""
    if future_qty.size == 0:
        return np.empty(0)
    fitted = _model(train_qty, train_price, state, variant).filter(state["params"])
    exog = _design(future_price, state, variant, start=len(train_qty))
    standardized = (future_qty - state["qty_mean"]) / state["qty_sd"]
    forecast = fitted.extend(standardized, exog=exog).get_prediction(information_set="predicted")
    return np.asarray(forecast.predicted_mean) * state["qty_sd"] + state["qty_mean"]


def candidate_quantities(raw_qty, observed_price, candidate_prices, state):
    if np.any(candidate_prices < state["price_min"] - 1e-9) or np.any(candidate_prices > state["price_max"] + 1e-9):
        raise ValueError("Candidate price is outside the observed training range")
    return np.maximum(0.0, raw_qty + state["price_slope"] * (candidate_prices - observed_price))


def _predict(state, variant, train, future):
    if state is None:
        return np.full(len(future[0]), np.nan)
    try:
        values = forecast_future(state, variant, *train, *future)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return np.full(len(future[0]), np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _fit_worker(task):
    sku, train, future = task
    states, forecasts = {}, {}
    for variant in FIT_MODELS:
        state, status = fit_series(*train, variant)
        states[variant] = (state, status)
        forecasts[variant] = _predict(state, variant, train, future)
    return sku, states, forecasts


def _test_worker(task):
    sku, train, future, states = task
    return sku, states, {variant: _predict(state[0], variant, train, future)
                         for variant, state in states.items()}


def _limit_threads():
    threadpool_limits(limits=1)


def _prepare_arrays(weekly, source_weeks):
    weeks = len(source_weeks)
    spec = weekly["split"].to_numpy().reshape(-1, weeks)
    dates = weekly["week"].to_numpy().reshape(-1, weeks)
    expected = pd.DatetimeIndex(source_weeks).to_numpy()
    skus = weekly["sku"].to_numpy().reshape(-1, weeks)
    if not (np.all(dates == expected) and np.all(skus == skus[:, :1]) and np.all(spec == spec[:1])):
        raise ValueError("M5 weekly grid is not common and ordered by SKU and date")
    price = weekly["price"].to_numpy(dtype=float).reshape(-1, weeks)
    qty = weekly["qty"].where(_observed(weekly)).to_numpy(dtype=float).reshape(-1, weeks)
    return list(skus[:, 0]), qty, price, spec[0]


def _last_observed(qty, train_count, end):
    timeline = qty[:end]
    positions = np.maximum.accumulate(np.where(np.isfinite(timeline), np.arange(end), -1))
    safe = positions[train_count - 1:end - 1]
    if np.any(safe < 0):
        return np.where(safe >= 0, timeline[np.maximum(safe, 0)], np.nan)
    return timeline[safe]


def _write_csv(path, frame):
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _verified_file(path, digest):
    if _sha256(path) != digest:
        raise ValueError(f"Artifact hash differs: {path}")
    return path


def _pack_frame(weekly, skus, results, futures, train_count, last_index, which, end, models):
    cols = [*KEYS, "qty", "price", "observed_sales"]
    test = weekly.loc[weekly["split"].eq(which), cols].copy().reset_index(drop=True)
    if len(test) != len(skus) * (end - last_index):
        raise ValueError("Forecast group count differs from M5 calendar")
    offset = slice(last_index - train_count, None)
    test["reference_last"] = np.concatenate(
        [_last_observed(q, train_count, end)[offset] for q in futures])
    for model in models:
        raw = np.concatenate([forecast[model][offset] for _, _, forecast in results])
        test[f"raw_{model}"] = raw
        if model != BASE:
            ranges = [((state or {}).get("price_min", np.nan), (state or {}).get("price_max", np.nan),
                       (state or {}).get("price_slope", np.nan))
                      for _, states, _ in results for state, _ in (states[model],)]
            repeated = np.repeat(np.asarray(ranges), end - last_index, axis=0)
            test[f"support_{model}"] = ((test["price"] >= repeated[:, 0] - 1e-7)
                                       & (test["price"] <= repeated[:, 1] + 1e-7))
            test[f"lo_{model}"], test[f"hi_{model}"], test[f"slope_{model}"] = repeated.T
    test = test.loc[test.pop("observed_sales").astype(bool)].copy()
    test["week"] = test["week"].dt.strftime("%Y-%m-%d")
    test.insert(0, "dataset", "m5")
    return test


def _train_shard(weekly, output, store, source_weeks, pool):
    skus, qty, price, splits = _prepare_arrays(weekly, source_weeks)
    train_count = int(np.sum(splits == "train"))
    end = train_count + int(np.sum(splits == "validation"))
    # Only train and validation arrays are passed to the fitting workers.
    tasks = ((sku, (q[:train_count], p[:train_count]), (q[train_count:end], p[train_count:end]))
             for sku, q, p in zip(skus, qty, price, strict=True))
    results = list(pool.map(_fit_worker, tasks, chunksize=32))
    if [sku for sku, _, _ in results] != skus:
        raise ValueError(f"{store}: training outputs differ from ordered series")
    frame = _pack_frame(weekly, skus, results, qty, train_count, train_count, "validation", end, FIT_MODELS)
    models = output / f"models_{store}.csv"
    predictions = output / f"validation_{store}.csv"
    _write_csv(models, pd.DataFrame(
        ({"sku": sku, "model": variant, "status": status,
          "parameters": json.dumps(state, allow_nan=False) if state else ""}
         for sku, states, _ in results for variant, (state, status) in states.items())))
    _write_csv(predictions, frame)
    return models, predictions


def _test_shard(weekly, parameters, selected, source_weeks, pool):
    skus, qty, price, splits = _prepare_arrays(weekly, source_weeks)
    train_count = int(np.sum(splits == "train"))
    start = train_count + int(np.sum(splits == "validation"))
    models = (BASE, selected)
    states = {sku: {} for sku in skus}
    for row in pd.read_csv(parameters, keep_default_na=False).itertuples(index=False):
        if row.model in models:
            states[row.sku][row.model] = (json.loads(row.parameters) if row.parameters else None, row.status)
    tasks = ((sku, (q[:train_count], p[:train_count]), (q[train_count:], p[train_count:]), states[sku])
             for sku, q, p in zip(skus, qty, price, strict=True))
    results = list(pool.map(_test_worker, tasks, chunksize=32))
    if [sku for sku, _, _ in results] != skus:
        raise ValueError("Test outputs differ from ordered series")
    return _pack_frame(weekly, skus, results, qty, train_count, start, "test", len(splits), models)


def _metric_frames(frame, reference, models):
    from .phase5_verify import KEYS as identity, align_predictions

    merged = reference.merge(frame.drop(columns=["qty", "price"]), on=identity, how="left", validate="one_to_one")
    if merged["reference_last"].isna().any() or len(merged) != len(reference):
        raise ValueError("Canonical keys lack a past observation for fallback")
    align_predictions(merged, frame.loc[:, [*identity, "qty", "price"]].merge(
        reference[identity], on=identity, validate="one_to_one"))
    merged.rename(columns={"predicted_qty": TRANSFORMER}, inplace=True)
    fallback = merged["raw_" + BASE].fillna(merged["reference_last"]).clip(lower=0)
    merged[BASE] = fallback
    for model in models:
        merged[model] = merged["raw_" + model].fillna(fallback).clip(lower=0)
    return merged


def _score_part(store, data, models, split, normalizers, overall):
    from .phase5_verify import sufficient, summarize

    common = data[[f"raw_{model}" for model in models]].notna().all(axis=1)
    inside = data[[f"support_{model}" for model in models]].all(axis=1)
    scores = []
    for scope, rows in (("all_keys_with_fallback", data), ("common_native_keys", data.loc[common]),
                        ("common_native_in_train_range", data.loc[common & inside])):
        if rows.empty:
            continue
        for model in (BASE, *models, TRANSFORMER):
            metric_rows = rows.assign(predicted_qty=rows[model])
            totals = sufficient(metric_rows).sum()
            overall[scope, model].append(totals)
            scores.append({"store": store, "split": split, "scope": scope, "model": model,
                           **summarize(totals, **normalizers)})
    counts = []
    for model in (BASE, *models):
        native = data["raw_" + model].notna()
        inside_model = native if model == BASE else native & data["support_" + model]
        counts.append({"split": split, "store": store, "model": model, "testable_keys": len(data),
                       "native_keys": int(native.sum()), "native_series": data.loc[native, "sku"].nunique(),
                       "native_within_train_price_keys": int(inside_model.sum())})
    return scores, counts


def _totals(overall, split, normalizers):
    from .phase5_verify import summarize

    return pd.DataFrame({"split": split, "scope": scope, "model": model,
                         **summarize(pd.concat(rows, axis=1).sum(axis=1), **normalizers)}
                        for (scope, model), rows in overall.items())


def _price_diagnostics(data, sku, models):
    rows = data.loc[data["sku"].eq(sku)].sort_values("week")
    positions = np.linspace(0, len(rows) - 1, min(5, len(rows)), dtype=int)
    records = []
    for row in rows.iloc[positions].to_dict("records"):
        for model in models:
            state = {"price_min": row[f"lo_{model}"], "price_max": row[f"hi_{model}"],
                     "price_slope": row[f"slope_{model}"]}
            raw = row[f"raw_{model}"]
            ok = np.isfinite(raw) and np.isfinite(list(state.values())).all()
            prices = np.linspace(state["price_min"], state["price_max"], 9) if ok else np.array([])
            predictions = candidate_quantities(raw, row["price"], prices, state) if ok else np.array([])
            records.append({"sku": sku, "week": row["week"], "model": model, "native": bool(ok),
                            "slope_units_per_usd": state["price_slope"],
                            "candidate_min": state["price_min"], "candidate_max": state["price_max"],
                            "quantity_span": float(np.ptp(predictions)) if ok else None,
                            "prediction_low": float(predictions[0]) if ok else None,
                            "prediction_high": float(predictions[-1]) if ok else None})
    return records


def run(root, output, cache, transformer_reference, step2, workers):
    from .phase5_verify import _csv

    began = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(root, LOCK_PATH)
    root = contract["root"]
    step2_report = _read_json(step2 / "m5_test_audit.json")
    validation_ref = _read_json(transformer_reference / "validation_run.json")
    frozen = contract["lock"]["frozen_test_keys"]["m5"]
    if (step2_report["fit_scope"] != "train_only" or step2_report["test_keys_sha256"] != frozen["sha256"]
            or validation_ref["checkpoint_sha256"] != step2_report["checkpoint_sha256"]
            or validation_ref["phase1_sha256"] != contract["phase1_sha256"]):
        raise ValueError("Transformer comparator does not match the train-only Step 2 checkpoint")
    normalizers = {"train_mean": validation_ref["train_qty_mean"], "train_std": validation_ref["train_qty_std"]}
    identity = {"phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"],
                "transformer_checkpoint_sha256": step2_report["checkpoint_sha256"],
                "validation_reference_sha256": _sha256(transformer_reference / "validation_run.json"),
                "source_sha256": _sha256(Path(__file__)), "statsmodels": statsmodels.__version__,
                "numpy": np.__version__, "model_order": [1, 0, 0], "min_train_pairs": MIN_PAIRS,
                "maxiter": MAXITER, "year_weeks": YEAR_WEEKS, "fit_scope": "train_only"}
    marker = output / "run_identity.json"
    if marker.is_file() and _read_json(marker) != identity:
        raise ValueError("Use a new output folder for a different run")
    _write_json(marker, identity)
    cohort = {sku.rsplit("@", 1)[1]: sku for sku in contract["lock"]["pricing"]["cohort_skus"]}

    fit_markers, val_scores, coverage, diagnostics = {}, [], [], []
    overall = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_threads) as pool:
        for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
            source_weeks = weekly["week"].drop_duplicates()
            status = output / f"fit_{store}.json"
            if status.is_file():
                meta = _read_json(status)
                parameters = _verified_file(output / meta["models"], meta["models_sha256"])
                predictions = _verified_file(output / meta["validation"], meta["validation_sha256"])
            else:
                parameters, predictions = _train_shard(weekly, output, store, source_weeks, pool)
                meta = {"models": parameters.name, "validation": predictions.name,
                        "models_sha256": _sha256(parameters), "validation_sha256": _sha256(predictions)}
                _write_json(status, meta)
            fit_markers[store] = meta
            pred = validation_ref["prediction_files"][store]
            reference = _csv(_verified_file(transformer_reference / pred["file"], pred["sha256"]))
            part = _metric_frames(_csv(predictions), reference, VARIANTS)
            stats, counts = _score_part(store, part, VARIANTS, "validation", normalizers, overall)
            val_scores.extend(stats)
            coverage.extend(counts)
            diagnostics.extend(_price_diagnostics(part, cohort[store], VARIANTS))
            print(f"[Step 3 validation] {store}: {len(part)} matched keys, "
                  f"{counts[1]['native_series']} fitted price series", flush=True)

        validation = _totals(overall, "validation", normalizers)
        _write_csv(output / "validation_scores.csv", validation)
        _write_csv(output / "validation_by_store.csv", pd.DataFrame(val_scores))
        _write_csv(output / "validation_price_response.csv", pd.DataFrame(diagnostics))
        candidates = validation.loc[validation["scope"].eq("common_native_keys") & validation["model"].isin(VARIANTS)]
        if set(candidates["model"]) != set(VARIANTS):
            raise ValueError("Cannot compare both price variants on common validation keys")
        winner = candidates.sort_values(["rmse", "model"]).iloc[0]
        selected = winner["model"]
        selection = {"selected": selected, "validation_common_n": int(winner["n"]),
                     "validation_common_rmse": float(winner["rmse"]),
                     "rule": "minimum micro RMSE on shared native validation keys", "identity": identity}
        _write_json(output / "selection.json", selection)
        print(f"[Step 3 selection] {selected}, RMSE {winner['rmse']:.4f}; scoring frozen M5 test", flush=True)

        overall = defaultdict(list)
        test_scores, test_files = [], {}
        for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
            source_weeks = weekly["week"].drop_duplicates()
            export = output / f"test_{store}.csv"
            status = output / f"test_{store}.json"
            if status.is_file():
                saved = _read_json(status)
                if saved["selected"] != selected:
                    raise ValueError(f"{store}: saved test predictions belong to another selection")
                _verified_file(export, saved["sha256"])
            else:
                parameters = output / fit_markers[store]["models"]
                _write_csv(export, _test_shard(weekly, parameters, selected, source_weeks, pool))
                saved = {"selected": selected, "sha256": _sha256(export)}
                _write_json(status, saved)
            pred = step2_report["prediction_files"][store]
            reference = _csv(_verified_file(step2 / "full_m5" / pred["file"], pred["sha256"]))
            part = _metric_frames(_csv(export), reference, (selected,))
            if len(part) != contract["test_key_shards"]["m5"][store]["n"]:
                raise ValueError(f"{store}: test is missing frozen keys")
            stats, counts = _score_part(store, part, (selected,), "test", normalizers, overall)
            test_scores.extend(stats)
            coverage.extend(counts)
            test_files[store] = {"file": export.name, "sha256": saved["sha256"], "n": len(part)}
            print(f"[Step 3 test] {store}: {len(part)} verified keys", flush=True)

    test = _totals(overall, "test", normalizers)
    if sum(entry["n"] for entry in test_files.values()) != frozen["count"]:
        raise ValueError("Full M5 test coverage does not match the frozen contract")
    _write_csv(output / "test_scores.csv", test)
    _write_csv(output / "test_by_store.csv", pd.DataFrame(test_scores))
    _write_csv(output / "coverage.csv", pd.DataFrame(coverage))
    report = {"status": "complete", **selection, "n_test": frozen["count"], "test_keys_sha256": frozen["sha256"],
              "models_by_store": fit_markers, "test_predictions": test_files,
              "wall_seconds": perf_counter() - began}
    _write_json(output / "run.json", report)
    print(f"[Step 3 complete] {frozen['count']} verified test keys", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=PHASE5 / "data_cache")
    parser.add_argument("--transformer-reference", type=Path, default=OUTPUT / "transformer_reference")
    parser.add_argument("--step2-dir", type=Path, default=STEP2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.phase1_root, args.output_dir, args.cache_dir, args.transformer_reference, args.step2_dir, args.workers)


if __name__ == "__main__":
    main()
