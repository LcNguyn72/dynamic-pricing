"""Chronological, shared-row benchmark for one-week retail demand forecasts.

This module is separate from the existing application model.  Every method is
scored in original units/week on the same observed SKU-week rows.  Validation
selects hyperparameters when the chosen split profile provides it; the final
test period is only evaluated after model selection.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MODELS = (
    "naive_last",
    "arima",
    "xgboost",
    "xgboost_price",
    "lstm",
    "lstm_price",
    "transformer",
    "transformer_price",
)
KNOWN_CALENDAR_COVARIATES = ("event_days", "snap_days")


def _validate_windows(windows: pd.DataFrame, lookback: int) -> pd.DataFrame:
    required = {"sku", "week", "qty", "price", "split", "observed_sales", "month", "week_of_year"}
    required.update(f"qty_lag_{lag}" for lag in range(1, lookback + 1))
    required.update(f"price_lag_{lag}" for lag in range(1, lookback + 1))
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Window table lacks required columns: {sorted(missing)}")
    data = windows
    if data["sku"].dtype != object or (len(data) and not isinstance(data["sku"].iloc[0], str)):
        data = data.copy()
        data["sku"] = data["sku"].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(data["week"]):
        if data is windows:
            data = data.copy()
        data["week"] = pd.to_datetime(data["week"])
    # Prepared panels are already sorted. Avoid sorting a second full copy of
    # millions of M5 rows; only repair order for ad-hoc caller-provided panels.
    sku_values = data["sku"].to_numpy()
    week_values = data["week"].to_numpy()
    if len(data) > 1:
        sku_reversed = sku_values[1:] < sku_values[:-1]
        same_sku = sku_values[1:] == sku_values[:-1]
        week_reversed = week_values[1:] < week_values[:-1]
        if np.any(sku_reversed | (same_sku & week_reversed)):
            data = data.sort_values(["sku", "week"]).reset_index(drop=True)
    if data.duplicated(["sku", "week"]).any():
        raise ValueError("Window table has duplicate SKU-week rows")
    if not set(data["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Split labels must be train, validation, or test")
    if data.loc[data["split"] == "train"].empty or data.loc[data["split"] == "test"].empty:
        raise ValueError("Both train and test periods must have windows")
    week_splits = data.groupby("week")["split"].nunique()
    if (week_splits > 1).any():
        raise ValueError("A calendar week appears in multiple data splits")
    # A later training week would invalidate the prospective test claim.
    train_last = data.loc[data["split"] == "train", "week"].max()
    test_first = data.loc[data["split"] == "test", "week"].min()
    if train_last >= test_first:
        raise ValueError("Test must begin after every training week")
    val = data[data["split"] == "validation"]
    if not val.empty and not (train_last < val["week"].min() <= val["week"].max() < test_first):
        raise ValueError("Validation weeks must be between train and test")
    return data


def _observed(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["observed_sales"].astype(bool)]


def _score(rows: pd.DataFrame, prediction: np.ndarray) -> dict:
    actual = rows["qty"].to_numpy(dtype=float)
    forecast = np.clip(np.asarray(prediction, dtype=float), 0, None)
    if len(actual) != len(forecast):
        raise ValueError("Prediction count does not match the observed test rows")
    if not np.isfinite(forecast).all():
        raise ValueError("Forecast contains non-finite values")
    error = forecast - actual
    nonzero = actual > 0
    return {
        "n": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "wape_pct": float(100 * np.abs(error).sum() / actual.sum()) if actual.sum() > 0 else None,
        "mape_pct": float(100 * np.mean(np.abs(error[nonzero]) / actual[nonzero])) if nonzero.any() else None,
        "mape_n": int(nonzero.sum()),
    }


def _prediction_rows(model_name: str, split: str, rows: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["model", "split", "sku", "week", "price", "y_true", "y_pred", "tier"])
    if len(rows) != len(prediction):
        raise ValueError(f"{model_name}: prediction rows are misaligned")
    return pd.DataFrame({
        "model": model_name,
        "split": split,
        "sku": rows["sku"].to_numpy(),
        "week": rows["week"].to_numpy(),
        "price": rows["price"].to_numpy(dtype=float),
        "y_true": rows["qty"].to_numpy(dtype=float),
        "y_pred": np.clip(np.asarray(prediction, dtype=float), 0, None),
        "tier": rows["tier"].to_numpy() if "tier" in rows else "",
    })


def _naive_predictions(data: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    out = {}
    for split in ("validation", "test"):
        rows = _observed(data[data["split"] == split])
        out[split] = (rows, rows["qty_lag_1"].to_numpy(dtype=float))
    return out


def _arima_phase(group: pd.DataFrame, train_labels: set[str], phase: str, *, strict_fit: bool = False):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = group[group["split"].isin(train_labels)]
    future = group[group["split"] == phase]
    if future.empty:
        return future, np.empty(0), None
    endog = train["qty"].where(train["observed_sales"].astype(bool), np.nan).to_numpy(dtype=float)
    if np.isfinite(endog).sum() < 12:
        raise ValueError("ARIMA requires at least 12 observed training weeks per SKU")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            endog, order=(1, 0, 1), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=80)
    if strict_fit and not fitted.mle_retvals.get("converged", False):
        raise ValueError("ARIMA optimization did not converge")
    predictions = []
    for _, row in future.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yhat = float(fitted.forecast(1)[0])
            observed = float(row["qty"]) if bool(row["observed_sales"]) else np.nan
            fitted = fitted.append([observed], refit=False)
        predictions.append(float("nan") if strict_fit and not math.isfinite(yhat) else max(0.0, yhat))
    return future, np.asarray(predictions, dtype=float), None


def _arima_sku_task(task):
    sku, group = task
    records = []
    failures = []
    group = group.sort_values("week")
    for phase, labels in (("validation", {"train"}),
                          ("test", {"train", "validation"})):
        if group[group["split"] == phase].empty:
            continue
        try:
            rows, predictions, _ = _arima_phase(group, labels, phase)
            records.extend(
                (phase, pd.Timestamp(week), bool(observed), float(prediction))
                for week, observed, prediction in zip(
                    rows["week"], rows["observed_sales"], predictions
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"sku": str(sku), "split": phase, "error": str(exc)})
    return str(sku), records, failures


def _arima_predictions(data: pd.DataFrame):
    out_by_split: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}
    failures = []
    # Each SKU's ARIMA fit is independent. Two worker processes use available
    # CPU parallelism while keeping memory headroom for the full M5 panel and
    # later neural models on typical student hardware.
    tasks = ((str(sku), group) for sku, group in data.groupby("sku", sort=True))
    workers = min(2, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sku, records, group_failures in pool.map(_arima_sku_task, tasks, chunksize=8):
            failures.extend(group_failures)
            for split in ("validation", "test"):
                part = [(sku, week, observed, prediction)
                        for phase, week, observed, prediction in records if phase == split]
                if part:
                    out_by_split[split].extend(part)
    test_failures = [failure for failure in failures if failure["split"] == "test"]
    if test_failures:
        raise RuntimeError(f"ARIMA failed on {len(test_failures)} test SKU groups: {test_failures[:3]}")
    validation_failures = [failure for failure in failures if failure["split"] == "validation"]
    details = {
        "validation_rows_omitted": 0,
        "validation_skus_omitted": 0,
        "validation_omission_reason": None,
    }
    out = {}
    for split in ("validation", "test"):
        rows = _observed(data[data["split"] == split])
        if rows.empty:
            out[split] = (rows, np.empty(0))
            continue
        # ARIMA needs at least 12 observed training weeks. Keep every SKU in
        # the common test cohort; if any series cannot support a validation fit,
        # omit ARIMA's validation score as a whole instead of reporting it on a
        # smaller and potentially biased subset.
        if split == "validation" and validation_failures:
            skipped_skus = {failure["sku"] for failure in validation_failures}
            details.update({
                "validation_rows_omitted": int(rows["sku"].isin(skipped_skus).sum()),
                "validation_skus_omitted": len(skipped_skus),
                "validation_omission_reason": "some SKUs have fewer than 12 observed train weeks",
            })
            out[split] = (rows.iloc[:0], np.empty(0))
            continue
        predictions = pd.DataFrame(
            out_by_split[split], columns=["sku", "week", "observed_sales", "prediction"]
        )
        merged = rows[["sku", "week"]].merge(predictions[["sku", "week", "prediction"]], on=["sku", "week"], how="left", validate="one_to_one")
        if merged["prediction"].isna().any():
            raise ValueError(f"ARIMA lacks predictions for {split}")
        out[split] = (rows, merged["prediction"].to_numpy(dtype=float))
    return out, details


def _tabular_features(
    rows: pd.DataFrame,
    skus: Sequence[str],
    scalers: Mapping[str, Mapping[str, float]],
    lookback: int,
    with_price: bool,
    calendar_covariates: Sequence[str] | None = None,
) -> np.ndarray:
    if rows.empty:
        return np.empty((0, 0), dtype=np.float32)
    sku = rows["sku"].astype(str).to_numpy()
    q_mean = np.array([scalers[s]["qty_mean"] for s in sku], dtype=np.float32)
    q_std = np.array([scalers[s]["qty_std"] for s in sku], dtype=np.float32)
    p_mean = np.array([scalers[s]["price_mean"] for s in sku], dtype=np.float32)
    p_std = np.array([scalers[s]["price_std"] for s in sku], dtype=np.float32)
    if np.any(q_std <= 0) or np.any(p_std <= 0):
        raise ValueError("Train-only scaler standard deviations must be positive")
    parts = [
        ((rows[f"qty_lag_{lag}"].to_numpy(dtype=np.float32) - q_mean) / q_std)[:, None]
        for lag in range(1, lookback + 1)
    ]
    if with_price:
        parts += [
            ((rows[f"price_lag_{lag}"].to_numpy(dtype=np.float32) - p_mean) / p_std)[:, None]
            for lag in range(1, lookback + 1)
        ]
    month = rows["month"].to_numpy(dtype=np.float32)
    week = rows["week_of_year"].to_numpy(dtype=np.float32)
    parts += [
        np.sin(2 * math.pi * month / 12)[:, None],
        np.cos(2 * math.pi * month / 12)[:, None],
        np.sin(2 * math.pi * week / 52.18)[:, None],
        np.cos(2 * math.pi * week / 52.18)[:, None],
    ]
    if calendar_covariates is None:
        calendar_covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in rows.columns)
    missing_covariates = set(calendar_covariates) - set(rows.columns)
    if missing_covariates:
        raise ValueError(f"Missing known calendar covariates: {sorted(missing_covariates)}")
    for covariate in calendar_covariates:
        values = rows[covariate].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 7)):
            raise ValueError(f"{covariate} must contain day counts from 0 to 7")
        parts.append((values / 7.0)[:, None])
    # Keep the compact dense representation for the original small cohorts.
    # Full M5 has tens of thousands of item-store series; a dense one-hot matrix
    # would dominate memory, so encode its single active SKU feature sparsely.
    if len(skus) > 256:
        from scipy.sparse import csr_matrix, hstack

        categories = pd.Index(skus)
        codes = categories.get_indexer(sku)
        if (codes < 0).any():
            raise ValueError("Unknown SKU in XGBoost features")
        sku_sparse = csr_matrix(
            (np.ones(len(codes), dtype=np.float32), codes,
             np.arange(len(codes) + 1, dtype=np.int64)),
            shape=(len(codes), len(skus)),
        )
    else:
        indices = {item: i for i, item in enumerate(skus)}
        onehot = np.eye(len(skus), dtype=np.float32)[np.array([indices[s] for s in sku])]
        parts.append(onehot)
    if with_price:
        parts.append(((rows["price"].to_numpy(dtype=np.float32) - p_mean) / p_std)[:, None])
    dense = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(dense).all():
        raise ValueError("Non-finite XGBoost features")
    if len(skus) > 256:
        from scipy.sparse import hstack

        return hstack([dense, sku_sparse], format="csr", dtype=np.float32)
    return dense


def _xgb_predictions(data: pd.DataFrame, manifest: Mapping, with_price: bool, seed: int, model_dir: Path):
    from xgboost import XGBRegressor

    scalers = manifest["scalers"]
    lookback = int(manifest.get("lookback_weeks", manifest.get("lookback", 12)))
    calendar_covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in data.columns)
    skus = sorted(data["sku"].unique())
    train = _observed(data[data["split"] == "train"])
    val = _observed(data[data["split"] == "validation"])
    test = _observed(data[data["split"] == "test"])
    if train.empty or test.empty:
        raise ValueError("XGBoost needs observed train and test rows")
    x_train = _tabular_features(train, skus, scalers, lookback, with_price, calendar_covariates)
    y_train = np.log1p(train["qty"].to_numpy(dtype=np.float32))
    x_val = _tabular_features(val, skus, scalers, lookback, with_price, calendar_covariates) if not val.empty else None
    x_test = _tabular_features(test, skus, scalers, lookback, with_price, calendar_covariates)
    candidates = [(3, 150), (5, 150), (3, 300), (5, 300)] if not val.empty else [(3, 150)]
    best = None
    best_rmse = float("inf")
    trials = []
    for depth, trees in candidates:
        model = XGBRegressor(
            n_estimators=trees, max_depth=depth, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.9,
            objective="reg:squarederror", tree_method="hist", random_state=seed,
            n_jobs=2, verbosity=0,
        )
        model.fit(x_train, y_train)
        if val.empty:
            best = (model, depth, trees)
            trials.append({"max_depth": depth, "n_estimators": trees,
                           "validation_rmse": None})
            break
        pred = np.clip(np.expm1(model.predict(x_val)), 0, None)
        rmse = _score(val, pred)["rmse"]
        trials.append({"max_depth": depth, "n_estimators": trees,
                       "validation_rmse": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best = (model, depth, trees)
    assert best is not None
    tuned_model, depth, trees = best
    val_pred = np.clip(np.expm1(tuned_model.predict(x_val)), 0, None) if not val.empty else np.empty(0)

    if not val.empty:
        combined = pd.concat([train, val], ignore_index=True)
        x_fit = _tabular_features(combined, skus, scalers, lookback, with_price, calendar_covariates)
        y_fit = np.log1p(combined["qty"].to_numpy(dtype=np.float32))
        final_model = XGBRegressor(
            n_estimators=trees, max_depth=depth, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.9,
            objective="reg:squarederror", tree_method="hist", random_state=seed,
            n_jobs=2, verbosity=0,
        )
        final_model.fit(x_fit, y_fit)
    else:
        final_model = tuned_model
    test_pred = np.clip(np.expm1(final_model.predict(x_test)), 0, None)
    name = "xgboost_price" if with_price else "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_dir / f"{name}.json")
    return {"validation": (val, val_pred), "test": (test, test_pred)}, {
        "max_depth": depth, "n_estimators": trees, "validation_rmse": best_rmse if not val.empty else None,
        "validation_trials": trials,
        "candidate_grid": [{"max_depth": d, "n_estimators": n} for d, n in candidates],
        "fit_scope_for_test": "train+validation" if not val.empty else "train",
        "fixed_hyperparameters": {"learning_rate": 0.05, "min_child_weight": 3,
                                  "subsample": 0.8, "colsample_bytree": 0.9,
                                  "objective": "reg:squarederror", "tree_method": "hist"},
        "with_price": with_price,
        "calendar_covariates": list(calendar_covariates),
    }


def run_forecast_benchmark(
    windows_df: pd.DataFrame,
    output_dir: str | Path,
    manifest: Mapping,
    models: Sequence[str] | None = None,
    seed: int = 42,
    torch_epochs: int = 25,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the selected algorithms and save directly comparable test outputs.

    `manifest` is returned by `experiments.data.load_experiment_data` and must
    hold train-only per-SKU scalers.  Without a validation split, models use fixed
    settings and no test rows are used to select hyperparameters.
    """
    if not isinstance(manifest, Mapping) or "scalers" not in manifest:
        raise ValueError("A data manifest with train-only SKU scalers is required")
    requested = tuple(models or DEFAULT_MODELS)
    unknown = set(requested) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unknown forecast algorithms: {sorted(unknown)}")
    data = _validate_windows(windows_df, int(manifest.get("lookback_weeks", manifest.get("lookback", 12))))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    pred_parts = []
    metric_rows = []
    details = {}
    errors = {}
    for name in requested:
        started = time.perf_counter()
        try:
            if name == "naive_last":
                outputs = _naive_predictions(data)
                info = {"description": "previous week's observed-or-zero units"}
            elif name == "arima":
                outputs, arima_info = _arima_predictions(data)
                info = {"order": [1, 0, 1], "trend": "constant",
                        "updated_with_past_observations": True, **arima_info}
            elif name in {"xgboost", "xgboost_price"}:
                outputs, info = _xgb_predictions(data, manifest, name.endswith("_price"), seed, model_dir)
            else:
                try:
                    from .forecast_torch import save_sequence_model, train_sequence_model
                except ImportError as exc:
                    raise RuntimeError("PyTorch is required for LSTM and Transformer benchmarks") from exc
                architecture = "lstm" if name.startswith("lstm") else "transformer"
                with_price = name.endswith("_price")
                model, val_pred, test_pred, info, rows, train_only_model = train_sequence_model(
                    data, manifest, architecture, with_price, seed=seed, max_epochs=torch_epochs,
                )
                save_sequence_model(model, info, model_dir, name)
                if name == "transformer_price":
                    train_only_info = dict(info)
                    train_only_info["checkpoint_scope"] = "train_only"
                    train_only_info.pop("final_training", None)
                    train_only_info["training_rows"] = int(
                        ((data["split"] == "train") & data["observed_sales"].astype(bool)).sum()
                    )
                    save_sequence_model(
                        train_only_model, train_only_info, model_dir,
                        "transformer_price_train_only",
                    )
                outputs = {
                    "validation": (rows["val"], val_pred),
                    "test": (rows["test"], test_pred),
                }
            for split in ("validation", "test"):
                rows, forecast = outputs[split]
                if rows.empty:
                    continue
                pred_parts.append(_prediction_rows(name, split, rows, forecast))
                metric_rows.append({"model": name, "split": split, **_score(rows, forecast)})
            details[name] = {**info, "runtime_seconds": time.perf_counter() - started}
            print(f"[forecast] {name}: complete")
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"[forecast] {name}: FAILED - {errors[name]}")
            if strict:
                raise
    if not pred_parts:
        raise RuntimeError(f"No forecast method completed; errors: {errors}")
    predictions = pd.concat(pred_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["split", "rmse", "model"]).reset_index(drop=True)
    predictions = predictions.sort_values(["split", "model", "sku", "week"]).reset_index(drop=True)
    expected = _observed(data[data["split"] == "test"])[["sku", "week"]].sort_values(["sku", "week"]).reset_index(drop=True)
    for name in details:
        actual = predictions[(predictions["split"] == "test") & (predictions["model"] == name)][["sku", "week"]].reset_index(drop=True)
        if not actual.equals(expected):
            raise AssertionError(f"{name} was not evaluated on the common observed test rows")
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    predictions.to_csv(out / "forecast_predictions.csv", index=False)
    by_sku = []
    for (model, split, sku), group in predictions.groupby(["model", "split", "sku"], sort=True):
        by_sku.append({
            "model": model, "split": split, "sku": sku,
            "tier": str(group["tier"].iloc[0]),
            **_score(group.rename(columns={"y_true": "qty"}), group["y_pred"].to_numpy()),
        })
    pd.DataFrame(by_sku).to_csv(out / "forecast_metrics_by_sku.csv", index=False)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest.get("dataset"),
        "prepared_manifest_sha256": manifest.get("prepared_manifest_sha256"),
        "prepared_windows_sha256": manifest.get("prepared_windows_sha256"),
        "weekly_source": manifest.get("weekly_source"),
        "weekly_source_sha256": manifest.get("weekly_source_sha256"),
        "sales_source": manifest.get("sales_source"),
        "price_source": manifest.get("price_source"),
        "calendar_source": manifest.get("calendar_source"),
        "selection_source": manifest.get("selection_source"),
        "selection_source_sha256": manifest.get("selection_source_sha256"),
        "selection_scope_note": manifest.get("selection_scope_note"),
        "data_split_profile": manifest.get("split_profile"),
        "split_weeks": manifest.get("split_weeks"),
        "n_skus": manifest.get("n_skus", manifest.get("n_series")),
        "calendar_covariates": [c for c in KNOWN_CALENDAR_COVARIATES if c in data.columns],
        "lookback": int(manifest.get("lookback_weeks", manifest.get("lookback", 12))),
        "target_train_transform": "log1p(qty) for XGBoost/LSTM/Transformer; raw qty for ARIMA and naive",
        "metric_scale": "observed units per SKU-week",
        "mape_rule": "exclude rows with zero actual quantity",
        "selection_policy": (
            "XGBoost grid and deep-model early stopping on chronological validation; "
            "refit on train+validation before test"
            if (data["split"] == "validation").any()
            else "Fixed XGBoost and deep-model settings on train; no tuning on test"
        ),
        "seed": int(seed),
        "torch_epochs_max": int(torch_epochs),
        "requested_models": requested,
        "completed_models": list(details),
        "models": details,
        "errors": errors,
        "test_observed_rows": len(expected),
    }
    with (out / "forecast_run.json").open("w", encoding="utf-8") as stream:
        json.dump(run, stream, ensure_ascii=False, indent=2, default=str)
    return metrics, predictions
