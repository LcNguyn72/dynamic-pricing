"""Select forecasting configurations on the Phase 4 V2 validation cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from .data import make_windows
from .forecast_benchmark import KNOWN_CALENDAR_COVARIATES, _arima_phase, _score, _tabular_features
from .forecast_torch import (
    SequenceForecaster, _fit_epochs, _predict_array, feature_arrays as sequence_arrays,
    save_sequence_model,
)
from .phase1_finalize import _observed
from .phase2_song import (
    M5_COLUMNS, PHASE1_ROOT, PROTOCOL, OUTPUT_ROOT as PHASE2_ROOT,
    _cache_data, _fit, _read_json, _uci_windows, _verify, _write_json,
)
from .song_forecast import LOOKBACK, SongTransformer, _scalers, _sha256, feature_arrays as song_arrays

OUTPUT_ROOT = PHASE1_ROOT.with_name("phase4_v2")
UCI_SERIES = 120
SEED = 42
EPOCHS = 3
XGB_GRID = ((3, 150), (5, 150), (3, 300), (5, 300))
SEQUENCE_RATES = (1e-3, 5e-4)
KEYS = ["sku", "week"]


def only_validation(frame: pd.DataFrame) -> None:
    if not set(frame["split"]).issubset({"train", "validation"}):
        raise ValueError("Phase 4 accepts only train and validation rows")


def checked_protocol(root: Path) -> tuple[dict, dict]:
    phase1 = _read_json(root / "phase1_manifest.json")
    _verify(PROTOCOL, phase1["protocol_sha256"])
    protocol = _read_json(PROTOCOL)
    if phase1["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Phase 1 manifest is from a different protocol")
    return phase1, protocol


def checked_m5_checkpoint(root: Path, phase2: Path) -> tuple[dict, dict]:
    run = _read_json(phase2 / "run.json")
    checkpoint = phase2 / "song_transformer.pt"
    _verify(checkpoint, run["checkpoint_sha256"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["dataset"] != "m5" or saved["fit_scope"] != "train_only"
            or saved["phase1_manifest_sha256"] != _sha256(root / "phase1_manifest.json")):
        raise ValueError("Phase 4 requires the verified M5 V2 train-only checkpoint")
    return saved, run


def calendar_features(rows: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(rows["week"])
    return rows.assign(month=dates.dt.month, week_of_year=dates.dt.isocalendar().week.astype(int))


def _uci_cohort(root: Path, report: dict, config: dict):
    windows, scalers, _ = _uci_windows(root / "uci" / "panel", report)
    ranked = sorted(scalers, key=lambda sku: (
        -scalers[sku]["qty_mean"] * scalers[sku]["n_observed_train"], sku))
    chosen = []
    for band in np.array_split(np.asarray(ranked), 3):
        indices = np.linspace(0, len(band) - 1, min(len(band), UCI_SERIES // 3), dtype=int)
        chosen.extend(str(band[index]) for index in indices)
    windows = windows.loc[windows["sku"].isin(chosen)]

    weekly_path = root / "uci" / "weekly.csv"
    _verify(weekly_path, report["weekly_sha256"])
    observed = pd.read_csv(weekly_path, usecols=[*KEYS, "qty"], dtype={"sku": str}, parse_dates=["week"])
    last = pd.Timestamp(config["validation_end"])
    observed = observed.loc[observed["sku"].isin(chosen) & observed["week"].le(last)]
    dates = pd.date_range(end=last, periods=config["train_weeks"] + config["validation_weeks"], freq="7D")
    grid = pd.MultiIndex.from_product([chosen, dates], names=KEYS).to_frame(index=False)
    weekly = grid.merge(observed, on=KEYS, how="left", validate="one_to_one")
    weekly["observed_sales"] = weekly["qty"].notna()
    weekly["qty"] = weekly["qty"].fillna(0.0)
    weekly["split"] = np.where(weekly["week"].le(pd.Timestamp(config["train_end"])), "train", "validation")
    return weekly, windows, {sku: scalers[sku] for sku in chosen}


def _m5_cohort(root: Path, report: dict, saved: dict, run: dict):
    scalers = saved["scalers"]
    if len(scalers) != 120 or len(run["shards"]) != 10:
        raise ValueError("Phase 4 expects the 120-series Phase 2 M5 screening cohort")
    parts = []
    for store in report["stores"]:
        name = store["store"]
        path = root / "m5" / f"weekly_{name}.csv"
        _verify(path, store["weekly_sha256"])
        panel = pd.read_csv(path, usecols=M5_COLUMNS, dtype={"sku": str}, parse_dates=["week"])
        parts.append(panel.loc[panel["sku"].isin(scalers) & panel["split"].isin(("train", "validation"))])
    weekly = pd.concat(parts, ignore_index=True)
    if set(weekly["sku"]) != set(scalers):
        raise ValueError("Phase 2 M5 cohort and Phase 4 data differ")
    windows = make_windows(weekly, LOOKBACK)
    windows = windows.loc[_observed(windows)]
    if windows.groupby("split").size().to_dict() != run["windows"]:
        raise ValueError("Phase 2 M5 training and validation windows do not match")
    return weekly, windows, scalers


def arima_validation(weekly: pd.DataFrame, validation: pd.DataFrame) -> tuple[np.ndarray, dict]:
    only_validation(weekly)
    if set(validation["split"]) != {"validation"}:
        raise ValueError("ARIMA scoring requires validation rows")
    estimates = []
    for sku, group in weekly.sort_values(KEYS).groupby("sku", sort=False):
        known = group["qty"].where(_observed(group)).ffill().shift(1)
        future = group["split"].eq("validation")
        rows = group.loc[future, KEYS].copy()
        if rows.empty:
            continue
        fallback = known.loc[future].to_numpy(dtype=float)
        try:
            _, prediction, _ = _arima_phase(group, {"train"}, "validation", strict_fit=True)
            used_fallback = ~np.isfinite(prediction)
            rows["prediction"] = np.where(used_fallback, fallback, prediction)
            rows["fallback"] = used_fallback
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            rows["prediction"] = fallback
            rows["fallback"] = True
        estimates.append(rows)
    aligned = validation[KEYS].merge(pd.concat(estimates, ignore_index=True), on=KEYS,
                                     how="left", validate="one_to_one")
    if not np.isfinite(aligned["prediction"]).all():
        raise ValueError("ARIMA validation lacks earlier observed demand for a fallback")
    return aligned["prediction"].to_numpy(), {
        "rows": int(aligned["fallback"].sum()),
        "skus": int(aligned.loc[aligned["fallback"], "sku"].nunique()),
    }


def metrics(validation: pd.DataFrame, predictions: np.ndarray) -> dict:
    scores = _score(validation, predictions)
    actual = validation["qty"].to_numpy(dtype=float)
    sst = float(np.square(actual - actual.mean()).sum())
    return {**scores, "r2": 1.0 - scores["rmse"] ** 2 * len(actual) / sst if sst > 0 else None}


def _xgb(train, val, scalers, skus, covariates, price, out: Path, name: str):
    x_train = _tabular_features(train, skus, scalers, LOOKBACK, price, covariates)
    x_val = _tabular_features(val, skus, scalers, LOOKBACK, price, covariates)
    label = np.log1p(train["qty"].to_numpy(dtype=np.float32))
    fixed = dict(learning_rate=0.05, min_child_weight=3, subsample=0.8, colsample_bytree=0.9,
                 objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=2)
    trials, best = [], None
    for depth, trees in XGB_GRID:
        started = perf_counter()
        params = dict(max_depth=depth, n_estimators=trees, **fixed)
        model = XGBRegressor(**params).fit(x_train, label)
        pred = np.expm1(model.predict(x_val)).clip(0)
        score = metrics(val, pred)
        trials.append({"model": name, "max_depth": depth, "n_estimators": trees,
                       "seconds": perf_counter() - started, **score})
        if best is None or score["rmse"] < best[0]:
            best = score["rmse"], pred, model, params
    _, pred, model, params = best
    model.save_model(out / f"{name}.json")
    info = {"parameters": params, "calendar_covariates": list(covariates),
            "with_price": price, "skus": skus, "scalers": scalers, "fit_scope": "train_only"}
    _write_json(out / f"{name}_meta.json", info)
    return pred, trials, {"parameters": params}


def _sequence(train, val, scalers, skus, covariates, name: str, out: Path):
    price = name.endswith("_price")
    mapping = {sku: i for i, sku in enumerate(skus)}
    th, ta = sequence_arrays(train, mapping, scalers, LOOKBACK, price, covariates)
    vh, va = sequence_arrays(val, mapping, scalers, LOOKBACK, price, covariates)
    ty = np.log1p(train["qty"].to_numpy(dtype=np.float32))
    vy = np.log1p(val["qty"].to_numpy(dtype=np.float32))
    architecture = name.split("_")[0]
    config = dict(architecture=architecture, lookback=LOOKBACK, history_dim=th.shape[2],
                  auxiliary_dim=ta.shape[1])
    trials, best = [], None
    for lr in SEQUENCE_RATES:
        started = perf_counter()
        torch.manual_seed(SEED)
        model, training = _fit_epochs(SequenceForecaster(**config), th, ta, ty,
                                      EPOCHS, 64, lr, SEED, validation=(vh, va, vy))
        pred = np.expm1(_predict_array(model, vh, va, batch_size=512).numpy()).clip(0)
        stats = metrics(val, pred)
        trial = {"model": name, "learning_rate": lr, "best_epoch": training["best_epoch"],
                 "seconds": perf_counter() - started, **stats}
        trials.append(trial)
        if best is None or stats["rmse"] < best[0]:
            best = stats["rmse"], pred, model, lr, training
    _, prediction, model, lr, training = best
    meta = {"config": config, "sku_to_idx": mapping, "scalers": scalers, "sku_encoding": "onehot",
            "calendar_covariates": list(covariates), "with_price": price, "seed": SEED,
            "batch_size": 64, "learning_rate": lr, "fit_scope": "train_only", "target": "log1p(quantity)",
            "validation_training": training}
    save_sequence_model(model, meta, out, name)
    return prediction, trials, {
        "model_config": config, "learning_rate": lr, "batch_size": 64,
        "epoch_cap": EPOCHS, "best_epoch": training["best_epoch"],
    }


def _song(windows, val, scalers, out: Path, saved, phase2_run):
    if saved is None:
        with TemporaryDirectory(dir=out) as tmp:
            cache = Path(tmp)
            shards, _, _ = _cache_data(cache, [(windows, scalers, {})])
            state, _ = _fit(cache, shards, SEED, EPOCHS, 64, torch.device("cpu"))
        torch.save({**state, "scalers": scalers, "fit_scope": "train_only"}, out / "song_transformer.pt")
    else:
        state = saved
    model = SongTransformer().eval()
    model.load_state_dict(state["state_dict"])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(val), 256):
            part = val.iloc[start:start + 256]
            h, c = song_arrays(part, scalers)
            mean, std, _, _ = _scalers(part, scalers)
            pred = model(torch.from_numpy(h), torch.from_numpy(c)).numpy()
            predictions.extend(np.maximum(0, pred * std + mean))
    return np.asarray(predictions), {
        "model": "SongTransformer", "layers": 4, "heads": 8, "width": 128,
        "learning_rate": 1e-3,
        "batch_size": phase2_run["batch_size"] if saved is not None else 64,
        "epoch_cap": phase2_run["epoch_cap"] if saved is not None else EPOCHS,
        "best_epoch": state["best_epoch"],
        "reused_phase2_checkpoint_sha256": phase2_run["checkpoint_sha256"] if saved is not None else None,
    }


def run_forecast(dataset: str, root: Path, phase2: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    phase1, protocol = checked_protocol(root)
    saved, phase2_run = None, None
    if dataset == "uci":
        weekly, windows, scalers = _uci_cohort(root, phase1["uci"], protocol["datasets"]["uci_online_retail"])
    else:
        saved, phase2_run = checked_m5_checkpoint(root, phase2)
        if saved["seed"] != SEED:
            raise ValueError("M5 Phase 2 checkpoint seed differs from the screening seed")
        weekly, windows, scalers = _m5_cohort(root, phase1["m5"], saved, phase2_run)
    only_validation(weekly)
    only_validation(windows)
    windows = calendar_features(windows).sort_values(KEYS).reset_index(drop=True)
    skus = sorted(scalers)
    train = windows.loc[windows["split"].eq("train")]
    val = windows.loc[windows["split"].eq("validation")].reset_index(drop=True)
    if train.empty or val.empty or val.duplicated(KEYS).any():
        raise ValueError("Phase 4 needs unique validation keys and observed train windows")
    covariates = tuple(c for c in KNOWN_CALENDAR_COVARIATES if c in windows.columns)
    print(f"[Phase 4] {dataset}: {len(skus)} series, {len(train)} train and {len(val)} validation windows", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase4-forecast-", dir=output.parent) as temp:
        out = Path(temp) / "run"
        models = out / "models"
        models.mkdir(parents=True)
        summaries, all_trials, all_predictions, settings = [], [], [], {}
        names = ("naive_last", "arima_sparse_fallback", "xgboost", "xgboost_price",
                 "lstm", "lstm_price", "transformer", "transformer_price", "song_transformer")
        for name in names:
            started = perf_counter()
            trials = []
            if name == "naive_last":
                pred, params = val["qty_lag_1"].to_numpy(), {"rule": "previous panel week"}
            elif name == "arima_sparse_fallback":
                pred, fallback = arima_validation(weekly, val)
                params = {"order": [1, 0, 1], "minimum_observed_train": 12,
                          "require_convergence": True, "fallback": fallback}
            elif name.startswith("xgboost"):
                pred, trials, params = _xgb(train, val, scalers, skus, covariates,
                                           name.endswith("_price"), models, name)
            elif name == "song_transformer":
                pred, params = _song(windows, val, scalers, models, saved, phase2_run)
            else:
                pred, trials, params = _sequence(train, val, scalers, skus, covariates, name, models)
            scores = metrics(val, pred)
            summaries.append({"dataset": dataset, "model": name, "split": "validation",
                              "seconds": perf_counter() - started, **scores})
            all_trials.extend(trials)
            settings[name] = params
            all_predictions.append(val[[*KEYS, "qty", "price"]].assign(model=name, predicted_qty=pred))
            print(f"[Phase 4] {dataset} {name}: RMSE {scores['rmse']:.5f}", flush=True)
        board = pd.DataFrame(summaries).sort_values(["rmse", "model"])
        board.to_csv(out / "forecast_validation.csv", index=False)
        val[KEYS].to_csv(out / "validation_keys.csv", index=False)
        pd.DataFrame(all_trials).to_csv(out / "forecast_trials.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(out / "forecast_predictions.csv", index=False)
        result = {
            "dataset": dataset, "split": "validation", "fit_scope": "train_only", "seed": SEED,
            "protocol_id": phase1["protocol_id"], "phase1_manifest_sha256": _sha256(root / "phase1_manifest.json"),
            "scope": ("120 Phase 2 M5 series" if dataset == "m5" else
                      "40 UCI series sampled evenly by rank within each train-volume tercile"),
            "skus": skus, "train_rows": len(train), "validation_rows": len(val),
            "train_window_skus": int(train["sku"].nunique()),
            "validation_window_skus": int(val["sku"].nunique()),
            "validation_keys_sha256": _sha256(out / "validation_keys.csv"),
            "same_validation_sku_week_keys": True, "settings": settings,
            "best_by_validation_rmse": str(board.iloc[0]["model"]),
            "deferred_models": {"TFT": "no V2 implementation in this screening budget",
                                "Informer": "no V2 implementation in this screening budget"},
            "xgboost_price_model_sha256": _sha256(models / "xgboost_price.json"),
            "xgboost_price_metadata_sha256": _sha256(models / "xgboost_price_meta.json"),
        }
        _write_json(out / "forecast_run.json", result)
        out.rename(output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "m5"), required=True)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_ROOT / "m5_seed42")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_ROOT / f"forecast_{args.dataset}"
    result = run_forecast(args.dataset, args.phase1_root, args.phase2_dir, output)
    print(f"[Phase 4] forecast {args.dataset}: {result['best_by_validation_rmse']} selected for shortlist", flush=True)


if __name__ == "__main__":
    main()
