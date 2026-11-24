"""Refit and score one locked Phase 5 forecast model and seed on V2 test keys."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from .phase1_finalize import _observed, _sha256

DEFAULT_PHASE1 = Path(__file__).resolve().parent / "artifacts" / "phase1_v2"
DEFAULT_OUTPUT = DEFAULT_PHASE1.with_name("phase5_v2")
DEFAULT_LOCK = Path(__file__).resolve().parent / "phase5_settings_v2.json"
DETERMINISTIC = {"naive_last", "arima_sparse_fallback"}
FORECAST_VERSION = 1
KEYS = ["sku", "week"]
DATASET_NAMES = {"uci": "uci_online_retail", "m5": "m5"}


def arima_series_predictions(fit: np.ndarray, observed_test: np.ndarray,
                             minimum_observed: int = 12, order: tuple = (1, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    """One-step ARIMA with a frozen refit and the last known quantity fallback."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    history = np.concatenate((fit, observed_test))
    positions = np.where(np.isfinite(history), np.arange(len(history)), -1)
    latest = np.concatenate((np.array([-1]), np.maximum.accumulate(positions)))
    # The forecast at week t can only access the value saved at t-1 or earlier.
    last_known = latest[len(fit):len(history)]
    fallback = np.where(last_known >= 0, history[np.maximum(last_known, 0)], np.nan)
    if np.isfinite(fit).sum() < minimum_observed:
        return fallback, np.ones(len(fallback), dtype=bool)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SARIMAX(fit, order=order, trend="c", enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False, maxiter=80)
            if not result.mle_retvals.get("converged", False):
                return fallback, np.ones(len(fallback), dtype=bool)
            # Kalman extension filters forward once under the frozen parameters.
            # The predicted information set excludes that period's observed value.
            forecast = np.asarray(result.extend(observed_test).get_prediction(
                information_set="predicted").predicted_mean, dtype=float)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return fallback, np.ones(len(fallback), dtype=bool)
    use_fallback = ~np.isfinite(forecast)
    return np.where(use_fallback, fallback, np.maximum(forecast, 0.0)), use_fallback


def _arima_worker(task):
    sku, dates, fit, test, minimum_observed, order = task
    forecast, used = arima_series_predictions(fit, test, minimum_observed, order)
    return sku, dates, forecast, used


def _limit_blas() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


def _arima_shard(weekly: pd.DataFrame, target: pd.DataFrame, settings: dict,
                  pool: ProcessPoolExecutor | None) -> tuple[np.ndarray, dict]:
    target_skus = set(target["sku"])

    def tasks():
        for sku, group in weekly.groupby("sku", sort=False, observed=True):
            if sku not in target_skus:
                continue
            sales = group["qty"].where(_observed(group)).to_numpy(dtype=float)
            test = group["split"].eq("test").to_numpy()
            yield (str(sku), group.loc[test, "week"].to_numpy(), sales[~test], sales[test],
                   settings["minimum_observed_train"], tuple(settings["order"]))

    rows = (map(_arima_worker, tasks()) if pool is None
            else pool.map(_arima_worker, tasks(), chunksize=16))
    pieces = [pd.DataFrame({"sku": sku, "week": dates, "forecast": forecast, "fallback": used})
              for sku, dates, forecast, used in rows]
    alignment = target[KEYS].merge(pd.concat(pieces, ignore_index=True), on=KEYS, how="left",
                                    sort=False, validate="one_to_one")
    pred = alignment["forecast"].to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("ARIMA test forecast lacks an observed earlier quantity for fallback")
    used = alignment["fallback"].astype(bool)
    return pred, {"rows": int(used.sum()), "skus": int(alignment.loc[used, "sku"].nunique())}


def _atomic_json(path: Path, value: dict) -> None:
    working = path.with_suffix(path.suffix + ".tmp")
    working.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                       encoding="utf-8")
    working.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_rows(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.loc[windows["split"].eq("test") & _observed(windows)].sort_values(KEYS)


def _write_shard(path: Path, dataset: str, target: pd.DataFrame, predicted: np.ndarray) -> dict:
    prediction = np.asarray(predicted, dtype=float)
    if prediction.shape != (len(target),) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Forecast output is not one finite nonnegative prediction per test key")
    if target.duplicated(KEYS).any():
        raise ValueError("A forecast shard has repeated SKU-week keys")
    output = target[[*KEYS, "qty", "price"]].copy()
    output.insert(0, "dataset", DATASET_NAMES[dataset])
    output["predicted_qty"] = prediction
    working = path.with_suffix(".csv.tmp")
    output.to_csv(working, index=False, lineterminator="\n", date_format="%Y-%m-%d")
    working.replace(path)
    return {"sha256": _sha256(path), "rows": len(output)}


def run_forecast(dataset: str, model: str, seed: int, phase1_root: Path, lock_path: Path,
                 output: Path, cache_root: Path, *, device: str = "auto", cpu_threads: int = 4,
                 arima_workers: int = 4, cuda_graph: bool = False) -> dict:
    from .phase5_data import iter_weekly, iter_windows, load_contract, windows_from_weekly

    start = perf_counter()
    contract = load_contract(phase1_root, lock_path)
    lock = contract["lock"]
    settings = lock["forecast"][dataset]["settings_by_model"][model]
    allowed = lock["phase5_seeds"][:1] if model in DETERMINISTIC else lock["phase5_seeds"]
    if seed not in allowed:
        raise ValueError(f"{model}: seed {seed} is not a scheduled run seed {allowed}")
    if cpu_threads < 1 or arima_workers < 1:
        raise ValueError("Worker and thread counts must be positive")
    shards = ["uci"] if dataset == "uci" else list(contract["protocol"]["datasets"]["m5"]["store_ids"])
    expected = lock["frozen_test_keys"][dataset]
    execution_device = "cpu"
    if model not in DETERMINISTIC and not model.startswith("xgboost"):
        from .phase5_training import resolve_device
        execution_device = str(resolve_device(device))
    if cuda_graph and (dataset != "m5" or execution_device != "cuda"):
        raise ValueError("--cuda-graph requires a Phase 5 M5 neural model and CUDA")
    identity = {
        "schema_version": FORECAST_VERSION, "dataset": dataset, "model": model, "seed": seed,
        "phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"],
        "protocol_sha256": contract["protocol_sha256"], "test_keys_sha256": expected["sha256"],
        "fit_scope": "train+validation", "preprocessing_fit_scope": "train_only",
        "settings": settings, "device": execution_device,
        "compute_threads": cpu_threads,
    }
    if cuda_graph:
        from .phase5_cuda_graph import MODE
        identity["execution_mode"] = MODE
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.is_file():
        state = _read_json(manifest)
        if (any(state.get(key) != value for key, value in identity.items())
                or state.get("execution_mode") != identity.get("execution_mode")):
            raise ValueError(f"Existing run identity differs from the requested run {output}")
    else:
        state = {**identity, "status": "running", "shards": {}, "seconds": 0.0}
        _atomic_json(manifest, state)
    remaining = []
    for shard in shards:
        path = output / f"predictions_{shard}.csv"
        done = state["shards"].get(shard, {})
        if not path.is_file() or _sha256(path) != done.get("sha256"):
            remaining.append(shard)
    if state["status"] == "complete" and not remaining:
        checkpoint = state.get("checkpoint")
        if checkpoint and _sha256(output / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Completed model checkpoint no longer matches its recorded hash")
        return state

    print(f"[Phase 5 forecast] {dataset} {model} seed={seed} device={execution_device} "
          f"shards_remaining={len(remaining)}/{len(shards)}", flush=True)
    prior_seconds = state["seconds"]
    state["status"] = "running"
    _atomic_json(manifest, state)
    if model not in DETERMINISTIC:
        if cuda_graph:
            from .phase5_cuda_graph import fit_or_load_graph as fit_or_load
        else:
            from .phase5_training import fit_or_load
        predictor, checkpoint = fit_or_load(dataset, model, seed, phase1_root, contract, settings,
                                            output, cache_root, execution_device, cpu_threads)
        state["checkpoint"] = checkpoint

    if remaining:
        if model == "arima_sparse_fallback":
            pool_context = (ProcessPoolExecutor(max_workers=arima_workers, initializer=_limit_blas)
                            if arima_workers > 1 else nullcontext(None))
            with pool_context as pool:
                for shard, weekly, _ in iter_weekly(phase1_root, dataset, contract, shards=remaining,
                                                     cache_root=cache_root):
                    target = _target_rows(windows_from_weekly(weekly))
                    from .phase5_data import verify_shard_test_keys
                    verify_shard_test_keys(phase1_root, dataset, contract, shard, target)
                    predicted, fallback = _arima_shard(weekly, target, settings, pool)
                    path = output / f"predictions_{shard}.csv"
                    state["shards"][shard] = {**_write_shard(path, dataset, target, predicted),
                                              "fallback": fallback}
                    state["seconds"] = prior_seconds + perf_counter() - start
                    _atomic_json(manifest, state)
                    print(f"[Phase 5 forecast] {dataset}/{model}/{seed}: {shard} "
                          f"rows={len(target)} fallback={fallback['rows']}", flush=True)
        else:
            for shard, windows, _ in iter_windows(phase1_root, dataset, contract, shards=remaining,
                                                    cache_root=cache_root):
                target = _target_rows(windows)
                from .phase5_data import verify_shard_test_keys
                verify_shard_test_keys(phase1_root, dataset, contract, shard, target)
                predicted = (target["qty_lag_1"].to_numpy() if model == "naive_last" else predictor(target))
                path = output / f"predictions_{shard}.csv"
                state["shards"][shard] = _write_shard(path, dataset, target, predicted)
                state["seconds"] = prior_seconds + perf_counter() - start
                _atomic_json(manifest, state)
                print(f"[Phase 5 forecast] {dataset}/{model}/{seed}: {shard} rows={len(target)}", flush=True)

    if set(state["shards"]) != set(shards) or sum(row["rows"] for row in state["shards"].values()) != expected["count"]:
        raise ValueError("Forecast coverage does not match the full frozen test cohort")
    state.update(status="complete", seconds=prior_seconds + perf_counter() - start,
                 prediction_files={shard: state["shards"][shard]["sha256"] for shard in shards})
    if model == "arima_sparse_fallback":
        state["fallback"] = {key: sum(row["fallback"][key] for row in state["shards"].values())
                             for key in ("rows", "skus")}
    _atomic_json(manifest, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT / "data_cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-graph", action="store_true", help="Opt in to CUDA Graph + fused Adam for M5 neural fits")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--arima-workers", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int)
    parser.add_argument("--calibration-shard", default="CA_1")
    args = parser.parse_args()
    if args.calibration_batches is not None:
        if args.cuda_graph:
            from .phase5_cuda_graph import calibrate_graph as calibrate_neural
        else:
            from .phase5_training import calibrate_neural
        if args.dataset != "m5":
            parser.error("Calibration is available only for a M5 neural model")
        calibrated = calibrate_neural(args.model, args.seed, args.phase1_root, args.lock_path,
                                      args.cache_dir, args.calibration_shard, args.calibration_batches,
                                      args.device, args.cpu_threads)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(args.output_dir / "calibration.json", calibrated)
        print(json.dumps(calibrated, indent=2), flush=True)
        return
    output = args.output_dir or DEFAULT_OUTPUT / "forecast" / args.dataset / args.model / f"seed{args.seed}"
    run_forecast(args.dataset, args.model, args.seed, args.phase1_root, args.lock_path, output,
                 args.cache_dir, device=args.device, cpu_threads=args.cpu_threads,
                 arima_workers=args.arima_workers, cuda_graph=args.cuda_graph)


if __name__ == "__main__":
    main()
