"""Compare train-only ARIMAX with the UCI rolling-mean baseline on frozen V2 keys."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import statsmodels

from final_system.bundle import DEFAULT_PREPARED_ROOT, atomic_write, file_sha256
from final_system.engine import _price_point
from final_system.pipeline.estimators import ARIMAX, MAXITER, MIN_PAIRS, _limit_threads, fit_series
from final_system.schemas import WeekObservation

from .phase5_data import iter_weekly, load_contract, verify_shard_test_keys, windows_from_weekly
from .phase5_verify import sufficient, summarize


OUTPUT = Path(__file__).resolve().parent / "artifacts" / "uci_arimax_v2"
PRICE_TOLERANCE = 1e-8


def _fit(task):
    sku, qty, prices = task
    state, status = fit_series(qty, prices, ARIMAX)
    return sku, state, status


def _candidate(row, state):
    if state is None:
        return None, "no_train_fit"
    if not row.previous_observed:
        return None, "missing_observed_lag"
    low, high = state["price_min"], state["price_max"]
    if not all(low - PRICE_TOLERANCE <= price <= high + PRICE_TOLERANCE
               for price in (row.price, row.previous_price)):
        return None, "price_outside_train_support"
    history = WeekObservation(week=(row.week - pd.Timedelta(days=7)).date(),
                              qty=row.previous_qty, price=row.previous_price)
    return _price_point(state, history, row.price)["predicted_qty"], "arimax"


def _score(split, scope, model, rows, column, train_mean, train_std):
    metrics = summarize(sufficient(rows.assign(predicted_qty=rows[column])).sum(),
                        train_mean=train_mean, train_std=train_std)
    return {"split": split, "scope": scope, "model": model, "sku_count": rows["sku"].nunique(), **metrics}


def run(root, output, workers):
    started = time.monotonic()
    contract = load_contract(root)
    shard, panel, _ = next(iter_weekly(root, "uci", contract))
    panel = panel.sort_values(["sku", "week"]).reset_index(drop=True)
    windows = windows_from_weekly(panel)
    keys = verify_shard_test_keys(root, "uci", contract, shard, windows)
    observed_train = panel["split"].eq("train") & panel["observed_sales"]
    mean = float(panel.loc[observed_train, "qty"].mean())
    std = float(panel.loc[observed_train, "qty"].std(ddof=0))
    train = panel.loc[panel["split"].eq("train")].copy()
    train["qty"] = train["qty"].where(train["observed_sales"])
    tasks = ((str(sku), rows["qty"].to_numpy(dtype=float), rows["price"].to_numpy(dtype=float))
             for sku, rows in train.groupby("sku", sort=False))
    print(f"Fitting UCI ARIMAX on the 37-week train calendar for {train['sku'].nunique():,} eligible SKU", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_threads) as pool:
        results = list(pool.map(_fit, tasks, chunksize=24))
    states = {sku: state for sku, state, _ in results if state is not None}
    print(f"Fit complete: {len(states):,} ARIMAX parameters", flush=True)

    prior = panel.groupby("sku", sort=False)[["observed_sales", "qty", "price"]].shift(1)
    prior.columns = ["previous_observed", "previous_qty", "previous_price"]
    prior = panel[["sku", "week"]].join(prior)
    rows = windows.loc[windows["observed_sales"] & windows["split"].isin(("validation", "test"))].copy()
    rows["mean4"] = rows[[f"qty_lag_{lag}" for lag in range(1, 5)]].mean(axis=1)
    rows = rows[["sku", "week", "split", "qty", "price", "mean4"]].merge(prior, on=["sku", "week"],
                                                                         validate="one_to_one")
    values = [_candidate(row, states.get(row.sku)) for row in rows.itertuples(index=False)]
    rows[["arimax", "status"]] = pd.DataFrame(values, index=rows.index)
    rows["arimax_or_mean4"] = rows["arimax"].fillna(rows["mean4"])
    scores, support = [], {}
    for split, subset in rows.groupby("split", sort=False):
        overlap = subset.loc[subset["status"].eq("arimax")]
        comparisons = (("all_eligible_keys", subset, "mean4"),
                       ("all_eligible_keys", subset, "arimax_or_mean4"),
                       ("arimax_overlap", overlap, "mean4"),
                       ("arimax_overlap", overlap, "arimax"))
        scores.extend(_score(split, scope, model, comparison, model, mean, std)
                      for scope, comparison, model in comparisons)
        support[split] = {name: int(count) for name, count in subset["status"].value_counts().items()}

    output.mkdir(parents=True, exist_ok=True)
    files = {"predictions.csv": rows, "scores.csv": pd.DataFrame(scores),
             "fit_status.csv": pd.DataFrame(((sku, status) for sku, _, status in results), columns=["sku", "status"])}
    for name, table in files.items():
        atomic_write(output / name, table.to_csv(index=False).encode())
    report = {"dataset": "uci_online_retail", "protocol": "song_full_scope_v2", "fit_scope": "train_only",
              "phase1_sha256": contract["phase1_sha256"], "test_keys": keys,
              "target_price": "observed historical price, conditional forecast",
              "order": [1, 0, 0], "min_train_pairs": MIN_PAIRS, "maxiter": MAXITER,
              "solver": {"numpy": np.__version__, "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
              "eligible_sku_count": len(results), "train_fit_status": dict(Counter(status for _, _, status in results)),
              "prediction_status": support, "seconds": round(time.monotonic() - started, 2),
              "files": {name: file_sha256(output / name) for name in files}}
    packed = json.dumps(report, indent=2, allow_nan=False).encode()
    atomic_write(output / "run.json", packed + b"\n")
    print(packed.decode(), flush=True)
    print(pd.DataFrame(scores).drop(columns=["rmse_train_mean_relative", "rmse_train_z"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=4)
    args = parser.parse_args()
    run(args.prepared_root, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
