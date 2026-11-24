"""Run the shared forecast benchmark over all prebuilt M5 store panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import make_windows
from .forecast_benchmark import DEFAULT_MODELS, run_forecast_benchmark


def _accumulate(predictions: pd.DataFrame, totals: dict) -> None:
    for (model, split), group in predictions.groupby(["model", "split"], sort=True):
        actual = group["y_true"].to_numpy(dtype=np.float64)
        error = group["y_pred"].to_numpy(dtype=np.float64) - actual
        nonzero = actual > 0
        key = (str(model), str(split))
        row = totals.setdefault(key, {
            "n": 0, "mape_n": 0, "sum_squared_error": 0.0,
            "sum_absolute_error": 0.0, "sum_actual": 0.0,
            "sum_absolute_percentage_error": 0.0,
        })
        row["n"] += int(len(actual))
        row["mape_n"] += int(nonzero.sum())
        row["sum_squared_error"] += float(np.square(error).sum())
        row["sum_absolute_error"] += float(np.abs(error).sum())
        row["sum_actual"] += float(actual.sum())
        if nonzero.any():
            row["sum_absolute_percentage_error"] += float(
                (np.abs(error[nonzero]) / actual[nonzero]).sum()
            )


def _metric_frame(totals: dict) -> pd.DataFrame:
    rows = []
    for (model, split), sums in sorted(totals.items()):
        n = sums["n"]
        mape_n = sums["mape_n"]
        rows.append({
            "model": model,
            "split": split,
            "n": n,
            "rmse": math.sqrt(sums["sum_squared_error"] / n) if n else None,
            "mae": sums["sum_absolute_error"] / n if n else None,
            "wape_pct": (100 * sums["sum_absolute_error"] / sums["sum_actual"]
                         if sums["sum_actual"] else None),
            "mape_pct": (100 * sums["sum_absolute_percentage_error"] / mape_n
                         if mape_n else None),
            "mape_n": mape_n,
        })
    return pd.DataFrame(rows).sort_values(["split", "rmse", "model"]).reset_index(drop=True)


def _train_only_scalers(weekly: pd.DataFrame) -> dict:
    train = weekly.loc[weekly["split"].eq("train") & weekly["observed_sales"].astype(bool)]
    scalers = {}
    for sku, group in train.groupby("sku", sort=False):
        qty_mean = float(group["qty"].mean())
        price_mean = float(group["price"].mean())
        qty_std = float(group["qty"].std(ddof=0))
        price_std = float(group["price"].std(ddof=0))
        qty_floor = max(1e-8, abs(qty_mean) * 1e-8)
        price_floor = max(1e-8, abs(price_mean) * 1e-8)
        scalers[str(sku)] = {
            "qty_mean": qty_mean,
            "qty_std": qty_std if np.isfinite(qty_std) and qty_std > qty_floor else 1.0,
            "price_mean": price_mean,
            "price_std": price_std if np.isfinite(price_std) and price_std > price_floor else 1.0,
            "n_observed_train": int(len(group)),
        }
    return scalers


def run(
    prepared_root: str | Path,
    output_root: str | Path,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seed: int = 42,
    torch_epochs: int = 25,
    stores: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    prepared_root, output_root = Path(prepared_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    weekly_paths = sorted(prepared_root.glob("weekly_*.csv"))
    if stores:
        store_filter = set(stores)
        weekly_paths = [path for path in weekly_paths if path.stem.removeprefix("weekly_") in store_filter]
    if not weekly_paths:
        raise FileNotFoundError(f"No M5 weekly_<store>.csv panels found in {prepared_root}")

    totals: dict = {}
    for weekly_path in weekly_paths:
        store = weekly_path.stem.removeprefix("weekly_")
        manifest_path = prepared_root / f"manifest_{store}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing store manifest: {manifest_path}")
        store_out = output_root / store
        run_path = store_out / "forecast_run.json"
        predictions_path = store_out / "forecast_predictions.csv"
        requested = list(models)

        reusable = False
        if run_path.is_file() and predictions_path.is_file():
            prior_run = json.loads(run_path.read_text(encoding="utf-8"))
            reusable = (prior_run.get("requested_models") == requested
                        and prior_run.get("completed_models") == requested
                        and prior_run.get("seed") == seed
                        and prior_run.get("torch_epochs_max") == torch_epochs)
        print(f"[full-m5] {store}: {'reusing completed output' if reusable else 'loading panel'}", flush=True)
        if reusable:
            for chunk in pd.read_csv(predictions_path, chunksize=200_000):
                _accumulate(chunk, totals)
        else:
            weekly = pd.read_csv(
                weekly_path, dtype={"sku": str}, parse_dates=["week"],
            )
            windows = make_windows(weekly, lookback=12, forecast_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["lookback_weeks"] = 12
            if "scalers" not in manifest:
                manifest["scalers"] = _train_only_scalers(weekly)
                manifest["scaler_fit_policy"] = "Reconstructed per-SKU population scaler from observed train weeks only"
            print(f"[full-m5] {store}: {len(windows):,} windows; running {len(models)} models", flush=True)
            _, predictions = run_forecast_benchmark(
                windows, store_out, manifest, models=models, seed=seed,
                torch_epochs=torch_epochs, strict=True,
            )
            _accumulate(predictions, totals)
            del predictions, windows, weekly
        summary = _metric_frame(totals)
        summary.to_csv(output_root / "forecast_metrics_full_scope.csv", index=False)
        print(f"[full-m5] {store}: complete", flush=True)

    summary = _metric_frame(totals)
    summary.to_csv(output_root / "forecast_metrics_full_scope.csv", index=False)
    details = {
        "dataset": "M5 Forecasting Accuracy",
        "stores": [path.stem.removeprefix("weekly_") for path in weekly_paths],
        "models": list(models),
        "seed": seed,
        "torch_epochs_max": torch_epochs,
        "split_profile": "paper_70_15_15",
        "metric_scale": "observed units per SKU-week, aggregated across stores",
        "mape_rule": "exclude rows with zero actual quantity",
        "store_output_dirs": [str(output_root / path.stem.removeprefix("weekly_")
                                   ) for path in weekly_paths],
        "summary_file": str(output_root / "forecast_metrics_full_scope.csv"),
    }
    (output_root / "full_m5_run.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-epochs", type=int, default=25)
    parser.add_argument("--stores", nargs="+")
    args = parser.parse_args()
    result = run(
        args.prepared_root, args.output_root, models=tuple(args.models),
        seed=args.seed, torch_epochs=args.torch_epochs,
        stores=tuple(args.stores) if args.stores else None,
    )
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
