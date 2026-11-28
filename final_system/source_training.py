"""Train and evaluate an isolated SKU demand model from UCI or Lazada data.

Set ``FINAL_SYSTEM_DATA_SOURCE=uci|lazada`` in the environment or in
``final_system/.env``, then run ``python -m final_system.source_training`` from
the project directory. Every invocation writes a new, source-specific run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import RUNTIME, file_sha256
from .pipeline.lazada import cap_spikes_after_aggregation, load_lazada_panel


PACKAGE_ROOT = Path(__file__).resolve().parent
ENV_FILE = PACKAGE_ROOT / ".env"
PREPARED_UCI = RUNTIME / "prepared" / "uci"
OUTPUT_ROOT = RUNTIME / "source_runs"
SOURCES = ("uci", "lazada")
MIN_TRAIN_ROWS_PER_SKU = 5
TEST_FRACTION = 0.20


def _load_env():
    if not ENV_FILE.is_file():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _load_uci_panel():
    weekly_path = PREPARED_UCI / "weekly.csv"
    manifest_path = PREPARED_UCI / "preprocessing_manifest.json"
    if not weekly_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"UCI panel not found under {PREPARED_UCI}; run "
            "python -m final_system.data_pipeline prepare first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(weekly_path) != manifest.get("weekly_sha256"):
        raise ValueError("Prepared UCI weekly.csv does not match its preprocessing manifest")
    frame = pd.read_csv(weekly_path, dtype={"sku": str}, parse_dates=["week"])
    required = {"sku", "week", "qty", "price"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prepared UCI weekly.csv missing columns: {sorted(required - set(frame))}")
    frame = frame[["sku", "week", "qty", "price"]].copy()
    frame["observed_sales"] = True
    audit = {
        "source": "prepared UCI Online Retail 352",
        "weekly_file": str(weekly_path.resolve()),
        "weekly_sha256": manifest["weekly_sha256"],
        "preprocessing_manifest_sha256": file_sha256(manifest_path),
        "period_days": 7,
        "periods": int(frame["week"].nunique()),
        "skus": int(frame["sku"].nunique()),
    }
    return frame, audit


def load_source_panel(source: str):
    if source == "uci":
        panel, audit = _load_uci_panel()
    elif source == "lazada":
        panel, _, audit = load_lazada_panel()
    else:
        raise ValueError(f"source must be one of {SOURCES}")
    panel = panel[["sku", "week", "qty", "price", "observed_sales"]].copy()
    panel["sku"] = panel["sku"].astype(str)
    panel["week"] = pd.to_datetime(panel["week"], errors="raise")
    panel["qty"] = pd.to_numeric(panel["qty"], errors="coerce")
    panel["price"] = pd.to_numeric(panel["price"], errors="coerce")
    panel = panel.dropna(subset=["sku", "week", "qty", "price"])
    panel = panel.loc[np.isfinite(panel["qty"]) & np.isfinite(panel["price"])]
    panel = panel.loc[panel["qty"].ge(0) & panel["price"].gt(0)]
    if panel.duplicated(["sku", "week"]).any():
        raise ValueError(f"{source}: duplicate SKU-period rows after normalization")
    if panel.empty:
        raise ValueError(f"{source}: no valid SKU-period rows")
    return panel.sort_values(["week", "sku"]).reset_index(drop=True), audit


def _fit_sku(train: pd.DataFrame, period_index: dict[pd.Timestamp, int]):
    usable = train.loc[train["price"].gt(0)].copy()
    if len(usable) < MIN_TRAIN_ROWS_PER_SKU:
        return None
    x = np.column_stack([
        np.ones(len(usable), dtype=float),
        np.log(usable["price"].to_numpy(dtype=float)),
        np.asarray([period_index[pd.Timestamp(d)] for d in usable["week"]], dtype=float),
    ])
    y = np.log1p(usable["qty"].to_numpy(dtype=float))
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1] or not np.isfinite(coefficients).all():
        return None
    return {"intercept": float(coefficients[0]), "log_price": float(coefficients[1]),
            "trend": float(coefficients[2]), "train_rows": int(len(usable))}


def predict_quantity(model: dict, sku: str, price: float, period_index: int) -> float:
    """Predict units for one SKU using a saved source-training model."""
    if price <= 0:
        raise ValueError("price must be positive")
    state = model["models"].get(str(sku))
    if state is None:
        raise KeyError(f"No trained series for SKU {sku}")
    log_qty = (state["intercept"] + state["log_price"] * np.log(price)
               + state["trend"] * period_index)
    return float(max(0.0, np.expm1(np.clip(log_qty, -30, 30))))


def train_and_evaluate(source: str, output_root: Path = OUTPUT_ROOT) -> dict:
    panel, source_audit = load_source_panel(source)
    periods = pd.DatetimeIndex(sorted(panel["week"].unique()))
    n_test = max(1, int(np.ceil(len(periods) * TEST_FRACTION)))
    n_train = len(periods) - n_test
    if n_train < MIN_TRAIN_ROWS_PER_SKU or n_test < 1:
        raise ValueError(
            f"{source}: need at least {MIN_TRAIN_ROWS_PER_SKU + 1} periods; found {len(periods)}"
        )
    train_periods, test_periods = periods[:n_train], periods[n_train:]
    if source == "lazada":
        panel, spike_audit = cap_spikes_after_aggregation(panel, train_periods)
        source_audit["spike_normalization"] = spike_audit
    period_index = {pd.Timestamp(period): index for index, period in enumerate(periods)}
    panel["split"] = np.where(panel["week"].isin(train_periods), "train", "test")

    models = {}
    train = panel.loc[panel["split"].eq("train")]
    for sku, group in train.groupby("sku", sort=True):
        state = _fit_sku(group, period_index)
        if state is not None:
            models[str(sku)] = state
    if not models:
        raise ValueError(f"{source}: no SKU had enough train observations for a model")

    test = panel.loc[panel["split"].eq("test") & panel["sku"].isin(models)].copy()
    if test.empty:
        raise ValueError(f"{source}: no held-out rows belong to a trained SKU")
    test["period_index"] = test["week"].map(period_index).astype(int)
    test["predicted_qty"] = [
        predict_quantity({"models": models}, row.sku, row.price, row.period_index)
        for row in test.itertuples()
    ]
    if "qty_raw" in test:
        test["normalized_qty"] = test["qty"]
        test["error"] = test["predicted_qty"] - test["qty_raw"]
        test["normalized_error"] = test["predicted_qty"] - test["normalized_qty"]
        test["qty"] = test["qty_raw"]
    else:
        test["error"] = test["predicted_qty"] - test["qty"]
    metrics = {
        "source": source,
        "train_periods": int(len(train_periods)),
        "test_periods": int(len(test_periods)),
        "train_end": pd.Timestamp(train_periods[-1]).date().isoformat(),
        "test_start": pd.Timestamp(test_periods[0]).date().isoformat(),
        "eligible_models": int(len(models)),
        "test_rows": int(len(test)),
        "test_skus": int(test["sku"].nunique()),
        "test_mae_qty": float(test["error"].abs().mean()),
        "test_rmse_qty": float(np.sqrt(np.mean(test["error"] ** 2))),
        "test_mae_log1p_qty": float(np.mean(np.abs(
            np.log1p(test["predicted_qty"]) - np.log1p(test["qty"])
        ))),
    }
    if "normalized_error" in test:
        metrics["test_mae_qty_normalized"] = float(test["normalized_error"].abs().mean())
        metrics["test_rmse_qty_normalized"] = float(np.sqrt(np.mean(test["normalized_error"] ** 2)))
        metrics["test_mae_log1p_qty_normalized"] = float(np.mean(np.abs(
            np.log1p(test["predicted_qty"]) - np.log1p(test["normalized_qty"])
        )))
        metrics["lazada_spike_normalization"] = source_audit["spike_normalization"]["method"]

    created = datetime.now(timezone.utc)
    run_id = f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{source}"
    run_dir = Path(output_root) / source / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    panel.to_csv(run_dir / "panel.csv", index=False)
    test.drop(columns=["error", "normalized_error"], errors="ignore").to_csv(
        run_dir / "test_predictions.csv", index=False
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    model = {
        "schema_version": 1,
        "model_type": "per_sku_log1p_price_trend_ols",
        "source": source,
        "period_days": int(source_audit["period_days"]),
        "train_end": metrics["train_end"],
        "test_start": metrics["test_start"],
        "period_index": {pd.Timestamp(key).date().isoformat(): value for key, value in period_index.items()},
        "models": models,
    }
    model_path = run_dir / "model.json"
    model_path.write_text(json.dumps(model, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": created.isoformat(),
        "source": source,
        "source_audit": source_audit,
        "split": {"method": "chronological", "test_fraction": TEST_FRACTION,
                  "train_end": metrics["train_end"], "test_start": metrics["test_start"]},
        "files": {name: file_sha256(run_dir / name)
                  for name in ("panel.csv", "test_predictions.csv", "metrics.json", "model.json")},
        "model_type": model["model_type"],
        "notes": [
            "Train and test use only the selected source; no rows are mixed across sources.",
            "Lazada observations are aggregated into three-day periods; UCI rows remain weekly.",
            "The fitted SKU-level price coefficient is observational and is not a causal estimate.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), "metrics": metrics}, indent=2), flush=True)
    return {"run_dir": run_dir, "metrics": metrics, "manifest": manifest}


def main():
    _load_env()
    default_source = os.getenv("FINAL_SYSTEM_DATA_SOURCE", "uci").strip().lower()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default=default_source,
                        help="Overrides FINAL_SYSTEM_DATA_SOURCE")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.source not in SOURCES:
        parser.error(f"FINAL_SYSTEM_DATA_SOURCE must be one of {SOURCES}")
    train_and_evaluate(args.source, args.output_root)


if __name__ == "__main__":
    main()
