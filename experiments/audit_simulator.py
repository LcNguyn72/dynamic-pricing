"""Evaluate the M5 price simulator at observed validation/test prices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_prepared import audit as audit_panel
from .forecast_benchmark import _score
from .pricing_benchmark import validate_simulator
from .pricing_simulator import KernelDemandSimulator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(prepared_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    prepared_root, output_dir = Path(prepared_root), Path(output_dir)
    panel = audit_panel(prepared_root)
    if not panel["dataset"].startswith("M5"):
        raise ValueError("This simulator audit is for the M5 pricing panel")
    weekly = pd.read_csv(prepared_root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(
        prepared_root / "windows.csv", dtype={"sku": str}, parse_dates=["week"],
        usecols=["sku", "week", "qty_lag_1"],
    )
    simulator = KernelDemandSimulator().fit(weekly)
    checked = validate_simulator(simulator, weekly)
    checked = checked.merge(windows, on=["sku", "week"], how="left", validate="one_to_one")
    if checked["qty_lag_1"].isna().any():
        raise ValueError("A scored simulator row lacks last-week demand")
    train_mean = (weekly.loc[weekly["split"].eq("train") & weekly["observed_sales"]]
                  .groupby("sku")["qty"].mean())
    checked["train_mean_qty"] = checked["sku"].map(train_mean)
    bounds = simulator.diagnostics().set_index("sku")
    checked["train_price_min"] = checked["sku"].map(bounds["price_min"])
    checked["train_price_max"] = checked["sku"].map(bounds["price_max"])
    checked["price_in_train_support"] = (
        checked["price"].ge(checked["train_price_min"] - 1e-8)
        & checked["price"].le(checked["train_price_max"] + 1e-8)
    )
    metrics = []
    for split, split_rows in checked.groupby("split"):
        for segment, rows in (("all", split_rows),
                              ("in_support", split_rows.loc[split_rows["price_in_train_support"]]),
                              ("out_of_support", split_rows.loc[~split_rows["price_in_train_support"]])):
            if rows.empty:
                continue
            for model, column in (("kernel_simulator", "simulator_pred_qty"),
                                  ("naive_last", "qty_lag_1"),
                                  ("train_mean", "train_mean_qty")):
                score = _score(rows, rows[column].to_numpy())
                metrics.append({
                    "split": split, "segment": segment, "model": model,
                    "price_in_train_support_pct": float(100 * rows["price_in_train_support"].mean()),
                    **score,
                })
    frame = pd.DataFrame(metrics).sort_values(["split", "segment", "rmse", "model"])
    by_sku = []
    for (split, sku), rows in checked.groupby(["split", "sku"]):
        by_sku.append({
            "split": split, "sku": sku, "n": len(rows),
            "simulator_rmse": _score(rows, rows["simulator_pred_qty"].to_numpy())["rmse"],
            "naive_rmse": _score(rows, rows["qty_lag_1"].to_numpy())["rmse"],
            "price_in_train_support_pct": float(100 * rows["price_in_train_support"].mean()),
        })
    validation = frame.loc[frame["split"].eq("validation")
                           & frame["segment"].eq("in_support")].set_index("model")
    necessary_gate_passed = bool(
        validation.loc["kernel_simulator", "rmse"] <= validation.loc["naive_last", "rmse"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "observed_price_metrics.csv", index=False)
    checked.to_csv(output_dir / "observed_price_predictions.csv", index=False)
    pd.DataFrame(by_sku).to_csv(output_dir / "observed_price_by_sku.csv", index=False)
    simulator.diagnostics().to_csv(output_dir / "train_price_support.csv", index=False)
    (output_dir / "audit_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": panel["dataset"],
        "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        "prepared_weekly_sha256": _sha256(prepared_root / "weekly.csv"),
        "simulator_fit_scope": "observed, fully priced train weeks only",
        "necessary_validation_gate": "kernel simulator RMSE at observed in-support validation prices <= last-week naive RMSE",
        "necessary_validation_gate_passed": necessary_gate_passed,
        "interpretation": "Passing the gate is not evidence of correct demand at unobserved prices or causal uplift",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.prepared_root, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
