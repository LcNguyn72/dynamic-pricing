"""Check the saved research panel, chronological split, lineage and lag rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dataset = manifest.get("dataset", "")
    _require(bool(dataset) and not dataset.startswith("Unverified"), "Dataset lineage is unknown")
    weekly = pd.read_csv(root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(
        root / "windows.csv", dtype={"sku": str}, parse_dates=["week"],
        usecols=["sku", "week", "split", "observed_sales", "qty_lag_1", "price_lag_1"],
    )
    _require(not weekly[["sku", "week"]].duplicated().any(), "Duplicate SKU-week in weekly panel")
    _require(not windows[["sku", "week"]].duplicated().any(), "Duplicate SKU-week in forecast windows")
    _require(set(weekly["split"]) == {"train", "validation", "test"}, "Missing or unknown split")
    selected = set(manifest.get("skus", manifest.get("selected_skus", [])))
    _require(bool(selected) and selected == set(weekly["sku"]), "Cohort differs from manifest")
    labels = weekly[["week", "split"]].drop_duplicates()
    _require(not labels["week"].duplicated().any(), "One calendar week has multiple splits")
    train_end = labels.loc[labels["split"].eq("train"), "week"].max()
    valid_start = labels.loc[labels["split"].eq("validation"), "week"].min()
    valid_end = labels.loc[labels["split"].eq("validation"), "week"].max()
    test_start = labels.loc[labels["split"].eq("test"), "week"].min()
    _require(train_end < valid_start <= valid_end < test_start, "Splits are not chronological")
    _require(manifest["split_weeks"]["train_end"] == train_end.date().isoformat(),
             "Train boundary differs from manifest")
    _require(manifest["split_weeks"]["test_start"] == test_start.date().isoformat(),
             "Test boundary differs from manifest")

    for sku, scaler in manifest["scalers"].items():
        rows = weekly.loc[weekly["sku"].eq(sku) & weekly["split"].eq("train")
                          & weekly["observed_sales"]]
        _require(len(rows) == scaler["n_observed_train"], f"Train count differs for {sku}")
        for field, column in (("qty_mean", "qty"), ("price_mean", "price")):
            _require(np.isclose(float(rows[column].mean()), scaler[field], rtol=1e-7, atol=1e-7),
                     f"Scaler {field} was not fit on observed train rows for {sku}")

    source_checks = []
    for path_key, hash_key in (("weekly_source", "weekly_source_sha256"),
                               ("selection_source", "selection_source_sha256"),
                               ("upstream_preprocessing_manifest", "upstream_preprocessing_manifest_sha256")):
        if manifest.get(path_key) and manifest.get(hash_key):
            path = Path(manifest[path_key])
            _require(path.is_file() and _sha256(path) == manifest[hash_key],
                     f"Source hash differs: {path_key}")
            source_checks.append(path_key)
    for key, filename in (("weekly_sha256", "weekly.csv"), ("windows_sha256", "windows.csv")):
        if key in manifest:
            _require(_sha256(root / filename) == manifest[key], f"Saved {filename} hash differs")
            source_checks.append(filename)
    if isinstance(manifest.get("source_sha256"), dict):
        for name, expected in manifest["source_sha256"].items():
            source = Path(manifest[{
                "sales_train_validation.csv": "sales_source",
                "sell_prices.csv": "price_source",
                "calendar.csv": "calendar_source",
            }[name]])
            _require(source.is_file() and _sha256(source) == expected, f"M5 source hash differs: {name}")
            source_checks.append(name)

    prior = weekly.sort_values(["sku", "week"])[["sku", "week", "qty", "price"]].copy()
    prior["expected_qty_lag_1"] = prior.groupby("sku")["qty"].shift(1)
    prior["expected_price_lag_1"] = prior.groupby("sku")["price"].shift(1)
    joined = windows.merge(prior, on=["sku", "week"], how="left", validate="one_to_one")
    _require(len(joined) == len(windows) and joined["expected_qty_lag_1"].notna().all(),
             "Forecast windows do not match the weekly panel")
    for observed, expected in (("qty_lag_1", "expected_qty_lag_1"),
                               ("price_lag_1", "expected_price_lag_1")):
        _require(np.allclose(joined[observed], joined[expected], rtol=1e-6, atol=1e-6),
                 f"Past-only lag check failed: {observed}")
    summary = {
        "dataset": dataset,
        "series": len(selected),
        "calendar_weeks": int(labels["week"].nunique()),
        "weekly_rows": len(weekly),
        "forecast_windows": len(windows),
        "observed_targets": {split: int((windows["split"].eq(split) & windows["observed_sales"]).sum())
                             for split in ("train", "validation", "test")},
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "verified_hashes": source_checks,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
