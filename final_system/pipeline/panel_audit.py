"""Audit protocol-V2 weekly panels and freeze the held-out forecast keys."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..bundle import DEFAULT_PREPARED_ROOT, PROTOCOL_PATH
from .m5_panel import STORE_IDS
from .uci_panel import make_windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _observed(frame: pd.DataFrame) -> pd.Series:
    values = frame["observed_sales"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    mapped = values.astype("string").str.lower().map({"true": True, "false": False})
    _require(mapped.notna().all(), "observed_sales must contain booleans")
    return mapped.astype(bool)


def _weekly_dates(frame: pd.DataFrame) -> pd.Series:
    weeks = pd.to_datetime(frame["week"], errors="raise")
    _require(weeks.notna().all() and weeks.dt.day_of_week.eq(0).all(),
             "Week keys must be non-null Mondays")
    return weeks


def _require_full_week_grid(frame: pd.DataFrame, *, series: int, weeks: int) -> None:
    _require(not frame[["sku", "week"]].duplicated().any(), "Duplicate SKU-week in panel")
    _require(frame["sku"].nunique() == series and frame["week"].nunique() == weeks
             and len(frame) == series * weeks,
             "Weekly panel has an incomplete SKU-week grid")


def _require_panel_matches_source(panel: pd.DataFrame, source: pd.DataFrame) -> None:
    observed = panel.loc[_observed(panel), ["sku", "week", "qty", "price"]]
    expected = source.loc[source["sku"].isin(panel["sku"]),
                          ["sku", "week", "qty", "price"]]
    joined = observed.merge(expected, on=["sku", "week"], how="outer",
                            suffixes=("_panel", "_upstream"),
                            validate="one_to_one", indicator=True)
    _require(joined["_merge"].eq("both").all(),
             "Panel observed targets differ from upstream SKU-week keys")
    for field in ("qty", "price"):
        _require(np.allclose(joined[f"{field}_panel"], joined[f"{field}_upstream"],
                             rtol=1e-8, atol=1e-8),
                 f"Panel {field} differs from upstream weekly source")


def _require_past_lags(windows: pd.DataFrame, weekly: pd.DataFrame,
                       *, lookback: int) -> None:
    ordered = weekly.sort_values(["sku", "week"])
    expected = ordered[["sku", "week"]].copy()
    fields = [name for name in ("qty", "price", "qty_z", "price_z")
              if name in ordered and f"{name}_lag_1" in windows]
    for lag in range(1, lookback + 1):
        for field in fields:
            column = f"{field}_lag_{lag}"
            _require(column in windows, f"Forecast windows lack {column}")
            expected[column] = ordered.groupby("sku", sort=False)[field].shift(lag)
    joined = windows.merge(expected, on=["sku", "week"], how="left",
                           suffixes=("_saved", "_expected"), validate="one_to_one")
    _require(len(joined) == len(windows), "Forecast windows differ from weekly panel keys")
    for lag in range(1, lookback + 1):
        for field in fields:
            column = f"{field}_lag_{lag}"
            _require(np.allclose(joined[f"{column}_saved"],
                                 joined[f"{column}_expected"],
                                 rtol=1e-5, atol=1e-5, equal_nan=True),
                     f"Forecast windows have an invalid {column}")


def write_test_keys(frames: list[pd.DataFrame], dataset: str, path: str | Path) -> dict:
    """Write one canonical UTF-8 CSV and hash its exact LF-terminated bytes."""
    path = Path(path)
    parts = []
    for frame in frames:
        required = {"sku", "week", "split", "observed_sales"}
        _require(required.issubset(frame), f"Test-key frame lacks {sorted(required - set(frame))}")
        selected = frame.loc[frame["split"].eq("test") & _observed(frame), ["sku", "week"]].copy()
        selected["week"] = _weekly_dates(selected)
        _require(selected["sku"].notna().all(), "Test keys have null SKU")
        parts.append(selected)
    _require(bool(parts), "No frames supplied for test keys")
    keys = pd.concat(parts, ignore_index=True)
    _require(not keys.empty, "No observed test targets")
    _require(not keys[["sku", "week"]].duplicated().any(), "Duplicate test SKU-week key")
    keys["week"] = keys["week"].dt.strftime("%Y-%m-%d")
    keys = keys.sort_values(["sku", "week"], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("dataset", "sku", "week"))
        writer.writerows((dataset, sku, week) for sku, week in keys.itertuples(index=False, name=None))
    return {
        "path": str(path.resolve()), "count": int(len(keys)),
        "series": int(keys["sku"].nunique()),
        "first_week": keys["week"].min(), "last_week": keys["week"].max(),
        "sha256": _sha256(path),
        "serialization": "UTF-8 CSV; dataset,sku,week header; sorted dataset/sku/ISO-Monday-week; LF endings",
    }


def select_pricing_series(
    weekly: pd.DataFrame, store_id: str, *, test_weeks: int = 20, lookback: int = 12
) -> dict:
    """Pick the train-volume leader; verify test feasibility without reselection."""
    _require(test_weeks > 0 and lookback > 0, "test_weeks and lookback must be positive")
    required = {"sku", "week", "split", "observed_sales", "qty", "price"}
    _require(required.issubset(weekly), f"Pricing panel lacks {sorted(required - set(weekly))}")
    work = weekly.copy()
    work["week"] = _weekly_dates(work)
    _require(work["sku"].astype(str).str.endswith("@" + store_id).all(),
             f"Pricing panel contains a SKU outside {store_id}")
    _require(not work[["sku", "week"]].duplicated().any(), "Duplicate pricing SKU-week")
    observed = _observed(work)
    train = work.loc[work["split"].eq("train") & observed & work["price"].gt(0)]
    stats = train.groupby("sku", sort=False).agg(
        train_units=("qty", "sum"),
        train_observed_weeks=("week", "nunique"),
        train_price_levels=("price", "nunique"),
    ).reset_index()
    stats = stats.loc[
        stats["train_observed_weeks"].ge(lookback)
        & stats["train_price_levels"].ge(2)
    ].sort_values(["train_units", "sku"], ascending=[False, True])
    _require(not stats.empty, f"No train-eligible pricing series in {store_id}")
    chosen = stats.iloc[0]
    sku = str(chosen["sku"])
    global_test_weeks = sorted(work.loc[work["split"].eq("test"), "week"].unique())
    _require(len(global_test_weeks) >= test_weeks,
             f"{sku} has fewer than {test_weeks} calendar test weeks")
    required_weeks = pd.DatetimeIndex(global_test_weeks[:test_weeks])
    selected = work.loc[work["sku"].eq(sku)].sort_values("week")
    history = selected.loc[selected["week"].lt(required_weeks[0])].tail(lookback)
    expected_history = pd.date_range(end=required_weeks[0] - pd.Timedelta(days=7),
                                     periods=lookback, freq="7D")
    _require(len(history) == lookback
             and history["week"].reset_index(drop=True).equals(pd.Series(expected_history))
             and history["price"].notna().all() and history["price"].gt(0).all(),
             f"{sku} lacks complete priced {lookback}-week history before test")
    decisions = selected.loc[selected["week"].isin(required_weeks)]
    _require(len(decisions) == test_weeks
             and decisions["week"].reset_index(drop=True).equals(pd.Series(required_weeks))
             and _observed(decisions).all() and decisions["price"].gt(0).all(),
             f"{sku} lacks a valid test decision; no test-based replacement is allowed")
    return {
        "store": store_id, "sku": sku,
        "train_units": float(chosen["train_units"]),
        "train_observed_weeks": int(chosen["train_observed_weeks"]),
        "train_price_levels": int(chosen["train_price_levels"]),
        "first_test_week": required_weeks[0].date().isoformat(),
        "last_test_week": required_weeks[-1].date().isoformat(),
        "test_decisions": int(len(decisions)),
    }


def _split_counts(weekly: pd.DataFrame, expected: dict) -> dict:
    labels = weekly[["week", "split"]].drop_duplicates()
    _require(not labels["week"].duplicated().any(), "Calendar week has multiple split labels")
    dates = pd.DatetimeIndex(sorted(labels["week"].unique()))
    _require(dates.equals(pd.date_range(dates[0], dates[-1], freq="7D")),
             "Weekly panel contains a calendar gap")
    counts = labels["split"].value_counts().to_dict()
    for split, key in (("train", "train_weeks"), ("validation", "validation_weeks"),
                       ("test", "test_weeks")):
        _require(counts.get(split) == expected[key], f"Unexpected {split} calendar weeks")
    _require(labels.loc[labels["split"].eq("train"), "week"].max().date().isoformat()
             == expected["train_end"], "Train end differs from protocol")
    _require(labels.loc[labels["split"].eq("test"), "week"].min().date().isoformat()
             == expected["test_start"], "Test start differs from protocol")
    for split, edge, key in (("validation", "min", "validation_start"),
                             ("validation", "max", "validation_end")):
        if key in expected:
            values = labels.loc[labels["split"].eq(split), "week"]
            actual = (values.min() if edge == "min" else values.max()).date().isoformat()
            _require(actual == expected[key], f"{key} differs from protocol")
    return {name: int(counts[name]) for name in ("train", "validation", "test")}


def _check_scalers(weekly: pd.DataFrame, scalers: dict | None) -> None:
    train = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    actual = train.groupby("sku").agg(
        qty_mean=("qty", "mean"), price_mean=("price", "mean"),
        n_observed_train=("week", "size"),
        qty_std=("qty", lambda values: values.std(ddof=0)),
        price_std=("price", lambda values: values.std(ddof=0)),
    )
    _require(set(actual.index) == set(weekly["sku"]), "Scaler train cohort lacks panel SKUs")
    for field in ("qty", "price"):
        mean, std = actual[f"{field}_mean"], actual[f"{field}_std"]
        floor = np.maximum(1e-8, mean.abs() * 1e-8)
        actual[f"{field}_std"] = std.where(np.isfinite(std) & std.gt(floor), 1.0)
    if scalers is not None:
        stated = pd.DataFrame.from_dict(scalers, orient="index")
        _require(set(actual.index) == set(stated.index), "Scaler SKU set differs from train cohort")
        stated = stated.loc[actual.index]
        for field in ("qty_mean", "price_mean", "qty_std", "price_std"):
            _require(np.allclose(actual[field], stated[field], rtol=1e-4, atol=1e-5),
                     f"{field} was not fitted from observed train weeks")
        _require(np.array_equal(actual["n_observed_train"].to_numpy(),
                                stated["n_observed_train"].to_numpy()),
                 "Scaler observed-train counts differ")
    parameters = actual.loc[weekly["sku"]]
    for field in ("qty", "price"):
        expected = ((weekly[field].to_numpy() - parameters[f"{field}_mean"].to_numpy())
                    / parameters[f"{field}_std"].to_numpy())
        _require(np.allclose(weekly[f"{field}_z"].to_numpy(), expected,
                             rtol=1e-4, atol=1e-4, equal_nan=True),
                 f"{field}_z differs from the observed-train scaler")


def _uci_panel(uci_root: Path, protocol: dict) -> tuple[dict, pd.DataFrame]:
    source_manifest = json.loads((uci_root / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    panel_root = uci_root / "panel"
    panel_manifest = json.loads((panel_root / "manifest.json").read_text(encoding="utf-8"))
    cfg = protocol["datasets"]["uci_online_retail"]
    _require(source_manifest.get("outlier_application") == "train_only",
             "UCI V2 must retain valid held-out outliers")
    _require(source_manifest["source_sha256"] == cfg["source_sha256"],
             "UCI workbook hash differs from protocol")
    _require(_sha256(Path(source_manifest["source"])) == cfg["source_sha256"],
             "UCI workbook bytes differ from protocol")
    for panel_key, upstream_key in (("weekly_source_sha256", "weekly_sha256"),
                                    ("selection_source_sha256", "selected_sha256")):
        _require(panel_manifest[panel_key] == source_manifest[upstream_key],
                 f"UCI panel {panel_key} differs from upstream manifest")
    _require(panel_manifest["upstream_preprocessing_manifest_sha256"]
             == _sha256(uci_root / "preprocessing_manifest.json"),
             "UCI panel links to a different preprocessing manifest")
    for name, key in (("weekly.csv", "weekly_sha256"),
                      ("selected_products.csv", "selected_sha256")):
        _require(_sha256(uci_root / name) == source_manifest[key], f"UCI {name} hash differs")
    source = pd.read_csv(uci_root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    train_end = pd.Timestamp(cfg["train_end"])
    heldout = source.loc[source["week"].gt(train_end)]
    _require(heldout["qty"].eq(heldout["qty_all"]).all(),
             "UCI validation/test quantity was altered by outlier filtering")
    weekly = pd.read_csv(panel_root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    _require_full_week_grid(weekly, series=cfg["currently_prepared_skus"],
                            weeks=cfg["calendar_weeks"])
    _require_panel_matches_source(weekly, source)
    _require(weekly["week"].min().date().isoformat() == source_manifest["first_week"]
             and weekly["week"].max().date().isoformat() == source_manifest["last_week"],
             "UCI panel calendar differs from full weekly source")
    splits = _split_counts(weekly, cfg)
    _check_scalers(weekly, panel_manifest["scalers"])
    observed = _observed(weekly)
    train_counts = weekly.loc[weekly["split"].eq("train") & observed].groupby("sku").size()
    fit_counts = weekly.loc[weekly["split"].ne("test") & observed].groupby("sku").size()
    lag_columns = [f"{field}_lag_{lag}" for lag in range(1, 13)
                   for field in ("qty", "price", "qty_z", "price_z")]
    windows = pd.read_csv(panel_root / "windows.csv", dtype={"sku": str}, parse_dates=["week"],
                          usecols=["sku", "week", "split", "observed_sales", *lag_columns])
    _require_past_lags(windows, weekly, lookback=12)
    report = {
        "source_sha256": source_manifest["source_sha256"],
        "weekly_sha256": _sha256(uci_root / "weekly.csv"),
        "windows_sha256": _sha256(panel_root / "windows.csv"),
        "panel_manifest_sha256": _sha256(panel_root / "manifest.json"),
        "eligible_skus": int(weekly["sku"].nunique()),
        "calendar_weeks": int(weekly["week"].nunique()),
        "split_weeks": splits,
        "valid_heldout_observed_weeks": int(len(heldout)),
        "arima_series_under_12_observed_train_weeks": int(train_counts.lt(12).sum()),
        "arima_series_under_12_observed_train_validation_weeks": int(fit_counts.lt(12).sum()),
        "heldout_outlier_lines_retained": source_manifest["audit"]["heldout_outlier_lines_retained"],
        "partial_boundary_weeks": {
            "first_transaction": source_manifest["first_transaction"],
            "last_transaction": source_manifest["last_transaction"],
            "note": "First and last UCI calendar weeks contain fewer than seven source days",
        },
    }
    return report, windows


def _m5_store(store: str, root: Path, data_dir: Path,
              cfg: dict, lookback: int) -> tuple[dict, pd.DataFrame, dict]:
    path = root / f"weekly_{store}.csv"
    manifest_path = root / f"manifest_{store}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("protocol_id") == "song_full_scope_v2"
             and manifest.get("store_id") == store,
             f"{store} panel was not built for protocol V2")
    for name, source_key, hash_key in (
        ("sales_train_evaluation.csv", "sales_source", "sales_sha256"),
        ("sell_prices.csv", "price_source", "price_sha256"),
        ("calendar.csv", "calendar_source", "calendar_sha256"),
    ):
        _require(Path(manifest[source_key]).resolve() == (data_dir / name).resolve()
                 and manifest["source_sha256"][name] == cfg[hash_key],
                 f"{store} {name} provenance differs from protocol")
    panel_hash = _sha256(path)
    _require(manifest.get("weekly_sha256") == panel_hash,
             f"{store} weekly panel hash differs from its manifest")
    _require(manifest["n_weeks"] == cfg["calendar_weeks"] and manifest["split_profile"] == "paper_70_15_15",
             f"{store} calendar or split profile differs from protocol")
    _require(manifest["split_weeks"]["test_start"] == cfg["test_start"],
             f"{store} test boundary differs from protocol")
    header = pd.read_csv(path, nrows=0).columns
    _require("promotion_flag" not in header, "M5 panel unexpectedly claims observed promotion")
    columns = ["sku", "week", "split", "observed_sales", "qty", "price",
               "priced_days", "qty_z", "price_z", "month", "week_of_year",
               "event_days", "snap_days", "promotion_proxy"]
    weekly = pd.read_csv(path, usecols=columns, dtype={"sku": str}, parse_dates=["week"])
    _require_full_week_grid(weekly, series=manifest["n_series"],
                            weeks=cfg["calendar_weeks"])
    _require(weekly["sku"].str.endswith("@" + store).all(), f"{store} panel contains other stores")
    _require(weekly["qty"].ge(0).all(), f"{store} contains negative quantity")
    observed = _observed(weekly)
    _require(observed.eq(weekly["priced_days"].eq(7)).all(),
             f"{store} observed target flag differs from seven priced days")
    _require(weekly.loc[observed, "price"].gt(0).all(),
             f"{store} observed target has invalid price")
    splits = _split_counts(weekly, cfg)
    _check_scalers(weekly, manifest.get("scalers"))
    train_counts = weekly.loc[weekly["split"].eq("train") & observed].groupby("sku").size()
    fit_counts = weekly.loc[weekly["split"].ne("test") & observed].groupby("sku").size()
    chosen = select_pricing_series(
        weekly, store, test_weeks=20, lookback=lookback)
    windows = make_windows(weekly, lookback=lookback, forecast_only=True)
    test_keys = windows.loc[windows["split"].eq("test") & _observed(windows),
                            ["sku", "week", "split", "observed_sales"]].copy()
    report = {
        "store": store, "requested_series": int(manifest["requested_series"]),
        "eligible_series": int(manifest["n_series"]),
        "calendar_weeks": int(weekly["week"].nunique()), "split_weeks": splits,
        "observed_test_targets": int(len(test_keys)),
        "zero_sales_test_targets": int((weekly["split"].eq("test") & observed & weekly["qty"].eq(0)).sum()),
        "arima_series_under_12_observed_train_weeks": int(train_counts.lt(12).sum()),
        "arima_series_under_12_observed_train_validation_weeks": int(fit_counts.lt(12).sum()),
        "promotion_proxy_rows": int(weekly["promotion_proxy"].sum()),
        "weekly_sha256": panel_hash, "manifest_sha256": _sha256(manifest_path),
    }
    return report, test_keys, chosen


def finalize(
    uci_root: str | Path, m5_root: str | Path, m5_data_dir: str | Path,
    output_root: str | Path = DEFAULT_PREPARED_ROOT,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> dict:
    uci_root, m5_root = Path(uci_root), Path(m5_root)
    m5_data_dir, output_root = Path(m5_data_dir), Path(output_root)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    _require(protocol["protocol_id"] == "song_full_scope_v2", "Unexpected protocol")
    cfg = protocol["datasets"]["m5"]
    for file, hash_key in (("sales_train_evaluation.csv", "sales_sha256"),
                           ("sell_prices.csv", "price_sha256"),
                           ("calendar.csv", "calendar_sha256")):
        _require(_sha256(m5_data_dir / file) == cfg[hash_key], f"M5 {file} hash differs")
    output_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads((m5_root / "full_m5_preparation.json").read_text(encoding="utf-8"))
    _require(summary.get("protocol_id") == protocol["protocol_id"]
             and summary.get("source_sha256") == {
                 "sales_train_evaluation.csv": cfg["sales_sha256"],
                 "sell_prices.csv": cfg["price_sha256"],
                 "calendar.csv": cfg["calendar_sha256"]},
             "M5 aggregate preparation manifest differs from protocol")
    _require({row["store"] for row in summary["stores"]} == set(STORE_IDS)
             and len(summary["stores"]) == len(STORE_IDS),
             "M5 aggregate preparation manifest does not cover ten stores")
    uci_report, uci_windows = _uci_panel(uci_root, protocol)
    uci_keys = write_test_keys([uci_windows], "uci_online_retail",
                               output_root / "test_keys_uci.csv")
    uci_cfg = protocol["datasets"]["uci_online_retail"]
    _require(uci_keys["first_week"] == uci_cfg["test_start"],
             "UCI first scored test week differs from protocol")
    _require(uci_keys["count"] == uci_cfg["primary_test_key_count"]
             and uci_keys["sha256"] == uci_cfg["primary_test_key_sha256"],
             "UCI frozen test keys differ from protocol")
    store_reports, key_frames, cohort = [], [], []
    for store in STORE_IDS:
        print(f"[phase1] validating {store}", flush=True)
        store_report, keys, chosen = _m5_store(
            store, m5_root, m5_data_dir, cfg, protocol["split"]["lookback_weeks"])
        store_reports.append(store_report)
        key_frames.append(keys)
        cohort.append(chosen)
    _require(sum(row["requested_series"] for row in store_reports) == cfg["raw_item_store_series"],
             "M5 raw series count differs from protocol")
    _require(sum(row["eligible_series"] for row in store_reports) == cfg["currently_prepared_series"],
             "M5 train-eligible series count differs from protocol")
    for row in store_reports:
        prepared = next(item for item in summary["stores"] if item["store"] == row["store"])
        _require(prepared["weekly_sha256"] == row["weekly_sha256"]
                 and prepared["manifest_sha256"] == row["manifest_sha256"],
                 f"{row['store']} aggregate hashes differ from per-store artifacts")
    m5_keys = write_test_keys(key_frames, "m5", output_root / "test_keys_m5.csv")
    _require(m5_keys["first_week"] == cfg["test_start"],
             "M5 first scored test week differs from protocol")
    _require(m5_keys["series"] == cfg["currently_prepared_series"]
             and m5_keys["count"] == cfg["currently_prepared_series"] * cfg["test_weeks"],
             "M5 forecast test-key grid is incomplete")
    _require(m5_keys["count"] == cfg["primary_test_key_count"]
             and m5_keys["sha256"] == cfg["primary_test_key_sha256"],
             "M5 frozen test keys differ from protocol")
    _require([row["sku"] for row in cohort] == protocol["pricing"]["cohort_skus"],
             "Train-only M5 pricing cohort differs from protocol")
    cohort_path = output_root / "m5_pricing_cohort.csv"
    pd.DataFrame(cohort).to_csv(cohort_path, index=False, lineterminator="\n")
    result = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(Path(protocol_path)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "phase1_data_verified_no_model_training",
        "uci": {**uci_report, "test_keys": uci_keys},
        "m5": {
            "source_sha256": {name: cfg[key] for name, key in (
                ("sales_train_evaluation.csv", "sales_sha256"),
                ("sell_prices.csv", "price_sha256"), ("calendar.csv", "calendar_sha256"))},
            "stores": store_reports,
            "test_keys": m5_keys,
            "pricing_cohort_path": str(cohort_path.resolve()),
            "pricing_cohort_sha256": _sha256(cohort_path),
            "pricing_decisions": sum(row["test_decisions"] for row in cohort),
            "promotion_policy": "No observed M5 promotion_flag; Song feature is zero. Price-derived promotion_proxy is retained only as an auxiliary panel column.",
        },
    }
    summary_path = output_root / "phase1_manifest.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase1] complete: {summary_path}", flush=True)
    return result
