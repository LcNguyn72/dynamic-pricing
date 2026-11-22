"""Build the chronological UCI weekly panel and causal windows.

The split uses calendar weeks, never shuffled rows.  Product standardisation is
learned from observed training weeks only.  A missing SKU-week is present in
the history with quantity zero, but ``observed_sales=False`` means its zero is
not a confirmed demand observation.  Offline metrics should therefore use
only rows with ``observed_sales=True`` as targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


from ..bundle import DEFAULT_PREPARED_ROOT


DEFAULT_WEEKLY_PATH = DEFAULT_PREPARED_ROOT / "uci" / "weekly.csv"
DEFAULT_SELECTED_PATH = DEFAULT_PREPARED_ROOT / "uci" / "selected_products.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PREPARED_ROOT / "uci" / "panel"
SPLIT_PROFILES = ("paper_70_15_15", "retail2_80_20", "retail2_last12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_std(values: pd.Series) -> float:
    std = float(values.std(ddof=0))
    mean = float(values.mean())
    # Floating-point reductions of a constant non-integer series can return a
    # tiny positive value instead of zero, exploding held-out z scores.
    floor = max(1e-8, abs(mean) * 1e-8)
    return std if np.isfinite(std) and std > floor else 1.0


def load_experiment_data(
    weekly_path: str | Path = DEFAULT_WEEKLY_PATH,
    selected_path: str | Path = DEFAULT_SELECTED_PATH,
    lookback: int = 12,
    split_profile: str = "paper_70_15_15",
) -> tuple[pd.DataFrame, dict]:
    """Return complete selected-SKU weekly rows and a reproducibility manifest.

    The ``weekly_df`` columns include ``sku``, ``week``, ``qty``, ``price``,
    ``split``, ``observed_sales``, ``price_imputed``, ``qty_z``, ``price_z``,
    ``month``, and ``week_of_year``.  ``price`` is carried forward only from
    earlier observed weeks.  It remains missing before a SKU's first sale.
    """
    weekly_path, selected_path = Path(weekly_path), Path(selected_path)
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if split_profile not in SPLIT_PROFILES:
        raise ValueError(f"split_profile must be one of {SPLIT_PROFILES}")
    for path in (weekly_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = pd.read_csv(selected_path, dtype={"sku": str})
    if "sku" not in selected or selected["sku"].isna().any():
        raise ValueError("Selected-products file must contain non-null sku values")
    sku_order = selected["sku"].astype(str).tolist()
    if not sku_order or len(sku_order) != len(set(sku_order)):
        raise ValueError("Selected-products file must contain unique SKUs")

    source = pd.read_csv(weekly_path, dtype={"sku": str}, parse_dates=["week"])
    required = {"sku", "week", "qty", "price"}
    if not required.issubset(source.columns):
        raise ValueError(f"Weekly file is missing columns: {sorted(required - set(source))}")
    if source["week"].isna().any() or (source["week"].dt.day_of_week != 0).any():
        raise ValueError("Weekly source dates must be non-null Monday week starts")
    calendar_start, calendar_end = source["week"].min(), source["week"].max()
    source = source.loc[source["sku"].isin(sku_order), ["sku", "week", "qty", "price"]].copy()
    absent = set(sku_order) - set(source["sku"])
    if absent:
        raise ValueError(f"Selected SKUs absent from weekly data: {sorted(absent)}")
    if source[["sku", "week"]].duplicated().any():
        raise ValueError("Weekly source contains duplicate SKU-week rows")
    source["qty"] = pd.to_numeric(source["qty"], errors="raise")
    source["price"] = pd.to_numeric(source["price"], errors="raise")
    if source[["week", "qty", "price"]].isna().any().any():
        raise ValueError("Observed weekly rows cannot contain missing week, qty or price")
    if (source["qty"] <= 0).any() or (source["price"] <= 0).any():
        raise ValueError("Observed weekly rows must have positive qty and price")
    if (source["week"].dt.day_of_week != 0).any():
        raise ValueError("Weekly dates must be Monday week starts")

    weeks = pd.date_range(calendar_start, calendar_end, freq="7D")
    if len(weeks) <= lookback + 2:
        raise ValueError("Too few calendar weeks for lookback and chronological split")
    if split_profile == "paper_70_15_15":
        n_train = int(np.floor(0.70 * len(weeks)))
        n_valid_end = int(np.floor(0.85 * len(weeks)))
        split_rule = "Global chronological weeks; boundaries floor(0.70*N), floor(0.85*N)"
    elif split_profile == "retail2_80_20":
        n_train = int(np.floor(0.80 * len(weeks)))
        n_valid_end = n_train
        split_rule = "Global chronological weeks; first floor(0.80*N) train, remainder test; rolling CV folds are available inside train but are not automatically used by benchmark models"
    else:
        n_train = len(weeks) - 12
        n_valid_end = n_train
        split_rule = "Global chronological weeks; last 12 weeks test, earlier weeks train; rolling CV folds are available inside train but are not automatically used by benchmark models. This holdout size is inspired by an Online Retail II thesis, not an exact reproduction of its 90-train/4-buffer/12-test aggregate 12-step task."
    if (n_train <= lookback or n_valid_end < n_train or n_valid_end >= len(weeks)
            or (split_profile == "paper_70_15_15" and n_valid_end == n_train)):
        raise ValueError("Each split needs weeks, and train must exceed lookback")
    split_by_week = {
        week: ("train" if i < n_train else "validation" if i < n_valid_end else "test")
        for i, week in enumerate(weeks)
    }

    grid = pd.MultiIndex.from_product([sku_order, weeks], names=["sku", "week"]).to_frame(index=False)
    weekly = grid.merge(source, on=["sku", "week"], how="left", validate="one_to_one", indicator=True)
    weekly["observed_sales"] = weekly["_merge"].eq("both")
    weekly = weekly.drop(columns="_merge").sort_values(["sku", "week"]).reset_index(drop=True)
    weekly["price_imputed"] = ~weekly["observed_sales"]
    weekly["qty"] = weekly["qty"].fillna(0.0).astype(float)
    weekly["price"] = weekly.groupby("sku", sort=False)["price"].ffill().astype(float)
    weekly["split"] = weekly["week"].map(split_by_week)
    weekly["month"] = weekly["week"].dt.month.astype(int)
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)
    if "tier" in selected.columns:
        tier_by_sku = dict(zip(selected["sku"], selected["tier"].fillna("")))
        weekly["tier"] = weekly["sku"].map(tier_by_sku)

    scalers: dict[str, dict] = {}
    weekly["qty_z"] = np.nan
    weekly["price_z"] = np.nan
    for sku in sku_order:
        mask = weekly["sku"].eq(sku)
        train_observed = weekly.loc[mask & weekly["split"].eq("train") & weekly["observed_sales"]]
        if len(train_observed) < 2:
            raise ValueError(f"SKU {sku} has fewer than two observed training weeks")
        qty_mean = float(train_observed["qty"].mean())
        qty_std = _positive_std(train_observed["qty"])
        price_mean = float(train_observed["price"].mean())
        price_std = _positive_std(train_observed["price"])
        weekly.loc[mask, "qty_z"] = (weekly.loc[mask, "qty"] - qty_mean) / qty_std
        weekly.loc[mask, "price_z"] = (weekly.loc[mask, "price"] - price_mean) / price_std
        scalers[sku] = {
            "qty_mean": qty_mean,
            "qty_std": qty_std,
            "price_mean": price_mean,
            "price_std": price_std,
            "n_observed_train": int(len(train_observed)),
        }

    counts = {
        split: {
            "weeks": int((pd.Series(weeks).map(split_by_week) == split).sum()),
            "sku_weeks": int(weekly["split"].eq(split).sum()),
            "observed_sku_weeks": int((weekly["split"].eq(split) & weekly["observed_sales"]).sum()),
        }
        for split in ("train", "validation", "test")
    }
    upstream_manifest_path = selected_path.parent / "preprocessing_manifest.json"
    upstream_manifest = None
    if upstream_manifest_path.is_file() and selected_path.name == "selected_products.csv":
        with upstream_manifest_path.open("r", encoding="utf-8") as stream:
            candidate_manifest = json.load(stream)
        if candidate_manifest.get("selected_skus") != sku_order:
            raise ValueError(
                "Selected SKUs do not match preprocessing_manifest.json; rerun preprocessing "
                "for this cohort instead of silently labeling it as Online Retail II"
            )
        for key, path in (("weekly_sha256", weekly_path), ("selected_sha256", selected_path)):
            if key in candidate_manifest and candidate_manifest[key] != _sha256(path):
                raise ValueError(f"{path.name} does not match {key} in preprocessing manifest")
        if not candidate_manifest.get("dataset"):
            raise ValueError("Preprocessing manifest must name its dataset")
        upstream_manifest = candidate_manifest
    manifest = {
        "schema_version": 1,
        "dataset": (upstream_manifest["dataset"] if upstream_manifest else
                    "Unverified weekly source: preprocessing manifest absent"),
        "weekly_source": str(weekly_path.resolve()),
        "weekly_source_sha256": _sha256(weekly_path),
        "selection_source": str(selected_path.resolve()),
        "selection_source_sha256": _sha256(selected_path),
        "selection_scope_note": (upstream_manifest.get("selection_note", upstream_manifest.get("eligibility"))
                                 if upstream_manifest else
                                 "Existing selected-products CSV is reused as a fixed cohort. Its upstream selection rule is not verified by this adapter."),
        "upstream_preprocessing_manifest": str(upstream_manifest_path.resolve()) if upstream_manifest else None,
        "upstream_preprocessing_manifest_sha256": _sha256(upstream_manifest_path) if upstream_manifest else None,
        "upstream_preprocessing_deviations": upstream_manifest.get("deviations_from_song_paper") if upstream_manifest else None,
        "skus": sku_order,
        "n_skus": len(sku_order),
        "lookback_weeks": lookback,
        "split_profile": split_profile,
        "comparison_target_note": (
            f"Our selected-SKU panel has {len(weeks)} calendar weeks and predicts one SKU-week at a time. "
            f"The retail2_last12 profile uses {n_train} train / {len(weeks)-n_train} test weeks; "
            "the cited thesis used 90 train / 4-week buffer / 12 test for an aggregate 12-step forecast. "
            "Compare protocol inspiration only."
            if split_profile == "retail2_last12" else None
        ),
        "first_week": weeks[0].date().isoformat(),
        "last_week": weeks[-1].date().isoformat(),
        "split_rule": split_rule,
        "split_weeks": {
            "train_end": weeks[n_train - 1].date().isoformat(),
            "validation_start": weeks[n_train].date().isoformat() if n_valid_end > n_train else None,
            "validation_end": weeks[n_valid_end - 1].date().isoformat() if n_valid_end > n_train else None,
            "test_start": weeks[n_valid_end].date().isoformat(),
        },
        "counts": counts,
        "missing_sku_weeks": int((~weekly["observed_sales"]).sum()),
        "zero_week_policy": "Insert missing SKU-week with qty=0 for lag context; do not treat as confirmed demand target in primary offline metrics",
        "price_fill_policy": "Forward fill from earlier observed price within SKU only; no backward fill",
        "scaler_fit_policy": "Per-SKU population z-score from observed train weeks only; zero std replaced by 1",
        "scalers": scalers,
    }
    return weekly, manifest


def make_rolling_validation_folds(
    weekly_df: pd.DataFrame,
    n_folds: int = 3,
    horizon_weeks: int = 12,
    min_train_weeks: int = 24,
) -> list[dict]:
    """Create expanding-origin validation folds strictly within fixed train weeks.

    Returned week boundaries can be applied to any algorithm.  The fixed test
    interval is never used for hyperparameter selection.
    """
    if n_folds < 1 or horizon_weeks < 1 or min_train_weeks < 1:
        raise ValueError("n_folds, horizon_weeks and min_train_weeks must be positive")
    train_weeks = sorted(pd.to_datetime(weekly_df.loc[weekly_df["split"].eq("train"), "week"].unique()))
    if len(train_weeks) < min_train_weeks + n_folds * horizon_weeks:
        raise ValueError("Too few training weeks for requested rolling folds")
    folds = []
    first_origin = len(train_weeks) - n_folds * horizon_weeks
    for fold_index in range(n_folds):
        origin = first_origin + fold_index * horizon_weeks
        valid_end = origin + horizon_weeks
        folds.append({
            "fold": fold_index + 1,
            "train_start": pd.Timestamp(train_weeks[0]).date().isoformat(),
            "train_end": pd.Timestamp(train_weeks[origin - 1]).date().isoformat(),
            "validation_start": pd.Timestamp(train_weeks[origin]).date().isoformat(),
            "validation_end": pd.Timestamp(train_weeks[valid_end - 1]).date().isoformat(),
        })
    return folds


def make_windows(
    weekly_df: pd.DataFrame,
    lookback: int = 12,
    *,
    forecast_only: bool = False,
) -> pd.DataFrame:
    """Return one-step rows; optionally omit lags used only by pricing code.

    ``forecast_only`` keeps the compact quantity/price history consumed by the
    demand benchmarks. The default retains the wider history used by pricing
    environments and existing artifacts.
    """
    if lookback < 1:
        raise ValueError("lookback must be positive")
    required = {"sku", "week", "qty", "price", "qty_z", "price_z", "split", "observed_sales"}
    if not required.issubset(weekly_df.columns):
        raise ValueError(f"Weekly table is missing columns: {sorted(required - set(weekly_df))}")
    windows = weekly_df.sort_values(["sku", "week"]).reset_index(drop=True).copy()
    by_sku = windows.groupby("sku", sort=False)
    optional_history = [name for name in ("event_days", "snap_days", "promotion_flag")
                        if name in windows.columns]
    for lag in range(1, lookback + 1):
        lag_features = ("qty", "price") if forecast_only else ("qty", "price", "qty_z", "price_z")
        for column in lag_features:
            windows[f"{column}_lag_{lag}"] = by_sku[column].shift(lag)
        if not forecast_only:
            for column in optional_history:
                windows[f"{column}_lag_{lag}"] = by_sku[column].shift(lag)
            windows[f"observed_sales_lag_{lag}"] = by_sku["observed_sales"].shift(lag)
    lag_columns = [f"{column}_lag_{lag}" for lag in range(1, lookback + 1)
                   for column in (("qty", "price") if forecast_only
                                  else ("qty", "price", "qty_z", "price_z"))]
    windows = windows.dropna(subset=["price", *lag_columns]).reset_index(drop=True)
    if forecast_only:
        optional_current = [c for c in ("event_days", "snap_days") if c in windows.columns]
        keep = ["sku", "week", "qty", "price", "split", "observed_sales",
                "month", "week_of_year", *optional_current]
        keep.extend(f"{name}_lag_{lag}" for lag in range(1, lookback + 1)
                    for name in ("qty", "price"))
        windows = windows.loc[:, keep]
        lag_names = [f"{name}_lag_{lag}" for lag in range(1, lookback + 1)
                     for name in ("qty", "price")]
        windows[lag_names] = windows[lag_names].astype(np.float32)
    # A missing sale has no observed current price; historical forward fill is
    # retained as context, but such rows must be excluded from primary scoring.
    return windows


def prepare_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    weekly_path: str | Path = DEFAULT_WEEKLY_PATH,
    selected_path: str | Path = DEFAULT_SELECTED_PATH,
    lookback: int = 12,
    split_profile: str = "paper_70_15_15",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build, save and return the weekly table, forecast windows and manifest."""
    weekly, manifest = load_experiment_data(weekly_path, selected_path, lookback, split_profile)
    windows = make_windows(weekly, lookback)
    if split_profile != "paper_70_15_15":
        manifest["rolling_validation_folds"] = make_rolling_validation_folds(weekly)
    manifest["window_counts"] = {
        split: {
            "all": int(windows["split"].eq(split).sum()),
            "observed_targets": int((windows["split"].eq(split) & windows["observed_sales"]).sum()),
        }
        for split in ("train", "validation", "test")
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly.csv", index=False)
    windows.to_csv(output_dir / "windows.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return weekly, windows, manifest
