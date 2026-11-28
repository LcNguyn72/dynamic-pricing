"""Build the per-item-store weekly M5 panel from the official CSVs.

Read the *official, locally supplied* M5 CSVs.  Each ``sku`` is an item-store
pair, because the same item can have different prices and demand by store.
Calendar weeks start on Monday; incomplete boundary weeks are discarded.  The
weekly ``price`` is the mean of the seven daily listed prices, rather than a
sales-weighted price, so zero-sales weeks still have a usable price.  M5 has no
explicit promotion flag: ``promotion_proxy`` indicates a price drop from the
previous complete week, while SNAP and event columns are separate covariates.

``observed_sales`` means all seven days have a listed price, including weeks
with zero units sold. Pre-release and partially priced weeks remain in the
calendar with ``observed_sales=False``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SPLIT_PROFILES = ("paper_70_15_15", "retail2_80_20", "retail2_last12")
SALES_METADATA = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
CALENDAR_COLUMNS = (
    "date", "wm_yr_wk", "d", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
)
PRICE_COLUMNS = ("store_id", "item_id", "wm_yr_wk", "sell_price")


def _required_columns(path: Path, required: Sequence[str]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return columns


def _filter_values(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    result = {str(value) for value in values}
    if not result:
        raise ValueError("A supplied item/store filter cannot be empty")
    return result


def _positive_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    floor = max(1e-8, abs(float(series.mean())) * 1e-8)
    return value if np.isfinite(value) and value > floor else 1.0


def _joined_names(group: pd.DataFrame, columns: tuple[str, str]) -> str:
    names = set()
    for column in columns:
        names.update(group[column].dropna().astype(str))
    return " | ".join(sorted(names))


def _calendar_data(path: Path, day_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _required_columns(path, CALENDAR_COLUMNS)
    calendar = pd.read_csv(
        path, usecols=list(CALENDAR_COLUMNS), dtype={"d": str, "wm_yr_wk": str}
    )
    if calendar["d"].duplicated().any():
        raise ValueError("calendar.csv has duplicate d keys")
    missing_days = set(day_columns) - set(calendar["d"])
    if missing_days:
        raise ValueError(f"calendar.csv lacks {len(missing_days)} sales day keys")
    calendar = calendar.loc[calendar["d"].isin(day_columns)].copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    if calendar["date"].isna().any() or calendar["date"].duplicated().any():
        raise ValueError("Calendar dates must be present and unique")
    if calendar["wm_yr_wk"].isna().any():
        raise ValueError("Calendar M5 week keys cannot be missing")
    calendar["week"] = calendar["date"] - pd.to_timedelta(calendar["date"].dt.day_of_week, unit="D")
    day_counts = calendar.groupby("week")["date"].agg(["count", "min", "max"])
    full = day_counts.loc[
        day_counts["count"].eq(7)
        & (day_counts["max"] - day_counts["min"]).eq(pd.Timedelta(days=6))
    ].index
    calendar = calendar.loc[calendar["week"].isin(full)].copy()
    if calendar.empty:
        raise ValueError("No complete Monday-Sunday calendar week in sales file")

    for state in ("CA", "TX", "WI"):
        column = f"snap_{state}"
        calendar[column] = pd.to_numeric(calendar[column], errors="raise")
        if calendar[column].isna().any() or not calendar[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain only 0/1 values")

    calendar["event_day"] = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1)
    event_cov = calendar.groupby("week", sort=True).agg(event_days=("event_day", "sum")).reset_index()
    names = pd.DataFrame([
        {
            "week": week,
            "event_names": _joined_names(group, ("event_name_1", "event_name_2")),
            "event_types": _joined_names(group, ("event_type_1", "event_type_2")),
        }
        for week, group in calendar.groupby("week", sort=True)
    ])
    event_cov = event_cov.merge(names, on="week", validate="one_to_one")
    event_cov["has_event"] = event_cov["event_days"].gt(0)

    snap_cov = pd.concat(
        [calendar.groupby("week", sort=True)[f"snap_{state}"].sum().rename("snap_days")
         .reset_index().assign(state_id=state)
         for state in ("CA", "TX", "WI")],
        ignore_index=True,
    )
    snap_cov["snap_days"] = snap_cov["snap_days"].astype(int)
    return calendar[["d", "week", "wm_yr_wk"]], event_cov, snap_cov


def _sales_metadata(path: Path, item_ids: set[str] | None, store_ids: set[str] | None) -> pd.DataFrame:
    metadata = pd.read_csv(path, usecols=list(SALES_METADATA), dtype=str)
    if metadata[list(SALES_METADATA)].isna().any().any():
        raise ValueError("Sales metadata cannot contain null values")
    if metadata["id"].duplicated().any() or metadata[["item_id", "store_id"]].duplicated().any():
        raise ValueError("Sales file must have one row per item-store pair")
    if item_ids is not None:
        metadata = metadata.loc[metadata["item_id"].isin(item_ids)]
    if store_ids is not None:
        metadata = metadata.loc[metadata["store_id"].isin(store_ids)]
    metadata = metadata.copy()
    if metadata.empty:
        raise ValueError("No M5 item-store series match the requested filters")
    invalid_states = set(metadata["state_id"]) - {"CA", "TX", "WI"}
    if invalid_states:
        raise ValueError(f"Unsupported M5 state_id values: {sorted(invalid_states)}")
    metadata["sku"] = metadata["item_id"] + "@" + metadata["store_id"]
    return metadata


def _price_data(path: Path, metadata: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    _required_columns(path, PRICE_COLUMNS)
    pairs = pd.MultiIndex.from_frame(metadata[["item_id", "store_id"]])
    items, stores = set(metadata["item_id"]), set(metadata["store_id"])
    selected = []
    for chunk in pd.read_csv(
        path, usecols=list(PRICE_COLUMNS),
        dtype={"item_id": str, "store_id": str, "wm_yr_wk": str},
        chunksize=max(100_000, chunksize * 1000),
    ):
        chunk = chunk.loc[chunk["item_id"].isin(items) & chunk["store_id"].isin(stores)]
        if not chunk.empty:
            chunk = chunk.loc[pd.MultiIndex.from_frame(chunk[["item_id", "store_id"]]).isin(pairs)]
        if not chunk.empty:
            selected.append(chunk)
    if not selected:
        raise ValueError("No M5 prices found for selected item-store pairs")
    prices = pd.concat(selected, ignore_index=True)
    if prices[["item_id", "store_id", "wm_yr_wk"]].isna().any().any():
        raise ValueError("M5 price keys cannot be null")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="raise")
    if (not np.isfinite(prices["sell_price"]).all()
            or not prices["sell_price"].gt(0).all()):
        raise ValueError("Listed M5 prices must be finite and positive")
    if prices[["item_id", "store_id", "wm_yr_wk"]].duplicated().any():
        raise ValueError("sell_prices.csv has duplicate item-store-M5-week keys")
    return prices


def _split_weeks(weeks: pd.DatetimeIndex, profile: str) -> tuple[dict[pd.Timestamp, str], dict]:
    if profile not in SPLIT_PROFILES:
        raise ValueError(f"split_profile must be one of {SPLIT_PROFILES}")
    n = len(weeks)
    if profile == "paper_70_15_15":
        train_end, valid_end = int(np.floor(0.70 * n)), int(np.floor(0.85 * n))
    elif profile == "retail2_80_20":
        train_end = valid_end = int(np.floor(0.80 * n))
    else:
        train_end = valid_end = n - 12
    if train_end < 2 or valid_end >= n or (profile == "paper_70_15_15" and valid_end == train_end):
        raise ValueError("Too few complete M5 weeks for the requested chronological split")
    labels = {
        week: ("train" if index < train_end else "validation" if index < valid_end else "test")
        for index, week in enumerate(weeks)
    }
    boundaries = {
        "train_end": weeks[train_end - 1].date().isoformat(),
        "validation_start": weeks[train_end].date().isoformat() if valid_end > train_end else None,
        "validation_end": weeks[valid_end - 1].date().isoformat() if valid_end > train_end else None,
        "test_start": weeks[valid_end].date().isoformat(),
    }
    return labels, boundaries


def load_m5_weekly(
    data_dir: str | Path,
    *,
    sales_file: str = "sales_train_evaluation.csv",
    item_ids: Sequence[str] | None = None,
    store_ids: Sequence[str] | None = None,
    split_profile: str = "paper_70_15_15",
    cold_start_policy: str = "drop",
    chunksize: int = 128,
) -> tuple[pd.DataFrame, dict]:
    """Return item-store weekly M5 rows and a manifest, without data download.

    ``data_dir`` must contain the selected sales file plus ``sell_prices.csv``
    and ``calendar.csv``.  Pass ``item_ids``/``store_ids`` to keep the output
    manageable; an unfiltered M5 table has millions of item-store-week rows.
    Splits are shared chronological calendar weeks, with per-series z-scores
    fitted only on complete, priced training weeks.  Series with fewer than two
    such training weeks are dropped and reported, or rejected when
    ``cold_start_policy='error'``.  Eligibility uses only training data.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if cold_start_policy not in ("drop", "error"):
        raise ValueError("cold_start_policy must be 'drop' or 'error'")
    root = Path(data_dir)
    sales_path = root / sales_file
    price_path = root / "sell_prices.csv"
    calendar_path = root / "calendar.csv"
    sales_columns = _required_columns(sales_path, SALES_METADATA)
    day_columns = sorted(
        (name for name in sales_columns if re.fullmatch(r"d_\d+", name)),
        key=lambda name: int(name[2:]),
    )
    if not day_columns:
        raise ValueError("Sales file has no M5 d_* demand columns")
    item_filter, store_filter = _filter_values(item_ids), _filter_values(store_ids)
    metadata = _sales_metadata(sales_path, item_filter, store_filter)
    calendar, event_cov, snap_cov = _calendar_data(calendar_path, day_columns)
    prices = _price_data(price_path, metadata, chunksize)
    full_days = set(calendar["d"])
    selected_ids = set(metadata["id"])
    weekly_chunks: list[pd.DataFrame] = []
    dtype = {name: "float32" for name in day_columns}
    dtype.update({name: str for name in SALES_METADATA})
    for chunk in pd.read_csv(
        sales_path, usecols=[*SALES_METADATA, *day_columns], dtype=dtype,
        chunksize=chunksize,
    ):
        chunk = chunk.loc[chunk["id"].isin(selected_ids)]
        if chunk.empty:
            continue
        daily = chunk[["id", "item_id", "store_id", *day_columns]].melt(
            id_vars=["id", "item_id", "store_id"], value_vars=day_columns,
            var_name="d", value_name="qty",
        )
        daily = daily.loc[daily["d"].isin(full_days)].copy()
        if not np.isfinite(daily["qty"]).all() or (daily["qty"] < 0).any():
            raise ValueError("Daily M5 units must be finite and nonnegative")
        if (daily["qty"] != np.floor(daily["qty"])).any():
            raise ValueError("Daily M5 units must be whole counts")
        daily = daily.merge(calendar, on="d", how="left", validate="many_to_one")
        daily = daily.merge(
            prices, on=["item_id", "store_id", "wm_yr_wk"], how="left", validate="many_to_one"
        )
        if (daily["qty"].gt(0) & daily["sell_price"].isna()).any():
            raise ValueError("Positive M5 sales have no listed price")
        daily["revenue"] = daily["qty"] * daily["sell_price"].fillna(0.0)
        grouped = daily.groupby(["id", "week"], sort=False).agg(
            qty=("qty", "sum"), price=("sell_price", "mean"),
            priced_days=("sell_price", "count"), calendar_days=("d", "count"),
            revenue=("revenue", "sum"),
        ).reset_index()
        if not grouped["calendar_days"].eq(7).all():
            raise ValueError("An item-store series lacks a day in a complete week")
        weekly_chunks.append(grouped)
    if not weekly_chunks:
        raise ValueError("No selected M5 sales rows were read")
    weekly = pd.concat(weekly_chunks, ignore_index=True)
    if weekly[["id", "week"]].duplicated().any():
        raise ValueError("Duplicate item-store-week rows after aggregation")
    weekly = weekly.merge(metadata, on="id", how="left", validate="many_to_one")
    weekly = weekly.merge(event_cov, on="week", validate="many_to_one")
    weekly = weekly.merge(snap_cov, on=["week", "state_id"], validate="many_to_one")
    weekly["snap_share"] = weekly["snap_days"] / 7.0
    weekly["observed_sales"] = weekly["priced_days"].eq(7)
    weekly["partial_price_week"] = weekly["priced_days"].between(1, 6)
    weekly["price_imputed"] = False
    weekly = weekly.sort_values(["sku", "week"]).reset_index(drop=True)
    previous_price = weekly.groupby("sku", sort=False)["price"].shift(1)
    previous_complete = weekly.groupby("sku", sort=False)["observed_sales"].shift(1).fillna(False)
    valid_change = weekly["observed_sales"] & previous_complete & previous_price.gt(0)
    weekly["price_change_pct"] = (weekly["price"] / previous_price - 1).where(valid_change)
    weekly["promotion_proxy"] = weekly["price_change_pct"].lt(-1e-6)
    weeks = pd.DatetimeIndex(sorted(weekly["week"].unique()))
    split_labels, split_boundaries = _split_weeks(weeks, split_profile)
    weekly["split"] = weekly["week"].map(split_labels)
    training_counts = weekly.loc[
        weekly["split"].eq("train") & weekly["observed_sales"]
    ].groupby("sku").size()
    excluded = sorted(set(weekly["sku"]) - set(training_counts.loc[training_counts.ge(2)].index))
    if excluded:
        if cold_start_policy == "error":
            raise ValueError(f"M5 series lack two fully priced training weeks: {excluded[:10]}")
        weekly = weekly.loc[~weekly["sku"].isin(excluded)].reset_index(drop=True)
        if weekly.empty:
            raise ValueError("No M5 series has two fully priced training weeks")
    weekly["month"] = weekly["week"].dt.month.astype(int)
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)
    weekly["qty_z"] = np.nan
    weekly["price_z"] = np.nan
    scalers = {}
    for sku, group in weekly.groupby("sku", sort=False):
        train = group.loc[group["split"].eq("train") & group["observed_sales"]]
        q_mean, q_std = float(train["qty"].mean()), _positive_std(train["qty"])
        p_mean, p_std = float(train["price"].mean()), _positive_std(train["price"])
        indices = group.index
        weekly.loc[indices, "qty_z"] = (group["qty"] - q_mean) / q_std
        weekly.loc[indices, "price_z"] = (group["price"] - p_mean) / p_std
        scalers[sku] = {
            "qty_mean": q_mean, "qty_std": q_std,
            "price_mean": p_mean, "price_std": p_std,
            "n_observed_train": int(len(train)),
        }
    manifest = {
        "schema_version": 1,
        "dataset": "M5 Forecasting Accuracy (official local CSVs)",
        "sales_source": str(sales_path.resolve()),
        "price_source": str(price_path.resolve()),
        "calendar_source": str(calendar_path.resolve()),
        "n_series": int(weekly["sku"].nunique()),
        "requested_series": int(len(metadata)),
        "excluded_cold_start_skus": excluded,
        "cold_start_policy": cold_start_policy,
        "n_weeks": int(len(weeks)),
        "first_week": weeks[0].date().isoformat(),
        "last_week": weeks[-1].date().isoformat(),
        "split_profile": split_profile,
        "split_weeks": split_boundaries,
        "rows_by_split": {name: int(weekly["split"].eq(name).sum())
                          for name in ("train", "validation", "test")},
        "fully_priced_rows": int(weekly["observed_sales"].sum()),
        "price_policy": "Mean of listed daily prices within each complete Monday-Sunday week; no imputation",
        "target_policy": "Real zero-unit weeks are valid when all seven days have a listed price",
        "promotion_policy": "Price decrease versus preceding fully priced week; proxy, not an M5 promotion label",
        "scaler_fit_policy": "Per-series population z-score from fully priced training weeks only",
        "scalers": scalers,
    }
    return weekly, manifest
