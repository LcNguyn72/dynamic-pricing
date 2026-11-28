"""Normalize Lazada crawler snapshots into a source panel for final_system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..bundle import PROJECT_ROOT, file_sha256


DEFAULT_DAILY = PROJECT_ROOT / "crawler" / "data" / "daily_product_info.csv"
DEFAULT_PRODUCTS = PROJECT_ROOT / "crawler" / "data" / "ref_products.csv"
PERIOD_DAYS = 3


def cap_spikes_after_aggregation(panel, train_periods, modified_z_threshold=3.5):
    """Flag and cap high SKU-period quantities using train-only log-MAD limits.

    The three-day panel is already aggregated when this is called. Raw quantities
    are retained in ``qty_raw``; ``qty`` is the capped target for model fitting.
    """
    result = panel.copy()
    result["qty_raw"] = result["qty"].astype(float)
    result["qty_spike"] = False
    result["qty_cap"] = np.nan
    train_mask = result["week"].isin(train_periods)
    audit_rows = []

    for sku, indices in result.groupby("sku", sort=True).groups.items():
        sku_train = result.loc[indices].loc[train_mask.loc[indices], "qty_raw"]
        if len(sku_train) < 5:
            audit_rows.append({"sku": str(sku), "status": "too_few_train_periods"})
            continue
        log_values = np.log1p(sku_train.to_numpy(dtype=float))
        median = float(np.median(log_values))
        mad = float(np.median(np.abs(log_values - median)))
        if not np.isfinite(mad) or mad <= 0:
            audit_rows.append({"sku": str(sku), "status": "zero_mad_no_cap"})
            continue
        log_cap = median + modified_z_threshold * 1.4826 * mad
        cap = float(np.expm1(log_cap))
        result.loc[indices, "qty_cap"] = cap
        spike_mask = result.loc[indices, "qty_raw"].gt(cap)
        spike_indices = spike_mask.index[spike_mask]
        result.loc[spike_indices, "qty_spike"] = True
        result.loc[spike_indices, "qty"] = cap
        audit_rows.append({
            "sku": str(sku), "status": "capped" if len(spike_indices) else "no_spikes",
            "train_periods": int(len(sku_train)), "log_median": median,
            "log_mad": mad, "quantity_cap": cap,
            "capped_periods": int(len(spike_indices)),
        })

    audit = {
        "method": "per_sku_log1p_median_plus_3.5_scaled_MAD",
        "threshold_multiplier": float(modified_z_threshold),
        "threshold_fit_periods": int(len(train_periods)),
        "thresholds_fit_on_train_only": True,
        "panel_rows_flagged": int(result["qty_spike"].sum()),
        "sku_thresholds": audit_rows,
    }
    return result, audit


def load_lazada_panel(daily_path=DEFAULT_DAILY, products_path=DEFAULT_PRODUCTS):
    """Return normalized three-day SKU periods, product info, and source audit."""
    daily_path, products_path = Path(daily_path), Path(products_path)
    if not daily_path.is_file():
        raise FileNotFoundError(daily_path)
    if not products_path.is_file():
        raise FileNotFoundError(products_path)

    daily = pd.read_csv(daily_path, encoding="utf-8-sig")
    products = pd.read_csv(products_path, encoding="utf-8-sig")
    daily.columns = daily.columns.astype(str).str.strip()
    products.columns = products.columns.astype(str).str.strip()
    required = {"crawl_date", "id", "price", "daily_sold"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily_product_info.csv missing columns: {sorted(missing)}")
    if "id" not in products:
        raise ValueError("ref_products.csv missing id column")

    n_raw = len(daily)
    daily["crawl_date"] = pd.to_datetime(daily["crawl_date"], errors="coerce")
    daily["id"] = daily["id"].astype("string").str.strip()
    for column in ("price", "daily_sold", "review", "rating_score", "sold"):
        if column in daily:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.dropna(subset=["crawl_date", "id", "price", "daily_sold"]).copy()
    daily = daily.loc[daily["price"].gt(0) & daily["daily_sold"].ge(0)]
    daily = daily.sort_values(["id", "crawl_date"])
    duplicate_rows = int(daily.duplicated(["id", "crawl_date"], keep="last").sum())
    daily = daily.drop_duplicates(["id", "crawl_date"], keep="last")
    if daily.empty:
        raise ValueError("No valid Lazada rows remain after cleaning")

    daily["crawl_day"] = daily["crawl_date"].dt.normalize()
    start = daily["crawl_day"].min()
    offsets = (daily["crawl_day"] - start).dt.days
    daily["week"] = start + pd.to_timedelta((offsets // PERIOD_DAYS) * PERIOD_DAYS, unit="D")
    aggregations = {
        "qty": ("daily_sold", "sum"),
        "price": ("price", "median"),
        "days_observed": ("crawl_day", "nunique"),
    }
    for column, operation in (("review", "max"), ("rating_score", "mean"),
                              ("sold", "max"), ("is_sponsored", "max")):
        if column in daily:
            aggregations[column] = (column, operation)
    panel = daily.groupby(["id", "week"], sort=True).agg(**aggregations).reset_index()
    panel = panel.rename(columns={"id": "sku"})
    panel["observed_sales"] = True
    panel["split"] = "unassigned"

    products["id"] = products["id"].astype("string").str.strip()
    products = products.drop_duplicates("id", keep="last").rename(
        columns={"name": "description", "ratingScore": "rating_score",
                 "isSponsored": "is_sponsored"}
    )
    product_info = panel.groupby("sku", sort=True).agg(
        periods=("week", "nunique"), price_median=("price", "median"),
        price_min=("price", "min"), price_max=("price", "max"),
        qty_mean=("qty", "mean"), qty_total=("qty", "sum"),
    ).reset_index().merge(products, left_on="sku", right_on="id", how="left")
    product_info = product_info.drop(columns="id", errors="ignore")

    audit = {
        "source": "Lazada crawler daily snapshots",
        "daily_file": str(daily_path.resolve()),
        "daily_sha256": file_sha256(daily_path),
        "products_file": str(products_path.resolve()),
        "products_sha256": file_sha256(products_path),
        "raw_rows": n_raw,
        "valid_daily_rows": int(len(daily)),
        "duplicate_rows_removed": duplicate_rows,
        "date_start": daily["crawl_date"].min().date().isoformat(),
        "date_end": daily["crawl_date"].max().date().isoformat(),
        "period_days": PERIOD_DAYS,
        "panel_rows": int(len(panel)),
        "periods": int(panel["week"].nunique()),
        "skus": int(panel["sku"].nunique()),
    }
    return panel, product_info, audit
