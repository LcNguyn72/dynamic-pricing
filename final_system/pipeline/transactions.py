"""Transaction validity, train-only outlier rules and weekly aggregation."""

import numpy as np
import pandas as pd


RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price"]


def remove_sheet_overlap(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep sheet 2 for the overlapping date range, as the legacy importer does."""
    if "sheet" not in raw:
        raise ValueError("Raw rows need a sheet index to resolve the two-year overlap")
    observed_sheets = set(pd.to_numeric(raw["sheet"], errors="raise").dropna().astype(int))
    if not observed_sheets.issubset({0, 1}):
        raise ValueError(f"Expected sheet indexes 0 and 1, got {sorted(observed_sheets)}")
    first, second = raw.loc[raw["sheet"] == 0], raw.loc[raw["sheet"] == 1]
    if second.empty:
        return raw.copy(), {"overlap_start": None, "discarded_first_sheet_rows": 0}
    cutoff = second["InvoiceDate"].min().normalize()
    discarded = int((first["InvoiceDate"] >= cutoff).sum())
    deduped = pd.concat([first.loc[first["InvoiceDate"] < cutoff], second], ignore_index=True)
    return deduped, {
        "overlap_start": cutoff.date().isoformat(),
        "discarded_first_sheet_rows": discarded,
    }


def build_weekly_from_transactions(
    raw: pd.DataFrame, *, outlier_application: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return all valid SKU-weeks, train-only descriptions, and audit counts.

    Outlier limits use positive transaction quantities from the *first 70% of
    calendar weeks only*, keeping test observations out of preprocessing fit.
    ``train_only`` applies these limits only to train, preserving every valid
    held-out sale for the V2 primary test; ``all`` keeps the historical demo.
    An IQR of zero leaves that SKU unfiltered rather than deleting all nonmodal
    sales.  A retained line's revenue is summed in integer thousandths, which
    is exact for the supplied dataset's prices (at most three decimal places).
    """
    if outlier_application not in ("all", "train_only"):
        raise ValueError("outlier_application must be 'all' or 'train_only'")
    missing = set([*RAW_COLUMNS, "sheet"]) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    work = raw.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="raise")
    work, overlap = remove_sheet_overlap(work)
    n_after_overlap = len(work)
    work = work.rename(columns={
        "StockCode": "sku", "Quantity": "qty", "Price": "price",
        "InvoiceDate": "date", "Description": "description",
    })
    work["sku"] = work["sku"].astype("string")
    work["Invoice"] = work["Invoice"].astype("string")
    work["qty"] = pd.to_numeric(work["qty"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["week"] = work["date"].dt.to_period("W-SUN").dt.start_time
    if work["week"].isna().any():
        raise ValueError("Some invoice dates could not be mapped to weeks")
    weeks = pd.date_range(work["week"].min(), work["week"].max(), freq="7D")
    n_train = int(np.floor(0.70 * len(weeks)))
    train_end = weeks[n_train - 1]

    cancel = work["Invoice"].str.upper().str.startswith("C", na=False) | work["qty"].lt(0)
    # Match the legacy project convention: five leading digits; a colour or
    # variant suffix such as 84029G is still a distinct valid product code.
    product_code = work["sku"].str.match(r"^\d{5}", na=False)
    valid = product_code & work["qty"].gt(0) & work["price"].gt(0) & ~cancel
    sales = work.loc[valid, ["sku", "description", "qty", "price", "week"]].copy()
    if sales.empty:
        raise ValueError("No positive product sales remain after basic cleaning")
    if not np.allclose(sales["qty"], np.rint(sales["qty"]), rtol=0, atol=1e-8):
        raise ValueError("Quantity contains non-integer values")
    if not np.allclose(sales["price"] * 1000, np.rint(sales["price"] * 1000), rtol=0, atol=1e-7):
        raise ValueError("Raw prices exceed the supported 0.001 precision")
    sales["qty"] = sales["qty"].astype(np.int64)
    sales["price_mills"] = np.rint(sales["price"] * 1000).astype(np.int64)

    fit_rows = sales.loc[sales["week"] <= train_end]
    quartiles = fit_rows.groupby("sku")["qty"].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["q1", "q3"]
    quartiles["iqr"] = quartiles["q3"] - quartiles["q1"]
    sales = sales.join(quartiles, on="sku")
    outlier = (
        sales["iqr"].gt(0)
        & (sales["qty"].lt((sales["q1"] - 3 * sales["iqr"]).clip(lower=0))
           | sales["qty"].gt(sales["q3"] + 3 * sales["iqr"]))
    ).fillna(False)
    removed = outlier if outlier_application == "all" else outlier & sales["week"].le(train_end)
    n_outlier = int(removed.sum())
    retained = sales.loc[~removed].copy()
    retained["line_revenue_mills"] = retained["qty"] * retained["price_mills"]

    weekly = retained.groupby(["sku", "week"], sort=True).agg(
        qty=("qty", "sum"),
        price=("price", "mean"),
        revenue_mills=("line_revenue_mills", "sum"),
        n_lines=("qty", "size"),
        n_price_levels=("price_mills", "nunique"),
    ).reset_index()
    weekly["observed_revenue"] = weekly.pop("revenue_mills") / 1000
    qty_before_outlier = sales.groupby(["sku", "week"])["qty"].sum().rename("qty_all")
    weekly = weekly.join(qty_before_outlier, on=["sku", "week"])
    weekly = weekly[["sku", "week", "qty", "price", "qty_all", "observed_revenue", "n_lines", "n_price_levels"]]
    # The description and all selection statistics are restricted to train.
    descriptions = (
        retained.loc[retained["week"] <= train_end, ["sku", "description"]]
        .dropna(subset=["description"])
        .groupby("sku")["description"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "")
        .rename("description")
        .reset_index()
    )
    audit = {
        "raw_rows": int(len(raw)),
        "rows_after_overlap": int(n_after_overlap),
        **overlap,
        "cancellation_or_negative_qty_rows": int(cancel.sum()),
        "non_5_digit_prefix_sku_rows": int((~product_code).sum()),
        "nonpositive_or_missing_qty_price_rows": int((~(work["qty"].gt(0) & work["price"].gt(0))).sum()),
        "positive_valid_product_lines": int(len(sales)),
        "outlier_application": outlier_application,
        "quantity_outlier_lines_flagged": int(outlier.sum()),
        "quantity_outlier_lines_removed": n_outlier,
        "heldout_outlier_lines_retained": int((outlier & sales["week"].gt(train_end) & ~removed).sum()),
        "retained_sales_lines": int(len(retained)),
        "weekly_rows": int(len(weekly)),
        "calendar_weeks": int(len(weeks)),
        "fit_train_end": train_end.date().isoformat(),
        "outlier_fit_rows": int(len(fit_rows)),
        "outlier_skus_with_train_history": int(len(quartiles)),
    }
    return weekly, descriptions, audit
