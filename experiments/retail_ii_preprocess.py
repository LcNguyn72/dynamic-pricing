"""Build a documented Online Retail II variant for the research pipeline.

This module is separate from ``1_data_processing_online_retail_2y.py`` and
never rewrites its artifacts or the existing 20-product demo.  It follows the
paper where the PDF is specific (returns contribute zero demand, 3*IQR outlier
rule, weekly mean observed price) and makes the unspecified decisions explicit.
Online Retail II is not the paper's one-year Online Retail dataset; the output
is an adaptation, not an exact reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import DEFAULT_SELECTED_PATH, PROJECT_ROOT


DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "online_retail_II_sheets.csv"
DEFAULT_XLSX_PATH = PROJECT_ROOT / "data" / "online_retail_II.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "artifacts" / "retail_ii_paper"
RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_transactions(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    xlsx_path: str | Path = DEFAULT_XLSX_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Read the existing raw cache, or both XLSX sheets if the cache is absent."""
    cache_path, xlsx_path = Path(cache_path), Path(xlsx_path)
    if cache_path.is_file():
        raw = pd.read_csv(
            cache_path,
            usecols=[*RAW_COLUMNS, "sheet"],
            dtype={"Invoice": str, "StockCode": str, "Description": str},
            parse_dates=["InvoiceDate"],
        )
        return raw, cache_path
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Neither raw cache nor XLSX exists: {cache_path}, {xlsx_path}")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    frames = []
    for sheet_index, (_, frame) in enumerate(sheets.items()):
        frame = frame[[*RAW_COLUMNS]].copy()
        frame["sheet"] = sheet_index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), xlsx_path


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


def select_products_from_train(
    weekly: pd.DataFrame,
    descriptions: pd.DataFrame,
    train_end: str,
    n_per_tier: int = 5,
    min_week_fraction: float = 0.85,
    min_price_cv: float = 0.05,
) -> pd.DataFrame:
    """Apply the legacy five-per-price-tier score using training weeks only."""
    train_end = pd.Timestamp(train_end)
    train = weekly.loc[weekly["week"] <= train_end].copy()
    all_train_weeks = pd.date_range(train["week"].min(), train_end, freq="7D")
    min_weeks = int(np.ceil(min_week_fraction * len(all_train_weeks)))
    stats = train.groupby("sku").agg(
        weeks=("week", "nunique"),
        n_levels=("price", "nunique"),
        qty_mean=("qty", "mean"),
        price_median=("price", "median"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        price_std=("price", "std"),
        price_mean=("price", "mean"),
    ).reset_index()
    stats["price_cv"] = (stats["price_std"] / stats["price_mean"]).fillna(0.0)
    stats = stats.loc[stats["weeks"].ge(min_weeks) & stats["price_cv"].ge(min_price_cv)].copy()
    if len(stats) < 4 * n_per_tier:
        raise ValueError(f"Only {len(stats)} products meet train-only selection criteria")
    stats["score"] = (stats["weeks"].rank() + stats["n_levels"].rank() + stats["qty_mean"].rank())
    stats = stats.merge(descriptions, on="sku", how="left")
    stats["description"] = stats["description"].fillna(stats["sku"])
    stats = stats.sort_values(["score", "sku"], ascending=[False, True]).drop_duplicates("description")
    labels = ["re", "vua_thap", "vua_cao", "dat"]
    stats["tier"] = pd.qcut(stats["price_median"].rank(method="first"), 4, labels=labels)
    selected = pd.concat(
        [group.nlargest(n_per_tier, "score") for _, group in stats.groupby("tier", observed=True)],
        ignore_index=True,
    )
    if len(selected) != 4 * n_per_tier:
        raise ValueError("Unable to select the requested number of products per price tier")
    return selected.drop(columns=["price_std", "price_mean"]).sort_values(["tier", "score"], ascending=[True, False])


def prepare_paper_aligned(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    xlsx_path: str | Path = DEFAULT_XLSX_PATH,
    selection_mode: str = "train_only",
    existing_selection_path: str | Path = DEFAULT_SELECTED_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Save weekly.csv, selected_products.csv and preprocessing_manifest.json."""
    if selection_mode not in ("train_only", "existing"):
        raise ValueError("selection_mode must be 'train_only' or 'existing'")
    raw, source_path = load_raw_transactions(cache_path, xlsx_path)
    weekly, descriptions, audit = build_weekly_from_transactions(raw)
    if selection_mode == "train_only":
        selected = select_products_from_train(weekly, descriptions, audit["fit_train_end"])
        selection_note = "20 SKUs selected by legacy-style score within first 70% calendar weeks only"
    else:
        existing_selection_path = Path(existing_selection_path)
        selected = pd.read_csv(existing_selection_path, dtype={"sku": str})
        absent = set(selected["sku"]) - set(weekly["sku"])
        if absent:
            raise ValueError(f"Existing selected SKUs absent after preprocessing: {sorted(absent)}")
        selection_note = "Legacy 20-SKU list reused; its upstream selection used full-period statistics"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly.csv", index=False)
    selected.to_csv(output_dir / "selected_products.csv", index=False)
    manifest = {
        "schema_version": 1,
        "dataset": "UCI Online Retail II 2009–2011; adaptation of Song (2026), not the paper's one-year Online Retail dataset",
        "source": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "selection_mode": selection_mode,
        "selection_note": selection_note,
        "selected_skus": selected["sku"].astype(str).tolist(),
        "selected_n": int(len(selected)),
        "audit": audit,
        "rules": {
            "sheet_overlap": "Keep sheet 0 before first date of sheet 1, then all of sheet 1",
            "product_code": "StockCode beginning with five digits, including alphanumeric suffixes, matching the legacy project; paper does not specify this filter",
            "returns": "Cancellation invoices and negative quantities contribute zero demand/revenue; no invoice-level return matching",
            "outliers": "Per-SKU transaction quantity outside Q1−3*IQR or Q3+3*IQR; thresholds fit on first 70% calendar weeks only; zero-IQR SKU unfiltered",
            "price": "Simple arithmetic mean of positive retained transaction line prices per SKU-week",
            "quantity": "Sum of positive retained transaction line quantities per SKU-week",
            "observed_revenue": "Exact sum of retained line qty*price using integer thousandths; gross revenue after cleaning, not net invoice revenue",
            "promotion": "No observed promotion indicator in Online Retail II; not fabricated",
        },
        "deviations_from_song_paper": [
            "Online Retail II covers two years; Song uses one-year UCI Online Retail and M5.",
            "The PDF does not specify whether its 3*IQR filter applies to transactions or weekly totals; we apply it to transaction quantity, using train-fitted thresholds.",
            "The PDF does not specify its SKU-code and product-selection filters; we use the legacy five-digit-prefix rule and an optional train-only 5-per-price-tier cohort.",
            "Online Retail II lacks observed promotion, inventory, competitor-price and counterfactual demand fields.",
        ],
    }
    with (output_dir / "preprocessing_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return weekly, selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-mode", choices=("train_only", "existing"), default="train_only")
    parser.add_argument("--existing-selection", type=Path, default=DEFAULT_SELECTED_PATH)
    args = parser.parse_args()
    weekly, selected, manifest = prepare_paper_aligned(
        args.output_dir, args.cache, args.xlsx, args.selection_mode, args.existing_selection)
    print(json.dumps({
        "weekly_rows": len(weekly),
        "selected_skus": len(selected),
        "selected_tiers": selected["tier"].astype(str).value_counts().to_dict() if "tier" in selected else None,
        "audit": manifest["audit"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
