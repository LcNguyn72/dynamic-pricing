"""Prepare Song's one-year UCI Online Retail forecast dataset.

The article does not publish code or a complete SKU eligibility rule. This
module keeps the unspecified choices explicit in its manifest. Protocol V2
uses the train-fitted 3*IQR filter only on training sales; the historical pilot
mode remains available. This module never reads the project's two-year Online
Retail II workbook.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .download_paper_data import DATA_ROOT
from .retail_ii_preprocess import build_weekly_from_transactions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = DATA_ROOT / "online_retail" / "Online Retail.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "artifacts" / "song_online_retail"
PAPER_URL = "https://doi.org/10.1007/s10791-026-10745-2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_train_eligible(weekly: pd.DataFrame, descriptions: pd.DataFrame,
                          train_end: pd.Timestamp, min_train_weeks: int = 2,
                          max_skus: int | None = None) -> pd.DataFrame:
    """Keep SKUs with enough observed training weeks, without seeing future sales."""
    train = weekly.loc[weekly["week"].le(train_end)]
    stats = train.groupby("sku").agg(
        n_observed_train_weeks=("week", "nunique"),
        train_qty=("qty", "sum"),
    ).reset_index()
    stats = stats.loc[stats["n_observed_train_weeks"].ge(min_train_weeks)]
    if max_skus is not None:
        if max_skus < 1:
            raise ValueError("max_skus must be positive")
        stats = stats.sort_values(
            ["n_observed_train_weeks", "train_qty", "sku"],
            ascending=[False, False, True],
        ).head(max_skus)
    else:
        stats = stats.sort_values("sku")
    selected = stats.reset_index(drop=True)
    selected = selected.merge(descriptions, on="sku", how="left", validate="one_to_one")
    return selected


def prepare(workbook: str | Path = DEFAULT_WORKBOOK,
            output_dir: str | Path = DEFAULT_OUTPUT,
            min_train_weeks: int = 2,
            max_skus: int | None = None,
            outlier_application: str = "all") -> dict:
    workbook, output_dir = Path(workbook), Path(output_dir)
    if min_train_weeks < 2:
        raise ValueError("At least two observed training weeks are needed for per-SKU scalers")
    if max_skus is not None and max_skus < 1:
        raise ValueError("max_skus must be positive")
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    raw = pd.read_excel(workbook, dtype={"InvoiceNo": str, "StockCode": str})
    expected = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"UCI workbook is missing columns: {sorted(expected - set(raw))}")
    if len(raw) != 541_909:
        raise ValueError(f"Expected 541,909 transactions from UCI Online Retail, found {len(raw):,}")
    first_date = pd.to_datetime(raw["InvoiceDate"], errors="raise").min()
    last_date = pd.to_datetime(raw["InvoiceDate"], errors="raise").max()
    if first_date.date().isoformat() != "2010-12-01" or last_date.date().isoformat() != "2011-12-09":
        raise ValueError(f"Unexpected UCI dates: {first_date}..{last_date}")
    normalized = raw.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price"})
    normalized = normalized[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price"]].copy()
    normalized["sheet"] = 0
    weekly, descriptions, audit = build_weekly_from_transactions(
        normalized, outlier_application=outlier_application)
    train_end = pd.Timestamp(audit["fit_train_end"])
    eligible_before_limit = int(
        weekly.loc[weekly["week"].le(train_end)]
        .groupby("sku")["week"].nunique().ge(min_train_weeks).sum()
    )
    selected = select_train_eligible(weekly, descriptions, train_end, min_train_weeks, max_skus)
    if selected.empty:
        raise ValueError("No SKU has enough observed training weeks")
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_path = output_dir / "weekly.csv"
    selected_path = output_dir / "selected_products.csv"
    weekly.to_csv(weekly_path, index=False)
    selected.to_csv(selected_path, index=False)
    selection_note = (
        f"At least {min_train_weeks} observed positive-sale training weeks; "
        + (f"top {max_skus} ranked by train-only observed weeks, then train quantity, then SKU"
           if max_skus is not None else "all eligible SKUs, ordered by SKU")
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_URL,
        "dataset": "UCI Online Retail ID 352, 2010-12-01 to 2011-12-09",
        "source": str(workbook.resolve()),
        "source_sha256": _sha256(workbook),
        "raw_transaction_rows": len(raw),
        "first_transaction": first_date.isoformat(),
        "last_transaction": last_date.isoformat(),
        "first_week": weekly["week"].min().date().isoformat(),
        "last_week": weekly["week"].max().date().isoformat(),
        "weekly_rows_all_valid_skus": len(weekly),
        "weekly_sha256": _sha256(weekly_path),
        "eligible_skus": eligible_before_limit,
        "selected_skus": selected["sku"].astype(str).tolist(),
        "selected_sha256": _sha256(selected_path),
        "selected_n": len(selected),
        "max_skus": max_skus,
        "outlier_application": outlier_application,
        "selection_note": selection_note,
        "eligibility": selection_note,
        "audit": audit,
        "known_paper_ambiguities": [
            "The article does not publish its SKU eligibility or product-code exclusions; five-digit-prefix product codes and two observed train weeks are explicit project assumptions.",
            "The article specifies 3*IQR but not whether it filters transaction quantities or weekly totals; this implementation filters positive transaction quantities using thresholds fit in train.",
            "No true promotion flag exists in the UCI workbook; none is invented.",
            "No holiday calendar is bundled with UCI Online Retail; the holiday/event input is zero for this dataset.",
            "Weeks without a transaction are context with zero quantity, but are not scored as confirmed demand targets in the current shared window builder.",
        ],
        "deviations_from_song_paper": [
            "The article does not publish exact SKU eligibility or cohort; this cohort is chosen using training weeks only.",
            "The article does not specify whether 3*IQR applies to transactions or weekly totals; this implementation filters transaction quantity using train-fitted thresholds.",
            "UCI Online Retail has no observed promotion or holiday flags; these are not fabricated.",
        ],
        "processing": (
            "Cancellation and negative-quantity lines contribute zero; train positive lines use train-fitted 3*IQR filter; "
            "validation/test retain all valid positive lines; SKU-week price is arithmetic mean of retained line prices"
            if outlier_application == "train_only" else
            "Cancellation and negative-quantity lines contribute zero; positive valid lines use train-fitted 3*IQR filter in every split; SKU-week price is arithmetic mean of retained line prices"
        ),
        "provenance_note": "Only the downloaded one-year UCI workbook is read; the existing Online Retail II demo is unchanged",
    }
    with (output_dir / "preprocessing_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[UCI Online Retail] {len(raw):,} transactions, {len(weekly):,} SKU-weeks, {len(selected):,} train-eligible SKUs", flush=True)
    print(output_dir / "preprocessing_manifest.json", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    parser.add_argument("--max-skus", type=int, default=None,
                        help="Optional train-ranked cohort size for a CPU-limited common benchmark")
    parser.add_argument("--outlier-application", choices=("all", "train_only"), default="all",
                        help="Use train_only for protocol V2; all preserves the historical pilot")
    args = parser.parse_args()
    prepare(args.workbook, args.output_dir, args.min_train_weeks, args.max_skus,
            args.outlier_application)


if __name__ == "__main__":
    main()
