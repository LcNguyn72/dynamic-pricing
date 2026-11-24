"""Make a bounded real-M5 pipeline smoke test from a train-only SKU ranking.

This output is explicitly a pilot, not Song's full 3,049-item/10-store M5 run.
The selected item IDs come only from the first 70% of complete calendar weeks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .data import make_windows
from .download_paper_data import DATA_ROOT, file_hash
from .m5 import load_m5_weekly


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"


def select_items(m5_dir: Path, store_id: str, count: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    calendar = pd.read_csv(m5_dir / "calendar.csv", usecols=["date", "d"])
    calendar = calendar.loc[calendar["d"].str.split("_").str[1].astype(int).le(1913)].copy()
    calendar["week"] = pd.to_datetime(calendar["date"]).dt.to_period("W-SUN").dt.start_time
    complete = calendar.groupby("week")["d"].size()
    calendar = calendar.loc[calendar["week"].isin(complete.loc[complete.eq(7)].index)].copy()
    weeks = sorted(calendar["week"].unique())
    train_end = weeks[int(0.70 * len(weeks)) - 1]
    train_days = [str(d) for d in calendar.loc[calendar["week"].le(train_end), "d"]]
    scored = []
    with (m5_dir / "sales_train_validation.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["store_id"] == store_id:
                scored.append((sum(int(row[day]) for day in train_days), row["item_id"]))
    if not scored:
        raise ValueError(f"No M5 series for store {store_id}")
    return [item for _, item in sorted(scored, key=lambda value: (-value[0], value[1]))[:count]]


def prepare(m5_dir: str | Path = DATA_ROOT / "m5", output_dir: str | Path = DEFAULT_OUTPUT,
            store_id: str = "CA_1", count: int = 20) -> dict:
    m5_dir, output_dir = Path(m5_dir), Path(output_dir)
    # A pricing simulator needs at least two observed training prices. Rank
    # demand using training days, then apply this eligibility rule using only
    # training weeks; a future price change must not make a series eligible.
    candidates = select_items(m5_dir, store_id, min(3049, max(count * 2, count + 10)))
    weekly, manifest = load_m5_weekly(m5_dir, item_ids=candidates, store_ids=[store_id])
    eligible = set(
        weekly.loc[weekly["split"].eq("train") & weekly["observed_sales"]
                   & weekly["price"].gt(0)]
        .groupby("item_id")["price"].nunique()
        .loc[lambda levels: levels.ge(2)].index
    )
    items = [item for item in candidates if item in eligible][:count]
    if len(items) != count:
        raise ValueError(f"Only {len(items)} of {count} requested M5 series have two training prices")
    weekly = weekly.loc[weekly["item_id"].isin(items)].copy()
    manifest["n_series"] = int(weekly["sku"].nunique())
    manifest["requested_series"] = count
    manifest["rows_by_split"] = {split: int(weekly["split"].eq(split).sum())
                                 for split in ("train", "validation", "test")}
    manifest["fully_priced_rows"] = int(weekly["observed_sales"].sum())
    manifest["scalers"] = {sku: scale for sku, scale in manifest["scalers"].items()
                           if sku in set(weekly["sku"])}
    windows = make_windows(weekly, lookback=12)
    manifest.update({
        "scope": "PILOT ONLY; train-ranked items from one store, not Song's complete M5 experiment",
        "pilot_store": store_id, "pilot_items": items,
        "selected_skus": [f"{item}@{store_id}" for item in items],
        "selection_rule": "Top item IDs by units in first 70% complete calendar weeks, requiring at least two observed training-week prices",
        "selection_candidates": len(candidates),
        "ineligible_candidate_items": [item for item in candidates if item not in eligible],
        "lookback_weeks": 12,
        "window_counts": {split: int((windows["split"].eq(split) & windows["observed_sales"]).sum())
                          for split in ("train", "validation", "test")},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_path = output_dir / "weekly.csv"
    windows_path = output_dir / "windows.csv"
    weekly.to_csv(weekly_path, index=False)
    windows.to_csv(windows_path, index=False)
    manifest["source_sha256"] = {
        name: file_hash(m5_dir / name, "sha256")
        for name in ("sales_train_validation.csv", "sell_prices.csv", "calendar.csv")
    }
    manifest["weekly_sha256"] = file_hash(weekly_path, "sha256")
    manifest["windows_sha256"] = file_hash(windows_path, "sha256")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M5 pilot] {len(items)} item-store series, {len(weekly)} weeks, "
          f"{manifest['window_counts']} observed windows", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m5-dir", type=Path, default=DATA_ROOT / "m5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--store-id", default="CA_1")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    prepare(args.m5_dir, args.output_dir, args.store_id, args.count)


if __name__ == "__main__":
    main()
