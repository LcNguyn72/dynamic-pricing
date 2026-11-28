"""Import the verified final_system M5 weekly panel into SQL Server."""

from __future__ import annotations

import argparse
from contextlib import closing
from decimal import Decimal
from itertools import islice
from pathlib import Path

import pandas as pd

from final_system.bundle import DEFAULT_PREPARED_ROOT, file_sha256, read_json

from .m5_connection import connect_m5


DEMO_STORES = ("CA_1", "TX_1")
COLUMNS = ["store_id", "state_id", "item_id", "dept_id", "cat_id", "week", "qty", "price", "revenue", "observed_sales"]
BATCH_SIZE = 10_000


def read_panel(prepared: Path, store: str, record: dict) -> pd.DataFrame:
    path = prepared / "m5" / f"weekly_{store}.csv"
    if file_sha256(path) != record["weekly_sha256"]:
        raise ValueError(f"The prepared weekly file does not match the manifest: {path}")
    panel = pd.read_csv(
        path, usecols=COLUMNS, parse_dates=["week"],
        dtype={"qty": "int32", "price": "float64", "revenue": "float64", "observed_sales": bool},
    )[COLUMNS]
    products = panel["item_id"].nunique()
    if (set(panel["store_id"]) != {store} or panel["state_id"].nunique() != 1
            or products != record["eligible_series"] or len(panel) != products * record["calendar_weeks"]):
        raise ValueError(f"Unexpected panel identifiers or row count for {store}")
    return panel


def weekly_rows(panel: pd.DataFrame):
    columns = ["store_id", "item_id", "week", "qty", "price", "revenue", "observed_sales"]
    for store, item, week, qty, price, revenue, observed in panel[columns].itertuples(index=False, name=None):
        yield store, item, week.date(), qty, float(price) if pd.notna(price) else None, Decimal(f"{revenue:.4f}"), observed


def import_store(connection, prepared: Path, record: dict) -> int:
    store = record["store"]
    digest = record["weekly_sha256"]
    cursor = connection.cursor()
    cursor.execute("""
        DECLARE @result int;
        EXEC @result = sp_getapplock @Resource = 'm5_panel_import', @LockMode = 'Exclusive',
                                     @LockOwner = 'Transaction', @LockTimeout = 15000;
        IF @result < 0 THROW 50001, 'Could not lock the M5 panel for import.', 1;
    """)
    existing = cursor.execute("SELECT source_sha256 FROM dbo.M5PanelImports WHERE store_id = ?", store).fetchone()
    if existing:
        if existing.source_sha256 != digest:
            raise ValueError(f"SQL Server already holds a different verified panel for {store}.")
        return 0

    panel = read_panel(prepared, store, record)
    cursor.execute("INSERT dbo.M5Stores (store_id, state_id) VALUES (?, ?)", store, panel["state_id"].iat[0])
    products = panel[["item_id", "dept_id", "cat_id"]].drop_duplicates()
    cursor.fast_executemany = True
    cursor.executemany("""
        INSERT dbo.M5Products (item_id, department_id, category_id)
        SELECT ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM dbo.M5Products WHERE item_id = ?)
    """, [(item, dept, category, item) for item, dept, category in products.itertuples(index=False, name=None)])
    cursor.executemany("INSERT dbo.M5StoreProducts (store_id, item_id) VALUES (?, ?)",
                       [(store, item) for item in products["item_id"]])

    source_rows = weekly_rows(panel)
    while rows := list(islice(source_rows, BATCH_SIZE)):
        cursor.executemany("""
            INSERT dbo.M5WeeklySales (store_id, item_id, week_start, quantity, price, revenue, observed_sales)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

    cursor.execute("""
        INSERT dbo.M5PanelImports (store_id, source_sha256, product_count, row_count, first_week, last_week)
        VALUES (?, ?, ?, ?, ?, ?)
    """, store, digest, len(products), len(panel), panel["week"].min().date(), panel["week"].max().date())
    return len(panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED_ROOT,
                        help="Path to the verified Phase 1 output")
    stores = parser.add_mutually_exclusive_group()
    stores.add_argument("--stores", nargs="+", metavar="STORE", help="Stores to import (default: CA_1 TX_1)")
    stores.add_argument("--all", action="store_true", help="Import all stores in the verified M5 panel")
    args = parser.parse_args()

    manifest = read_json(args.prepared / "phase1_manifest.json")
    records = {record["store"]: record for record in manifest["m5"]["stores"]}
    selected = list(records) if args.all else (args.stores or DEMO_STORES)
    if any(store not in records for store in selected):
        parser.error(f"Allowed stores: {', '.join(records)}")

    with closing(connect_m5()) as connection:
        for store in dict.fromkeys(selected):
            count = import_store(connection, args.prepared, records[store])
            connection.commit()
            print(f"{store}: imported {count:,} weekly rows" if count else f"{store}: verified panel already imported",
                  flush=True)


if __name__ == "__main__":
    main()
