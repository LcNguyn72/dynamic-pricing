"""Checks of overlap removal, train-fitted cleaning and cohort selection."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments.retail_ii_preprocess import (
    build_weekly_from_transactions,
    select_products_from_train,
)


class RetailIIPreprocessTests(unittest.TestCase):
    def test_train_only_outlier_mode_preserves_heldout_sales_and_revenue(self) -> None:
        weeks = pd.date_range("2020-01-06", periods=20, freq="7D")
        rows = [
            {"Invoice": f"N{i}", "StockCode": "12345A", "Description": "A",
             "Quantity": 1 if i % 2 == 0 else 2, "InvoiceDate": week,
             "Price": 2.0, "sheet": 0}
            for i, week in enumerate(weeks)
        ]
        for i in (0, 15, 19):
            rows.append({"Invoice": f"SPIKE{i}", "StockCode": "12345A",
                         "Description": "A", "Quantity": 100,
                         "InvoiceDate": weeks[i], "Price": 2.0, "sheet": 0})

        weekly, _, audit = build_weekly_from_transactions(
            pd.DataFrame(rows), outlier_application="train_only")
        by_week = weekly.set_index("week")
        self.assertEqual(by_week.loc[weeks[0], "qty"], 1)
        self.assertEqual(by_week.loc[weeks[0], "qty_all"], 101)
        self.assertEqual(by_week.loc[weeks[15], "qty"], 102)
        self.assertEqual(by_week.loc[weeks[15], "qty_all"], 102)
        self.assertEqual(by_week.loc[weeks[19], "qty"], 102)
        self.assertEqual(by_week.loc[weeks[19], "observed_revenue"], 204)
        self.assertEqual(audit["quantity_outlier_lines_removed"], 1)
        self.assertEqual(audit["heldout_outlier_lines_retained"], 2)

    def test_overlap_cancellation_outlier_and_line_revenue(self) -> None:
        rows = []
        weeks = pd.date_range("2020-01-06", periods=20, freq="7D")
        for i, week in enumerate(weeks):
            for sheet in ((0, 1) if i == 10 else (0,) if i < 10 else (1,)):
                rows.append({
                    "Invoice": f"N{i}-{sheet}", "StockCode": "12345A",
                    "Description": "A", "Quantity": 1 if i % 2 == 0 else 2,
                    "InvoiceDate": week, "Price": 1.0, "sheet": sheet,
                })
        rows.extend([
            {"Invoice": "EXTRA", "StockCode": "12345A", "Description": "A",
             "Quantity": 2, "InvoiceDate": weeks[0], "Price": 1.234, "sheet": 0},
            {"Invoice": "C123", "StockCode": "12345A", "Description": "A",
             "Quantity": -5, "InvoiceDate": weeks[0], "Price": 1.0, "sheet": 0},
            {"Invoice": "HIGH", "StockCode": "12345A", "Description": "A",
             "Quantity": 100, "InvoiceDate": weeks[15], "Price": 1.0, "sheet": 1},
        ])
        weekly, _, audit = build_weekly_from_transactions(pd.DataFrame(rows))
        first = weekly.loc[weekly["week"] == weeks[0]].iloc[0]
        overlap_week = weekly.loc[weekly["week"] == weeks[10]].iloc[0]
        self.assertEqual(audit["discarded_first_sheet_rows"], 1)
        self.assertEqual(audit["cancellation_or_negative_qty_rows"], 1)
        self.assertEqual(audit["quantity_outlier_lines_removed"], 1)
        self.assertEqual(first["qty"], 3)
        self.assertAlmostEqual(first["price"], 1.117)
        self.assertAlmostEqual(first["observed_revenue"], 3.468)
        self.assertEqual(overlap_week["qty"], 1)

    def test_train_only_selection_ignores_future_quantity_spike(self) -> None:
        rows = []
        weeks = pd.date_range("2020-01-06", periods=13, freq="7D")
        for tier in range(4):
            for position in range(6):
                sku = f"{tier}{position:04d}"
                for i, week in enumerate(weeks):
                    rows.append({
                        "sku": sku, "week": week,
                        "qty": 10000 if i == 12 and position == 0 else position + 1,
                        "price": float(10 * (tier + 1)) + (0.1 if i % 2 else 0.0),
                    })
        weekly = pd.DataFrame(rows)
        descriptions = pd.DataFrame({"sku": weekly["sku"].unique()})
        descriptions["description"] = descriptions["sku"]
        selected = select_products_from_train(
            weekly, descriptions, str(weeks[9].date()),
            n_per_tier=5, min_week_fraction=0.8, min_price_cv=0.0)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected["tier"].value_counts().tolist(), [5, 5, 5, 5])
        self.assertNotIn("00000", set(selected["sku"]))


if __name__ == "__main__":
    unittest.main()
