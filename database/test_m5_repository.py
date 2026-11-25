"""SQL integration checks for M5 feedback and catalog price changes."""

from contextlib import closing
from datetime import date, timedelta
from decimal import Decimal
import os
import unittest

from .m5_connection import connect_m5
from .m5_repository import M5DataError, M5Repository


@unittest.skipUnless(os.environ.get("M5_SQL_TEST") == "1", "requires imported CA_1 in SQL Server")
class M5ReplayTests(unittest.TestCase):
    def test_archived_feedback_is_idempotent_and_excluded_from_its_own_forecast(self):
        week = date(2015, 8, 3)
        store, sku = "CA_1", "FOODS_3_090"
        with closing(connect_m5()) as connection:
            repo = M5Repository(connection)
            first = repo.save_feedback(store, sku, week, source="m5_history")
            retry = repo.save_feedback(store, sku, week, source="m5_history")
            self.assertEqual(first, retry)
            self.assertGreaterEqual(first["observed_qty"], 0)

            history, feedback = repo.model_inputs(store, sku, week)
            self.assertEqual(history[-1]["week"], week - timedelta(weeks=1))
            self.assertTrue(all(row["week"] < week for row in feedback))

            history, feedback = repo.model_inputs(store, sku, week + timedelta(weeks=1))
            self.assertEqual(history[-1]["week"], week)
            recorded = next(row for row in feedback if row["week"] == week)
            self.assertEqual(recorded["qty"], first["observed_qty"])
            self.assertEqual(recorded["previous_qty"], history[-2]["qty"])
            connection.rollback()

    def test_catalog_edit_is_immediate_and_preserves_m5_history(self):
        sku, store = "FOODS_1_096", "CA_1"
        with closing(connect_m5()) as connection:
            repo = M5Repository(connection)
            catalog = repo.catalog()
            self.assertEqual(len(catalog), 20)
            self.assertTrue(all(item["store_count"] == 10 for item in catalog))
            original = repo.catalog_item(sku)
            old_offers = {offer["store_id"]: offer for offer in original["offers"]}
            previous = old_offers[store]["price"]
            observed_qty = connection.execute("""
                SELECT quantity FROM dbo.M5WeeklySales WHERE item_id = ? AND store_id = ? AND week_start = ?
            """, sku, store, original["week"]).fetchval()
            self.assertEqual(old_offers[store]["observed_qty"], observed_qty)
            new_price = previous - Decimal("0.01") if previous > Decimal("0.01") else Decimal("0.02")

            self.assertTrue(repo.set_catalog_price(sku, store, new_price)["changed"])
            self.assertFalse(repo.set_catalog_price(sku, store, new_price)["changed"])
            updated = repo.catalog_item(sku)
            new_offers = {offer["store_id"]: offer for offer in updated["offers"]}
            self.assertEqual(new_offers[store]["price"], new_price)
            self.assertEqual(new_offers[store]["reference_price"], old_offers[store]["reference_price"])
            self.assertEqual(new_offers[store]["observed_qty"], observed_qty)
            self.assertEqual(new_offers["TX_1"], old_offers["TX_1"])
            self.assertEqual(updated["changes"][0]["old_price"], previous)
            self.assertEqual(updated["changes"][0]["new_price"], new_price)
            self.assertEqual(updated["changes"][0]["store_id"], store)
            connection.rollback()

    def test_batch_price_edit_rolls_back_on_invalid_store_and_updates_selected_offers(self):
        sku = "FOODS_3_219"
        stores = {"CA_1", "TX_1"}
        with closing(connect_m5()) as connection:
            repo = M5Repository(connection)
            original = repo.catalog_item(sku)
            old_prices = {offer["store_id"]: offer["price"] for offer in original["offers"]}
            new_price = next(Decimal(value) for value in ("9.91", "9.92", "9.93")
                             if Decimal(value) not in {old_prices[store] for store in stores})

            with self.assertRaises(M5DataError):
                repo.set_catalog_prices(sku, ["CA_1", "ZZ_9"], new_price)
            self.assertEqual(repo.catalog_item(sku), original)

            result = repo.set_catalog_prices(sku, ["TX_1", "CA_1"], new_price)
            self.assertTrue(all(row["changed"] for row in result))
            updated = repo.catalog_item(sku)
            for offer in updated["offers"]:
                store = offer["store_id"]
                self.assertEqual(offer["price"], new_price if store in stores else old_prices[store])
            self.assertEqual({row["store_id"] for row in updated["changes"][:2]}, stores)
            connection.rollback()

    def test_home_basket_reflects_current_price_and_rank_uses_historical_sales(self):
        sku, store = "FOODS_3_219", "CA_1"
        with closing(connect_m5()) as connection:
            repo = M5Repository(connection)
            original = repo.home_highlights()
            self.assertEqual(len(original["basket"]), 10)
            self.assertTrue(all(row["item_count"] == 20 for row in original["basket"]))
            self.assertEqual(len(original["best_sellers"]), 5)
            self.assertEqual(original["week"], date(2016, 5, 16))
            previous = next(offer["price"] for offer in repo.catalog_item(sku)["offers"]
                            if offer["store_id"] == store)
            delta = Decimal("0.25")
            repo.set_catalog_price(sku, store, previous + delta)
            updated = repo.home_highlights()
            old_totals = {row["store_id"]: row["total"] for row in original["basket"]}
            old_totals[store] += delta
            self.assertEqual({row["store_id"]: row["total"] for row in updated["basket"]}, old_totals)
            self.assertEqual(updated["best_sellers"], original["best_sellers"])
            connection.rollback()
