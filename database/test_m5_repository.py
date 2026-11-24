"""SQL integration checks for the M5 replay boundary, after importing CA_1."""

from contextlib import closing
from datetime import date, timedelta
import os
import unittest

from .m5_connection import connect_m5
from .m5_repository import M5Repository


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
