"""Catalog, weekly M5 observations, and confirmed pricing outcomes."""

from __future__ import annotations

from datetime import date, timedelta
from math import isclose

from .m5_connection import connect_m5


class M5DataError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


class M5Repository:
    def __init__(self, db):
        self.db = db

    def stores(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT s.store_id, s.state_id, p.product_count
            FROM dbo.M5Stores s JOIN dbo.M5PanelImports p ON p.store_id = s.store_id
            ORDER BY s.store_id
        """)
        return [{"id": row.store_id, "state": row.state_id, "item_count": row.product_count} for row in rows]

    def _archive_end(self, store: str) -> date:
        row = self.db.execute("SELECT last_week FROM dbo.M5PanelImports WHERE store_id = ?", store).fetchone()
        if row is None:
            raise M5DataError("unknown_store", "Cửa hàng chưa được nhập dữ liệu M5.", 404)
        return row.last_week

    def _check_item(self, store: str, item: str, *, lock=False) -> None:
        hint = " WITH (UPDLOCK, HOLDLOCK)" if lock else ""
        query = f"SELECT 1 FROM dbo.M5StoreProducts{hint} WHERE store_id = ? AND item_id = ?"
        if self.db.execute(query, store, item).fetchone() is None:
            raise M5DataError("unknown_product", "Sản phẩm chưa có dữ liệu tại cửa hàng đã chọn.", 404)

    def products(self, store: str, search: str, limit: int, offset: int) -> dict:
        self._archive_end(store)
        match = """
            FROM dbo.M5StoreProducts sp JOIN dbo.M5Products p ON p.item_id = sp.item_id
            WHERE sp.store_id = ? AND (
                ? = '' OR CHARINDEX(?, p.item_id) > 0 OR CHARINDEX(?, p.category_id) > 0 OR CHARINDEX(?, p.department_id) > 0
            )
        """
        params = store, search, search, search, search
        total = self.db.execute(f"SELECT COUNT_BIG(*) {match}", *params).fetchval()
        rows = self.db.execute(f"""
            WITH ProductPage AS (
                SELECT sp.store_id, sp.item_id, p.category_id, p.department_id {match}
                ORDER BY sp.item_id OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            )
            SELECT p.item_id, p.category_id, p.department_id, latest.week_start, latest.price
            FROM ProductPage p
            OUTER APPLY (
                SELECT TOP (1) observations.week_start, observations.price
                FROM (
                    SELECT archived.week_start, archived.price FROM (
                        SELECT TOP (1) week_start, price
                        FROM dbo.M5WeeklySales
                        WHERE store_id = p.store_id AND item_id = p.item_id AND observed_sales = 1
                        ORDER BY week_start DESC
                    ) archived
                    UNION ALL
                    SELECT actual.week_start, actual.applied_price FROM (
                        SELECT TOP (1) week_start, applied_price
                        FROM dbo.M5Feedback
                        WHERE store_id = p.store_id AND item_id = p.item_id AND source = 'actual'
                        ORDER BY week_start DESC
                    ) actual
                ) observations
                ORDER BY observations.week_start DESC
            ) latest
            ORDER BY p.item_id
        """, *params, offset, limit)
        return {"store_id": store, "total": total, "items": [
            {"sku": row.item_id, "category": row.category_id, "department": row.department_id,
             "last_price": row.price, "latest_week": row.week_start} for row in rows
        ]}

    def product_weeks(self, store: str, item: str, limit: int, before: date | None = None) -> dict:
        self._check_item(store, item)
        rows = self.db.execute("""
            WITH Weeks AS (
                SELECT week_start, quantity, price, revenue, observed_sales
                FROM dbo.M5WeeklySales WHERE store_id = ? AND item_id = ?
                UNION ALL
                SELECT week_start, observed_qty, applied_price,
                       CAST(applied_price * observed_qty AS decimal(24, 4)), CAST(1 AS bit)
                FROM dbo.M5Feedback WHERE store_id = ? AND item_id = ? AND source = 'actual'
            )
            SELECT TOP (?) week_start, quantity, price, revenue, observed_sales
            FROM Weeks WHERE (? IS NULL OR week_start < ?) ORDER BY week_start DESC
        """, store, item, store, item, limit, before, before)
        return {"items": [
            {"week": row.week_start, "qty": row.quantity, "price": row.price,
             "revenue": float(row.revenue), "observed_sales": bool(row.observed_sales)} for row in rows
        ]}

    def stats(self, store: str) -> dict:
        self._archive_end(store)
        product_count = self.db.execute("SELECT product_count FROM dbo.M5PanelImports WHERE store_id = ?", store).fetchval()
        rows = self.db.execute("""
            WITH Sales AS (
                SELECT week_start, quantity, revenue FROM dbo.M5WeeklySales WHERE store_id = ?
                UNION ALL
                SELECT week_start, observed_qty, CAST(applied_price * observed_qty AS decimal(24, 4))
                FROM dbo.M5Feedback WHERE store_id = ? AND source = 'actual'
            )
            SELECT week_start, SUM(CAST(quantity AS bigint)) AS qty, SUM(revenue) AS revenue
            FROM Sales GROUP BY week_start ORDER BY week_start
        """, store, store).fetchall()
        return {"store_id": store, "product_count": product_count, "total_qty": sum(row.qty for row in rows),
                "total_revenue": float(sum(row.revenue for row in rows)), "latest_week": rows[-1].week_start,
                "weekly": [{"week": row.week_start, "qty": row.qty, "revenue": float(row.revenue)} for row in rows]}

    def model_inputs(self, store: str, item: str, before: date) -> tuple[list[dict], list[dict]]:
        self._check_item(store, item)
        observations = self.db.execute("""
            WITH Observations AS (
                SELECT week_start, quantity, price FROM dbo.M5WeeklySales
                WHERE store_id = ? AND item_id = ? AND observed_sales = 1 AND week_start < ?
                UNION ALL
                SELECT week_start, observed_qty, applied_price FROM dbo.M5Feedback
                WHERE store_id = ? AND item_id = ? AND source = 'actual' AND week_start < ?
            )
            SELECT TOP (277) o.week_start, o.quantity, o.price, f.week_start AS feedback_week
            FROM Observations o
            LEFT JOIN dbo.M5Feedback f ON f.store_id = ? AND f.item_id = ? AND f.week_start = o.week_start
            ORDER BY o.week_start DESC
        """, store, item, before, store, item, before, store, item).fetchall()
        by_week = {row.week_start: row for row in observations}
        feedback = []
        for row in observations:
            if row.feedback_week is None:
                continue
            previous = by_week.get(row.week_start - timedelta(weeks=1))
            if previous is not None:
                feedback.append({"week": row.week_start, "qty": row.quantity, "price": row.price,
                                 "previous_qty": previous.quantity, "previous_price": previous.price})
            if len(feedback) == 32:
                break
        history = [{"week": row.week_start, "qty": row.quantity, "price": row.price}
                   for row in reversed(observations)]
        return history, feedback[::-1]

    def save_feedback(self, store: str, item: str, week: date, *, source: str,
                      applied_price: float | None = None, observed_qty: int | None = None) -> dict:
        archive_end = self._archive_end(store)
        self._check_item(store, item, lock=True)
        if week <= archive_end:
            archived = self.db.execute("""
                SELECT quantity, price FROM dbo.M5WeeklySales
                WHERE store_id = ? AND item_id = ? AND week_start = ? AND observed_sales = 1
            """, store, item, week).fetchone()
            if archived is None:
                raise M5DataError("archive_week_unavailable", "Tuần này chưa có quan sát M5 đầy đủ.")
            if source == "actual" and (observed_qty != archived.quantity or not isclose(
                    applied_price, archived.price, rel_tol=1e-8, abs_tol=1e-8)):
                raise M5DataError("archive_feedback_mismatch", "Kết quả nhập khác số liệu đã ghi trong M5.", 409)
            applied_price, observed_qty = archived.price, archived.quantity
            source = "m5_history"
        elif source == "m5_history":
            raise M5DataError("archive_week_unavailable", "Tuần này nằm ngoài dữ liệu lịch sử M5.")

        history, _ = self.model_inputs(store, item, week)
        if not history or history[-1]["week"] != week - timedelta(weeks=1):
            raise M5DataError("missing_previous_week", "Cần giá và lượng bán đã biết của tuần liền trước.")
        existing = self.db.execute("""
            SELECT applied_price, observed_qty, source FROM dbo.M5Feedback
            WHERE store_id = ? AND item_id = ? AND week_start = ?
        """, store, item, week).fetchone()
        if existing:
            if existing.observed_qty != observed_qty or existing.applied_price != applied_price:
                raise M5DataError("feedback_already_recorded", "Tuần này đã có kết quả thực tế khác.", 409)
        else:
            self.db.execute("""
                INSERT dbo.M5Feedback (store_id, item_id, week_start, applied_price, observed_qty, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, store, item, week, applied_price, observed_qty, source)
        return {"store_id": store, "sku": item, "week": week,
                "applied_price": applied_price, "observed_qty": observed_qty, "source": source}
