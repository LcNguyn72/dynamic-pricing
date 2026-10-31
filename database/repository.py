"""Queries used by the local pricing API."""

import json
from datetime import date
from decimal import Decimal

from database.connection import connect


def _value(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _rows(cursor):
    names = [column[0] for column in cursor.description]
    return [{name: _value(value) for name, value in zip(names, row)} for row in cursor.fetchall()]


def database_ready():
    connection = connect()
    try:
        return connection.cursor().execute("SELECT 1").fetchval() == 1
    finally:
        connection.close()


def overview():
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""SELECT COUNT(DISTINCT p.sku) AS n_products,
            COUNT(w.week_start) AS n_weekly_rows,
            MIN(w.week_start) AS first_week, MAX(w.week_start) AS last_week
            FROM dbo.Products AS p LEFT JOIN dbo.WeeklySales AS w ON w.sku=p.sku
            WHERE p.is_selected=1""")
        summary = _rows(cursor)[0]
        cursor.execute("""SELECT COUNT(*) FROM dbo.RecommendationHistory AS r
            INNER JOIN dbo.Products AS p ON p.sku=r.sku WHERE p.is_selected=1""")
        summary["n_recommendations"] = cursor.fetchval()
        return summary
    finally:
        connection.close()


def products(model_version):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""SELECT p.sku, p.description, p.tier, p.weeks, p.n_levels,
            p.price_median, p.price_min, p.price_max, p.qty_mean,
            s.p_ref AS reference_price, s.eps_mean AS elasticity_mean,
            s.eps_sd AS elasticity_sd
            FROM dbo.Products AS p INNER JOIN dbo.ProductModelStats AS s
            ON s.sku=p.sku AND s.model_version=?
            WHERE p.is_selected=1
            ORDER BY CASE p.tier WHEN N're' THEN 1 WHEN N'vua_thap' THEN 2
                WHEN N'vua_cao' THEN 3 WHEN N'dat' THEN 4 ELSE 5 END,
                p.selection_score DESC, p.sku""", model_version)
        return _rows(cursor)
    finally:
        connection.close()


def product(sku, model_version):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""SELECT p.sku, p.description, p.tier, p.weeks, p.n_levels,
            p.price_median, p.price_min, p.price_max, p.qty_mean,
            s.p_ref AS reference_price, s.eps_mean AS elasticity_mean,
            s.eps_sd AS elasticity_sd, s.n_train_weeks
            FROM dbo.Products AS p INNER JOIN dbo.ProductModelStats AS s
            ON s.sku=p.sku AND s.model_version=?
            WHERE p.sku=? AND p.is_selected=1""", model_version, sku)
        found = _rows(cursor)
        return found[0] if found else None
    finally:
        connection.close()


def weekly_sales(sku):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""SELECT w.week_start AS week, w.qty, w.price, w.qty_all
            FROM dbo.WeeklySales AS w INNER JOIN dbo.Products AS p ON p.sku=w.sku
            WHERE w.sku=? AND p.is_selected=1 ORDER BY w.week_start""", sku)
        return _rows(cursor)
    finally:
        connection.close()


def latest_price(sku, as_of):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""SELECT TOP (1) w.week_start AS week, w.price, w.qty
            FROM dbo.WeeklySales AS w INNER JOIN dbo.Products AS p ON p.sku=w.sku
            WHERE w.sku=? AND p.is_selected=1 AND w.week_start<=?
            ORDER BY w.week_start DESC""", sku, as_of)
        found = _rows(cursor)
        return found[0] if found else None
    finally:
        connection.close()


def save_recommendation(result, price_source, max_change):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""INSERT INTO dbo.RecommendationHistory
            (model_version, sku, scenario_date, current_price, price_source, max_change,
             recommended_price, target_price, action, prob_gain, expected_gain_pct,
             units_current, units_recommended, revenue_current, revenue_recommended, result_json)
            OUTPUT INSERTED.recommendation_id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            result["model_version"], result["sku"], result["date"], result["current_price"],
            price_source, max_change, result["recommended_price"], result["target_price"],
            result["action"], result["confidence"]["prob_gain"],
            result["confidence"]["expected_gain_pct"], result["expected"]["units_current"],
            result["expected"]["units_recommended"], result["expected"]["revenue_current"],
            result["expected"]["revenue_recommended"], json.dumps(result, ensure_ascii=False))
        recommendation_id = int(cursor.fetchval())
        connection.commit()
        return recommendation_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def recommendations(limit=20):
    limit = max(1, min(int(limit), 100))
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute(f"""SELECT TOP ({limit}) r.recommendation_id, r.created_at,
            r.sku, p.description, r.scenario_date, r.current_price, r.price_source,
            r.recommended_price, r.action, r.prob_gain, r.expected_gain_pct,
            r.revenue_current, r.revenue_recommended
            FROM dbo.RecommendationHistory AS r
            INNER JOIN dbo.Products AS p ON p.sku=r.sku
            WHERE p.is_selected=1
            ORDER BY r.recommendation_id DESC""")
        return _rows(cursor)
    finally:
        connection.close()
