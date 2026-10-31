"""Import only the 20 products selected by 3_select_products_2y.py."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from database.connection import connect
from pricing_engine import MODEL_VERSION


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models_2y"


def load_inputs():
    selected = pd.read_csv(MODEL_DIR / "selected_products_online_retail.csv", dtype={"sku": str})
    weekly = pd.read_csv(MODEL_DIR / "weekly_online_retail.csv", dtype={"sku": str}, parse_dates=["week"])
    params = pd.read_csv(MODEL_DIR / "pricing_params_online_retail.csv", dtype={"sku": str})
    with (MODEL_DIR / "product_stats_online_retail.json").open(encoding="utf-8") as f:
        stats = json.load(f)
    with (MODEL_DIR / "pricing_model_online_retail.json").open(encoding="utf-8") as f:
        model_info = json.load(f)
    with (MODEL_DIR / "model_metrics_online_retail.json").open(encoding="utf-8") as f:
        metrics = json.load(f)

    skus = set(selected["sku"])
    if len(selected) != 20 or len(skus) != 20:
        raise ValueError("File lựa chọn phải có đúng 20 SKU không trùng nhau.")
    weekly = weekly[weekly["sku"].isin(skus)].copy()
    params = params[params["sku"].isin(skus)].copy()
    if set(weekly["sku"]) != skus or set(params["sku"]) != skus or not skus.issubset(stats):
        raise ValueError("Thiếu dữ liệu tuần, tham số hoặc đặc trưng cho một trong 20 SKU.")
    if weekly.duplicated(["sku", "week"]).any():
        raise ValueError("Dữ liệu tuần có khóa (sku, week) trùng.")
    return selected, weekly, params, stats, model_info, metrics


def import_products(cursor, selected):
    cursor.execute("UPDATE dbo.Products SET is_selected = 0 WHERE is_selected = 1")
    update_sql = """UPDATE dbo.Products SET description=?, tier=?, weeks=?, n_levels=?,
        price_median=?, price_min=?, price_max=?, qty_mean=?, selection_score=?, is_selected=1
        WHERE sku=?"""
    insert_sql = """INSERT INTO dbo.Products
        (sku, description, tier, weeks, n_levels, price_median, price_min, price_max,
         qty_mean, selection_score, is_selected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)"""
    for row in selected.itertuples(index=False):
        values = (str(row.description).strip(), str(row.tier), int(row.weeks), int(row.n_levels),
                  float(row.price_median), float(row.price_min), float(row.price_max),
                  float(row.qty_mean), float(row.score))
        cursor.execute(update_sql, *values, str(row.sku))
        if cursor.rowcount == 0:
            cursor.execute(insert_sql, str(row.sku), *values)


def import_model(cursor, model_info, metrics):
    values = (datetime.fromisoformat(model_info["created"]), model_info["method"],
              float(model_info["global_elasticity"]), float(model_info["tau2"]),
              float(model_info["sigma_extra"]), float(metrics["test_r2"]),
              float(metrics["test_mae_qty"]), float(metrics["baseline_r2"]),
              float(metrics["baseline_mae_qty"]))
    cursor.execute("""UPDATE dbo.ModelRuns SET trained_at=?, elasticity_method=?,
        global_elasticity=?, tau2=?, sigma_extra=?, test_r2=?, test_mae_qty=?,
        baseline_r2=?, baseline_mae_qty=? WHERE model_version=?""", *values, MODEL_VERSION)
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO dbo.ModelRuns
            (model_version, trained_at, elasticity_method, global_elasticity, tau2, sigma_extra,
             test_r2, test_mae_qty, baseline_r2, baseline_mae_qty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", MODEL_VERSION, *values)


def import_weekly(cursor, weekly):
    cursor.execute("""CREATE TABLE #WeeklyImport (
        sku NVARCHAR(20) NOT NULL, week_start DATE NOT NULL,
        qty BIGINT NOT NULL, price DECIMAL(18,4) NOT NULL, qty_all BIGINT NOT NULL)""")
    rows = [(str(r.sku), r.week.date(), int(r.qty), float(r.price), int(r.qty_all))
            for r in weekly.itertuples(index=False)]
    cursor.fast_executemany = True
    cursor.executemany("INSERT INTO #WeeklyImport VALUES (?, ?, ?, ?, ?)", rows)
    cursor.fast_executemany = False
    cursor.execute("""UPDATE target SET qty=source.qty, price=source.price, qty_all=source.qty_all
        FROM dbo.WeeklySales AS target INNER JOIN #WeeklyImport AS source
        ON target.sku=source.sku AND target.week_start=source.week_start""")
    cursor.execute("""INSERT INTO dbo.WeeklySales (sku, week_start, qty, price, qty_all)
        SELECT source.sku, source.week_start, source.qty, source.price, source.qty_all
        FROM #WeeklyImport AS source LEFT JOIN dbo.WeeklySales AS target
        ON target.sku=source.sku AND target.week_start=source.week_start
        WHERE target.sku IS NULL""")
    cursor.execute("DROP TABLE #WeeklyImport")


def import_stats(cursor, params, stats):
    columns = ("p_ref", "n_train_weeks", "sku_base", "sku_lp_med", "sku_lq_std", "sku_lp_std",
               "sku_q4_effect", "eps_raw", "se_raw", "prior_mean", "weight_own", "eps_mean", "eps_sd")
    set_clause = ", ".join(f"{column}=?" for column in columns)
    column_clause = ", ".join(columns)
    markers = ", ".join("?" for _ in columns)
    update_sql = f"UPDATE dbo.ProductModelStats SET {set_clause} WHERE model_version=? AND sku=?"
    insert_sql = ("INSERT INTO dbo.ProductModelStats (model_version, sku, " + column_clause +
                  ") VALUES (?, ?, " + markers + ")")
    for row in params.itertuples(index=False):
        sku = str(row.sku)
        st = stats[sku]
        values = (float(st["p_ref"]), int(st["n_train_weeks"]), float(st["sku_base"]),
                  float(st["sku_lp_med"]), float(st["sku_lq_std"]), float(st["sku_lp_std"]),
                  float(st["sku_q4_effect"]), float(row.eps_raw), float(row.se_raw),
                  float(row.prior_mean), float(row.weight_own), float(row.eps_mean),
                  float(row.eps_sd))
        cursor.execute(update_sql, *values, MODEL_VERSION, sku)
        if cursor.rowcount == 0:
            cursor.execute(insert_sql, MODEL_VERSION, sku, *values)


def main():
    selected, weekly, params, stats, model_info, metrics = load_inputs()
    connection = connect()
    try:
        cursor = connection.cursor()
        import_products(cursor, selected)
        import_model(cursor, model_info, metrics)
        import_weekly(cursor, weekly)
        import_stats(cursor, params, stats)
        cursor.execute("""SELECT COUNT(*) FROM dbo.Products WHERE is_selected=1""")
        n_products = cursor.fetchval()
        cursor.execute("""SELECT COUNT(*) FROM dbo.WeeklySales AS w
            JOIN dbo.Products AS p ON p.sku=w.sku WHERE p.is_selected=1""")
        n_weeks = cursor.fetchval()
        cursor.execute("""SELECT COUNT(*) FROM dbo.ProductModelStats AS s
            JOIN dbo.Products AS p ON p.sku=s.sku
            WHERE p.is_selected=1 AND s.model_version=?""", MODEL_VERSION)
        n_stats = cursor.fetchval()
        if (n_products, n_weeks, n_stats) != (20, len(weekly), 20):
            raise RuntimeError(f"Đối chiếu sau nhập thất bại: {(n_products, n_weeks, n_stats)}")
        connection.commit()
        print(f"SQL Server: {n_products} sản phẩm, {n_weeks} dòng tuần, {n_stats} bộ tham số.")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
