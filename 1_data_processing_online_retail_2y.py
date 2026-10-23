"""
Xử lý dữ liệu 2 NĂM - Online Retail II (UCI)
============================================
Bước 1: đọc file Excel (2 sheet), bỏ phần CHỒNG LẤN giữa hai sheet, làm sạch, gộp theo (sản phẩm, tuần).

File online_retail_II.xlsx có 2 sheet:
    Year 2009-2010: 2009-12-01 -> 2010-12-09
    Year 2010-2011: 2010-12-01 -> 2011-12-09
Hai sheet CHỒNG LẤN 9 ngày (01-09/12/2010) nên phải bỏ một bản, nếu không lượng bán của tuần đó
bị tính gấp đôi. Script giữ sheet 2 cho đoạn chồng lấn (sheet 2 chính là dữ liệu 1 năm bạn đã dùng)
và in ra kiểm tra hai bản có giống nhau không.

Các bước làm sạch giống bản 1 năm: bỏ hóa đơn hủy, qty/giá <= 0, chỉ giữ mã sản phẩm thật (5 chữ số),
bỏ đơn sỉ lớn (top 5% số lượng của từng sản phẩm), gộp theo tuần (qty = tổng, price = trung vị).

Lần chạy đầu đọc Excel (mất vài phút) rồi lưu cache data/online_retail_II_sheets.csv.
Output (./models_2y): weekly_online_retail.csv, products_online_retail.csv
"""
import os

import numpy as np
import pandas as pd

from y2_config import (MIN_CV, MIN_WEEKS_FILTER, OUT_DIR, RAW_CACHE, RAW_XLSX, path)

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
        "Customer ID", "Country"]


def load_raw():
    print("[1] Đang tải dữ liệu gốc...")
    if os.path.exists(RAW_CACHE):
        print(f"    Đọc cache {RAW_CACHE}")
        return pd.read_csv(RAW_CACHE, dtype={"Invoice": str, "StockCode": str},
                           parse_dates=["InvoiceDate"])
    if not os.path.exists(RAW_XLSX):
        raise SystemExit(f"Không thấy {RAW_XLSX}")
    print("    Đọc file Excel (mất vài phút, chỉ lần đầu)...")
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    parts = []
    for i, (name, d) in enumerate(sheets.items()):
        d = d.copy()
        d["sheet"] = i
        print(f"    Sheet {i} '{name}': {d.shape}")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(RAW_CACHE, index=False)
    print(f"    Đã lưu cache {RAW_CACHE}")
    return df


def drop_sheet_overlap(df):
    print("[2] Xử lý phần chồng lấn...")
    s1, s2 = df[df["sheet"] == 0], df[df["sheet"] == 1]
    if s2.empty:
        print("    Chỉ có 1 sheet, bỏ qua")
        return df
    cut = s2["InvoiceDate"].min().normalize()
    ov1 = s1[s1["InvoiceDate"] >= cut]
    ov2 = s2[s2["InvoiceDate"] <= s1["InvoiceDate"].max()]
    a = ov1[COLS].sort_values(COLS).reset_index(drop=True)
    b = ov2[COLS].sort_values(COLS).reset_index(drop=True)
    print(f"    Đoạn chồng lấn: {cut:%Y-%m-%d} -> {s1['InvoiceDate'].max():%Y-%m-%d}")
    print(f"    Số dòng trong đoạn đó: sheet 1 = {len(ov1):,} | sheet 2 = {len(ov2):,} "
          f"| giống hệt nhau: {a.equals(b)}")
    out = pd.concat([s1[s1["InvoiceDate"] < cut], s2], ignore_index=True)
    print(f"    Giữ sheet 1 trước {cut:%Y-%m-%d} + toàn bộ sheet 2: {len(df):,} -> {len(out):,} dòng")
    return out


def clean(df):
    print("[3] Làm sạch...")
    df = df.rename(columns={"StockCode": "sku", "Quantity": "qty", "Price": "price",
                            "InvoiceDate": "date", "Description": "desc"})
    n0 = len(df)
    df = df[~df["Invoice"].astype(str).str.upper().str.startswith("C")]   # bỏ hóa đơn hủy
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    df["sku"] = df["sku"].astype(str)
    df = df[df["sku"].str.match(r"^\d{5}")]                                # chỉ giữ mã sản phẩm thật
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    print(f"    {n0:,} -> {len(df):,} dòng | {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    return df


def build_weekly(df):
    print("[4] Gộp theo tuần (bỏ đơn sỉ, giá trung vị)...")
    cap = df.groupby("sku")["qty"].transform(lambda s: s.quantile(0.95))
    retail = df[df["qty"] <= cap]
    w = retail.groupby(["sku", "week"]).agg(qty=("qty", "sum"), price=("price", "median")).reset_index()
    w_all = df.groupby(["sku", "week"])["qty"].sum().rename("qty_all").reset_index()
    w = w.merge(w_all, on=["sku", "week"], how="left")
    print(f"    {len(w):,} dòng (sản phẩm x tuần) | {w['week'].nunique()} tuần")
    return w


def filter_products(w, df):
    print(f"[5] Lọc sản phẩm: >= {MIN_WEEKS_FILTER} tuần có bán, cv giá >= {MIN_CV}...")
    g = w.groupby("sku")["price"]
    st = pd.DataFrame({"weeks": g.size(), "cv": g.std() / g.mean()})
    keep = st[(st["weeks"] >= MIN_WEEKS_FILTER) & (st["cv"] >= MIN_CV)].index
    w = w[w["sku"].isin(keep)].copy()
    info = w.groupby("sku").agg(
        weeks=("week", "nunique"), price_median=("price", "median"),
        price_min=("price", "min"), price_max=("price", "max"), qty_mean=("qty", "mean"),
    ).reset_index()
    desc = (df[df["sku"].isin(keep)].groupby("sku")["desc"]
            .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "")
            .rename("description").reset_index())
    info = info.merge(desc, on="sku", how="left")
    print(f"    {len(keep)} sản phẩm / {len(st)}")
    return w, info


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw()
    df = drop_sheet_overlap(df)
    df = clean(df)
    w = build_weekly(df)
    w, info = filter_products(w, df)
    w.to_csv(path("weekly_online_retail.csv"), index=False)
    info.to_csv(path("products_online_retail.csv"), index=False)
    print(f"\n[6] Đã lưu vào {OUT_DIR}/:")
    print(f"    weekly_online_retail.csv   - {len(w):,} dòng")
    print(f"    products_online_retail.csv - {len(info):,} sản phẩm")
    print("\n5 dòng đầu (weekly):")
    print(w.head().to_string(index=False))
    print("\n5 sản phẩm đầu:")
    print(info.head().to_string(index=False))


if __name__ == "__main__":
    main()
