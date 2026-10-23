"""
Xử lý dữ liệu (ML) - Online Retail II (2 NĂM, UCI)
==================================================
Bước 1 của pipeline: chuẩn bị X, y cho mô hình dự báo lượng bán theo tuần.

Đầu vào (từ data_processing_online_retail.py):
    models/weekly_online_retail.csv    (sku, week, qty, price, qty_all)
    models/products_online_retail.csv  (sku, price_min, price_max, description, ...)

Đầu ra (vào ./models):
    X_online_retail.npy, y_online_retail.npy, is_test_online_retail.npy
    feature_cols_online_retail.json
    product_stats_online_retail.json   (cho optimizer: sku_base, p_ref, khoảng giá)
    model_table_online_retail.csv      (bảng đầy đủ để kiểm tra bằng mắt)

Thiết kế:
    y = ln(qty)   (lượng bán theo tuần, lấy log cho bớt lệch phải)
    X = 8 đặc trưng:
        log_price_ratio : ln(giá tuần / giá trung vị của sản phẩm)   <- đặc trưng GIÁ
        sku_lp_med      : ln(giá trung vị của sản phẩm)              <- phân khúc giá
        sku_base        : trung bình ln(qty) của sản phẩm            <- mức bán nền
        month, woy      : tháng, tuần trong năm                      <- mùa vụ chung
        sku_lq_std      : độ biến động ln(lượng bán) của sản phẩm
        sku_lp_std      : độ biến động ln(giá) của sản phẩm
        sku_q4_effect   : sản phẩm bán nhiều hơn bao nhiêu ở tháng 11-12 so với còn lại
    KHÔNG dùng lag / rolling: lần trước chúng khiến mô hình chỉ nhìn doanh số
    hôm qua và bỏ qua giá.
    Thống kê của sản phẩm (sku_base, sku_lp_med) chỉ tính trên các tuần TRAIN
    để tránh rò rỉ thông tin từ tập test.
"""
import json
import os

import numpy as np
import pandas as pd

from y2_config import OUT_DIR as MODEL_DIR, MIN_TRAIN_WEEKS, TEST_EVERY
WEEKLY_PATH = os.path.join(MODEL_DIR, "weekly_online_retail.csv")
PRODUCTS_PATH = os.path.join(MODEL_DIR, "products_online_retail.csv")

FEATURES = ["log_price_ratio", "sku_lp_med", "sku_base", "month", "woy",
            "sku_lq_std", "sku_lp_std", "sku_q4_effect"]


# ============================================================================
# 1. TẢI DỮ LIỆU
# ============================================================================
def load_weekly(path):
    print("[1] Loading weekly data...")
    w = pd.read_csv(path, parse_dates=["week"], dtype={"sku": str})
    print(f"    Shape: {w.shape} | Columns: {list(w.columns)}")
    w = w[(w["qty"] > 0) & (w["price"] > 0)].copy()
    w["lq"] = np.log(w["qty"])
    w["lp"] = np.log(w["price"])
    w["month"] = w["week"].dt.month
    w["woy"] = w["week"].dt.isocalendar().week.astype(int)
    print(f"    Sau khi lọc qty>0, price>0: {len(w):,} dòng, "
          f"{w['sku'].nunique()} sản phẩm")
    return w


# ============================================================================
# 2. CHIA TRAIN / TEST (theo tuần)
# ============================================================================
def add_split(w):
    """Giữ ra mỗi tuần thứ 5 làm test (dữ liệu chỉ có ~1 năm nên không thể cắt
    theo thời gian: tháng 11 sẽ không nằm trong train)."""
    print("\n[2] Chia train/test theo tuần...")
    weeks = np.sort(w["week"].unique())
    test_weeks = pd.to_datetime(weeks[::TEST_EVERY])
    w["is_test"] = w["week"].isin(test_weeks)
    print(f"    Tổng số tuần: {len(weeks)} | tuần test: {len(test_weeks)}")
    print(f"    Train: {(~w['is_test']).sum():,} dòng | Test: {w['is_test'].sum():,} dòng")
    return w


# ============================================================================
# 3. THỐNG KÊ SẢN PHẨM (chỉ trên train)
# ============================================================================
def product_stats_from_train(w):
    print("\n[3] Thống kê từng sản phẩm (chỉ trên tuần train)...")
    train = w[~w["is_test"]]
    st = train.groupby("sku").agg(
        sku_base=("lq", "mean"),
        sku_lp_med=("lp", "median"),
        sku_lq_std=("lq", "std"),
        sku_lp_std=("lp", "std"),
        n_train=("lq", "size"),
    ).reset_index()
    # Mùa vụ riêng: ln(lượng) trung bình ở tháng 11-12 trừ các tháng còn lại
    q4 = train["month"].isin([11, 12])
    a = train[q4].groupby("sku")["lq"].mean().rename("q4_mean")
    b = train[~q4].groupby("sku")["lq"].mean().rename("rest_mean")
    st = st.merge(a, on="sku", how="left").merge(b, on="sku", how="left")
    st["sku_q4_effect"] = (st["q4_mean"] - st["rest_mean"]).fillna(0.0)
    st[["sku_lq_std", "sku_lp_std"]] = st[["sku_lq_std", "sku_lp_std"]].fillna(0.0)
    st = st.drop(columns=["q4_mean", "rest_mean"])
    before = len(st)
    st = st[st["n_train"] >= MIN_TRAIN_WEEKS]
    print(f"    {len(st)}/{before} sản phẩm có >= {MIN_TRAIN_WEEKS} tuần train")
    return st


# ============================================================================
# 4. ĐẶC TRƯNG (features)
# ============================================================================
def build_features(w, st):
    print("\n[4] Tạo features...")
    t = w.merge(st, on="sku", how="inner")
    t["log_price_ratio"] = t["lp"] - t["sku_lp_med"]
    t = t.sort_values(["sku", "week"]).reset_index(drop=True)

    # Kiểm tra nhanh: trong từng sản phẩm, giá tương đối và lượng bán (đã trừ mức nền)
    # nên có tương quan ÂM (khác với sales_data.csv cũ, tương quan dương)
    tr = t[~t["is_test"]]
    corr = np.corrcoef(tr["log_price_ratio"], tr["lq"] - tr["sku_base"])[0, 1]
    print(f"    Tương quan (train) giữa log_price_ratio và ln(qty) đã trừ mức nền: {corr:.3f}")
    if corr >= 0:
        print("    CẢNH BÁO: tương quan không âm -> dữ liệu có vấn đề giống sales_data.csv")
    return t


# ============================================================================
# 5. X, y
# ============================================================================
def prepare_X_y(t):
    print("\n[5] Preparing X and y...")
    X = t[FEATURES].values.astype(np.float32)
    y = t["lq"].values.astype(np.float32)
    is_test = t["is_test"].values.astype(bool)

    print(f"    X shape: {X.shape} | y shape: {y.shape}")
    print(f"    Feature names: {FEATURES}")
    print(f"    y = ln(qty): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    print(f"    (tương ứng qty: min={np.exp(y.min()):.0f}, "
          f"trung bình ln -> {np.exp(y.mean()):.1f})")
    missing = int(np.isnan(X).sum())
    print(f"    Giá trị thiếu trong X: {missing}")
    return X, y, is_test


# ============================================================================
# 6. LƯU
# ============================================================================
def save_outputs(X, y, is_test, t, st):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n[6] Saving to {MODEL_DIR}/...")

    np.save(os.path.join(MODEL_DIR, "X_online_retail.npy"), X)
    np.save(os.path.join(MODEL_DIR, "y_online_retail.npy"), y)
    np.save(os.path.join(MODEL_DIR, "is_test_online_retail.npy"), is_test)
    print(f"    X_online_retail.npy {X.shape}, y_online_retail.npy {y.shape}, "
          f"is_test_online_retail.npy {is_test.shape}")

    with open(os.path.join(MODEL_DIR, "feature_cols_online_retail.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"    feature_cols_online_retail.json - {len(FEATURES)} features")

    # Thông tin sản phẩm cho optimizer
    prod = pd.read_csv(PRODUCTS_PATH, dtype={"sku": str})
    keep = [c for c in ["sku", "description", "price_min", "price_max"] if c in prod.columns]
    info = st.merge(prod[keep], on="sku", how="left")
    stats = {}
    for _, r in info.iterrows():
        stats[r["sku"]] = {
            "description": "" if pd.isna(r.get("description")) else str(r["description"]),
            "sku_base": round(float(r["sku_base"]), 4),
            "sku_lq_std": round(float(r["sku_lq_std"]), 4),
            "sku_lp_std": round(float(r["sku_lp_std"]), 4),
            "sku_q4_effect": round(float(r["sku_q4_effect"]), 4),
            "sku_lp_med": round(float(r["sku_lp_med"]), 4),
            "p_ref": round(float(np.exp(r["sku_lp_med"])), 4),
            "price_min": None if pd.isna(r.get("price_min")) else round(float(r["price_min"]), 4),
            "price_max": None if pd.isna(r.get("price_max")) else round(float(r["price_max"]), 4),
            "n_train_weeks": int(r["n_train"]),
        }
    with open(os.path.join(MODEL_DIR, "product_stats_online_retail.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"    product_stats_online_retail.json - {len(stats)} products")

    cols = ["sku", "week", "qty", "price", "is_test"] + FEATURES + ["lq"]
    t[cols].to_csv(os.path.join(MODEL_DIR, "model_table_online_retail.csv"), index=False)
    print(f"    model_table_online_retail.csv - {len(t):,} rows")


# ============================================================================
# 7. CHẠY CHÍNH
# ============================================================================
def main():
    print("=" * 70)
    print("XỬ LÝ DỮ LIỆU (ML): Online Retail")
    print("=" * 70)
    for p in (WEEKLY_PATH, PRODUCTS_PATH):
        if not os.path.exists(p):
            print(f"Không thấy file: {p}")
            print("Chạy data_processing_online_retail.py trước.")
            return None

    w = load_weekly(WEEKLY_PATH)
    w = add_split(w)
    st = product_stats_from_train(w)
    t = build_features(w, st)
    X, y, is_test = prepare_X_y(t)
    save_outputs(X, y, is_test, t, st)

    print("\n" + "=" * 70)
    print("Data processing completed!")
    print("=" * 70)
    print(f"Rows: {len(t):,} | Products: {t['sku'].nunique()} | X: {X.shape} | y: {y.shape}")
    print("\n5 dòng đầu sau xử lý:")
    print(t[["sku", "week", "qty", "price", "is_test"] + FEATURES + ["lq"]]
          .head().to_string(index=False))
    print("=" * 70)
    return X, y, is_test, t


if __name__ == "__main__":
    main()
