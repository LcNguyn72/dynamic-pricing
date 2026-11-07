"""
BƯỚC 3: BA BẢN DỰ BÁO (MỖI CÁCH CHIA MỘT BẢN)
=============================================
Với mỗi cách chia, chạy đúng pipeline 2_ và 4_ hiện có, chỉ khác ở:
  - cột is_test theo cách chia (xen kẽ / chiều 1->2 / chiều 2->1), thay cho cờ mặc định của 2_
  - thư mục ghi kết quả: models_2y/<cách chia>/
  - tham số XGBoost cố định: 273 cây, bộ đã chọn (không tune lại)

Được ghi vào models_2y/<cách chia>/:
  model_table_online_retail.csv, product_stats_online_retail.json, feature_cols_online_retail.json,
  X/y/is_test (.npy), demand_model_online_retail.pkl, model_metrics_online_retail.json

Thống kê sản phẩm và đặc trưng chỉ tính trên phần train của từng cách chia.
Elasticity KHÔNG tính ở bước này (dùng chung, bước 2).

Kiểm tra: bản xen_ke phải cho MAE test theo số lượng khoảng 20.412 (đã đo với cùng cấu hình).
Chưa có mốc đã đo cho hai chiều năm với bộ 273 cây, nên chỉ ghi số liệu, không kiểm tra.

Chạy:  python buoc3_ba_ban_du_bao.py
Cần: khung_cach_chia.py, buoc2 đã chạy (có chung/weekly, chung/products),
     2_data_processing_online_retail_ml_2y.py và 4_train_final_model_online_retail_2y.py cùng thư mục.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import khung_cach_chia as KC
from y2_config import OOT_CUT_WEEK, TEST_EVERY

FILE_2 = "2_data_processing_online_retail_ml_2y.py"
FILE_4 = "4_train_final_model_online_retail_2y.py"

# Bộ tham số đã chọn cho pipeline (giống cấu hình bạn đã chốt). monotone_constraints được thêm sau khi có feature_cols.
PARAMS_273 = {
    "n_estimators": 273,
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 5,
    "random_state": 42,
    "verbosity": 0,
}
MOC_MAE_XEN_KE = 20.412
DUNG_SAI_MOC = 0.01


def nap_module(ten_file, ten_module):
    """Nạp file có tên bắt đầu bằng chữ số (2_..., 4_...) như một module Python."""
    if not os.path.exists(ten_file):
        raise SystemExit(f"Thiếu file {ten_file} trong thư mục hiện tại.")
    spec = importlib.util.spec_from_file_location(ten_module, ten_file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gan_cot_is_test(w, cach_chia):
    """Gán cột is_test theo cách chia. True = dòng thuộc tập test."""
    w = w.copy()
    if cach_chia == "xen_ke":
        tuan = np.sort(w["week"].unique())
        tuan_test = pd.to_datetime(tuan[::TEST_EVERY])
        w["is_test"] = w["week"].isin(tuan_test)
    elif cach_chia == "chieu_1_2":
        cut = pd.Timestamp(OOT_CUT_WEEK)
        w["is_test"] = w["week"] >= cut          # test = năm 2
    elif cach_chia == "chieu_2_1":
        cut = pd.Timestamp(OOT_CUT_WEEK)
        w["is_test"] = w["week"] < cut           # test = năm 1
    else:
        raise ValueError(f"Cách chia không hợp lệ: {cach_chia}")
    return w


def chay_mot_cach_chia(cach_chia, d2, d4, weekly_path, products_path):
    thu_muc = KC.thu_muc_cach_chia(cach_chia)
    print(f"\n--- {cach_chia}: {KC.CACH_CHIA[cach_chia]}")
    print(f"    ghi vào: {thu_muc}")

    w = d2.load_weekly(weekly_path)
    w = gan_cot_is_test(w, cach_chia)
    print(f"    train: {(~w['is_test']).sum():,} dòng | test: {w['is_test'].sum():,} dòng")

    st = d2.product_stats_from_train(w)
    t = d2.build_features(w, st)
    X, y, is_test = d2.prepare_X_y(t)

    # ghi bảng, thống kê, đặc trưng vào thư mục của cách chia
    d2.MODEL_DIR = thu_muc
    d2.PRODUCTS_PATH = products_path
    d2.save_outputs(X, y, is_test, t, st)

    feats = d2.FEATURES
    params = dict(PARAMS_273)
    params["monotone_constraints"] = tuple(-1 if c == "log_price_ratio" else 0 for c in feats)

    X_tr, X_te = X[~is_test], X[is_test]
    y_tr, y_te = y[~is_test], y[is_test]
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)

    m = d4.evaluate(model, X_tr, X_te, y_tr, y_te, feats)
    d4.save_model(model, params, m, output_dir=thu_muc)
    KC.tao_manifest(cach_chia)
    return m


def main():
    print("=" * 78)
    print("BƯỚC 3: BA BẢN DỰ BÁO THEO CÁCH CHIA (273 CÂY)")
    print("=" * 78)
    weekly = os.path.join(KC.thu_muc_chung(), "weekly_online_retail.csv")
    products = os.path.join(KC.thu_muc_chung(), "products_online_retail.csv")
    for f in (weekly, products):
        if not os.path.exists(f):
            raise SystemExit(f"Thiếu {f}. Chạy buoc2_ban_dung_chung.py trước.")

    d2 = nap_module(FILE_2, "dp2")
    d4 = nap_module(FILE_4, "dp4")

    ket = {}
    for cach in KC.CACH_CHIA:
        ket[cach] = chay_mot_cach_chia(cach, d2, d4, weekly, products)

    print("\n" + "=" * 78)
    print("TÓM TẮT (test)")
    print("=" * 78)
    print(f"{'Cách chia':<12}{'MAE':>10}{'R² (ln)':>10}{'Baseline':>11}{'Elasticity':>12}{'Phẳng %':>10}")
    for cach, m in ket.items():
        print(f"{cach:<12}{m['test_mae_qty']:>10.3f}{m['test_r2']:>10.3f}{m['baseline_mae_qty']:>11.3f}"
              f"{m['elasticity_pm10']:>12.3f}{m['pct_flat_pm10']:>10.2f}")

    lech = abs(ket["xen_ke"]["test_mae_qty"] - MOC_MAE_XEN_KE)
    if lech <= DUNG_SAI_MOC:
        print(f"\nKiểm tra mốc xen_ke: {ket['xen_ke']['test_mae_qty']:.3f} khớp {MOC_MAE_XEN_KE} -> OK")
    else:
        print(f"\nKiểm tra mốc xen_ke: {ket['xen_ke']['test_mae_qty']:.3f} KHÔNG khớp {MOC_MAE_XEN_KE} "
              f"(lệch {lech:.3f}). Dừng lại kiểm tra dữ liệu hoặc pipeline trước khi sang bước 4.")
        sys.exit(1)
    print("Bước 3 hoàn tất. Ba bản đã nằm trong models_2y/xen_ke, chieu_1_2, chieu_2_1.")


if __name__ == "__main__":
    main()
