"""
Train Final Demand Model - Online Retail II (2 NĂM)
========================================
Bước 2 của pipeline (giống train_final_model_retail.py):
    - Đọc X, y, is_test từ data_processing_online_retail_ml.py
    - Train XGBoost dự báo y = ln(qty) từ 8 đặc trưng
    - Ràng buộc ĐƠN ĐIỆU cho giá: giá tương đối tăng -> dự báo không tăng
    - Đánh giá trên tập test (mỗi tuần thứ 5), so với baseline (chỉ dùng sku_base)
    - Lưu demand_model_online_retail.pkl

Đầu vào: X_online_retail.npy, y_online_retail.npy, is_test_online_retail.npy,
        feature_cols_online_retail.json
Đầu ra : demand_model_online_retail.pkl, model_metrics_online_retail.json
"""
import json
import os
import pickle
import warnings

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

from y2_config import OUT_DIR as MODEL_DIR
PRICE_FEATURE = "log_price_ratio"


# ============================================================================
# 1. TẢI DỮ LIỆU
# ============================================================================
def load_data(data_dir=MODEL_DIR):
    print("[1] Loading data...")
    X = np.load(os.path.join(data_dir, "X_online_retail.npy"))
    y = np.load(os.path.join(data_dir, "y_online_retail.npy"))
    is_test = np.load(os.path.join(data_dir, "is_test_online_retail.npy"))
    with open(os.path.join(data_dir, "feature_cols_online_retail.json")) as f:
        feature_cols = json.load(f)
    print(f"    X: {X.shape} | y: {y.shape} (ln(qty): min={y.min():.2f}, "
          f"max={y.max():.2f}, mean={y.mean():.2f})")
    print(f"    Features: {feature_cols}")
    return X, y, is_test, feature_cols


# ============================================================================
# 2. CHIA DỮ LIỆU
# ============================================================================
def split_data(X, y, is_test):
    print("\n[2] Chia train/test (theo is_test: mỗi tuần thứ 5 là test)...")
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]
    print(f"    Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    return X_train, X_test, y_train, y_test


# ============================================================================
# 3. HUẤN LUYỆN MÔ HÌNH
# ============================================================================
def train_model(X_train, y_train, feature_cols):
    print("\n[3] Training XGBoost...")
    params = {
        # Cấu hình chọn bằng validation (tune_model_online_retail.py)
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
        # Giá tương đối tăng -> ln(qty) dự báo không tăng
        "monotone_constraints": tuple(-1 if c == PRICE_FEATURE else 0
                                      for c in feature_cols),
    }
    print(f"    Params: {params}")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    print("    Model trained")
    return model, params


# ============================================================================
# 4. ĐÁNH GIÁ
# ============================================================================
def price_response(model, X_test, feature_cols, step=0.10):
    """Đo phản ứng theo giá: đổi giá ±10%, xem ln(qty) dự báo đổi bao nhiêu.
    Trả về (elasticity trung bình, % dòng không phản ứng)."""
    i = feature_cols.index(PRICE_FEATURE)
    up, dn = X_test.copy(), X_test.copy()
    up[:, i] += np.log(1 + step)
    dn[:, i] += np.log(1 - step)
    p_up, p_dn = model.predict(up), model.predict(dn)
    e = (p_up - p_dn) / (np.log(1 + step) - np.log(1 - step))
    flat = np.isclose(p_up, p_dn).mean() * 100
    return float(e.mean()), float(flat)


def evaluate(model, X_train, X_test, y_train, y_test, feature_cols):
    print("\n[4] Evaluation...")
    pred_tr, pred_te = model.predict(X_train), model.predict(X_test)

    m = {
        "train_r2": r2_score(y_train, pred_tr),
        "train_mae_ln": mean_absolute_error(y_train, pred_tr),
        "train_rmse_ln": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
        "test_r2": r2_score(y_test, pred_te),
        "test_mae_ln": mean_absolute_error(y_test, pred_te),
        "test_rmse_ln": float(np.sqrt(mean_squared_error(y_test, pred_te))),
        "test_mae_qty": mean_absolute_error(np.exp(y_test), np.exp(pred_te)),
    }

    # Baseline: chỉ dùng mức bán nền của sản phẩm (không dùng giá, không mùa vụ)
    base = X_test[:, feature_cols.index("sku_base")]
    m["baseline_r2"] = r2_score(y_test, base)
    m["baseline_mae_ln"] = mean_absolute_error(y_test, base)
    m["baseline_mae_qty"] = mean_absolute_error(np.exp(y_test), np.exp(base))

    e, flat = price_response(model, X_test, feature_cols)
    m["elasticity_pm10"] = e
    m["pct_flat_pm10"] = flat

    print("\n    TRAIN METRICS (trên ln(qty)):")
    print(f"      R2:   {m['train_r2']:.4f}")
    print(f"      MAE:  {m['train_mae_ln']:.4f}")
    print(f"      RMSE: {m['train_rmse_ln']:.4f}")
    print("\n    TEST METRICS (trên ln(qty)):")
    print(f"      R2:   {m['test_r2']:.4f}")
    print(f"      MAE:  {m['test_mae_ln']:.4f}   | MAE theo số lượng: {m['test_mae_qty']:.2f}")
    print(f"      RMSE: {m['test_rmse_ln']:.4f}")
    print("\n    BASELINE (chỉ dùng mức bán nền sku_base):")
    print(f"      R2:   {m['baseline_r2']:.4f}")
    print(f"      MAE:  {m['baseline_mae_ln']:.4f}   | MAE theo số lượng: {m['baseline_mae_qty']:.2f}")
    print("\n    PHẢN ỨNG THEO GIÁ (đổi giá ±10%, trên tập test):")
    print(f"      Elasticity trung bình: {m['elasticity_pm10']:.2f} | "
          f"% dòng không phản ứng: {m['pct_flat_pm10']:.1f}%")
    print("      (chỉ để tham khảo: cây quyết định là hàm bậc thang)")

    print("\n    TOP FEATURES:")
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for i, (feat, v) in enumerate(imp, 1):
        print(f"      {i}. {feat:18s} {v:.4f}")
    return m


# ============================================================================
# 5. LƯU MÔ HÌNH
# ============================================================================
def save_model(model, params, metrics, output_dir=MODEL_DIR):
    print("\n[5] Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "demand_model_online_retail.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"    demand_model_online_retail.pkl (test R2 = {metrics['test_r2']:.4f})")
    with open(os.path.join(output_dir, "model_metrics_online_retail.json"), "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)
    print("    model_metrics_online_retail.json")


# ============================================================================
# 6. CHẠY CHÍNH
# ============================================================================
def main():
    print("=" * 70)
    print("TRAIN FINAL DEMAND MODEL - Online Retail")
    print("=" * 70)
    for p in ("X_online_retail.npy", "y_online_retail.npy",
              "is_test_online_retail.npy", "feature_cols_online_retail.json"):
        if not os.path.exists(os.path.join(MODEL_DIR, p)):
            print(f"Không thấy {p}. Chạy data_processing_online_retail_ml.py trước.")
            return None

    X, y, is_test, feature_cols = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, is_test)
    model, params = train_model(X_train, y_train, feature_cols)
    metrics = evaluate(model, X_train, X_test, y_train, y_test, feature_cols)
    save_model(model, params, metrics)

    print("\n" + "=" * 70)
    print("Model training completed!")
    print("=" * 70)
    print(f"TEST R2 (ln): {metrics['test_r2']:.4f}  | baseline: {metrics['baseline_r2']:.4f}")
    print(f"TEST MAE (qty): {metrics['test_mae_qty']:.2f}  | baseline: {metrics['baseline_mae_qty']:.2f}")
    print("=" * 70)
    return model, metrics


if __name__ == "__main__":
    main()
