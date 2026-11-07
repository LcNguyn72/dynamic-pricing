"""
CHỌN SỐ CÂY CHO BỘ THAM SỐ XGBOOST CHUNG, TRÊN VALIDATION
=========================================================
Bộ tham số cố định (đúng với bước 4, trừ n_estimators):
    learning_rate 0.02, max_depth 8, min_child_weight 10, subsample 0.8,
    colsample_bytree 0.9, reg_lambda 5, ràng buộc đơn điệu trên log_price_ratio.

Quy trình (không dùng TEST ở bất kỳ bước nào):
  1. Trên mỗi cách chia, tách sub-train (80% tuần đầu của train) và validation (20% tuần cuối),
     thống kê sản phẩm tính trên sub-train.
  2. Huấn luyện một lần với tối đa 1500 cây trên sub-train.
  3. Tại mỗi mốc số cây trong lưới, dự báo bằng iteration_range và đo MAE số lượng trên validation.
  4. Lấy MAE validation trung bình qua ba cách chia, cho mỗi mốc.
  5. Chọn mốc có MAE validation trung bình nhỏ nhất. Đây là số cây đưa vào bước 4.

Lưu ý: không early stopping, không dùng test. Số cây chọn ở đây là số cây CỐ ĐỊNH cho pipeline,
khác với số cây early stopping của tune (vốn khác nhau theo cách chia).

Đầu ra:
  so_sanh_out/chon_so_cay_luoi.csv      : MAE validation theo số cây và cách chia, kèm trung bình
  so_sanh_out/chon_so_cay_ket_qua.json  : số cây được chọn
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

import khung_so_sanh_2y as K

warnings.filterwarnings("ignore")

LUOI_CAY = [200, 300, 400, 600, 800, 1000, 1200, 1500]
MAX_CAY = max(LUOI_CAY)
MONO = tuple(-1 if f == "log_price_ratio" else 0 for f in K.FEATS)
THAM_SO = {
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 5,
}


def mae_theo_so_cay(du, chia):
    sub_f, val_f = K.tach_con_val(du, chia)
    m = xgb.XGBRegressor(n_estimators=MAX_CAY, monotone_constraints=MONO, random_state=42,
                         verbosity=0, **THAM_SO)
    m.fit(sub_f[K.FEATS].values.astype(np.float32), np.log(sub_f["qty"].values.astype(float)))
    Xv = val_f[K.FEATS].values.astype(np.float32)
    yv = val_f["qty"].values.astype(float)
    out = {}
    for n in LUOI_CAY:
        pred = np.exp(m.predict(Xv, iteration_range=(0, n)))
        out[n] = float(np.mean(np.abs(pred - yv)))
    return out


def main():
    print("=" * 78)
    print("CHỌN SỐ CÂY CHO BỘ THAM SỐ CHUNG (TRÊN VALIDATION, KHÔNG DÙNG TEST)")
    print("=" * 78)
    print("  Tham số cố định:", THAM_SO)
    print("  Lưới số cây:", LUOI_CAY)
    os.makedirs(K.OUT, exist_ok=True)
    du = K.nap_du_lieu()

    bang = {}
    for chia in K.CACH_CHIA:
        print(f"\n--- {K.CO.TEN_CHIA[chia]}")
        r = mae_theo_so_cay(du, chia)
        bang[chia] = r
        for n in LUOI_CAY:
            print(f"    {n:>5} cây: MAE validation {r[n]:.3f}")

    df = pd.DataFrame(bang)
    df.index.name = "so_cay"
    df["trung_binh"] = df.mean(axis=1)
    df.reset_index().to_csv(os.path.join(K.OUT, "chon_so_cay_luoi.csv"), index=False, encoding="utf-8-sig")

    tot = df["trung_binh"].idxmin()
    print("\n" + "=" * 78)
    print("MAE validation trung bình ba cách chia")
    print("=" * 78)
    print(df.round(3).to_string())
    print(f"\n=> Số cây được chọn: {int(tot)} (MAE validation trung bình {df.loc[tot, 'trung_binh']:.3f})")

    with open(os.path.join(K.OUT, "chon_so_cay_ket_qua.json"), "w", encoding="utf-8") as f:
        json.dump({"n_estimators": int(tot), "tham_so": THAM_SO, "luoi": LUOI_CAY}, f,
                  indent=2, ensure_ascii=False)
    print(f"Đã ghi {K.OUT}/chon_so_cay_luoi.csv và chon_so_cay_ket_qua.json")
    print("Đặt n_estimators = số cây trên vào bước 4, rồi chạy bước 4 MỘT lần để đo test.")


if __name__ == "__main__":
    main()
