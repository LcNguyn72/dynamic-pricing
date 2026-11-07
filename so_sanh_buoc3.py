"""
BƯỚC 3: MÔ HÌNH CÂY (RANDOM FOREST, LIGHTGBM, CATBOOST), SO VỚI XGBOOST
========================================================================
Thêm vào khung (khung_so_sanh_2y.py), cùng 8 đặc trưng, cùng biến mục tiêu ln(qty), cùng cách chia:
  - random_forest : sklearn RandomForestRegressor. KHÔNG hỗ trợ ràng buộc đơn điệu.
  - lightgbm      : LGBMRegressor, CÓ ràng buộc đơn điệu (-1) trên log_price_ratio.
  - catboost      : CatBoostRegressor, thử ràng buộc đơn điệu; nếu phiên bản không hỗ trợ, chạy không
                    ràng buộc và ghi rõ trong bảng.

Tham số: chọn gần với XGBoost (độ sâu 6, 400 cây, learning rate 0.05), cố định, KHÔNG tune trên test.
Để so công bằng, mọi mô hình cây đều dùng cùng số cây và cùng độ sâu cỡ tương đương.

Thư viện cần: scikit-learn (đã có), lightgbm, catboost.
Mô hình nào thiếu thư viện sẽ được bỏ qua và ghi lỗi, không làm dừng cả chuỗi.

Đọc kết quả:
  - Mô hình cây nào có MAE thấp hơn XGBoost ổn định ở CẢ ba cách chia (khoảng bootstrap < 0) mới đáng kể.
  - Mô hình không có ràng buộc đơn điệu (random_forest) có thể cho phản ứng giá sai dấu; xem cột elasticity.
  - Độ phẳng (% dòng không phản ứng) cao nghĩa là mô hình bỏ qua giá.

Đầu ra: ./so_sanh_out/ket_qua_buoc3.csv (gồm cả các mô hình bước 1 và 2 để so), chi_tiet/.
Cần khung_so_sanh_2y.py trong cùng thư mục.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import khung_so_sanh_2y as K
import so_sanh_buoc2  # noqa: F401  (đăng ký sku_base và ridge)

warnings.filterwarnings("ignore")

BO_QUA = {}          # mô hình không chạy được -> lý do
CO_RANG_BUOC = {}    # tên mô hình -> True/False (có ràng buộc đơn điệu hay không)
MONO_LIST = [-1 if f == "log_price_ratio" else 0 for f in K.FEATS]


# ----------------------------------------------------------------------------- random forest
def huan_random_forest(trn):
    m = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                              max_features=0.6, n_jobs=-1, random_state=42)
    m.fit(trn[K.FEATS].values.astype(np.float32), np.log(trn["qty"].values.astype(float)))
    return m


K.dang_ky("random_forest", huan_random_forest)
CO_RANG_BUOC["random_forest"] = False


# ----------------------------------------------------------------------------- lightgbm
try:
    import lightgbm as lgb

    def huan_lightgbm(trn):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, max_depth=6,
                              min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              monotone_constraints=MONO_LIST, random_state=42, verbose=-1)
        m.fit(trn[K.FEATS].values.astype(np.float32), np.log(trn["qty"].values.astype(float)))
        return m

    K.dang_ky("lightgbm", huan_lightgbm)
    CO_RANG_BUOC["lightgbm"] = True
except ImportError as e:
    BO_QUA["lightgbm"] = f"chưa cài thư viện: {e}"


# ----------------------------------------------------------------------------- catboost
try:
    from catboost import CatBoostRegressor

    def huan_catboost(trn):
        X = trn[K.FEATS].values.astype(np.float32)
        y = np.log(trn["qty"].values.astype(float))
        try:
            m = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, loss_function="RMSE",
                                  monotone_constraints=MONO_LIST, random_seed=42, verbose=False)
            m.fit(X, y)
            CO_RANG_BUOC["catboost"] = True
        except Exception:
            m = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, loss_function="RMSE",
                                  random_seed=42, verbose=False)
            m.fit(X, y)
            CO_RANG_BUOC["catboost"] = False
        return m

    K.dang_ky("catboost", huan_catboost)
except ImportError as e:
    BO_QUA["catboost"] = f"chưa cài thư viện: {e}"


def main():
    print("=" * 78)
    print("BƯỚC 3: RANDOM FOREST, LIGHTGBM, CATBOOST, SO VỚI XGBOOST")
    print("=" * 78)
    if BO_QUA:
        for ten, ly_do in BO_QUA.items():
            print(f"  BỎ QUA {ten}: {ly_do}")
    os.makedirs(K.CHI_TIET, exist_ok=True)
    du = K.nap_du_lieu()

    ten_chay = ["xgboost", "sku_base", "ridge"] + [t for t in ["random_forest", "lightgbm", "catboost"]
                                                    if t in K.MO_HINH]
    ket, chi_tiet = [], {}
    for chia in K.CACH_CHIA:
        print(f"\n--- {K.CO.TEN_CHIA[chia]}")
        for ten in ten_chay:
            chi, ct = K.chay(ten, du, chia)
            chi_tiet[(ten, chia)] = ct
            ket.append({"mo_hinh": ten, "cach_chia": chia, "co_rang_buoc_don_dieu": CO_RANG_BUOC.get(ten, True),
                        **chi})
            print(f"    {ten:<14}: MAE {chi['mae']:.3f} | RMSE {chi['rmse']:.3f} | R²(ln) {chi['r2_ln']:.3f} | "
                  f"lệch {chi['lech_tong_pct']:+.1f}% | elasticity {chi['elasticity_tb']:+.2f} | "
                  f"phẳng {chi['ty_le_phang_pct']:.1f}%")

    ket_df = pd.DataFrame(ket)
    so = []
    for _, r in ket_df.iterrows():
        moc = chi_tiet[("xgboost", r["cach_chia"])]
        cur = chi_tiet[(r["mo_hinh"], r["cach_chia"])]
        so.append(K.so_voi_moc(cur, moc))
    ket_df = pd.concat([ket_df, pd.DataFrame(so)], axis=1)

    for (ten, chia), ct in chi_tiet.items():
        ct.to_csv(os.path.join(K.CHI_TIET, f"{ten}_{chia}.csv"), index=False, encoding="utf-8-sig")
    ket_df.to_csv(os.path.join(K.OUT, "ket_qua_buoc3.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("SO SÁNH VỚI XGBOOST (chênh MAE: âm = tốt hơn XGBoost; khoảng không chứa 0 = có ý nghĩa)")
    print("=" * 78)
    cot = ["mo_hinh", "cach_chia", "mae", "r2_ln", "lech_tong_pct", "chenh_mae_so_voi_moc",
           "khoang_lo", "khoang_hi", "elasticity_tb", "co_rang_buoc_don_dieu"]
    print(ket_df[cot].round(3).to_string(index=False))

    # kết luận nhanh: mô hình nào vượt XGBoost ở cả ba cách chia
    print("\nMô hình vượt XGBoost có ý nghĩa (khoảng 95% < 0) ở cả ba cách chia:")
    tot = False
    for ten in ket_df["mo_hinh"].unique():
        if ten == "xgboost":
            continue
        r = ket_df[ket_df["mo_hinh"] == ten]
        if len(r) == 3 and (r["khoang_hi"] < 0).all():
            print(f"  {ten}")
            tot = True
    if not tot:
        print("  Không mô hình nào vượt XGBoost có ý nghĩa ở cả ba cách chia.")
    print(f"\nĐã ghi {K.OUT}/ket_qua_buoc3.csv và chi_tiet/")


if __name__ == "__main__":
    main()
