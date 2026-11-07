"""
BƯỚC 2: BASELINE sku_base VÀ RIDGE, SO VỚI XGBOOST
==================================================
Thêm vào khung (khung_so_sanh_2y.py):
  - sku_base : dự báo ln(qty) = sku_base (mức bán nền của sản phẩm, không dùng giá hay mùa vụ).
               Đây là mốc đo giá trị của toàn bộ phần giá và mùa vụ.
  - ridge    : hồi quy tuyến tính Ridge trên 8 đặc trưng (đã chuẩn hóa), alpha cố định = 1.0.
               Không tune alpha trên test. Không có ràng buộc đơn điệu, không có tương tác.

Đọc kết quả:
  - sku_base: kỳ vọng MAE ≈ 23.37 (xen kẽ), 24.56 (chiều 1→2), 28.49 (chiều 2→1). Đây là các số đã đo.
  - ridge: nếu gần XGBoost, phần phi tuyến đóng góp ít. Nếu xa, XGBoost học được quan hệ mà đường thẳng không có.
  - Độ nhạy giá: sku_base có elasticity = 0 (không phản ứng với giá). Ridge có hệ số tuyến tính trên giá.

Đầu ra: ./so_sanh_out/ket_qua_buoc2.csv (gồm cả xgboost để so), và chi_tiet/ cho từng mô hình.
Cần khung_so_sanh_2y.py trong cùng thư mục.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import khung_so_sanh_2y as K

warnings.filterwarnings("ignore")

ALPHA_RIDGE = 1.0
MOC_BASELINE_MAE = {"xen_ke": 23.369, "chieu_1_2": 24.56, "chieu_2_1": 28.49}
DUNG_SAI_BASELINE = 0.05


class MoHinhSkuBase:
    """Dự báo ln(qty) = sku_base. Không phụ thuộc giá, nên mọi dòng không phản ứng với giá."""

    def __init__(self):
        self.idx = K.FEATS.index("sku_base")

    def predict(self, X):
        return np.asarray(X)[:, self.idx]


def huan_sku_base(trn):
    return MoHinhSkuBase()


def huan_ridge(trn):
    """Ridge trên 8 đặc trưng, chuẩn hóa trước. Target là ln(qty)."""
    m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA_RIDGE))
    m.fit(trn[K.FEATS].values.astype(np.float64), np.log(trn["qty"].values.astype(float)))
    return _RidgeWrap(m)


class _RidgeWrap:
    """Bọc pipeline để predict nhận float32 như các mô hình khác trong khung."""

    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=np.float64))


K.dang_ky("sku_base", huan_sku_base)
K.dang_ky("ridge", huan_ridge)


def kiem_tra_baseline(ket):
    print("\n[Kiểm tra baseline] sku_base phải khớp MAE đã đo:")
    ok = True
    for chia in K.CACH_CHIA:
        r = ket[(ket["mo_hinh"] == "sku_base") & (ket["cach_chia"] == chia)]
        if r.empty:
            print(f"  {chia}: không có kết quả")
            ok = False
            continue
        mae = float(r.iloc[0]["mae"])
        lech = mae - MOC_BASELINE_MAE[chia]
        dat = abs(lech) <= DUNG_SAI_BASELINE
        ok = ok and dat
        print(f"  {chia}: MAE = {mae:.3f} | đã đo = {MOC_BASELINE_MAE[chia]:.3f} | lệch {lech:+.3f} -> "
              f"{'KHỚP' if dat else 'KHÔNG KHỚP'}")
    return ok


def main():
    print("=" * 78)
    print("BƯỚC 2: BASELINE sku_base VÀ RIDGE, SO VỚI XGBOOST")
    print("=" * 78)
    os.makedirs(K.CHI_TIET, exist_ok=True)
    du = K.nap_du_lieu()

    ket, chi_tiet = [], {}
    for chia in K.CACH_CHIA:
        print(f"\n--- {K.CO.TEN_CHIA[chia]}")
        for ten in ["xgboost", "sku_base", "ridge"]:
            chi, ct = K.chay(ten, du, chia)
            chi_tiet[(ten, chia)] = ct
            ket.append({"mo_hinh": ten, "cach_chia": chia, **chi})
            print(f"    {ten:<9}: MAE {chi['mae']:.3f} | RMSE {chi['rmse']:.3f} | R²(ln) {chi['r2_ln']:.3f} | "
                  f"lệch tổng {chi['lech_tong_pct']:+.1f}% | elasticity {chi['elasticity_tb']:+.2f} | "
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
    ket_df.to_csv(os.path.join(K.OUT, "ket_qua_buoc2.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("SO SÁNH VỚI XGBOOST (chênh MAE: dương = kém hơn XGBoost)")
    print("=" * 78)
    cot = ["mo_hinh", "cach_chia", "mae", "rmse", "r2_ln", "lech_tong_pct",
           "chenh_mae_so_voi_moc", "khoang_lo", "khoang_hi", "elasticity_tb"]
    print(ket_df[cot].round(3).to_string(index=False))

    kiem_tra_baseline(ket_df)
    print(f"\nĐã ghi {K.OUT}/ket_qua_buoc2.csv và chi_tiet/")


if __name__ == "__main__":
    main()
