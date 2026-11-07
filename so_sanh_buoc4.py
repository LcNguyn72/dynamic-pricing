"""
BƯỚC 4: EXPECTILE GAM (pygam), SO VỚI XGBOOST
=============================================
Mô hình: ExpectileGAM với tau (expectile) = 0.5, tức ước lượng trung bình có điều kiện của ln(qty).
  ln(qty) = f_1(log_price_ratio) + f_2(sku_lp_med) + ... + f_8(sku_q4_effect)
Mỗi f_i là một đường spline mượt. Không có tương tác giữa các đặc trưng (cộng tính).
Ràng buộc: f_1 (log_price_ratio) giảm đơn điệu (monotonic_dec), như XGBoost.

Tham số cố định, không tune trên test:
  n_splines = 15 cho mỗi đặc trưng, lam = 0.6 (độ phạt độ cong), max_iter = 100.

Ngoài chỉ số như các bước trước, script vẽ đường hiệu ứng giá trung bình: lấy 2000 dòng test ngẫu nhiên,
thay log_price_ratio bằng một lưới giá, giữ các đặc trưng khác, lấy trung bình dự báo. Kết quả lưu để vẽ.
Kiểm tra: đường có giảm không, và có bao nhiêu điểm tăng lên (vi phạm đơn điệu).

Đọc kết quả:
  - MAE so với XGBoost: như các bước trước, khoảng bootstrap không chứa 0 mới là khác biệt có ý nghĩa.
  - Đường hiệu ứng giá: nên trơn và giảm. Nếu có đoạn gồ ghề, có thể do n_splines quá lớn hoặc lam quá nhỏ.
  - Cộng tính không bắt tương tác giữa giá và mùa vụ. Nếu XGBoost tốt hơn đáng kể, đó là một phần lý do.

Cần: pygam (pip install pygam). Cần ket_qua_buoc3.csv (chạy bước 3 trước).
Đầu ra: so_sanh_out/ket_qua_buoc4.csv (gồm mọi mô hình), so_sanh_out/gam_hieu_ung_gia.csv, chi_tiet/.
"""
import os
import warnings

import numpy as np
import pandas as pd

import khung_so_sanh_2y as K
import so_sanh_buoc3  # noqa: F401  (đăng ký các mô hình bước 1-3, để tương thích)

warnings.filterwarnings("ignore")

N_SPLINES = 15
LAM = 0.6
MAX_ITER = 100
MAU_HIEU_UNG = 2000
SO_DIEM_LUOI = 41

try:
    from pygam import ExpectileGAM, s
    PYGAM_OK = True
    PYGAM_LOI = ""
except ImportError as e:
    PYGAM_OK = False
    PYGAM_LOI = str(e)

IDX_GIA = K.FEATS.index("log_price_ratio")


def huan_gam(trn):
    """ExpectileGAM tau = 0.5, spline cho 8 đặc trưng, giảm đơn điệu trên log_price_ratio."""
    X = trn[K.FEATS].values.astype(np.float64)
    y = np.log(trn["qty"].values.astype(float))
    terms = None
    for i in range(len(K.FEATS)):
        if i == IDX_GIA:
            t = s(i, n_splines=N_SPLINES, constraints="monotonic_dec")
        else:
            t = s(i, n_splines=N_SPLINES)
        terms = t if terms is None else terms + t
    m = ExpectileGAM(terms, expectile=0.5, lam=LAM, max_iter=MAX_ITER)
    m.fit(X, y)
    return _GamWrap(m)


class _GamWrap:
    """Bọc để predict nhận float32 như các mô hình khác trong khung."""

    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=np.float64))


if PYGAM_OK:
    K.dang_ky("gam", huan_gam)


def duong_hieu_ung_gia(model, tst, seed=0):
    """Hiệu ứng giá trung bình: thay log_price_ratio theo lưới, giữ các đặc trưng khác của mẫu, lấy trung bình."""
    rng = np.random.default_rng(seed)
    n = min(MAU_HIEU_UNG, len(tst))
    idx = rng.choice(len(tst), size=n, replace=False)
    X = tst[K.FEATS].values.astype(np.float64)[idx]
    lo, hi = np.quantile(X[:, IDX_GIA], [0.01, 0.99])
    luoi = np.linspace(lo, hi, SO_DIEM_LUOI)
    trung_binh = []
    for g in luoi:
        Xg = X.copy()
        Xg[:, IDX_GIA] = g
        trung_binh.append(float(np.mean(np.exp(model.predict(Xg)))))
    trung_binh = np.array(trung_binh)
    tang = int(np.sum(np.diff(trung_binh) > 1e-9))
    return pd.DataFrame({"log_price_ratio": luoi, "gia_tuong_doi": np.exp(luoi),
                         "luong_ban_tb": trung_binh}), tang


def main():
    print("=" * 78)
    print("BƯỚC 4: EXPECTILE GAM, SO VỚI XGBOOST")
    print("=" * 78)
    if not PYGAM_OK:
        raise SystemExit(f"Chưa cài pygam: {PYGAM_LOI}\nChạy: pip install pygam")

    buoc3 = os.path.join(K.OUT, "ket_qua_buoc3.csv")
    if not os.path.exists(buoc3):
        raise SystemExit("Thiếu ket_qua_buoc3.csv. Chạy so_sanh_buoc3.py trước.")
    ket_cu = pd.read_csv(buoc3)
    os.makedirs(K.CHI_TIET, exist_ok=True)
    du = K.nap_du_lieu()

    ket_moi, hieu_ung = [], []
    for chia in K.CACH_CHIA:
        print(f"\n--- {K.CO.TEN_CHIA[chia]}")
        chi, ct = K.chay("gam", du, chia)
        ket_moi.append({"mo_hinh": "gam", "cach_chia": chia, "co_rang_buoc_don_dieu": True, **chi})
        ct.to_csv(os.path.join(K.CHI_TIET, f"gam_{chia}.csv"), index=False, encoding="utf-8-sig")
        print(f"    gam: MAE {chi['mae']:.3f} | RMSE {chi['rmse']:.3f} | R²(ln) {chi['r2_ln']:.3f} | "
              f"lệch {chi['lech_tong_pct']:+.1f}% | elasticity {chi['elasticity_tb']:+.2f} | "
              f"phẳng {chi['ty_le_phang_pct']:.1f}%")

        # đường hiệu ứng giá: cần mô hình và dữ liệu test của cách chia này
        trn, tst, tb = K.tach(du, chia)
        m = huan_gam(trn)
        df_hu, tang = duong_hieu_ung_gia(m, tst)
        df_hu.insert(0, "cach_chia", chia)
        df_hu["so_diem_tang_len"] = tang
        hieu_ung.append(df_hu)
        print(f"    đường hiệu ứng giá: {tang} điểm tăng lên (0 = giảm đơn điệu đầy đủ)")

    ket_gam = pd.DataFrame(ket_moi)
    so = []
    for _, r in ket_gam.iterrows():
        moc = pd.read_csv(os.path.join(K.CHI_TIET, f"xgboost_{r['cach_chia']}.csv"))
        cur = pd.read_csv(os.path.join(K.CHI_TIET, f"gam_{r['cach_chia']}.csv"))
        so.append(K.so_voi_moc(cur, moc))
    ket_gam = pd.concat([ket_gam, pd.DataFrame(so)], axis=1)

    ket_cu = ket_cu[ket_cu["mo_hinh"] != "gam"]
    ket_tong = pd.concat([ket_cu, ket_gam], ignore_index=True)
    ket_tong.to_csv(os.path.join(K.OUT, "ket_qua_buoc4.csv"), index=False, encoding="utf-8-sig")
    pd.concat(hieu_ung, ignore_index=True).to_csv(os.path.join(K.OUT, "gam_hieu_ung_gia.csv"),
                                                  index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("GAM SO VỚI XGBOOST (âm = tốt hơn; khoảng không chứa 0 = có ý nghĩa)")
    print("=" * 78)
    print(ket_gam[["mo_hinh", "cach_chia", "mae", "r2_ln", "lech_tong_pct", "chenh_mae_so_voi_moc",
                   "khoang_lo", "khoang_hi", "elasticity_tb"]].round(3).to_string(index=False))
    print(f"\nĐã ghi {K.OUT}/ket_qua_buoc4.csv, gam_hieu_ung_gia.csv, chi_tiet/")


if __name__ == "__main__":
    main()
