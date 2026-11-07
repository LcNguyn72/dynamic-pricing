"""
BƯỚC 1: KHUNG ĐÁNH GIÁ DÙNG CHUNG CHO MỌI MÔ HÌNH
==================================================
File này không tự huấn luyện mô hình mới. Nó cung cấp:
  1. Dữ liệu theo đúng quy trình pipeline (cắt đơn sỉ 95%, gộp tuần, lọc sản phẩm cố định).
  2. Thống kê sản phẩm và 8 đặc trưng, tính trên train của từng cách chia.
  3. Đăng ký mô hình: mỗi mô hình là một hàm huan_luyen(trn) trả về đối tượng có predict(X) -> ln(qty).
  4. Chỉ số: R² (ln), MAE, RMSE, thiên lệch tổng (số lượng), MAE/RMSE chuẩn hóa theo sản phẩm.
  5. Độ nhạy giá: elasticity trung bình và tỷ lệ dòng không phản ứng (đổi giá ±10%).
  6. Bootstrap theo sản phẩm: chênh MAE so với mốc (XGBoost) và khoảng 95%.
  7. Kiểm tra mốc: XGBoost trên khung này phải ra đúng MAE đã biết.

Các mô hình sau (Ridge, cây, GAM) chỉ cần gọi dang_ky(...) rồi chay(...).

Đầu ra: ./so_sanh_out/ket_qua_buoc1.csv và ./so_sanh_out/chi_tiet/<mô_hình>_<cách_chia>.csv
Không ghi gì vào models_2y/. Không sửa pipeline.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import compare_features_2y as CF
import compare_outliers_2y as CO
from y2_config import path

warnings.filterwarnings("ignore")

OUT = "./so_sanh_out"
CHI_TIET = os.path.join(OUT, "chi_tiet")
CACH_CHIA = CO.CHIA
FEATS = CO.FEATS                       # 8 đặc trưng hiện tại, đúng thứ tự pipeline
IDX_GIA = FEATS.index("log_price_ratio")
STEP_GIA = 0.10                        # đổi giá ±10% để đo phản ứng
N_BOOT = CO.N_BOOT
MOC_XGB_MAE = {"xen_ke": 20.412, "chieu_1_2": 22.151, "chieu_2_1": 25.537}   # đã đo trước đó
DUNG_SAI_MOC = 0.05

MO_HINH = {}                           # tên -> hàm huan_luyen(trn) -> đối tượng có predict(X) trả ln(qty)
_CACHE_TACH = {}                       # cache dữ liệu theo cách chia, để nhiều mô hình dùng chung


# ----------------------------------------------------------------------------- 1. dữ liệu
def nap_du_lieu():
    """Dữ liệu theo đúng pipeline: cắt đơn sỉ 95% theo sản phẩm, gộp tuần, giữ 1306 sản phẩm cố định."""
    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    wk = CO.tuan_tu(CO.giao_dich_phien_ban(df, True))
    wk = wk[wk["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(wk["week"].unique()))
    print(f"  Dữ liệu: {len(wk):,} dòng tuần, {wk['sku'].nunique()} sản phẩm, {len(weeks_all)} tuần")
    return {"wk": wk, "weeks_all": weeks_all, "skus": skus}


def tach(du, chia):
    """Trả về (trn, tst, tb) cho một cách chia.
    trn: dòng train với 8 đặc trưng và qty. Thống kê sản phẩm chỉ tính trên train.
    tst: dòng test với 8 đặc trưng và qty (sắp theo sku, week).
    tb: trung bình qty train của sản phẩm, căn theo từng dòng tst (dùng cho chỉ số chuẩn hóa)."""
    if chia in _CACHE_TACH:
        return _CACHE_TACH[chia]
    wk, weeks_all = du["wk"], du["weeks_all"]
    test = np.asarray(CO.la_test(chia, wk["week"], weeks_all))
    wtr = wk[~test].reset_index(drop=True)
    wte = wk[test].reset_index(drop=True)
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st)
    tst = CO.dac_trung(wte, st).sort_values(["sku", "week"]).reset_index(drop=True)
    tb = wtr.groupby("sku")["qty"].mean().reindex(tst["sku"].values).fillna(1.0).values
    _CACHE_TACH[chia] = (trn, tst, tb)
    return trn, tst, tb


# ----------------------------------------------------------------------------- 2. đăng ký mô hình
def dang_ky(ten, huan_luyen):
    """Đăng ký một mô hình. huan_luyen(trn) phải trả về đối tượng có predict(X) -> ln(qty)."""
    MO_HINH[ten] = huan_luyen


def huan_xgboost(trn):
    """Mốc: XGBoost với 8 đặc trưng, cấu hình và ràng buộc đơn điệu như pipeline hiện tại."""
    return CF.mo_hinh(trn, FEATS)


dang_ky("xgboost", huan_xgboost)


# ----------------------------------------------------------------------------- 3. chỉ số
def chi_so(y_qty, p_qty, tb):
    """Các chỉ số chung cho mọi mô hình. R² tính trên thang ln, còn lại trên số lượng."""
    y_qty = np.asarray(y_qty, float)
    p_qty = np.clip(np.asarray(p_qty, float), 1e-6, None)
    y_ln, p_ln = np.log(np.clip(y_qty, 1e-6, None)), np.log(p_qty)
    return {
        "so_dong": int(len(y_qty)),
        "r2_ln": float(r2_score(y_ln, p_ln)),
        "mae": float(np.mean(np.abs(p_qty - y_qty))),
        "rmse": float(np.sqrt(np.mean((p_qty - y_qty) ** 2))),
        "lech_tong_pct": float((p_qty.sum() / y_qty.sum() - 1) * 100),
        "mae_chuan": float(np.mean(np.abs(p_qty / tb - y_qty / tb))),
        "rmse_chuan": float(np.sqrt(np.mean((p_qty / tb - y_qty / tb) ** 2))),
    }


# ----------------------------------------------------------------------------- 4. độ nhạy giá
def phan_ung_gia(predict_ln, X):
    """Đổi giá ±10% (chỉ đổi log_price_ratio), đo elasticity trung bình và tỷ lệ dòng không phản ứng."""
    up, dn = X.copy(), X.copy()
    up[:, IDX_GIA] += np.log(1 + STEP_GIA)
    dn[:, IDX_GIA] += np.log(1 - STEP_GIA)
    pu, pdn = predict_ln(up), predict_ln(dn)
    e = (pu - pdn) / (np.log(1 + STEP_GIA) - np.log(1 - STEP_GIA))
    return {"elasticity_tb": float(np.mean(e)), "ty_le_phang_pct": float(np.mean(np.isclose(pu, pdn)) * 100)}


# ----------------------------------------------------------------------------- 5. chạy một mô hình
def chay(ten, du, chia):
    """Huấn luyện mô hình ten trên cách chia, trả về (chỉ số, bảng chi tiết từng dòng test)."""
    trn, tst, tb = tach(du, chia)
    m = MO_HINH[ten](trn)
    X = tst[FEATS].values.astype(np.float32)
    p_qty = np.exp(m.predict(X))
    y_qty = tst["qty"].values.astype(float)
    chi = chi_so(y_qty, p_qty, tb)
    chi.update(phan_ung_gia(m.predict, X))
    chi_tiet = pd.DataFrame({"sku": tst["sku"].values, "week": tst["week"].values,
                             "thuc_te": y_qty, "du_bao": p_qty, "tb": tb})
    return chi, chi_tiet


# ----------------------------------------------------------------------------- 6. so với mốc
def so_voi_moc(chi_tiet, chi_tiet_moc):
    """Chênh MAE so với mốc (số dương = kém hơn mốc), khoảng 95% bootstrap theo sản phẩm.
    Ghép hai bảng theo (sku, week) để chắc chắn cùng dòng test."""
    g = chi_tiet.merge(chi_tiet_moc, on=["sku", "week"], suffixes=("", "_moc"))
    if len(g) == 0:
        return {"chenh_mae_so_voi_moc": np.nan, "khoang_lo": np.nan, "khoang_hi": np.nan, "ty_le_tot_hon_moc": np.nan}
    idx, uniq = pd.factorize(g["sku"])
    err_moc = np.abs(g["thuc_te"].values - g["du_bao_moc"].values)
    err = np.abs(g["thuc_te"].values - g["du_bao"].values)
    d, lo, hi, share = CO.bootstrap_so_a(err_moc, err, idx, len(uniq))
    return {"chenh_mae_so_voi_moc": d, "khoang_lo": lo, "khoang_hi": hi, "ty_le_tot_hon_moc": share}


# ----------------------------------------------------------------------------- 7. kiểm tra mốc
def kiem_tra_moc(ket_qua):
    """XGBoost trên khung này phải ra đúng MAE đã đo. Nếu lệch, khung đang khác pipeline."""
    print("\n[Kiểm tra mốc] XGBoost phải khớp MAE đã đo:")
    ok = True
    for chia in CACH_CHIA:
        r = ket_qua[(ket_qua["mo_hinh"] == "xgboost") & (ket_qua["cach_chia"] == chia)]
        if r.empty:
            print(f"  {chia}: không có kết quả XGBoost")
            ok = False
            continue
        mae = float(r.iloc[0]["mae"])
        lech = mae - MOC_XGB_MAE[chia]
        dat = abs(lech) <= DUNG_SAI_MOC
        ok = ok and dat
        print(f"  {chia}: MAE khung = {mae:.3f} | mốc = {MOC_XGB_MAE[chia]:.3f} | lệch {lech:+.3f} -> "
              f"{'KHỚP' if dat else 'KHÔNG KHỚP: kiểm tra lại dữ liệu hoặc khung'}")
    return ok


# ----------------------------------------------------------------------------- 8. validation cho tune
def tach_con_val(du, chia):
    """Sub-train (80% tuần đầu của train) và validation (20% tuần cuối). Thống kê tính trên sub-train.
    Dùng để chọn tham số (tune) mà không dùng tập test."""
    wk, weeks_all = du["wk"], du["weeks_all"]
    test = np.asarray(CO.la_test(chia, wk["week"], weeks_all))
    wtr = wk[~test].reset_index(drop=True)
    tuan = np.sort(wtr["week"].unique())
    cut = tuan[int(len(tuan) * 0.8)]
    sub = wtr[wtr["week"] < cut].reset_index(drop=True)
    val = wtr[wtr["week"] >= cut].reset_index(drop=True)
    st = CO.thong_ke(sub)
    return CO.dac_trung(sub, st), CO.dac_trung(val, st)


# ----------------------------------------------------------------------------- chạy chính
def main():
    print("=" * 78)
    print("BƯỚC 1: KHUNG ĐÁNH GIÁ DÙNG CHUNG")
    print("=" * 78)
    os.makedirs(CHI_TIET, exist_ok=True)
    du = nap_du_lieu()

    ket = []
    chi_tiet_luu = {}
    for chia in CACH_CHIA:
        print(f"\n--- {CO.TEN_CHIA[chia]}")
        for ten in MO_HINH:
            chi, ct = chay(ten, du, chia)
            chi_tiet_luu[(ten, chia)] = ct
            ket.append({"mo_hinh": ten, "cach_chia": chia, **chi})
            print(f"    {ten}: MAE {chi['mae']:.3f} | R²(ln) {chi['r2_ln']:.3f} | lệch tổng {chi['lech_tong_pct']:+.1f}% | "
                  f"elasticity {chi['elasticity_tb']:.2f} | phẳng {chi['ty_le_phang_pct']:.1f}%")

    ket_df = pd.DataFrame(ket)
    # so với mốc XGBoost
    so = []
    for _, r in ket_df.iterrows():
        moc = chi_tiet_luu[("xgboost", r["cach_chia"])]
        cur = chi_tiet_luu[(r["mo_hinh"], r["cach_chia"])]
        so.append(so_voi_moc(cur, moc))
    ket_df = pd.concat([ket_df, pd.DataFrame(so)], axis=1)

    for (ten, chia), ct in chi_tiet_luu.items():
        ct.to_csv(os.path.join(CHI_TIET, f"{ten}_{chia}.csv"), index=False, encoding="utf-8-sig")
    ket_df.to_csv(os.path.join(OUT, "ket_qua_buoc1.csv"), index=False, encoding="utf-8-sig")

    kiem_tra_moc(ket_df)
    print(f"\nĐã ghi {OUT}/ket_qua_buoc1.csv và thư mục chi_tiet/")


if __name__ == "__main__":
    main()
