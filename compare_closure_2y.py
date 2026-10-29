"""
Tuần NGHỈ (đóng cửa Giáng sinh): thêm doanh số 0 rồi đánh giá lại 4 cấu hình đặc trưng
=====================================================================================
Bối cảnh: chẩn đoán cho thấy hai tuần không có giao dịch nào (28/12/2009 và 27/12/2010) là tuần đóng cửa.
Mô hình chưa từng thấy trạng thái này.

Làm gì:
  1. Tìm các tuần nằm trong khoảng hoạt động nhưng không có giao dịch (tuần nghỉ).
  2. Thêm dòng doanh số 0 cho các sản phẩm đã bán ở tuần trước đó (giá = giá tuần trước).
  3. Huấn luyện trên tuần MỞ (không lấy log của 0). Mô hình không thể học trực tiếp trạng thái nghỉ.
  4. Đánh giá trên TẤT CẢ dòng test, gồm cả tuần nghỉ, với 2 cách:
       - không luật: dự báo như bình thường (đây là hành vi hiện tại của pipeline)
       - có luật đóng cửa: tuần nghỉ dự báo 0
  5. Câu hỏi chính: cấu hình lịch có giảm MAE trên TUẦN MỞ khi áp dụng luật đóng cửa không?
     Tiêu chí như trước: cải thiện ở CẢ HAI chiều năm (khoảng 95% < 0) và lệch tổng không xấu quá 2 điểm.

Lưu ý thiết kế:
  - Luật đóng cửa dựa trên việc tuần đó không có giao dịch trong dữ liệu. Khi dự báo tương lai, cần một
    luật lịch tương đương (ví dụ nghỉ từ 24/12 đến 01/01), và điều đó phải được xác nhận với nghiệp vụ.
  - Tuần nghỉ nằm trong tập train thì KHÔNG đưa vào huấn luyện (vì log(0) không xác định).
  - Thống kê sản phẩm chỉ tính trên tuần mở của train.

Đầu ra: ./compare_closure_out/ket_qua_dong_cua.csv, tuan_nghi.csv
Cần chan_doan_lich_2y.py, compare_features_2y.py, compare_outliers_2y.py trong cùng thư mục.
"""
import os
import warnings

import numpy as np
import pandas as pd

import chan_doan_lich_2y as CD
import compare_features_2y as CF
import compare_outliers_2y as CO
from y2_config import path

warnings.filterwarnings("ignore")

OUT = "./compare_closure_out"
CHIEU = ["chieu_1_2", "chieu_2_1"]
CAU_HINH_LICH = ["chi_giang_sinh", "chi_lich_khac", "tat_ca"]


def tuan_nghi_va_dong(wk, skus):
    full = pd.date_range(wk["week"].min(), wk["week"].max(), freq="7D")
    co = set(wk["week"].unique())
    nghi = [w for w in full if w not in co]
    rows = []
    for w in nghi:
        prev = w - pd.Timedelta(days=7)
        p = wk[wk["week"] == prev][["sku", "price"]]
        p = p[p["sku"].isin(skus)]
        rows.append(pd.DataFrame({"sku": p["sku"].values, "week": w,
                                  "qty": 0.0, "price": p["price"].values}))
    dong = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sku", "week", "qty", "price"])
    dong["dong_cua"] = 1
    return full, nghi, dong


def chay_chia(allw, weeks_all, lich, chia):
    test = np.asarray(CO.la_test(chia, allw["week"], weeks_all))
    mo = allw["dong_cua"].values == 0
    wtr = allw[(~test) & mo].reset_index(drop=True)
    wte = allw[test].reset_index(drop=True)
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st).merge(lich, on="week", how="left")
    tst = CO.dac_trung(wte, st).merge(lich, on="week", how="left")
    tst = tst.sort_values(["sku", "week"]).reset_index(drop=True)
    y = tst["qty"].values.astype(float)
    nghi = tst["dong_cua"].values == 1
    mo_t = ~nghi
    idx_all, uq_all = pd.factorize(tst["sku"])
    idx_mo, uq_mo = pd.factorize(tst["sku"][mo_t])

    pred = {}
    for ten, cols in CD.CAU_HINH.items():
        feats = CF.FEATS_GOC + cols
        m = CF.mo_hinh(trn, feats)
        X = tst[feats].values.astype(np.float32)
        p = np.exp(m.predict(X))
        eps, flat = CF.dap_ung(m, X, feats)
        for luat in (False, True):
            pp = p.copy()
            if luat:
                pp[nghi] = 0.0
            pred[(ten, luat)] = (pp, eps, flat)

    ref = pred[("goc", True)][0]                 # tham chiếu: 8 đặc trưng hiện tại + luật đóng cửa
    err_ref = np.abs(y - ref)
    rows = []
    for (ten, luat), (pp, eps, flat) in pred.items():
        err = np.abs(y - pp)
        r = {"cach_chia": chia, "cau_hinh": ten, "luat_dong_cua": luat,
             "so_dong_test": len(y), "so_dong_nghi_test": int(nghi.sum()),
             "mae_tat_ca": float(err.mean()), "mae_tuan_mo": float(err[mo_t].mean()),
             "mae_tuan_nghi": float(err[nghi].mean()) if nghi.any() else np.nan,
             "lech_tong_tuan_mo_pct": float((pp[mo_t].sum() / y[mo_t].sum() - 1) * 100),
             "eps_tuan_mo": float(np.mean(eps[mo_t])),
             "ty_le_phang_tuan_mo_pct": float(np.mean(flat[mo_t]) * 100)}
        if not (ten == "goc" and luat):
            d, lo, hi, _ = CO.bootstrap_so_a(err_ref, err, idx_all, len(uq_all))
            r.update({"chenh_mae_tat_ca": d, "khoang_tat_ca_lo": lo, "khoang_tat_ca_hi": hi})
            d2, lo2, hi2, sh2 = CO.bootstrap_so_a(err_ref[mo_t], err[mo_t], idx_mo, len(uq_mo))
            r.update({"chenh_mae_tuan_mo": d2, "khoang_mo_lo": lo2, "khoang_mo_hi": hi2, "ty_le_tot_hon_mo": sh2})
        rows.append(r)
    return pd.DataFrame(rows), int(nghi.sum())


def ket_luan(df):
    print("\n" + "=" * 78)
    print("TÁC ĐỘNG CỦA LUẬT ĐÓNG CỬA (cấu hình goc, không luật so với có luật)")
    print("=" * 78)
    for chia in CHIEU + ["xen_ke"]:
        g = df[(df["cach_chia"] == chia) & (df["cau_hinh"] == "goc") & (~df["luat_dong_cua"])]
        if g.empty or pd.isna(g.iloc[0].get("chenh_mae_tat_ca", np.nan)):
            continue
        g = g.iloc[0]
        print(f"  {CO.TEN_CHIA[chia]}: chênh MAE khi thêm luật = {g['chenh_mae_tat_ca']:+.3f} "
              f"[{g['khoang_tat_ca_lo']:+.3f}; {g['khoang_tat_ca_hi']:+.3f}] | "
              f"số dòng nghỉ trong test: {int(g['so_dong_nghi_test'])}")

    print("\n" + "=" * 78)
    print("ĐẶC TRƯNG LỊCH SAU KHI CÓ LUẬT ĐÓNG CỬA (so với goc + luật, trên TUẦN MỞ)")
    print("=" * 78)
    for chia in CHIEU:
        ref = df[(df["cach_chia"] == chia) & (df["cau_hinh"] == "goc") & (df["luat_dong_cua"])].iloc[0]
        for cfg in CAU_HINH_LICH:
            r = df[(df["cach_chia"] == chia) & (df["cau_hinh"] == cfg) & (df["luat_dong_cua"])].iloc[0]
            ok = (r["khoang_mo_hi"] < 0) and (abs(r["lech_tong_tuan_mo_pct"]) <= abs(ref["lech_tong_tuan_mo_pct"]) + 2.0)
            print(f"  {chia} | {cfg:<14}: chênh MAE mở {r['chenh_mae_tuan_mo']:+.3f} "
                  f"[{r['khoang_mo_lo']:+.3f}; {r['khoang_mo_hi']:+.3f}] | lệch tổng "
                  f"{ref['lech_tong_tuan_mo_pct']:+.1f}% -> {r['lech_tong_tuan_mo_pct']:+.1f}% | "
                  f"{'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    for cfg in CAU_HINH_LICH:
        dat = all(
            (df[(df["cach_chia"] == c) & (df["cau_hinh"] == cfg) & (df["luat_dong_cua"])].iloc[0]["khoang_mo_hi"] < 0)
            for c in CHIEU)
        print(f"  => {cfg} ở cả hai chiều: {'ĐẠT' if dat else 'KHÔNG ĐẠT'}")


def main():
    print("=" * 78)
    print("TUẦN NGHỈ: thêm doanh số 0, đánh giá 4 cấu hình có và không có luật đóng cửa")
    print("=" * 78)
    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    wk = CO.tuan_tu(CO.giao_dich_phien_ban(df, True))
    wk = wk[wk["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(wk["week"].unique()))
    full, nghi, dong_nghi = tuan_nghi_va_dong(wk, skus)
    print(f"[1] Tuần trong khoảng hoạt động: {len(full)} | tuần có giao dịch: {len(weeks_all)} | "
          f"tuần nghỉ: {len(nghi)}")
    for w in nghi:
        print(f"    nghỉ: {w.date()} (thêm {int((dong_nghi['week'] == w).sum())} dòng doanh số 0)")

    open_rows = wk.assign(dong_cua=0)
    allw = pd.concat([open_rows, dong_nghi], ignore_index=True)
    lich = CF.lich_tuan(full)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"tuan_bat_dau": nghi}).to_csv(os.path.join(OUT, "tuan_nghi.csv"), index=False,
                                               encoding="utf-8-sig")

    ket = []
    for chia in CO.CHIA:
        print(f"\n--- {CO.TEN_CHIA[chia]}")
        r, n_nghi_test = chay_chia(allw, weeks_all, lich, chia)
        print(f"    dòng nghỉ trong test: {n_nghi_test}")
        ket.append(r)
    kq = pd.concat(ket, ignore_index=True)
    kq.to_csv(os.path.join(OUT, "ket_qua_dong_cua.csv"), index=False, encoding="utf-8-sig")
    ket_luan(kq)
    print(f"\nĐã ghi {OUT}/ket_qua_dong_cua.csv và tuan_nghi.csv")


if __name__ == "__main__":
    main()
