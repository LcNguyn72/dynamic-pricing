"""
Kiểm tra độ tin cậy của mô hình (không sửa pipeline)
====================================================
Phần 1 - Rò rỉ ngưỡng cắt 5% đơn lớn:
  Pipeline tính phân vị 95% của từng sản phẩm trên TOÀN BỘ dữ liệu, gồm cả tuần test.
  So hai cách, cùng một cách chia:
    bao_cao_hien_tai : ngưỡng từ toàn bộ dữ liệu (như pipeline)
    trung_thuc       : ngưỡng chỉ từ các giao dịch thuộc tuần train, áp dụng cho cả train và test
    bao_cao_hien_tai_cham_tren_trung_thuc : mô hình của cách 1, chấm trên target của cách 2
  Mục tiêu: xem số liệu báo cáo lạc quan bao nhiêu khi ngưỡng không nhìn thấy tuần test.

Phần 2 - Độ bao phủ của khoảng dự báo (conformal trên thang ln):
  Train -> 80% tuần đầu để huấn luyện, 20% tuần cuối để hiệu chỉnh (phần dư tuyệt đối trên ln(qty)).
  Khoảng = dự báo ln ± q, q là phân vị conformal tại mức 80% và 95%.
  Đo tỷ lệ thực tế nằm trong khoảng trên tập test (tổng thể, theo nhóm bán, theo tháng).
  Dùng cùng dữ liệu trung thực (ngưỡng train-only) để không mang rò rỉ sang phần 2.

Đầu ra: ./do_tin_cay_out/ro_ri_nguong_cat.csv, do_bao_phu.csv
Cần compare_outliers_2y.py và compare_features_2y.py trong cùng thư mục.
"""
import math
import os
import warnings

import numpy as np
import pandas as pd

import compare_features_2y as CF
import compare_outliers_2y as CO
from y2_config import path

warnings.filterwarnings("ignore")

OUT = "./do_tin_cay_out"
FEATS = CF.FEATS_GOC
MUC = [0.80, 0.95]


def nguong(tx):
    return tx.groupby("sku")["qty"].quantile(0.95)


def loc(tx, cap):
    return tx[tx["qty"] <= tx["sku"].map(cap)]


def xu_ly(tx, chia, weeks_all):
    """Dựng bảng tuần từ giao dịch, huấn luyện trên train, dự báo test"""
    wk = CO.tuan_tu(tx)
    t = np.asarray(CO.la_test(chia, wk["week"], weeks_all))
    wtr = wk[~t].reset_index(drop=True)
    wte = wk[t].reset_index(drop=True)
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st)
    tst = CO.dac_trung(wte, st).sort_values(["sku", "week"]).reset_index(drop=True)
    m = CF.mo_hinh(trn, FEATS)
    tst["pred"] = np.exp(m.predict(tst[FEATS].values.astype(np.float32)))
    tb = wtr.groupby("sku")["qty"].mean()
    return tst, tb


def danh_gia(y, p, tb):
    r = CO.chi_so(np.asarray(y, float), np.asarray(p, float), np.asarray(tb, float))
    return {k: r[k] for k in ("mae", "rmse", "r2_qty", "bias_pct", "mae_chuan", "rmse_chuan")} | {"so_dong": len(y)}


# ----------------------------------------------------------------------------- phần 1
def phan_1(df, test_tx, weeks_all, chia, cap_full):
    tr_tx = df[~test_tx]
    cap_tr = nguong(tr_tx).reindex(cap_full.index).fillna(cap_full)
    tx_full = loc(df, cap_full)
    tx_tr = loc(df, cap_tr)

    # tỷ lệ giao dịch test bị loại khác nhau giữa hai ngưỡng
    te = df[test_tx]
    khac = ((te["qty"] <= te["sku"].map(cap_full)) != (te["qty"] <= te["sku"].map(cap_tr))).mean() * 100
    tom = {"cach_chia": chia, "nguong_tb_toan_bo": float(cap_full.mean()),
           "nguong_tb_chi_train": float(cap_tr.mean()), "ty_le_gd_test_khac_nguong_pct": float(khac)}

    tst_full, tb_full = xu_ly(tx_full, chia, weeks_all)
    tst_tr, tb_tr = xu_ly(tx_tr, chia, weeks_all)
    rows = []
    r = danh_gia(tst_full["qty"], tst_full["pred"], tb_full.reindex(tst_full["sku"]).fillna(1.0).values)
    rows.append({"cach_chia": chia, "phien_ban": "bao_cao_hien_tai", **r})
    r = danh_gia(tst_tr["qty"], tst_tr["pred"], tb_tr.reindex(tst_tr["sku"]).fillna(1.0).values)
    rows.append({"cach_chia": chia, "phien_ban": "trung_thuc", **r})
    ghep = tst_full[["sku", "week", "pred"]].merge(tst_tr[["sku", "week", "qty"]], on=["sku", "week"])
    r = danh_gia(ghep["qty"], ghep["pred"], tb_tr.reindex(ghep["sku"]).fillna(1.0).values)
    rows.append({"cach_chia": chia, "phien_ban": "bao_cao_hien_tai_cham_tren_trung_thuc", **r})
    return rows, tom


# ----------------------------------------------------------------------------- phần 2
def phan_2(df, test_tx, weeks_all, chia, cap_full):
    cap_tr = nguong(df[~test_tx]).reindex(cap_full.index).fillna(cap_full)
    wk = CO.tuan_tu(loc(df, cap_tr))
    t = np.asarray(CO.la_test(chia, wk["week"], weeks_all))
    wtr = wk[~t]
    wte = wk[t]
    cac_tuan = np.sort(wtr["week"].unique())
    cat = int(len(cac_tuan) * 0.8)
    wfit = wtr[wtr["week"].isin(cac_tuan[:cat])]
    wcal = wtr[wtr["week"].isin(cac_tuan[cat:])]

    st = CO.thong_ke(wfit)
    fit = CO.dac_trung(wfit, st)
    cal = CO.dac_trung(wcal, st)
    tst = CO.dac_trung(wte, st).sort_values(["sku", "week"]).reset_index(drop=True)
    m = CF.mo_hinh(fit, FEATS)
    phan_du = np.abs(np.log(cal["qty"].values.astype(float)) - m.predict(cal[FEATS].values.astype(np.float32)))
    pred_ln = m.predict(tst[FEATS].values.astype(np.float32))
    lq = np.log(tst["qty"].values.astype(float))

    tb = wtr.groupby("sku")["qty"].mean()
    nhom_ban = pd.qcut(tb, 4, labels=False, duplicates="drop")
    nhom = tst["sku"].map(nhom_ban).fillna(0).astype(int).values
    thang = tst["week"].dt.month.values
    n = len(phan_du)

    rows = []
    for muc in MUC:
        k = min(n, math.ceil((n + 1) * muc))
        q = float(np.sort(phan_du)[k - 1])          # phân vị conformal
        trong = np.abs(lq - pred_ln) <= q
        base = {"cach_chia": chia, "muc_tin_cay_mong_muon_pct": muc * 100, "q_ln": q,
                "khoang_nhan_quanh_du_bao": float(np.exp(q)), "so_dong_cal": n}
        rows.append({**base, "nhom": "tong_the", "so_dong_test": len(lq),
                     "ty_le_bao_phu_pct": float(trong.mean() * 100)})
        for g in sorted(set(nhom)):
            s = nhom == g
            rows.append({**base, "nhom": f"nhom_ban_{g + 1}", "so_dong_test": int(s.sum()),
                         "ty_le_bao_phu_pct": float(trong[s].mean() * 100)})
        for th in sorted(set(thang)):
            s = thang == th
            rows.append({**base, "nhom": f"thang_{th:02d}", "so_dong_test": int(s.sum()),
                         "ty_le_bao_phu_pct": float(trong[s].mean() * 100)})
    return rows


def main():
    print("=" * 78)
    print("KIỂM TRA ĐỘ TIN CẬY: rò rỉ ngưỡng cắt và độ bao phủ khoảng dự báo")
    print("=" * 78)
    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    cap_full = nguong(df)
    wk_full = CO.tuan_tu(loc(df, cap_full))
    wk_full = wk_full[wk_full["sku"].isin(skus)]
    weeks_all = pd.DatetimeIndex(sorted(wk_full["week"].unique()))
    os.makedirs(OUT, exist_ok=True)

    ro_ri, tom_tat, bao_phu = [], [], []
    for chia in CO.CHIA:
        print(f"\n--- {CO.TEN_CHIA[chia]}")
        test_tx = np.asarray(CO.la_test(chia, df["week"], weeks_all))
        r1, tt = phan_1(df, test_tx, weeks_all, chia, cap_full)
        ro_ri += r1
        tom_tat.append(tt)
        bao_phu += phan_2(df, test_tx, weeks_all, chia, cap_full)

    print("\n[1] RÒ RỈ NGƯỠNG CẮT 5% (MAE theo số lượng; chênh = trung thực − báo cáo hiện tại)")
    rr = pd.DataFrame(ro_ri)
    print(rr[["cach_chia", "phien_ban", "so_dong", "mae", "rmse", "r2_qty", "bias_pct"]].round(3).to_string(index=False))
    for chia in CO.CHIA:
        a = rr[(rr.cach_chia == chia) & (rr.phien_ban == "bao_cao_hien_tai_cham_tren_trung_thuc")]
        b = rr[(rr.cach_chia == chia) & (rr.phien_ban == "trung_thuc")]
        if len(a) and len(b):
            print(f"  {chia}: mô hình báo cáo, chấm trên target trung thực: MAE {a.iloc[0]['mae']:.3f} | "
                  f"mô hình trung thực: MAE {b.iloc[0]['mae']:.3f}")
    print("  Lưu ý: hai dòng cuối dùng cùng dòng test để so; dòng 'trung_thuc' có thể khác số dòng do thống kê sản phẩm.")

    print("\n[2] ĐỘ BAO PHỦ KHOẢNG DỰ BÁO (tỷ lệ thực tế nằm trong khoảng, trên tập test)")
    bp = pd.DataFrame(bao_phu)
    tong = bp[bp["nhom"] == "tong_the"]
    print(tong[["cach_chia", "muc_tin_cay_mong_muon_pct", "q_ln", "khoang_nhan_quanh_du_bao",
                "so_dong_test", "ty_le_bao_phu_pct"]].round(2).to_string(index=False))
    for _, r in tong.iterrows():
        lech = r["ty_le_bao_phu_pct"] - r["muc_tin_cay_mong_muon_pct"]
        if abs(lech) > 2:
            print(f"  CẢNH BÁO {r['cach_chia']} mức {r['muc_tin_cay_mong_muon_pct']:.0f}%: lệch {lech:+.1f} điểm")

    rr.to_csv(os.path.join(OUT, "ro_ri_nguong_cat.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(tom_tat).to_csv(os.path.join(OUT, "tom_tat_nguong.csv"), index=False, encoding="utf-8-sig")
    bp.to_csv(os.path.join(OUT, "do_bao_phu.csv"), index=False, encoding="utf-8-sig")
    print(f"\nĐã ghi {OUT}/ (ro_ri_nguong_cat.csv, tom_tat_nguong.csv, do_bao_phu.csv)")


if __name__ == "__main__":
    main()
