"""
Thử đặc trưng LỊCH để giảm sai số XGBoost (so với 8 đặc trưng hiện tại)
=======================================================================
Phiên bản so sánh (cùng dữ liệu, cùng cách chia, cùng tham số XGBoost, cùng tập test):
  goc  : 8 đặc trưng hiện tại (đúng như pipeline)
  lich : 8 đặc trưng hiện tại + 6 đặc trưng lịch

Đặc trưng lịch (tính từ ngày thật, không ước lượng từ dữ liệu bán):
  tuan_giang_sinh        : tuần có ngày 25/12
  tuan_truoc_giang_sinh  : tuần ngay trước tuần Giáng sinh
  tuan_sau_giang_sinh    : tuần ngay sau tuần Giáng sinh
  tuan_le_phuc_sinh      : tuần có Thứ Sáu Tuần Thánh hoặc Thứ Hai Phục Sinh
  so_ngay_le             : số ngày nghỉ lễ ngân hàng Anh trong tuần (kể cả rơi cuối tuần)
  so_ngay_lam_viec       : 5 trừ số ngày lễ rơi vào Thứ Hai - Thứ Sáu

Nguồn ngày lễ: danh sách ngày nghỉ lễ ngân hàng Anh (England & Wales) 2009-2011 theo gov.uk.
Kiểm tra lại trên https://www.gov.uk/bank-holidays trước khi báo cáo.

Kiểm định: giữ nguyên khung của compare_outliers_2y.py (ba cách chia, chấm bằng target của A,
bootstrap theo sản phẩm). Quy tắc chấp nhận: MAE giảm ở CẢ HAI chiều năm (khoảng 95% < 0)
và |thiên lệch tổng| không tăng quá 2 điểm.

Đầu ra: ./compare_features_out/ket_qua_features.csv, lich_tuan.csv
Không ghi gì vào models_2y/ và không sửa pipeline. Cần compare_outliers_2y.py trong cùng thư mục.
"""
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

import compare_outliers_2y as CO
from y2_config import path

warnings.filterwarnings("ignore")

OUT = "./compare_features_out"
FEATS_GOC = CO.FEATS
LICH = ["tuan_giang_sinh", "tuan_truoc_giang_sinh", "tuan_sau_giang_sinh",
        "tuan_le_phuc_sinh", "so_ngay_le", "so_ngay_lam_viec"]
FEATS_MO = FEATS_GOC + LICH

# Ngày nghỉ lễ ngân hàng Anh (England & Wales), 2009-2011, theo gov.uk
NGAY_LE_ANH = pd.DatetimeIndex(pd.to_datetime([
    "2009-01-01", "2009-04-10", "2009-04-13", "2009-05-04", "2009-05-25", "2009-08-31", "2009-12-25", "2009-12-28",
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
    "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-08-29", "2011-12-26", "2011-12-27",
]))
PHUC_SINH = pd.DatetimeIndex(pd.to_datetime([
    "2009-04-10", "2009-04-13", "2010-04-02", "2010-04-05", "2011-04-22", "2011-04-25"]))
GIANG_SINH = pd.DatetimeIndex(pd.to_datetime(["2009-12-25", "2010-12-25", "2011-12-25", "2012-12-25"]))


def lich_tuan(weeks):
    """Bảng đặc trưng lịch cho từng tuần (tuần bắt đầu Thứ Hai, khớp với cột week của dữ liệu)"""
    rows = []
    for w in pd.DatetimeIndex(weeks):
        ngay = pd.date_range(w, periods=7, freq="D")
        ngay_le = int(ngay.isin(NGAY_LE_ANH).sum())
        ngay_le_thuong = int(((ngay.weekday < 5) & ngay.isin(NGAY_LE_ANH)).sum())
        rows.append({
            "week": w,
            "tuan_giang_sinh": int(ngay.isin(GIANG_SINH).any()),
            "tuan_truoc_giang_sinh": int((ngay + pd.Timedelta(days=7)).isin(GIANG_SINH).any()),
            "tuan_sau_giang_sinh": int((ngay - pd.Timedelta(days=7)).isin(GIANG_SINH).any()),
            "tuan_le_phuc_sinh": int(ngay.isin(PHUC_SINH).any()),
            "so_ngay_le": ngay_le,
            "so_ngay_lam_viec": 5 - ngay_le_thuong,
        })
    return pd.DataFrame(rows)


def mo_hinh(trn, feats):
    mono = tuple(-1 if f == "log_price_ratio" else 0 for f in feats)
    m = xgb.XGBRegressor(**CO.PARAMS, monotone_constraints=mono)
    m.fit(trn[feats].values.astype(np.float32), np.log(trn["qty"].values.astype(float)))
    return m


def dap_ung(model, X, feats, step=0.10):
    i = feats.index("log_price_ratio")
    up, dn = X.copy(), X.copy()
    up[:, i] += np.log(1 + step)
    dn[:, i] += np.log(1 - step)
    pu, pd_ = model.predict(up), model.predict(dn)
    return (pu - pd_) / (np.log(1 + step) - np.log(1 - step)), np.isclose(pu, pd_)


def chay_chia(wk, weeks_all, lich, chia):
    test = CO.la_test(chia, wk["week"], weeks_all)
    wtr = wk[~test].reset_index(drop=True)
    wte = wk[test].reset_index(drop=True)
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st).merge(lich, on="week", how="left")
    tst = CO.dac_trung(wte, st).merge(lich, on="week", how="left")
    tst = tst.sort_values(["sku", "week"]).reset_index(drop=True)
    tb = wtr.groupby("sku")["qty"].mean().reindex(tst["sku"].values).fillna(1.0).values
    y = tst["qty"].values.astype(float)
    idx, uniq = pd.factorize(tst["sku"])

    ket = {}
    for nhan, feats in (("goc", FEATS_GOC), ("lich", FEATS_MO)):
        m = mo_hinh(trn, feats)
        X = tst[feats].values.astype(np.float32)
        p = np.exp(m.predict(X))
        eps, flat = dap_ung(m, X, feats)
        ket[nhan] = {"p": p, "eps": eps, "flat": flat, "model": m, "feats": feats,
                     "chi": CO.chi_so(y, p, tb)}

    err_g = np.abs(y - ket["goc"]["p"])
    err_l = np.abs(y - ket["lich"]["p"])
    d, lo, hi, share = CO.bootstrap_so_a(err_g, err_l, idx, len(uniq))
    rows = []
    for nhan in ("goc", "lich"):
        k = ket[nhan]
        r = {"cach_chia": chia, "bo_dac_trung": nhan, "so_dac_trung": len(k["feats"]),
             "so_san_pham": int(len(uniq)), "so_dong_test": int(len(y))}
        r.update(k["chi"])
        r.update({"eps_trung_binh": float(np.mean(k["eps"])), "ty_le_phang_pct": float(np.mean(k["flat"]) * 100)})
        if nhan == "goc":
            r.update({"chenh_mae_so_voi_goc": np.nan, "khoang_95_lo": np.nan, "khoang_95_hi": np.nan,
                      "ty_le_tot_hon_goc": np.nan})
        else:
            r.update({"chenh_mae_so_voi_goc": d, "khoang_95_lo": lo, "khoang_95_hi": hi, "ty_le_tot_hon_goc": share})
        rows.append(r)
    imp = pd.Series(ket["lich"]["model"].feature_importances_, index=FEATS_MO).loc[LICH]
    return rows, imp


def main():
    print("=" * 78)
    print("THỬ ĐẶC TRƯNG LỊCH: 8 đặc trưng hiện tại vs 8 + 6 đặc trưng lịch")
    print("=" * 78)
    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    wk = CO.tuan_tu(CO.giao_dich_phien_ban(df, True))
    wk = wk[wk["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(wk["week"].unique()))
    lich = lich_tuan(weeks_all)
    print(f"[2] {len(weeks_all)} tuần | tuần có Giáng sinh: {int(lich['tuan_giang_sinh'].sum())} | "
          f"tuần có Phục sinh: {int(lich['tuan_le_phuc_sinh'].sum())} | "
          f"tổng ngày lễ: {int(lich['so_ngay_le'].sum())}")
    os.makedirs(OUT, exist_ok=True)
    lich.to_csv(os.path.join(OUT, "lich_tuan.csv"), index=False, encoding="utf-8-sig")

    tong, imps = [], {}
    for chia in CO.CHIA:
        print(f"\n--- {CO.TEN_CHIA[chia]}")
        rows, imp = chay_chia(wk, weeks_all, lich, chia)
        tong += rows
        imps[chia] = imp
        g, l = rows[0], rows[1]
        print(f"    goc : MAE {g['mae']:.3f} | lệch tổng {g['bias_pct']:+.1f}%")
        print(f"    lich: MAE {l['mae']:.3f} | lệch tổng {l['bias_pct']:+.1f}% | "
              f"chênh MAE {l['chenh_mae_so_voi_goc']:+.3f} [{l['khoang_95_lo']:+.3f}; {l['khoang_95_hi']:+.3f}]")

    df_t = pd.DataFrame(tong)
    df_t.to_csv(os.path.join(OUT, "ket_qua_features.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("ĐỐI CHIẾU TIÊU CHÍ (MAE giảm ở CẢ HAI chiều năm, |lệch tổng| không tăng quá 2 điểm)")
    print("=" * 78)
    dat = True
    for chia in ("chieu_1_2", "chieu_2_1"):
        l = df_t[(df_t["cach_chia"] == chia) & (df_t["bo_dac_trung"] == "lich")].iloc[0]
        g = df_t[(df_t["cach_chia"] == chia) & (df_t["bo_dac_trung"] == "goc")].iloc[0]
        ok = (l["khoang_95_hi"] < 0) and (abs(l["bias_pct"]) <= abs(g["bias_pct"]) + 2.0)
        dat = dat and ok
        print(f"  {chia}: chênh MAE {l['chenh_mae_so_voi_goc']:+.3f} | lệch tổng {g['bias_pct']:+.1f}% -> "
              f"{l['bias_pct']:+.1f}% | {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    l = df_t[(df_t["cach_chia"] == "xen_ke") & (df_t["bo_dac_trung"] == "lich")].iloc[0]
    print(f"  xen_ke (tham khảo): chênh MAE {l['chenh_mae_so_voi_goc']:+.3f} "
          f"[{l['khoang_95_lo']:+.3f}; {l['khoang_95_hi']:+.3f}]")
    print(f"\n  KẾT LUẬN: {'CHẤP NHẬN đặc trưng lịch' if dat else 'KHÔNG chấp nhận: lịch không cải thiện đủ rõ'}")

    print("\nĐộ quan trọng của đặc trưng lịch (mô hình có lịch):")
    for chia, imp in imps.items():
        print(f"  {chia}: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.items()))
    print(f"\nĐã ghi {OUT}/ket_qua_features.csv và lich_tuan.csv")


if __name__ == "__main__":
    main()
