"""
Chẩn đoán đặc trưng lịch: cải thiện đến từ đâu?
==============================================
Ba câu hỏi:
  1. Vì sao số ngày lễ đếm được (12) thấp hơn số ngày lễ trong khoảng dữ liệu (16)?
     -> liệt kê các tuần KHÔNG có giao dịch nào, ngày lễ rơi vào tuần nào, tuần đó có bán hay không.
  2. Cải thiện có chỉ đến từ nhóm tuần Giáng sinh không?
     -> so 4 cấu hình: goc, chỉ cờ Giáng sinh, chỉ các cờ lịch khác, tất cả. Đo MAE riêng nhóm tuần Giáng sinh
        và nhóm còn lại, kèm bootstrap so với goc.
  3. Cải thiện có tập trung ở vài tuần không?
     -> đóng góp của từng tuần test vào cải thiện (goc - tất cả).

Không sửa pipeline. Cần compare_outliers_2y.py và compare_features_2y.py trong cùng thư mục.
Đầu ra: ./compare_features_diag_out/
"""
import os
import warnings

import numpy as np
import pandas as pd

import compare_features_2y as CF
import compare_outliers_2y as CO
from y2_config import path

warnings.filterwarnings("ignore")

OUT = "./compare_features_diag_out"
NHOM_NOEL = ["tuan_giang_sinh", "tuan_truoc_giang_sinh", "tuan_sau_giang_sinh"]
NHOM_KHAC = ["tuan_le_phuc_sinh", "so_ngay_le", "so_ngay_lam_viec"]
CAU_HINH = {
    "goc": [],
    "chi_giang_sinh": NHOM_NOEL,
    "chi_lich_khac": NHOM_KHAC,
    "tat_ca": NHOM_NOEL + NHOM_KHAC,
}


def xay_du_lieu(wk, weeks_all, lich, chia):
    test = CO.la_test(chia, wk["week"], weeks_all)
    wtr = wk[~test].reset_index(drop=True)
    wte = wk[test].reset_index(drop=True)
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st).merge(lich, on="week", how="left")
    tst = CO.dac_trung(wte, st).merge(lich, on="week", how="left")
    tst = tst.sort_values(["sku", "week"]).reset_index(drop=True)
    y = tst["qty"].values.astype(float)
    idx, uniq = pd.factorize(tst["sku"])
    return trn, tst, y, idx, len(uniq)


# ----------------------------------------------------------------------------- 1. tuần thiếu và ngày lễ
def bao_cao_tuan(df, wk, lich_all):
    print("\n[1] TUẦN THIẾU DỮ LIỆU VÀ NGÀY LỄ")
    tat_ca = pd.date_range(wk["week"].min(), wk["week"].max(), freq="7D")
    co = set(wk["week"].unique())
    dem = df.groupby("week").size()
    thieu = [w for w in tat_ca if w not in co]
    print(f"  Tuần trong khoảng dữ liệu: {len(tat_ca)} | tuần có giao dịch: {len(co)} | "
          f"tuần thiếu: {len(thieu)}")

    dong = []
    for w in thieu:
        ngay = pd.date_range(w, periods=7, freq="D")
        le = [d.strftime("%Y-%m-%d") for d in ngay if d in CF.NGAY_LE_ANH]
        flags = CF.lich_tuan([w]).iloc[0]
        dong.append({"tuan_bat_dau": w.date(), "ngay_le_trong_tuan": ";".join(le),
                     "so_giao_dich_tuan_truoc": int(dem.get(w - pd.Timedelta(days=7), 0)),
                     "so_giao_dich_tuan_sau": int(dem.get(w + pd.Timedelta(days=7), 0)),
                     "tuan_giang_sinh": int(flags["tuan_giang_sinh"]),
                     "tuan_truoc_giang_sinh": int(flags["tuan_truoc_giang_sinh"]),
                     "tuan_sau_giang_sinh": int(flags["tuan_sau_giang_sinh"])})
    tuan_thieu = pd.DataFrame(dong)
    if len(tuan_thieu):
        print(tuan_thieu.to_string(index=False))

    # ngày lễ nằm trong khoảng dữ liệu: rơi vào tuần có bán hay tuần thiếu?
    d0, d1 = wk["week"].min(), wk["week"].max() + pd.Timedelta(days=6)
    le_trong_data = CF.NGAY_LE_ANH[(CF.NGAY_LE_ANH >= d0) & (CF.NGAY_LE_ANH <= d1)]
    dong_le = []
    for d in le_trong_data:
        w = d - pd.Timedelta(days=d.weekday())
        dong_le.append({"ngay_le": d.date(), "tuan_bat_dau": w.date(),
                        "tuan_co_giao_dich": bool(w in co),
                        "so_giao_dich_tuan": int(dem.get(w, 0))})
    le_df = pd.DataFrame(dong_le)
    print(f"\n  Ngày lễ trong khoảng dữ liệu: {len(le_df)} | rơi vào tuần thiếu: "
          f"{int((~le_df['tuan_co_giao_dich']).sum())}")
    print(le_df.to_string(index=False))
    return tuan_thieu, le_df


# ----------------------------------------------------------------------------- 2. phân rã cải thiện
def phan_ra(wk, weeks_all, lich, chia):
    trn, tst, y, idx, n_sku = xay_du_lieu(wk, weeks_all, lich, chia)
    pred, err = {}, {}
    for ten, cols in CAU_HINH.items():
        feats = CF.FEATS_GOC + cols
        m = CF.mo_hinh(trn, feats)
        pred[ten] = np.exp(m.predict(tst[feats].values.astype(np.float32)))
        err[ten] = np.abs(y - pred[ten])

    noel = (tst[NHOM_NOEL].sum(axis=1) > 0).values
    rows = []
    for ten in CAU_HINH:
        r = {"cach_chia": chia, "cau_hinh": ten,
             "mae_tong": float(err[ten].mean()),
             "mae_tuan_giang_sinh": float(err[ten][noel].mean()) if noel.any() else np.nan,
             "mae_tuan_khac": float(err[ten][~noel].mean()),
             "so_dong_tuan_giang_sinh": int(noel.sum()), "so_dong_tuan_khac": int((~noel).sum())}
        if ten != "goc":
            d, lo, hi, share = CO.bootstrap_so_a(err["goc"], err[ten], idx, n_sku)
            r.update({"chenh_mae_tong": d, "khoang_95_lo": lo, "khoang_95_hi": hi, "ty_le_tot_hon_goc": share})
            if noel.any():
                dn, lon, hin, _ = CO.bootstrap_so_a(err["goc"][noel], err[ten][noel],
                                                   idx[noel], int(len(np.unique(idx[noel]))))
                r.update({"chenh_mae_tuan_giang_sinh": dn, "chenh_mae_tuan_khac": float(
                    np.mean(err[ten][~noel]) - np.mean(err["goc"][~noel]))})
        rows.append(r)

    # đóng góp theo tuần: dương = cấu hình "tat_ca" tốt hơn goc ở tuần đó
    tuan = pd.DataFrame({"week": tst["week"].values, "cai_thien": err["goc"] - err["tat_ca"]})
    tg = tuan.groupby("week").agg(tong_cai_thien=("cai_thien", "sum"), so_dong=("cai_thien", "size")).reset_index()
    tg = tg.merge(lich, on="week", how="left").sort_values("tong_cai_thien", ascending=False)
    tg.insert(0, "cach_chia", chia)
    return pd.DataFrame(rows), tg


def tom_tat(rows_all, tg_all):
    print("\n[2] PHÂN RÃ CẢI THIỆN (MAE; chênh âm = tốt hơn goc)")
    cols = ["cach_chia", "cau_hinh", "mae_tong", "mae_tuan_giang_sinh", "mae_tuan_khac",
            "chenh_mae_tong", "khoang_95_lo", "khoang_95_hi", "chenh_mae_tuan_khac"]
    print(rows_all[[c for c in cols if c in rows_all.columns]].round(3).to_string(index=False))

    print("\n[3] CẢI THIỆN TẬP TRUNG Ở ĐÂU?")
    for chia, g in tg_all.groupby("cach_chia"):
        tong = g["tong_cai_thien"].sum()
        pos = g[g["tong_cai_thien"] > 0]["tong_cai_thien"].sum()
        noel = g[(g["tuan_giang_sinh"] + g["tuan_truoc_giang_sinh"] + g["tuan_sau_giang_sinh"]) > 0]
        top3 = g.head(3)["tong_cai_thien"].sum()
        print(f"  {CO.TEN_CHIA[chia]}:")
        print(f"    tổng cải thiện (dương − âm): {tong:+.1f} | tổng phần dương: {pos:.1f}")
        print(f"    tuần Giáng sinh (gồm tuần trước/sau) đóng góp: {noel['tong_cai_thien'].sum():+.1f}")
        print(f"    3 tuần cải thiện nhất đóng góp: {top3:+.1f} "
              f"({top3 / pos * 100 if pos else np.nan:.0f}% phần dương)")
        print("    top 5 tuần:")
        print(g.head(5)[["week", "tong_cai_thien", "so_dong", "tuan_giang_sinh",
                         "tuan_truoc_giang_sinh", "tuan_sau_giang_sinh"]].to_string(index=False))


def main():
    print("=" * 78)
    print("CHẨN ĐOÁN ĐẶC TRƯNG LỊCH")
    print("=" * 78)
    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    wk = CO.tuan_tu(CO.giao_dich_phien_ban(df, True))
    wk = wk[wk["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(wk["week"].unique()))
    lich = CF.lich_tuan(weeks_all)
    os.makedirs(OUT, exist_ok=True)

    tuan_thieu, le_df = bao_cao_tuan(df, wk, lich)
    tuan_thieu.to_csv(os.path.join(OUT, "tuan_thieu.csv"), index=False, encoding="utf-8-sig")
    le_df.to_csv(os.path.join(OUT, "ngay_le_trong_du_lieu.csv"), index=False, encoding="utf-8-sig")

    rows_all, tg_all = [], []
    for chia in CO.CHIA:
        r, tg = phan_ra(wk, weeks_all, lich, chia)
        rows_all.append(r)
        tg_all.append(tg)
    rows_all = pd.concat(rows_all, ignore_index=True)
    tg_all = pd.concat(tg_all, ignore_index=True)
    tom_tat(rows_all, tg_all)

    rows_all.to_csv(os.path.join(OUT, "phan_ra_cai_thien.csv"), index=False, encoding="utf-8-sig")
    tg_all.to_csv(os.path.join(OUT, "dong_gop_theo_tuan.csv"), index=False, encoding="utf-8-sig")
    print(f"\nĐã ghi {OUT}/ (tuan_thieu.csv, ngay_le_trong_du_lieu.csv, phan_ra_cai_thien.csv, dong_gop_theo_tuan.csv)")


if __name__ == "__main__":
    main()
