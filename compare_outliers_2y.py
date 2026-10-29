"""
So sánh các cách lọc ngoại lai: 8 phiên bản x 3 cách chia
=========================================================
Phiên bản (cắt đơn lớn = cắt 5% số lượng lớn nhất của từng sản phẩm ở mức giao dịch):
  A        : pipeline hiện tại (có cắt đơn lớn, gộp tuần)              <- đối chứng
  A0       : như A nhưng KHÔNG cắt đơn lớn
  A_B      : A + lọc NGÀY (IQR 3x, chỉ tính trên ngày có bán)
  A_Bp     : A + lọc NGÀY (IQR 3x, tính trên TẤT CẢ ngày, kể cả ngày không bán)
  A_C      : A + lọc TUẦN (IQR 3x trên lượng bán đã chuẩn hóa theo trung bình train của sản phẩm)
  A0_B, A0_Bp, A0_C : như A_B, A_Bp, A_C nhưng không cắt đơn lớn

Cách chia:
  xen_ke    : mỗi tuần thứ 5 là test (như pipeline hiện tại)
  chieu_1_2 : train năm 1, test năm 2 (kiểm định ngoài năm)
  chieu_2_1 : train năm 2, test năm 1

Quy tắc cố định (để các phiên bản so được với nhau):
  - Danh sách sản phẩm: 1306 sản phẩm của models_2y/products_online_retail.csv (phiên bản A), không đổi.
  - Dòng test: lấy từ phiên bản A. Mọi phiên bản dự báo trên cùng dòng này.
  - Chấm điểm theo 2 cách: (1) target đã cắt đơn lớn của A (cách cũ); (2) target KHÔNG cắt (qty0) của cùng dòng.
    Cách (2) là câu hỏi chính về bước cắt 5%: dự báo của A0 có tốt hơn A khi chấm bằng số lượng thật chưa cắt không.
  - Lọc chỉ xóa dòng TRAIN. Ngưỡng IQR tính trên train. Tập test không tham gia tính ngưỡng.
  - Chuẩn hóa để lọc (phiên bản C và ngưỡng ngày): chia cho trung bình train của sản phẩm, tính TRƯỚC khi lọc.
  - Chuẩn hóa để đánh giá (MAE/RMSE chuẩn hóa): chia cho trung bình train của phiên bản A, để mọi phiên bản dùng cùng thang.
  - Thống kê sản phẩm và 8 đặc trưng tính lại trên train SAU khi lọc, riêng từng phiên bản x cách chia.
  - Chỉ đánh giá các sản phẩm đủ điều kiện ở TẤT CẢ phiên bản (giao các tập), để so sánh công bằng.

Đầu vào: data/online_retail_II_sheets.csv (cache bước 1), models_2y/ (để kiểm tra đối chiếu).
Đầu ra : ./compare_outliers_out/<phiên bản>/<cách chia>/ (product_stats.csv, .json, summary_loc.csv, summary_theo_nhom.csv)
         ./compare_outliers_out/ket_qua_tong_hop.csv
Không ghi gì vào models_2y/ và không sửa các file pipeline.
"""
import json
import os
import traceback
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from y2_config import MIN_TRAIN_WEEKS, OOT_CUT_WEEK, RAW_CACHE, TEST_EVERY, path

warnings.filterwarnings("ignore")

OUT = "./compare_outliers_out"
FEATS = ["log_price_ratio", "sku_lp_med", "sku_base", "month", "woy",
         "sku_lq_std", "sku_lp_std", "sku_q4_effect"]
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
N_BOOT = 300
MIN_TRAIN_ROWS = 500          # dưới ngưỡng này coi như không đủ dữ liệu train để huấn luyện
PHIEN_BAN = [
    ("A", True, None), ("A0", False, None),
    ("A_B", True, "ngay_ban"), ("A_Bp", True, "ngay_tat_ca"), ("A_C", True, "tuan"),
    ("A0_B", False, "ngay_ban"), ("A0_Bp", False, "ngay_tat_ca"), ("A0_C", False, "tuan"),
]
CHIA = ["xen_ke", "chieu_1_2", "chieu_2_1"]
TEN_CHIA = {"xen_ke": "xen kẽ (mỗi tuần thứ 5)", "chieu_1_2": "train năm 1 -> test năm 2",
            "chieu_2_1": "train năm 2 -> test năm 1"}


# ----------------------------------------------------------------------------- dữ liệu
def doc_giao_dich(skus):
    print("[1] Đọc và làm sạch giao dịch (giống bước 1 của pipeline)...")
    df = pd.read_csv(RAW_CACHE, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    if "sheet" in df.columns and (df["sheet"] == 1).any():
        s1, s2 = df[df["sheet"] == 0], df[df["sheet"] == 1]
        cut = s2["InvoiceDate"].min().normalize()
        df = pd.concat([s1[s1["InvoiceDate"] < cut], s2], ignore_index=True)
    df = df.rename(columns={"StockCode": "sku", "Quantity": "qty", "Price": "price", "InvoiceDate": "date"})
    df = df[~df["Invoice"].astype(str).str.upper().str.startswith("C")]
    df = df[(df["qty"] > 0) & (df["price"] > 0)].copy()
    df["sku"] = df["sku"].astype(str)
    df = df[df["sku"].str.match(r"^\d{5}")].copy()
    df = df[df["sku"].isin(skus)].copy()
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["day"] = df["date"].dt.normalize()
    print(f"    {len(df):,} dòng giao dịch sạch của {df['sku'].nunique()} sản phẩm")
    return df[["sku", "date", "day", "week", "qty", "price"]].reset_index(drop=True)


def giao_dich_phien_ban(df, cap_on):
    """Cắt đơn lớn: giữ giao dịch có qty <= phân vị 95% của sản phẩm (đúng như bước 1 của pipeline)"""
    if not cap_on:
        return df
    cap = df.groupby("sku")["qty"].transform(lambda s: s.quantile(0.95))
    return df[df["qty"] <= cap]


def tuan_tu(tx):
    if len(tx) == 0:
        return pd.DataFrame(columns=["sku", "week", "qty", "price"])
    return tx.groupby(["sku", "week"]).agg(qty=("qty", "sum"), price=("price", "median")).reset_index()


def la_test(chia, weeks, weeks_all):
    """Trả về mảng bool: dòng có tuần nằm trong tập test của cách chia"""
    weeks = pd.Series(weeks).reset_index(drop=True)
    if chia == "xen_ke":
        return weeks.isin(pd.DatetimeIndex(weeks_all[::TEST_EVERY])).values
    cut = pd.Timestamp(OOT_CUT_WEEK)
    if chia == "chieu_1_2":
        return (weeks >= cut).values
    return (weeks < cut).values


def bo_khoa(tx, bo, cols):
    """Bỏ các dòng giao dịch có khóa (cols) nằm trong bảng bo"""
    if bo is None or len(bo) == 0 or len(tx) == 0:
        return tx
    idx = pd.MultiIndex.from_frame(tx[cols])
    bo_idx = pd.MultiIndex.from_frame(bo[cols])
    return tx[~idx.isin(bo_idx)]


# ----------------------------------------------------------------------------- lọc ngoại lai
def loc_ngay(tx_tr, kieu, d0, d1):
    """IQR 3x trên lượng bán theo (sản phẩm, ngày) của train, sau khi chuẩn hóa theo trung bình của sản phẩm.
    kieu = 'ngay_ban': chỉ tính trên ngày có bán. 'ngay_tat_ca': tính trên mọi ngày trong khoảng train (kể cả 0).
    Trả về (bảng khóa bị bỏ, số ngày có bán, số ngày bị bỏ)."""
    ban = tx_tr.groupby(["sku", "day"])["qty"].sum().unstack("day", fill_value=0).astype(float)
    if kieu == "ngay_tat_ca":
        cols = pd.date_range(d0.normalize(), d1.normalize(), freq="D")
        ban = ban.reindex(columns=cols, fill_value=0.0)
    tb = ban.mean(axis=1)
    z = ban.div(tb, axis=0)
    q1, q3 = np.percentile(z.values.ravel(), [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    mask = ((z > hi) | (z < lo)) & (ban > 0)         # chỉ ngày có bán mới có thể bị bỏ
    stk = mask.stack()
    stk = stk[stk]
    bo = pd.DataFrame(list(stk.index), columns=["sku", "day"])
    n_ban = int((ban > 0).values.sum())
    return bo, n_ban, len(bo)


def loc_tuan(wk, tb):
    """IQR 3x trên lượng bán tuần đã chuẩn hóa theo trung bình train của sản phẩm (tính trên train trước khi lọc)"""
    z = wk["qty"] / wk["sku"].map(tb)
    q1, q3 = np.percentile(z.dropna(), [25, 75])
    iqr = q3 - q1
    bad = ((z > q3 + 3 * iqr) | (z < q1 - 3 * iqr)).fillna(False).values
    bo = wk.loc[bad, ["sku", "week"]].reset_index(drop=True)
    return bo, len(bo)


# ----------------------------------------------------------------------------- thống kê, đặc trưng, mô hình
STAT_COLS = ["sku", "sku_base", "sku_lp_med", "sku_lq_std", "sku_lp_std", "sku_q4_effect", "n_train"]


def thong_ke(w):
    """Bảng thống kê sản phẩm, giống 2_data_processing_online_retail_ml_2y.py nhưng tính trên train đã lọc"""
    if len(w) == 0:
        return pd.DataFrame(columns=STAT_COLS)
    w = w.assign(lq=np.log(w["qty"]), lp=np.log(w["price"]), month=w["week"].dt.month)
    st = w.groupby("sku").agg(sku_base=("lq", "mean"), sku_lp_med=("lp", "median"),
                              sku_lq_std=("lq", "std"), sku_lp_std=("lp", "std"),
                              n_train=("lq", "size")).reset_index()
    q4 = w["month"].isin([11, 12])
    a = w[q4].groupby("sku")["lq"].mean().rename("q4m").reset_index()
    b = w[~q4].groupby("sku")["lq"].mean().rename("restm").reset_index()
    st = st.merge(a, on="sku", how="left").merge(b, on="sku", how="left")
    st["sku_q4_effect"] = (st["q4m"] - st["restm"]).fillna(0.0)
    st[["sku_lq_std", "sku_lp_std"]] = st[["sku_lq_std", "sku_lp_std"]].fillna(0.0)
    st = st[st["n_train"] >= MIN_TRAIN_WEEKS]
    return st[STAT_COLS].reset_index(drop=True)


def dac_trung(w, st):
    t = w.merge(st, on="sku", how="inner")
    t["lp"] = np.log(t["price"])
    t["log_price_ratio"] = t["lp"] - t["sku_lp_med"]
    t["month"] = t["week"].dt.month
    t["woy"] = t["week"].dt.isocalendar().week.astype(int).values
    return t


def mo_hinh_xgb(trn):
    mono = tuple(-1 if f == "log_price_ratio" else 0 for f in FEATS)
    m = xgb.XGBRegressor(**PARAMS, monotone_constraints=mono)
    m.fit(trn[FEATS].values.astype(np.float32), np.log(trn["qty"].values.astype(float)))
    return m


def dap_ung_gia(model, X, step=0.10):
    """Độ co giãn khi đổi giá ±10% (trên thang ln) và tỷ lệ dòng không phản ứng"""
    i = FEATS.index("log_price_ratio")
    up, dn = X.copy(), X.copy()
    up[:, i] += np.log(1 + step)
    dn[:, i] += np.log(1 - step)
    pu, pd_ = model.predict(up), model.predict(dn)
    eps = (pu - pd_) / (np.log(1 + step) - np.log(1 - step))
    return eps, np.isclose(pu, pd_)


# ----------------------------------------------------------------------------- một phiên bản x một cách chia
def chay_mot(df, cap_on, loc, chia, weeks_all, ref_test):
    tx = giao_dich_phien_ban(df, cap_on)
    tx = tx.assign(la_test=la_test(chia, tx["week"], weeks_all))
    wk_pre = tuan_tu(tx)
    wk_pre_tr = wk_pre[~la_test(chia, wk_pre["week"], weeks_all)].reset_index(drop=True)
    tb = wk_pre_tr.groupby("sku")["qty"].mean()             # trung bình train TRƯỚC khi lọc
    tx_tr = tx[~tx["la_test"]].copy()
    dong_truoc = len(tx_tr)

    info = {"phien_ban_loc": loc or "khong_loc", "dong_train_truoc": dong_truoc,
            "tuan_train_truoc": len(wk_pre_tr), "san_pham_train_truoc": int(wk_pre_tr["sku"].nunique())}
    if loc in ("ngay_ban", "ngay_tat_ca") and len(tx_tr):
        d0, d1 = tx_tr["week"].min(), tx_tr["week"].max() + pd.Timedelta(days=6)
        bo, n_ban, n_bo = loc_ngay(tx_tr, loc, d0, d1)
        tx_tr = bo_khoa(tx_tr, bo, ["sku", "day"])
        info.update({"so_ngay_co_ban": n_ban, "so_ngay_bi_bo": n_bo})
    elif loc == "tuan" and len(wk_pre_tr):
        bo, n_bo = loc_tuan(wk_pre_tr, tb)
        tx_tr = bo_khoa(tx_tr, bo, ["sku", "week"])
        info.update({"so_tuan_bi_bo": n_bo})

    wk_tr = tuan_tu(tx_tr)
    dong_sau = len(tx_tr)
    info.update({"dong_train_sau": dong_sau, "dong_bi_bo": dong_truoc - dong_sau,
                 "ty_le_bo_dong_pct": round((dong_truoc - dong_sau) / dong_truoc * 100, 2) if dong_truoc else np.nan,
                 "tuan_train_sau": len(wk_tr), "san_pham_train_sau": int(wk_tr["sku"].nunique()),
                 "tuan_bi_mat_het": len(set(wk_pre_tr["week"]) - set(wk_tr["week"]))})

    st = thong_ke(wk_tr)
    info["san_pham_bi_loai_do_it_tuan"] = int(len(set(wk_pre_tr["sku"]) - set(st["sku"])))

    # bảng phân nhóm: tỷ lệ dòng tuần bị bỏ theo nhóm mức bán (4 nhóm theo trung bình train trước khi lọc)
    gmap = pd.Series(pd.qcut(tb, 4, labels=False, duplicates="drop"), index=tb.index)
    pre = wk_pre_tr.assign(nhom=wk_pre_tr["sku"].map(gmap)).groupby("nhom").size()
    post = (wk_tr.assign(nhom=wk_tr["sku"].map(gmap)).groupby("nhom").size()
            if len(wk_tr) else pd.Series(dtype=float))
    nhom = pd.DataFrame({"so_dong_tuan_truoc": pre, "so_dong_tuan_sau": post}).fillna(0).reset_index()
    nhom["ty_le_bo_pct"] = np.where(nhom["so_dong_tuan_truoc"] > 0,
                                    (1 - nhom["so_dong_tuan_sau"] / nhom["so_dong_tuan_truoc"]) * 100, np.nan)

    out = {"info": info, "stats": st, "tb": tb, "nhom": nhom, "status": "ok"}
    trn = dac_trung(wk_tr, st) if len(st) else None
    tst = dac_trung(ref_test, st) if len(st) else None
    if trn is None or len(trn) < MIN_TRAIN_ROWS or tst is None or len(tst) == 0:
        out["status"] = "khong_du_du_lieu_train"
        return out

    model = mo_hinh_xgb(trn)
    Xte = tst[FEATS].values.astype(np.float32)
    pred = np.exp(model.predict(Xte))
    eps, flat = dap_ung_gia(model, Xte)
    out["test"] = pd.DataFrame({"sku": tst["sku"].values, "week": tst["week"].values,
                                "qty": tst["qty"].values.astype(float),
                                "qty0": tst["qty0"].values.astype(float), "pred": pred,
                                "eps": eps, "flat": flat})
    return out


# ----------------------------------------------------------------------------- đánh giá
def chi_so(y, p, tb):
    p = np.clip(p, 1e-6, None)
    r = {"mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
         "r2_qty": float(r2_score(y, p)), "bias_pct": float((p.sum() / y.sum() - 1) * 100)}
    r["mae_chuan"] = float(np.mean(np.abs(p / tb - y / tb)))
    r["rmse_chuan"] = float(np.sqrt(np.mean((p / tb - y / tb) ** 2)))
    return r


def bootstrap_so_a(err_a, err_v, sku_idx, n_sku):
    cnt = np.bincount(sku_idx, minlength=n_sku).astype(float)
    ea = np.bincount(sku_idx, weights=err_a, minlength=n_sku)
    ev = np.bincount(sku_idx, weights=err_v, minlength=n_sku)
    rng = np.random.default_rng(0)
    C = rng.multinomial(n_sku, np.full(n_sku, 1.0 / n_sku), size=N_BOOT).astype(float)
    d = (C @ ev) / (C @ cnt) - (C @ ea) / (C @ cnt)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(np.mean(d < 0))


CACH_CHAM = [("cat", "qty"), ("khong_cat", "qty0")]


def danh_gia(res, chia, ref_tb):
    ok = {ten: res[(ten, chia)] for ten, _, _ in PHIEN_BAN if res[(ten, chia)]["status"] == "ok"}
    rows = []
    if "A" not in ok:
        print(f"  Không có phiên bản A cho {chia}, bỏ qua đánh giá")
        return rows
    common = set.intersection(*[set(v["test"]["sku"]) for v in ok.values()])
    print(f"  Sản phẩm đủ điều kiện ở mọi phiên bản: {len(common)}")
    for cc, cot_y in CACH_CHAM:
        base = None
        for ten, _, _ in PHIEN_BAN:
            r = {"phien_ban": ten, "cach_chia": chia, "cach_cham": cc, "trang_thai": res[(ten, chia)]["status"]}
            if ten not in ok:
                rows.append(r)
                continue
            t = ok[ten]["test"]
            t = t[t["sku"].isin(common)].sort_values(["sku", "week"]).reset_index(drop=True)
            y, p = t[cot_y].values.astype(float), t["pred"].values
            tb_ref = ref_tb.reindex(t["sku"].values).fillna(1.0).values
            r.update(chi_so(y, p, tb_ref))
            r.update({"so_san_pham_danh_gia": len(common), "so_dong_test": len(t),
                      "eps_trung_binh": float(np.nanmean(t["eps"])), "ty_le_phang_pct": float(t["flat"].mean() * 100)})
            if ten == "A":
                base = t
                r.update({"chenh_mae_so_voi_A": np.nan, "khoang_95_lo": np.nan,
                          "khoang_95_hi": np.nan, "ty_le_tot_hon_A": np.nan})
            elif base is None or not (np.array_equal(base["sku"].values, t["sku"].values)
                                      and np.array_equal(base["week"].values, t["week"].values)):
                r.update({"chenh_mae_so_voi_A": np.nan, "khoang_95_lo": np.nan, "khoang_95_hi": np.nan,
                          "ty_le_tot_hon_A": np.nan, "ghi_chu": "dòng test không khớp với A"})
            else:
                idx, uniq = pd.factorize(t["sku"])
                err_a = np.abs(base[cot_y].values.astype(float) - base["pred"].values)
                d, lo, hi, share = bootstrap_so_a(err_a, np.abs(y - p), idx, len(uniq))
                r.update({"chenh_mae_so_voi_A": d, "khoang_95_lo": lo, "khoang_95_hi": hi, "ty_le_tot_hon_A": share})
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- lưu kết quả
def luu_phien_ban(ten, chia, v):
    folder = os.path.join(OUT, ten, chia)
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame([v["info"]]).to_csv(os.path.join(folder, "summary_loc.csv"), index=False, encoding="utf-8-sig")
    v["nhom"].to_csv(os.path.join(folder, "summary_theo_nhom.csv"), index=False, encoding="utf-8-sig")
    if len(v.get("stats", [])):
        v["stats"].to_csv(os.path.join(folder, "product_stats.csv"), index=False, encoding="utf-8-sig")
        d = {}
        for _, r in v["stats"].iterrows():
            d[str(r["sku"])] = {c: round(float(r[c]), 4) for c in STAT_COLS[1:]}
        with open(os.path.join(folder, "product_stats.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)


def kiem_tra_voi_pipeline(res, weekly_A, skus):
    """Đối chiếu phiên bản A với kết quả pipeline đang dùng (models_2y) để xác nhận đúng dữ liệu"""
    print("\n[Kiểm tra] Đối chiếu phiên bản A với models_2y:")
    p_w = path("weekly_online_retail.csv")
    if os.path.exists(p_w):
        wk = pd.read_csv(p_w, dtype={"sku": str})
        print(f"  Số dòng tuần: pipeline {len(wk):,} | script này {len(weekly_A[weekly_A['sku'].isin(skus)]):,}")
    p_s = path("product_stats_online_retail.json")
    v = res.get(("A", "xen_ke"))
    if os.path.exists(p_s) and v and v["status"] == "ok":
        with open(p_s) as f:
            ps = json.load(f)
        st = v["stats"].set_index("sku")
        common = [s for s in st.index if s in ps]
        diff = max(abs(st.loc[s, "sku_base"] - ps[s]["sku_base"]) for s in common) if common else np.nan
        print(f"  Thống kê sản phẩm (sku_base): {len(common)} sản phẩm chung, lệch lớn nhất = {diff:.6f} "
              f"-> {'KHỚP' if diff < 1e-3 else 'LỆCH: kiểm tra lại'}")


def main():
    print("=" * 78)
    print("SO SÁNH CÁC CÁCH LỌC NGOẠI LAI (8 phiên bản x 3 cách chia)")
    print("=" * 78)
    if not os.path.exists(RAW_CACHE):
        raise SystemExit(f"Không thấy {RAW_CACHE}. Chạy 1_data_processing_online_retail_2y.py trước.")
    p_prod = path("products_online_retail.csv")
    if not os.path.exists(p_prod):
        raise SystemExit("Không thấy models_2y/products_online_retail.csv. Chạy bước 1 trước.")
    skus = set(pd.read_csv(p_prod, dtype={"sku": str})["sku"])
    print(f"Danh sách sản phẩm cố định: {len(skus)} sản phẩm (từ phiên bản A)")

    df = doc_giao_dich(skus)
    weekly_A = tuan_tu(giao_dich_phien_ban(df, True))
    weekly_A = weekly_A[weekly_A["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(weekly_A["week"].unique()))
    weekly_0 = tuan_tu(giao_dich_phien_ban(df, False))
    weekly_0 = weekly_0[weekly_0["sku"].isin(skus)][["sku", "week", "qty"]].rename(columns={"qty": "qty0"})
    ref_test = {}
    for c in CHIA:
        rt = weekly_A[la_test(c, weekly_A["week"], weeks_all)].reset_index(drop=True)
        rt = rt.merge(weekly_0, on=["sku", "week"], how="left")
        print(f"    target không cắt thiếu ở {c}: {int(rt['qty0'].isna().sum())} dòng (điền 0)")
        rt["qty0"] = rt["qty0"].fillna(0.0)
        ref_test[c] = rt
    print(f"[2] Tuần của phiên bản A: {len(weeks_all)} tuần | dòng test: " +
          " | ".join(f"{c}: {len(ref_test[c]):,}" for c in CHIA))

    res = {}
    for ten, cap_on, loc in PHIEN_BAN:
        for chia in CHIA:
            print(f"\n--- {ten} | {TEN_CHIA[chia]}")
            try:
                res[(ten, chia)] = chay_mot(df, cap_on, loc, chia, weeks_all, ref_test[chia])
                inf = res[(ten, chia)]["info"]
                print(f"    dòng train: {inf['dong_train_truoc']:,} -> {inf['dong_train_sau']:,} "
                      f"(bỏ {inf['ty_le_bo_dong_pct']}%) | sản phẩm: {inf['san_pham_train_sau']} | "
                      f"trạng thái: {res[(ten, chia)]['status']}")
            except Exception as e:
                print(f"    LỖI: {type(e).__name__}: {e}")
                traceback.print_exc()
                res[(ten, chia)] = {"status": f"loi_{type(e).__name__}", "info": {}, "nhom": pd.DataFrame()}

    kiem_tra_voi_pipeline(res, weekly_A, skus)

    tong_hop = []
    for chia in CHIA:
        ref_tb = res[("A", chia)].get("tb", pd.Series(dtype=float))
        print(f"\n[Đánh giá] {TEN_CHIA[chia]}")
        tong_hop += danh_gia(res, chia, ref_tb)

    for ten, _, _ in PHIEN_BAN:
        for chia in CHIA:
            v = res[(ten, chia)]
            if "info" in v and v["info"]:
                luu_phien_ban(ten, chia, v)
    os.makedirs(OUT, exist_ok=True)
    tong = pd.DataFrame(tong_hop)
    tong.to_csv(os.path.join(OUT, "ket_qua_tong_hop.csv"), index=False, encoding="utf-8-sig")

    cot = ["phien_ban", "cach_chia", "cach_cham", "trang_thai", "so_dong_test", "so_san_pham_danh_gia", "mae", "rmse", "r2_qty",
           "bias_pct", "mae_chuan", "rmse_chuan", "chenh_mae_so_voi_A", "khoang_95_lo", "khoang_95_hi", "ty_le_tot_hon_A"]
    print("\n" + "=" * 78)
    print("KẾT QUẢ TỔNG HỢP (chênh MAE so với A: âm = tốt hơn A)")
    print("=" * 78)
    if len(tong):
        print(tong[[c for c in cot if c in tong.columns]].round(3).to_string(index=False))
    print(f"\nĐã ghi {OUT}/ket_qua_tong_hop.csv và thư mục từng phiên bản.")


if __name__ == "__main__":
    main()
