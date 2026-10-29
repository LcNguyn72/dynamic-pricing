"""
Vẽ biểu đồ KẾT QUẢ MÔ HÌNH (10 biểu đồ)
=======================================
  1  Thực tế và dự báo theo tuần (3 sản phẩm: bán ổn định, đột biến, điển hình)
  2  Thực tế so với dự báo (toàn bộ tập test), thang thường và thang log
  3  Phân phối sai số dự báo
  4  Sai số theo nhóm giá và nhóm mức bán (có cả sai số chuẩn hóa theo sản phẩm)
  5  Sai số theo tháng
  6  Đường cầu dự báo của 4 sản phẩm
  7  Phản ứng giá: dự báo so với thực tế ở các lần đổi giá
  8  Phân phối elasticity và xác suất cầu co giãn
  9  Đề xuất giá cho 20 sản phẩm (theo doanh thu và theo lợi nhuận với giá vốn GIẢ ĐỊNH)
  10 Doanh thu kỳ vọng theo giá (4 sản phẩm)

Chạy:  python ve_ket_qua.py
Cần có trong ./models_2y: model_table, demand_model, feature_cols, product_stats, pricing_params, pricing_model, weekly.
Chạy lại mô hình để lấy dự báo (không huấn luyện lại), mất vài phút. Chỉ ĐỌC, không ghi vào models_2y.
Đầu ra: ./figures/*.png và *.csv đi kèm.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import ham_chung_bieu_do as H
from ham_chung_bieu_do import GIA_VON_GIA_DINH, MAU, NGAY_MAU, plt
from y2_config import path
from y2_tools import estimate_raw, load_weekly

NHAN_GIA = ["rẻ", "vừa thấp", "vừa cao", "đắt"]
NHAN_BAN = ["bán ít", "vừa thấp", "vừa cao", "bán nhiều"]
MAU_HANH_DONG = {"lower": MAU["xanh"], "raise": MAU["cam"], "keep": MAU["nhat"]}
TEN_HANH_DONG = {"lower": "giảm giá", "raise": "tăng giá", "keep": "giữ nguyên"}


# ----------------------------------------------------------------------------- chọn sản phẩm minh họa
def chon_3_san_pham(t):
    th = t.groupby("sku").agg(n=("qty", "size"), tb=("qty", "mean"), sd=("qty", "std")).reset_index()
    th = th[th["n"] >= 60].copy()
    th["cv"] = th["sd"] / th["tb"]
    on_dinh = th.loc[th["cv"].idxmin(), "sku"]
    dot_bien = th.loc[th["cv"].idxmax(), "sku"]
    con_lai = th[~th["sku"].isin([on_dinh, dot_bien])]
    dien_hinh = con_lai.loc[(con_lai["cv"] - th["cv"].median()).abs().idxmin(), "sku"]
    return [("bán ổn định", on_dinh), ("bán đột biến", dot_bien), ("điển hình", dien_hinh)]


def chon_4_sku_engine(engine, sku_chon):
    ds = sorted([s for s in sku_chon if s in engine.params.index and s in engine.stats],
                key=lambda s: engine.stats[s]["p_ref"])
    if len(ds) <= 4:
        return ds
    return [ds[i] for i in np.linspace(0, len(ds) - 1, 4).round().astype(int)]


def nhan_sp(engine, sku, dai=24):
    mo_ta = str(engine.stats[sku]["description"]).strip()
    return f"{sku} - {mo_ta[:dai]}"


# ----------------------------------------------------------------------------- 1
def ve_01(t, q10, q90):
    print("[1] Thực tế và dự báo theo tuần...")
    nhom = chon_3_san_pham(t)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5))
    rows = []
    for i, (ax, (ten_nhom, sku)) in enumerate(zip(axes, nhom)):
        d = t[t["sku"] == sku].sort_values("week")
        te = d[d["is_test"]]
        thap, cao = te["du_bao"] * np.exp(q10), te["du_bao"] * np.exp(q90)
        ax.plot(d["week"], d["qty"], color=MAU["xanh"], lw=1.2, label="Thực tế (mọi tuần)")
        ax.plot(d["week"], d["du_bao"], color=MAU["xam"], lw=1, alpha=0.7, label="Dự báo ở tuần train (đã học)")
        ax.vlines(te["week"], thap, cao, color=MAU["cam"], alpha=0.5, lw=3, label="Dải 10-90% (phân vị sai số ln của tập test)")
        ax.scatter(te["week"], te["du_bao"], color=MAU["do"], s=24, zorder=3, label="Dự báo ở tuần test")
        ax.scatter(te["week"], te["qty"], color=MAU["den"], s=18, marker="x", zorder=3, label="Thực tế ở tuần test")
        mae = float(np.mean(np.abs(te["du_bao"] - te["qty"])))
        ax.set_title(f"{sku} ({ten_nhom}) - MAE trên tuần test = {mae:.1f}", fontsize=10)
        ax.set_ylabel("Lượng bán / tuần")
        if i == 0:
            ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
        rows.append(d.assign(nhom=ten_nhom, du_bao_thap=np.where(d["is_test"], d["du_bao"] * np.exp(q10), np.nan),
                             du_bao_cao=np.where(d["is_test"], d["du_bao"] * np.exp(q90), np.nan))
                    [["nhom", "sku", "week", "is_test", "qty", "du_bao", "du_bao_thap", "du_bao_cao"]]
                    .rename(columns={"qty": "thuc_te"}))
    fig.tight_layout()
    H.luu(fig, "fig01_thuc_te_du_bao_theo_tuan", pd.concat(rows, ignore_index=True))


# ----------------------------------------------------------------------------- 2
def ve_02(te):
    print("[2] Thực tế so với dự báo...")
    y, p = te["qty"].values.astype(float), te["du_bao"].values
    m = H.chi_so(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    lim = float(np.percentile(np.maximum(y, p), 99))
    ax = axes[0]
    ax.scatter(p, y, s=4, alpha=0.18, color=MAU["xanh"])
    ax.plot([0, lim], [0, lim], color=MAU["do"], lw=1.2, label="y = x (dự báo đúng)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Dự báo (cắt ở P99)")
    ax.set_ylabel("Thực tế")
    ax.set_title("Thang thường")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05, f"MAE = {m['mae']:.2f}\nRMSE = {m['rmse']:.2f}\nR² (qty) = {m['r2_qty']:.3f}\nLệch tổng = {m['bias_pct']:+.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=MAU["nhat"]))
    ax = axes[1]
    ax.scatter(np.clip(p, 0.5, None), y, s=4, alpha=0.18, color=MAU["xanh"])
    lo, hi = 0.5, float(max(y.max(), p.max())) * 1.1
    ax.plot([lo, hi], [lo, hi], color=MAU["do"], lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dự báo (thang log)")
    ax.set_ylabel("Thực tế (thang log)")
    ax.set_title("Thang log (thấy rõ các sản phẩm bán ít)")
    fig.tight_layout()
    H.luu(fig, "fig02_thuc_te_vs_du_bao", te[["sku", "week", "qty", "du_bao"]].rename(columns={"qty": "thuc_te"}))


# ----------------------------------------------------------------------------- 3
def ve_03(te):
    print("[3] Phân phối sai số...")
    y, p = te["qty"].values.astype(float), te["du_bao"].values
    sai = p - y
    sai_ln = te["du_bao_ln"].values - te["lq"].values
    m = H.chi_so(y, p)
    gioi_han = float(np.percentile(np.abs(sai), 99))
    top5 = np.abs(sai) >= np.percentile(np.abs(sai), 95)
    chia_sq = float((sai[top5] ** 2).sum() / (sai ** 2).sum() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    ax = axes[0]
    cnt, edges, _ = ax.hist(sai[np.abs(sai) <= gioi_han], bins=80, color=MAU["xanh"], alpha=0.85)
    ax.axvline(0, color=MAU["do"], lw=1.3, label="Sai số = 0")
    ax.axvline(sai.mean(), color=MAU["cam"], ls="--", label=f"Trung bình = {sai.mean():+.1f}")
    ax.set_xlabel(f"Dự báo - thực tế (số lượng; hiển thị |sai số| ≤ P99 = {gioi_han:.0f})")
    ax.set_ylabel("Số dòng")
    ax.set_title("Sai số theo số lượng")
    ax.legend()
    ax.text(0.97, 0.95, f"MAE = {m['mae']:.2f}\nRMSE = {m['rmse']:.2f}\nRMSE / MAE = {m['rmse'] / m['mae']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=MAU["nhat"]))
    ax = axes[1]
    ax.hist(sai_ln, bins=80, color=MAU["xanh"], alpha=0.85)
    ax.axvline(0, color=MAU["do"], lw=1.3)
    ax.set_xlabel("ln(dự báo) - ln(thực tế)")
    ax.set_ylabel("Số dòng")
    ax.set_title(f"Sai số trên thang log (trung bình {sai_ln.mean():+.2f})")
    fig.text(0.5, -0.03, f"5% dòng có sai số tuyệt đối lớn nhất đóng góp {chia_sq:.0f}% tổng bình phương sai số -> đây là lý do RMSE lớn hơn MAE nhiều.",
             ha="center", fontsize=9)
    fig.tight_layout()
    bang = pd.DataFrame({"sai_so_tu": edges[:-1], "sai_so_den": edges[1:], "so_dong": cnt})
    H.luu(fig, "fig03_phan_bo_sai_so", bang)
    H.luu_csv(pd.DataFrame([{**m, "ty_le_rmse_mae": m["rmse"] / m["mae"], "dong_gop_5pct_sai_so_lon_nhat_pct": chia_sq}]),
              "fig03_phan_bo_sai_so_tom_tat")


# ----------------------------------------------------------------------------- 4
def _theo_nhom(te, cot_nhom):
    out = []
    for nhom, d in te.groupby(cot_nhom, observed=True):
        m = H.chi_so(d["qty"], d["du_bao"], d["tb_train"])
        out.append({"nhom": nhom, **m})
    return pd.DataFrame(out)


def ve_04(te):
    print("[4] Sai số theo nhóm...")
    sp = te.groupby("sku").agg(gia=("gia_tv", "first"), ban=("tb_train", "first"))
    te = te.copy()
    for cot, nhan, ten in (("gia", NHAN_GIA, "nhom_gia"), ("ban", NHAN_BAN, "nhom_ban")):
        ma = pd.qcut(sp[cot], 4, labels=False, duplicates="drop")
        te[ten] = te["sku"].map(ma).map(lambda i: nhan[int(i)] if pd.notna(i) else None)
    g, b = _theo_nhom(te, "nhom_gia"), _theo_nhom(te, "nhom_ban")
    g["nhom"] = pd.Categorical(g["nhom"], [x for x in NHAN_GIA if x in set(g["nhom"])], ordered=True)
    b["nhom"] = pd.Categorical(b["nhom"], [x for x in NHAN_BAN if x in set(b["nhom"])], ordered=True)
    g, b = g.sort_values("nhom"), b.sort_values("nhom")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for hang, (bang, ten) in enumerate(((g, "nhóm giá (theo giá trung vị của sản phẩm)"), (b, "nhóm mức bán (theo lượng bán trung bình ở tuần train)"))):
        x = np.arange(len(bang))
        for cot, (c1, c2, nhan_y, tieu_de) in enumerate((("mae", "rmse", "Số lượng / tuần", "Sai số theo số lượng"),
                                                          ("mae_chuan", "rmse_chuan", "Tỷ lệ so với trung bình train của sản phẩm", "Sai số chuẩn hóa theo sản phẩm"))):
            ax = axes[hang, cot]
            ax.bar(x - 0.2, bang[c1], width=0.4, color=MAU["xanh"], label="MAE")
            ax.bar(x + 0.2, bang[c2], width=0.4, color=MAU["cam"], label="RMSE")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n}\n(n={k:,})" for n, k in zip(bang["nhom"], bang["n"])], fontsize=9)
            ax.set_ylabel(nhan_y)
            ax.set_title(f"{tieu_de} - {ten}", fontsize=9)
            ax.legend()
    fig.tight_layout()
    ra = pd.concat([g.assign(loai_nhom="giá"), b.assign(loai_nhom="mức bán")], ignore_index=True)
    H.luu(fig, "fig04_sai_so_theo_nhom", ra[["loai_nhom", "nhom", "n", "mae", "rmse", "mae_chuan", "rmse_chuan", "r2_qty", "bias_pct"]])


# ----------------------------------------------------------------------------- 5
def ve_05(te):
    print("[5] Sai số theo tháng...")
    te = te.copy()
    te["thang"] = te["week"].dt.month
    out = []
    for th, d in te.groupby("thang"):
        m = H.chi_so(d["qty"], d["du_bao"], d["tb_train"])
        out.append({"thang": th, **m})
    bang = pd.DataFrame(out)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(bang["thang"], bang["mae"], marker="o", color=MAU["xanh"], label="MAE")
    ax.plot(bang["thang"], bang["rmse"], marker="s", color=MAU["cam"], label="RMSE")
    ax.set_ylabel("Số lượng / tuần")
    ax.set_title("Sai số theo tháng (tập test)")
    ax.legend()
    ax = axes[1]
    mau = [MAU["do"] if v < 0 else MAU["la"] for v in bang["bias_pct"]]
    ax.bar(bang["thang"], bang["bias_pct"], color=mau)
    ax.axhline(0, color=MAU["den"], lw=1)
    for th, v, n in zip(bang["thang"], bang["bias_pct"], bang["n"]):
        ax.text(th, v, f"n={n}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylabel("Lệch tổng (%)")
    ax.set_xlabel("Tháng")
    ax.set_xticks(range(1, 13))
    ax.set_title("Dự báo thiên cao (xanh lá) hay thiên thấp (đỏ) so với thực tế", fontsize=10)
    fig.tight_layout()
    H.luu(fig, "fig05_sai_so_theo_thang", bang)


# ----------------------------------------------------------------------------- 6 và 10 (dùng engine)
def duong_cau_co_gia_hien_tai(engine, sku, points=21):
    """Đường cầu với GIÁ HIỆN TẠI đúng (chưa làm tròn) nằm trong lưới giá.
    Cùng công thức với PricingEngine.demand_curve: Q = Q0 * (p/p0)^(eps + k*sd), k = 0, +-1.2816."""
    s, p = engine._get(sku)
    d = engine._date(NGAY_MAU)
    p0 = float(s["p_ref"])
    m, sd = float(p["eps_mean"]), max(float(p["eps_sd"]), 1e-6)
    lo, hi = engine._bounds(s, p0, engine.max_change)
    u0 = engine._level(sku, p0, d)
    grid = np.unique(np.append(np.linspace(lo, hi, points), p0))
    r = grid / p0
    q_mid = u0 * r ** m
    q_lo = u0 * r ** (m - 1.2816 * sd)
    q_hi = u0 * r ** (m + 1.2816 * sd)
    return pd.DataFrame({"price": grid, "units": q_mid, "units_low": np.minimum(q_lo, q_hi),
                         "units_high": np.maximum(q_lo, q_hi)}), p0, m, sd


def ve_06(engine, ds4):
    print("[6] Đường cầu...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = []
    for ax, sku in zip(axes.ravel(), ds4):
        d, p0, m, sd = duong_cau_co_gia_hien_tai(engine, sku)
        ax.fill_between(d["price"], d["units_low"], d["units_high"], color=MAU["cam"], alpha=0.25,
                        label="Dải 10-90% (chỉ bất định elasticity)")
        ax.step(d["price"], d["units"], where="mid", color=MAU["xanh"], lw=2, label="Lượng bán dự báo")
        ax.axvline(p0, color=MAU["do"], ls="--", label=f"Giá hiện tại = {p0:.4f}")
        ax.set_title(f"{nhan_sp(engine, sku)}\nε = {m:.2f} ± {sd:.2f}", fontsize=10)
        ax.set_xlabel("Giá")
        ax.set_ylabel("Lượng bán / tuần")
        rows.append(d.assign(sku=sku, gia_hien_tai=p0, elasticity_mean=m, elasticity_sd=sd))
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Đường cầu dự báo (ngày áp dụng {NGAY_MAU})", fontweight="bold")
    fig.text(0.5, -0.02, "Đường có bậc vì mức cầu dự báo của XGBoost là hàm bậc thang theo giá (không phải đường mượt). "
             "Dải chỉ phản ánh bất định của elasticity, KHÔNG gồm sai số của mô hình nhu cầu.",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    H.luu(fig, "fig06_duong_cau", pd.concat(rows, ignore_index=True))

def doanh_thu_co_gia_hien_tai(engine, sku, points=41):
    """Doanh thu kỳ vọng theo giá với GIÁ HIỆN TẠI chưa làm tròn nằm trong lưới.
    Doanh thu = p * Q, Q = Q0 * (p/p0)^(eps + k*sd), k = 0, +-1.2816 (cùng công thức với demand_curve)."""
    d, p0, m, sd = duong_cau_co_gia_hien_tai(engine, sku, points)
    d["revenue"] = d["price"] * d["units"]
    d["revenue_low"] = d["price"] * d["units_low"]
    d["revenue_high"] = d["price"] * d["units_high"]
    return d, p0, m, sd


def ve_10(engine, ds4):
    print("[10] Doanh thu kỳ vọng theo giá...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = []
    for ax, sku in zip(axes.ravel(), ds4):
        d, p0, m, sd = doanh_thu_co_gia_hien_tai(engine, sku)
        r = engine.recommend(sku, date=NGAY_MAU)
        p_rec = r["recommended_price"]
        rev0 = p0 * float(d.loc[np.isclose(d["price"], p0), "units"].iloc[0])
        ax.fill_between(d["price"], d["revenue_low"], d["revenue_high"], color=MAU["cam"], alpha=0.25,
                        label="Dải 10-90% (chỉ bất định elasticity)")
        ax.step(d["price"], d["revenue"], where="mid", color=MAU["xanh"], lw=2, label="Doanh thu kỳ vọng / tuần")
        ax.scatter([p0], [rev0], color=MAU["do"], s=70, zorder=4, label=f"Giá hiện tại {p0:.4f}")
        if r["action"] != "keep":
            rev_rec = p_rec * float(np.interp(p_rec, d["price"], d["units"]))
            ax.scatter([p_rec], [rev_rec], color=MAU["la"], s=70, marker="D", zorder=4,
                       label=f"Giá đề xuất {p_rec:.2f} ({TEN_HANH_DONG[r['action']]})")
        ax.set_title(f"{nhan_sp(engine, sku)}\nε = {m:.2f} ± {sd:.2f}; xác suất có lợi = "
                     f"{'-' if r['confidence']['prob_gain'] is None else format(r['confidence']['prob_gain'], '.0%')}", fontsize=10)
        ax.set_xlabel("Giá")
        ax.set_ylabel("Doanh thu / tuần")
        ax.legend(fontsize=8, loc="best")
        rows.append(d.assign(sku=sku, gia_de_xuat=p_rec, hanh_dong=r["action"]))
    fig.suptitle(f"Doanh thu kỳ vọng theo giá - mục tiêu doanh thu (ngày {NGAY_MAU})", fontweight="bold")
    fig.text(0.5, -0.02, "Doanh thu = giá x lượng bán dự báo. Đường có bậc vì mô hình nhu cầu là hàm bậc thang theo giá. "
             "Dải chỉ phản ánh bất định elasticity, không gồm sai số của mô hình nhu cầu.", ha="center", fontsize=8.5)
    fig.tight_layout()
    H.luu(fig, "fig10_duong_doanh_thu", pd.concat(rows, ignore_index=True))

# ----------------------------------------------------------------------------- 7
def ve_07(t, model, feats):
    print("[7] Phản ứng giá ở các lần đổi giá...")
    w_all = load_weekly()[0]
    month_eff = estimate_raw(w_all, 10)[2]
    ev = H.su_kien_doi_gia(w_all, month_eff)
    ev = ev.merge(t[["sku", "week", "is_test"] + feats], on=["sku", "week"], how="inner")
    ev = ev[ev["is_test"]].copy()
    params = pd.read_csv(path("pricing_params_online_retail.csv"), dtype={"sku": str}).set_index("sku")
    with open(path("pricing_model_online_retail.json")) as f:
        g = float(json.load(f)["global_elasticity"])
    ev["eps"] = ev["sku"].map(params["eps_mean"])
    ev = ev.dropna(subset=["eps", "dlq"])
    if len(ev) < 50:
        print(f"  BỎ QUA: chỉ có {len(ev)} lần đổi giá ở tuần test")
        return
    X_cur = ev[feats].values.astype(np.float32)
    X_prev = X_cur.copy()
    X_prev[:, feats.index("log_price_ratio")] = (ev["lp_prev"].values - ev["sku_lp_med"].values).astype(np.float32)
    ev["du_bao_xgb"] = model.predict(X_cur) - model.predict(X_prev)   # chỉ đổi giá, giữ nguyên mọi thứ khác của tuần t
    ev["du_bao_hoi_quy"] = ev["eps"] * ev["dlp"]
    ev["du_bao_chung"] = g * ev["dlp"]
    cor = {k: float(np.corrcoef(ev[k], ev["dlq"])[0, 1]) for k in ("du_bao_xgb", "du_bao_hoi_quy", "du_bao_chung")}
    cat = [-np.inf, -0.30, -0.20, -0.15, -0.10, 0.10, 0.15, 0.20, 0.30, np.inf]
    ev["nhom"] = pd.cut(ev["dlp"], cat)
    bin_ = ev.groupby("nhom", observed=True).agg(n=("dlq", "size"), dlp=("dlp", "mean"), thuc_te=("dlq", "mean"),
                                                 xgboost=("du_bao_xgb", "mean"), hoi_quy=("du_bao_hoi_quy", "mean"),
                                                 mot_con_so_chung=("du_bao_chung", "mean")).reset_index()
    bin_["nhom"] = bin_["nhom"].astype(str)
    hien = bin_[bin_["n"] >= 30]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    ax.axhline(0, color=MAU["den"], lw=0.8)
    ax.axvline(0, color=MAU["den"], lw=0.8)
    ax.plot(hien["dlp"], hien["thuc_te"], marker="o", color=MAU["den"], lw=2.2, label="Thực tế")
    ax.plot(hien["dlp"], hien["xgboost"], marker="s", color=MAU["cam"], label="XGBoost (chỉ đổi giá)")
    ax.plot(hien["dlp"], hien["hoi_quy"], marker="^", color=MAU["xanh"], label="Hồi quy elasticity từng sản phẩm*")
    ax.plot(hien["dlp"], hien["mot_con_so_chung"], color=MAU["xam"], ls="--", label=f"Một con số chung (ε = {g:.2f})*")
    ax.set_xlabel("Thay đổi ln(giá) so với tuần trước (trung bình của nhóm)")
    ax.set_ylabel("Thay đổi ln(lượng bán), đã trừ mùa vụ")
    ax.set_title("Trung bình theo nhóm mức đổi giá")
    ax.legend(fontsize=8)
    ax = axes[1]
    mau_ = ev.sample(min(4000, len(ev)), random_state=0)
    ax.scatter(mau_["du_bao_xgb"], mau_["dlq"], s=6, alpha=0.25, color=MAU["xanh"])
    lim = float(np.percentile(np.abs(np.r_[mau_["du_bao_xgb"], mau_["dlq"]]), 99))
    ax.plot([-lim, lim], [-lim, lim], color=MAU["do"], lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Thay đổi ln(lượng) do XGBoost dự báo")
    ax.set_ylabel("Thay đổi ln(lượng) thực tế")
    ax.set_title("Từng lần đổi giá (mẫu ngẫu nhiên)")
    ax.text(0.03, 0.97, f"{len(ev):,} lần đổi giá ≥ 10%\nTương quan với thực tế:\n  XGBoost = {cor['du_bao_xgb']:.3f}\n"
            f"  Hồi quy* = {cor['du_bao_hoi_quy']:.3f}\n  Một con số chung* = {cor['du_bao_chung']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=MAU["nhat"]))
    fig.text(0.5, -0.04, "* Hồi quy elasticity được ước lượng trên cả 2 năm (kể cả các lần đổi giá này) nên đây là so sánh MÔ TẢ, không phải kiểm định ngoài mẫu. "
             "Tuần t thuộc tập test của XGBoost.", ha="center", fontsize=8.5)
    fig.tight_layout()
    H.luu(fig, "fig07_phan_ung_gia", bin_)
    H.luu_csv(pd.DataFrame([{"so_lan_doi_gia": len(ev), **{f"tuong_quan_{k}": v for k, v in cor.items()}}]), "fig07_phan_ung_gia_tuong_quan")


# ----------------------------------------------------------------------------- 8
def ve_08():
    print("[8] Phân phối elasticity...")
    pr = pd.read_csv(path("pricing_params_online_retail.csv"), dtype={"sku": str})
    p_el = norm.cdf((-1 - pr["eps_mean"]) / pr["eps_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    ax = axes[0]
    cnt, edges, _ = ax.hist(pr["eps_mean"], bins=40, color=MAU["xanh"], alpha=0.85)
    ax.axvline(-1, color=MAU["do"], lw=1.5, label="Ngưỡng ε = -1")
    ax.axvline(pr["eps_mean"].median(), color=MAU["cam"], ls="--", label=f"Trung vị = {pr['eps_mean'].median():.2f}")
    ax.set_xlabel("Elasticity trung bình của sản phẩm (eps_mean)")
    ax.set_ylabel("Số sản phẩm")
    ax.set_title("Phân phối elasticity (sau khi co về mức chung)")
    ax.legend()
    ax = axes[1]
    ax.hist(p_el, bins=40, color=MAU["xanh"], alpha=0.85)
    ax.axvline(0.5, color=MAU["do"], lw=1.2, ls="--", label="50%")
    ax.set_xlabel("Xác suất cầu co giãn, P(ε < -1)")
    ax.set_ylabel("Số sản phẩm")
    ax.set_title(f"Xác suất cầu co giãn (độ bất định trung vị sd = {pr['eps_sd'].median():.2f})")
    ax.legend()
    fig.tight_layout()
    tom_tat = pd.DataFrame({"chi_tieu": ["phan_vi_10_eps_mean", "trung_vi_eps_mean", "phan_vi_90_eps_mean", "trung_vi_eps_sd",
                                         "ty_le_san_pham_P_co_gian_tren_70pct", "ty_le_san_pham_eps_mean_nho_hon_-1", "so_san_pham"],
                            "gia_tri": [pr["eps_mean"].quantile(.1), pr["eps_mean"].median(), pr["eps_mean"].quantile(.9),
                                        pr["eps_sd"].median(), float((p_el > 0.7).mean()), float((pr["eps_mean"] < -1).mean()), len(pr)]})
    H.luu(fig, "fig08_phan_bo_elasticity", tom_tat)
    H.luu_csv(pd.DataFrame({"eps_tu": edges[:-1], "eps_den": edges[1:], "so_san_pham": cnt}), "fig08_phan_bo_elasticity_bin")


# ----------------------------------------------------------------------------- 9
def ve_09(engine, sku_chon):
    print("[9] Đề xuất giá cho 20 sản phẩm...")
    ds = [s for s in sku_chon if s in engine.params.index and s in engine.stats]
    kq = []
    for s in ds:
        r1 = engine.recommend(s, date=NGAY_MAU)
        r2 = engine.recommend(s, unit_cost=GIA_VON_GIA_DINH * engine.stats[s]["p_ref"], date=NGAY_MAU)
        kq.append({"sku": s, "mo_ta": engine.stats[s]["description"], "gia_hien_tai": r1["current_price"],
                   "dt_gia_de_xuat": r1["recommended_price"], "dt_doi_pct": r1["price_change_pct"], "dt_hanh_dong": r1["action"],
                   "dt_xac_suat_co_loi": r1["confidence"]["prob_gain"], "dt_ky_vong_pct": r1["confidence"]["expected_gain_pct"],
                   "ln_gia_von_gia_dinh": round(GIA_VON_GIA_DINH * engine.stats[s]["p_ref"], 2),
                   "ln_gia_de_xuat": r2["recommended_price"], "ln_doi_pct": r2["price_change_pct"], "ln_hanh_dong": r2["action"],
                   "ln_xac_suat_co_loi": r2["confidence"]["prob_gain"], "ln_ky_vong_pct": r2["confidence"]["expected_gain_pct"]})
    bang = pd.DataFrame(kq).sort_values("dt_doi_pct").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.36 * len(bang) + 2)), sharey=True)
    nhan = [f"{r.sku} {str(r.mo_ta)[:20]}" for r in bang.itertuples()]
    for ax, tien_to, tieu_de in ((axes[0], "dt", "Mục tiêu DOANH THU (không cần giá vốn)"),
                                 (axes[1], "ln", f"Mục tiêu LỢI NHUẬN (giá vốn GIẢ ĐỊNH = {GIA_VON_GIA_DINH:.0%} giá tham chiếu)")):
        for i, r in bang.iterrows():
            hd, v, pg = r[f"{tien_to}_hanh_dong"], r[f"{tien_to}_doi_pct"], r[f"{tien_to}_xac_suat_co_loi"]
            alpha = 0.35 if hd == "keep" else 0.4 + 0.6 * float(pg or 0)
            ax.barh(i, v, color=MAU_HANH_DONG[hd], alpha=alpha)
            ax.text(v + (0.15 if v >= 0 else -0.15), i, "giữ" if hd == "keep" else f"{v:+.1f}% | P={pg:.0%}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=MAU["den"], lw=0.8)
        ax.set_title(tieu_de, fontsize=10)
        ax.set_xlabel("Đổi giá đề xuất (%)")
        lim = max(1.0, float(bang[[f"{tien_to}_doi_pct"]].abs().max().iloc[0])) * 1.6
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(range(len(bang)))
    axes[0].set_yticklabels(nhan, fontsize=8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=MAU_HANH_DONG[k], label=TEN_HANH_DONG[k]) for k in ("lower", "raise", "keep")], loc="lower right", fontsize=8)
    fig.suptitle(f"Đề xuất giá, ngày {NGAY_MAU} (độ đậm của màu = xác suất có lợi, P)", fontweight="bold")
    fig.text(0.5, -0.01, "Chế độ lợi nhuận dùng giá vốn GIẢ ĐỊNH để minh họa; hướng tăng/giảm giá phụ thuộc mạnh vào giá vốn thật.", ha="center", fontsize=8.5)
    fig.tight_layout()
    H.luu(fig, "fig09_de_xuat_gia", bang)


# ----------------------------------------------------------------------------- chạy chính
def main():
    print("=" * 70)
    print("VẼ BIỂU ĐỒ KẾT QUẢ MÔ HÌNH")
    print("=" * 70)
    can = ["model_table_online_retail.csv", "demand_model_online_retail.pkl", "feature_cols_online_retail.json",
           "pricing_params_online_retail.csv", "pricing_model_online_retail.json", "product_stats_online_retail.json"]
    thieu = [f for f in can if not os.path.exists(path(f))]
    if thieu:
        raise SystemExit(f"Thiếu file trong models_2y: {thieu}. Chạy các bước 1, 2, 4, 5 của pipeline trước.")
    from pricing_engine import PricingEngine

    model, feats = H.nap_mo_hinh()
    t = H.du_bao(H.nap_bang_mo_hinh(), model, feats)
    te = t[t["is_test"]].copy()
    print("Kiểm tra số liệu:")
    H.kiem_tra_so_lieu(te)
    q10, q90 = np.percentile(te["lq"] - te["du_bao_ln"], [10, 90])
    engine = PricingEngine()
    sku_chon, nguon = H.nap_sku_chon()
    if sku_chon is None:
        ds = sorted(engine.params.index, key=lambda s: engine.stats[s]["p_ref"] if s in engine.stats else 0)
        sku_chon = [ds[i] for i in np.linspace(0, len(ds) - 1, 20).round().astype(int)]
        print("  CẢNH BÁO: không thấy selected_products_online_retail.csv, dùng 20 sản phẩm cách đều theo giá")
    else:
        print(f"  Danh sách 20 sản phẩm: {nguon}")
    ds4 = chon_4_sku_engine(engine, sku_chon)

    ve_01(t, q10, q90)
    ve_02(te)
    ve_03(te)
    ve_04(te)
    ve_05(te)
    ve_06(engine, ds4)
    ve_07(t, model, feats)
    ve_08()
    ve_09(engine, sku_chon)
    ve_10(engine, ds4)
    H.cap_nhat_readme()
    print("Xong. Xem thư mục ./figures")


if __name__ == "__main__":
    main()
