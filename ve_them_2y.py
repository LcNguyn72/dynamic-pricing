"""
Vẽ 7 biểu đồ bổ sung: A, B, C, D, E, G, H
==========================================
  A  Thực tế, dự báo và baseline theo tuần (tổng toàn bộ sản phẩm). Đường đứng: tuần test.
  B  Sai số tích lũy theo thời gian: (tổng dự báo / tổng thực tế − 1), cộng dồn. Trôi xa 0 = thiên lệch.
  C  MAE và RMSE của mô hình so với baseline (sku_base), trên ba cách chia.
  D  Hồi quy log-log cho 4 sản phẩm: điểm (ln giá, ln lượng đã trừ mùa vụ) và đường có độ dốc b_i.
  E  Co về mức chung: elasticity thô b_i và sau co m_i, nối với nhau, so với mức chung g.
  G  Doanh thu kỳ vọng theo giá với elasticity m−sd, m, m+sd và đường kỳ vọng (lognormal).
  H  Xác suất có lợi P(giá) và độ mạnh f(giá), kèm ngưỡng 0.55 và 0.90.

Chỉ ĐỌC models_2y và data. Ghi ./figures (PNG + CSV). Cần các file của pipeline và engine.
Chạy:  python ve_them_2y.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import compare_features_2y as CF
import compare_outliers_2y as CO
import ham_chung_bieu_do as H
from ham_chung_bieu_do import MAU, NGAY_MAU, plt
from y2_config import path
from y2_tools import estimate_raw, load_weekly

warnings.filterwarnings("ignore")

SKU_DEMO = "15036"          # sản phẩm dùng cho G và H (đổi nếu muốn)
SKU_HOI_QUY = 4             # số sản phẩm cho D
MO_TA_MOI = [
    ("A", "figA_tong_theo_tuan", "Thực tế, dự báo và baseline theo tuần (tổng)", "model_table + demand_model",
     "Mô hình có bám xu hướng tổng thể và có vượt baseline không."),
    ("B", "figB_sai_so_tich_luy", "Sai số tích lũy theo thời gian", "model_table + demand_model",
     "Đường trôi xa 0 nghĩa là thiên lệch hệ thống theo thời gian."),
    ("C", "figC_mo_hinh_so_baseline", "MAE và RMSE: mô hình so với baseline, ba cách chia", "compare_features_2y (cùng khung)",
     "Mô hình có tốt hơn baseline ổn định qua các cách chia không."),
    ("D", "figD_hoi_quy_log_log", "Hồi quy log-log cho 4 sản phẩm", "y2_tools.estimate_raw",
     "Elasticity thô b_i là độ dốc của đám điểm; se_i cho biết độ chắc chắn."),
    ("E", "figE_co_ve_muc_chung", "Co elasticity về mức chung", "pricing_params + pricing_model",
     "Sản phẩm có se lớn bị kéo mạnh về mức chung g."),
    ("G", "figG_doanh_thu_theo_elasticity", "Doanh thu kỳ vọng theo giá với các elasticity", "pricing_engine",
     "Đỉnh doanh thu dịch chuyển theo elasticity; bất định làm đường kỳ vọng phẳng hơn."),
    ("H", "figH_xac_suat_co_loi_theo_gia", "Xác suất có lợi và độ mạnh theo giá", "pricing_engine",
     "Quy tắc f: P từ 0.55 đến 0.90 thì đề xuất đi dần về phía giá mục tiêu."),
]


def _thieu(*ten):
    thieu = [t for t in ten if not os.path.exists(path(t))]
    return thieu


# ----------------------------------------------------------------------------- A
def ve_A(t):
    print("[A] Tổng theo tuần...")
    g = t.groupby("week").agg(thuc_te=("qty", "sum"), du_bao=("du_bao", "sum"),
                              nen=("sku_base", lambda s: float(np.exp(s).sum())),
                              la_test=("is_test", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for w in g.loc[g["la_test"], "week"]:
        ax.axvspan(w - pd.Timedelta(days=3), w + pd.Timedelta(days=3), color=MAU["nhat"], alpha=0.25, lw=0)
    ax.plot(g["week"], g["thuc_te"], color=MAU["den"], lw=2, label="Thực tế (tổng toàn bộ sản phẩm)")
    ax.plot(g["week"], g["du_bao"], color=MAU["xanh"], lw=2, label="Mô hình dự báo")
    ax.plot(g["week"], g["nen"], color=MAU["xam"], lw=1.5, ls="--", label="Baseline (mức bán nền sku_base)")
    ax.set_ylabel("Tổng lượng bán / tuần")
    ax.set_title("Tổng theo tuần: mô hình và baseline so với thực tế (vùng xám = tuần test)")
    ax.legend(loc="upper right", fontsize=9)
    m = CO.chi_so(g["thuc_te"].values, g["du_bao"].values, np.ones(len(g)))
    fig.text(0.5, -0.02, f"Lệch tổng toàn kỳ: {m['bias_pct']:+.1f}% (dự báo trên tuần train là dự báo khớp đã học).",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figA_tong_theo_tuan", g)


# ----------------------------------------------------------------------------- B
def ve_B(t):
    print("[B] Sai số tích lũy...")
    g = t.groupby("week").agg(thuc_te=("qty", "sum"), du_bao=("du_bao", "sum"),
                              la_test=("is_test", "max")).reset_index().sort_values("week")
    g["cum_thuc"] = g["thuc_te"].cumsum()
    g["cum_du_bao"] = g["du_bao"].cumsum()
    g["lech_tich_luy_pct"] = (g["cum_du_bao"] / g["cum_thuc"] - 1) * 100
    fig, ax = plt.subplots(figsize=(13, 5))
    for w in g.loc[g["la_test"], "week"]:
        ax.axvspan(w - pd.Timedelta(days=3), w + pd.Timedelta(days=3), color=MAU["nhat"], alpha=0.25, lw=0)
    ax.axhline(0, color=MAU["den"], lw=1)
    ax.plot(g["week"], g["lech_tich_luy_pct"], color=MAU["do"], lw=2)
    ax.set_ylabel("Lệch tích lũy (%)")
    ax.set_title("Sai số tích lũy: (tổng dự báo đến tuần t) / (tổng thực tế đến tuần t) − 1")
    fig.text(0.5, -0.02, "Đường gần 0 là dự báo không thiên lệch; trôi xa 0 là thiên lệch hệ thống. "
             "Vùng xám là tuần test.", ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figB_sai_so_tich_luy", g[["week", "thuc_te", "du_bao", "cum_thuc", "cum_du_bao",
                                          "lech_tich_luy_pct", "la_test"]])


# ----------------------------------------------------------------------------- C
def chay_c(wk, weeks_all, chia):
    test = np.asarray(CO.la_test(chia, wk["week"], weeks_all))
    wtr, wte = wk[~test], wk[test]
    st = CO.thong_ke(wtr)
    trn = CO.dac_trung(wtr, st)
    tst = CO.dac_trung(wte, st)
    m = CF.mo_hinh(trn, CF.FEATS_GOC)
    p_mh = np.exp(m.predict(tst[CF.FEATS_GOC].values.astype(np.float32)))
    p_nen = np.exp(tst["sku_base"].values)
    y = tst["qty"].values.astype(float)
    tb = wtr.groupby("sku")["qty"].mean().reindex(tst["sku"].values).fillna(1.0).values
    return {"mo_hinh": CO.chi_so(y, p_mh, tb), "baseline": CO.chi_so(y, p_nen, tb), "n": len(y)}


def ve_C(wk, weeks_all):
    print("[C] Mô hình so với baseline, ba cách chia...")
    rows = []
    for chia in CO.CHIA:
        r = chay_c(wk, weeks_all, chia)
        for ten in ("mo_hinh", "baseline"):
            rows.append({"cach_chia": chia, "mo_hinh": ten, "so_dong": r["n"], **r[ten]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(CO.CHIA))
    for ax, cot, tieu_de in ((axes[0], "mae", "MAE theo số lượng"), (axes[1], "rmse", "RMSE theo số lượng")):
        a = df[df["mo_hinh"] == "mo_hinh"].set_index("cach_chia").loc[CO.CHIA, cot].values
        b = df[df["mo_hinh"] == "baseline"].set_index("cach_chia").loc[CO.CHIA, cot].values
        ax.bar(x - 0.2, b, width=0.4, color=MAU["xam"], label="Baseline (sku_base)")
        ax.bar(x + 0.2, a, width=0.4, color=MAU["xanh"], label="Mô hình")
        for i in range(len(x)):
            ax.text(x[i] + 0.2, a[i], f"{a[i]:.2f}\n({(1 - a[i] / b[i]) * 100:+.1f}%)", ha="center", va="bottom", fontsize=8)
            ax.text(x[i] - 0.2, b[i], f"{b[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([CO.TEN_CHIA[c] for c in CO.CHIA], fontsize=9)
        ax.set_title(tieu_de)
        ax.set_ylim(0, max(a.max(), b.max()) * 1.25)
        ax.legend(fontsize=8)
    fig.text(0.5, -0.03, "Phần trăm trong ngoặc là mức giảm so với baseline (dương = tốt hơn baseline).",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figC_mo_hinh_so_baseline", df)


# ----------------------------------------------------------------------------- D
def ve_D(w_all):
    print("[D] Hồi quy log-log cho 4 sản phẩm...")
    res, g, me = estimate_raw(w_all, 10)
    res = res.sort_values("se").reset_index(drop=True)
    idx = np.linspace(0, len(res) - 1, SKU_HOI_QUY).round().astype(int)
    chon = res.iloc[idx]
    d = w_all.copy()
    d["lq_adj"] = d["lq"] - d["month"].map(lambda m: me.get(f"m_{m}", 0.0))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    rows = []
    for ax, r in zip(axes.ravel(), chon.itertuples()):
        s = d[d["sku"] == r.sku]
        ax.scatter(s["lp"], s["lq_adj"], s=22, color=MAU["xanh"], alpha=0.7)
        xb, yb = s["lp"].mean(), s["lq_adj"].mean()
        xs = np.linspace(s["lp"].min(), s["lp"].max(), 50)
        ax.plot(xs, yb + r.b * (xs - xb), color=MAU["do"], lw=2, label=f"độ dốc b = {r.b:.2f}")
        ax.set_xlabel("ln(giá tuần)")
        ax.set_ylabel("ln(lượng đã trừ mùa vụ)")
        ax.set_title(f"{r.sku}: b = {r.b:.2f} ± {r.se:.2f} (n = {r.n} tuần)", fontsize=10)
        ax.legend(fontsize=8)
        rows.append({"sku": r.sku, "n_tuan": r.n, "b_thô": r.b, "se": r.se, "mức_chung_g": g})
    fig.suptitle("Hồi quy log-log: độ dốc là elasticity thô của sản phẩm (chọn 4 sản phẩm có se khác nhau)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    H.luu(fig, "figD_hoi_quy_log_log", pd.DataFrame(rows))


# ----------------------------------------------------------------------------- E
def ve_E():
    print("[E] Co về mức chung...")
    pr = pd.read_csv(path("pricing_params_online_retail.csv"), dtype={"sku": str})
    with open(path("pricing_model_online_retail.json")) as f:
        g = float(json.load(f)["global_elasticity"])
    lo, hi = -4.0, 1.0
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ngoai = int(((pr["eps_raw"] < lo) | (pr["eps_raw"] > hi)).sum())
    raw = pr.clip(lower=None)
    for _, r in pr.iterrows():
        ax.plot([r["se_raw"], r["se_raw"]], [r["eps_raw"], r["eps_mean"]], color=MAU["nhat"], lw=0.6, alpha=0.6, zorder=1)
    ax.scatter(pr["se_raw"], pr["eps_raw"].clip(lo, hi), s=12, color=MAU["xam"], label="b thô (riêng từng sản phẩm)", zorder=2)
    ax.scatter(pr["se_raw"], pr["eps_mean"].clip(lo, hi), s=14, color=MAU["xanh"], label="m sau co (dùng trong engine)", zorder=3)
    ax.axhline(g, color=MAU["do"], lw=2, label=f"Mức chung g = {g:.2f}")
    ax.set_xlabel("se (sai số chuẩn của b thô): lớn = ước lượng nhiễu")
    ax.set_ylabel("Elasticity")
    ax.set_ylim(lo, hi)
    ax.set_title("Co về mức chung: sản phẩm se lớn bị kéo mạnh về g; đoạn nối là độ dịch chuyển")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.5, -0.02, f"{ngoai} sản phẩm có b thô ngoài khoảng [{lo}, {hi}] được cắt vào biên khi vẽ.",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figE_co_ve_muc_chung", pr[["sku", "n_weeks", "eps_raw", "se_raw", "prior_mean", "weight_own",
                                          "eps_mean", "eps_sd"]])


# ----------------------------------------------------------------------------- G và H
def _thong_so(eng, sku):
    s, p = eng._get(sku)
    d = eng._date(NGAY_MAU)
    p0 = float(s["p_ref"])
    m, sd = float(p["eps_mean"]), max(float(p["eps_sd"]), 1e-6)
    lo, hi = eng._bounds(s, p0, eng.max_change)
    u0 = eng._level(sku, p0, d)
    return p0, m, sd, lo, hi, u0


def ve_G(eng, sku):
    print("[G] Doanh thu theo elasticity...")
    p0, m, sd, lo, hi, u0 = _thong_so(eng, sku)
    gia = np.linspace(lo, hi, 161)
    r = gia / p0
    lr = np.log(r)
    rev0 = p0 * u0
    curves = {
        f"ε = m − sd = {m - sd:.2f}": gia * u0 * r ** (m - sd),
        f"ε = m = {m:.2f}": gia * u0 * r ** m,
        f"ε = m + sd = {m + sd:.2f}": gia * u0 * r ** (m + sd),
        "kỳ vọng theo N(m, sd) (engine dùng)": gia * u0 * np.exp(m * lr + 0.5 * sd ** 2 * lr ** 2),
    }
    fig, ax = plt.subplots(figsize=(12.5, 6))
    mau = [MAU["xam"], MAU["xanh"], MAU["cam"], MAU["do"]]
    rows = {"gia": gia}
    for (ten, y), c, ls in zip(curves.items(), mau, ["--", "-", "--", "-"]):
        ax.plot(gia, y, color=c, ls=ls, lw=2, label=ten)
        i = int(np.argmax(y))
        ax.scatter([gia[i]], [y[i]], color=c, s=50, zorder=4)
        rows[ten] = y
    ax.axhline(rev0, color=MAU["den"], lw=1, ls=":", label=f"Doanh thu hiện tại = {rev0:.2f}")
    ax.axvline(p0, color=MAU["den"], lw=1, ls=":")
    ax.set_xlabel("Giá")
    ax.set_ylabel("Doanh thu / tuần")
    ax.set_title(f"Doanh thu theo giá, sản phẩm {sku} (điểm = đỉnh của mỗi đường)")
    ax.legend(fontsize=9)
    fig.text(0.5, -0.02, "Elasticity càng âm, đỉnh càng nghiêng về giá thấp. Đường kỳ vọng dùng toàn bộ phân phối, "
             "nên phẳng và trơn hơn các đường đơn.", ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figG_doanh_thu_theo_elasticity", pd.DataFrame(rows))


def ve_H(eng, sku):
    print("[H] Xác suất có lợi theo giá...")
    p0, m, sd, lo, hi, u0 = _thong_so(eng, sku)
    gia = np.linspace(lo, hi, 161)
    from pricing_engine import _ZQ
    eps = m + sd * _ZQ                                        # 401 mẫu elasticity
    rev0 = p0 * u0
    rev = gia[:, None] * u0 * (gia[:, None] / p0) ** eps[None, :]
    P = (rev > rev0).mean(axis=1)
    f = np.clip((P - eng.prob_floor) / (eng.prob_full - eng.prob_floor), 0, 1)
    lr = np.log(gia / p0)
    ev = gia * u0 * np.exp(m * lr + 0.5 * sd ** 2 * lr ** 2)
    p_t = float(gia[int(np.argmax(ev))])
    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.plot(gia, P, color=MAU["xanh"], lw=2.2, label="P: xác suất có lợi (tỷ lệ 401 mẫu ε)")
    ax.plot(gia, f, color=MAU["cam"], lw=2, label="f: độ mạnh = clip((P − 0.55)/0.35)")
    ax.axhline(eng.prob_floor, color=MAU["xam"], ls="--", lw=1, label="ngưỡng 0.55 (bắt đầu đổi giá)")
    ax.axhline(eng.prob_full, color=MAU["xam"], ls=":", lw=1, label="ngưỡng 0.90 (đổi hết)")
    ax.axvline(p0, color=MAU["den"], lw=1, ls=":", label=f"giá hiện tại p0 = {p0:.2f}")
    ax.axvline(p_t, color=MAU["do"], lw=1.5, label=f"giá mục tiêu p_t = {p_t:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Giá thử nghiệm p")
    ax.set_ylabel("Xác suất / độ mạnh")
    ax.set_title(f"Xác suất có lợi và độ mạnh theo giá, sản phẩm {sku} (có lợi = doanh thu tại p > tại p0)")
    ax.legend(fontsize=8, loc="center left")
    fig.text(0.5, -0.02, "Đề xuất = p0 + f × (p_t − p0). Với P dưới 0.55, f = 0 và giữ nguyên giá.",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figH_xac_suat_co_loi_theo_gia", pd.DataFrame({"gia": gia, "P": P, "f": f}))


# ----------------------------------------------------------------------------- chạy chính
def main():
    print("=" * 70)
    print("VẼ 7 BIỂU ĐỒ BỔ SUNG: A, B, C, D, E, G, H")
    print("=" * 70)
    can = ["model_table_online_retail.csv", "demand_model_online_retail.pkl", "feature_cols_online_retail.json",
           "product_stats_online_retail.json", "pricing_params_online_retail.csv",
           "pricing_model_online_retail.json", "weekly_online_retail.csv", "products_online_retail.csv"]
    if _thieu(*can):
        raise SystemExit(f"Thiếu file trong models_2y: {_thieu(*can)}")
    H.MO_TA.extend([(ma, ten, tieu, nguon, y) for ma, ten, tieu, nguon, y in MO_TA_MOI])

    model, feats = H.nap_mo_hinh()
    t = H.du_bao(H.nap_bang_mo_hinh(), model, feats)
    ve_A(t)
    ve_B(t)

    skus = set(pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})["sku"])
    df = CO.doc_giao_dich(skus)
    wk = CO.tuan_tu(CO.giao_dich_phien_ban(df, True))
    wk = wk[wk["sku"].isin(skus)].reset_index(drop=True)
    weeks_all = pd.DatetimeIndex(sorted(wk["week"].unique()))
    ve_C(wk, weeks_all)

    w_all = load_weekly()[0]
    ve_D(w_all)
    ve_E()

    from pricing_engine import PricingEngine
    eng = PricingEngine()
    sku = SKU_DEMO if SKU_DEMO in eng.params.index else str(eng.params.index[0])
    print(f"  Sản phẩm dùng cho G và H: {sku}")
    ve_G(eng, sku)
    ve_H(eng, sku)

    H.cap_nhat_readme()
    print("Xong. Xem ./figures")


if __name__ == "__main__":
    main()
