"""
Vẽ biểu đồ về DỮ LIỆU (4 biểu đồ)
=================================
  A2  Thác nước làm sạch dữ liệu (mỗi bước loại bao nhiêu dòng)
  B1  Phân phối lượng bán theo tuần (giải thích vì sao RMSE cao hơn MAE)
  B3  Số mức giá và số lần đổi giá của mỗi sản phẩm (giải thích vì sao elasticity riêng yếu)
  D3  20 sản phẩm được chọn có đại diện cho toàn bộ không

Chạy:  python ve_du_lieu.py        (cần đã chạy bước 1; A2 cần thêm file cache data/online_retail_II_sheets.csv)
Đầu ra: ./figures/*.png và *.csv đi kèm, figures/README.md. Chỉ ĐỌC dữ liệu, không ghi gì vào models_2y.
"""
import os

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import ham_chung_bieu_do as H
from ham_chung_bieu_do import MAU, plt
from y2_config import MIN_CV, MIN_WEEKS_FILTER, RAW_CACHE, path
from y2_tools import load_weekly


def _k(x, _):
    return f"{x / 1000:,.0f}k"


# ----------------------------------------------------------------------------- A2
def ve_a2():
    print("[A2] Thác nước làm sạch dữ liệu...")
    if not os.path.exists(RAW_CACHE):
        print(f"  BỎ QUA: không thấy {RAW_CACHE} (chạy 1_data_processing_online_retail_2y.py trước)")
        return
    df = pd.read_csv(RAW_CACHE, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    n_goc = len(df)
    if "sheet" in df.columns and (df["sheet"] == 1).any():          # bỏ phần chồng lấn giống bước 1
        s1, s2 = df[df["sheet"] == 0], df[df["sheet"] == 1]
        cut = s2["InvoiceDate"].min().normalize()
        df = pd.concat([s1[s1["InvoiceDate"] < cut], s2], ignore_index=True)
    n_chong = n_goc - len(df)
    df = df.rename(columns={"StockCode": "sku", "Quantity": "qty", "Price": "price", "InvoiceDate": "date"})

    n = len(df)
    df = df[~df["Invoice"].astype(str).str.upper().str.startswith("C")]
    n_huy, n = n - len(df), len(df)
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    n_qg, n = n - len(df), len(df)
    df["sku"] = df["sku"].astype(str)
    df = df[df["sku"].str.match(r"^\d{5}")]
    n_ma, n = n - len(df), len(df)
    cap = df.groupby("sku")["qty"].transform(lambda s: s.quantile(0.95))
    ban_le = df[df["qty"] <= cap].copy()
    n_si, n_con = n - len(ban_le), len(ban_le)

    ban_le["week"] = ban_le["date"].dt.to_period("W").dt.start_time
    w = ban_le.groupby(["sku", "week"]).agg(price=("price", "median")).reset_index()
    g = w.groupby("sku")["price"]
    st = pd.DataFrame({"weeks": g.size(), "cv": g.std() / g.mean()})
    keep = st[(st["weeks"] >= MIN_WEEKS_FILTER) & (st["cv"] >= MIN_CV)].index
    w_keep = w[w["sku"].isin(keep)]

    p_w = path("weekly_online_retail.csv")                          # đối chiếu với kết quả bước 1
    if os.path.exists(p_w):
        wk = pd.read_csv(p_w, dtype={"sku": str})
        ok = len(wk) == len(w_keep) and wk["sku"].nunique() == len(keep)
        print(f"  Đối chiếu bước 1: {len(wk):,} dòng tuần, {wk['sku'].nunique()} sản phẩm -> "
              f"{'KHỚP' if ok else f'LỆCH (script này tính {len(w_keep):,} dòng, {len(keep)} sản phẩm)'}")

    buoc = [("Dòng gốc\n(2 sheet)", n_goc, None), ("Bỏ phần\nchồng lấn", -n_chong, None),
            ("Bỏ hóa đơn\nhủy", -n_huy, None), ("Bỏ qty/giá\n≤ 0", -n_qg, None),
            ("Bỏ mã\nkhông phải SP", -n_ma, None), ("Bỏ đơn sỉ\n(top 5%)", -n_si, None),
            ("Còn lại\n(bán lẻ)", n_con, "cuoi")]
    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 2, width_ratios=[1.7, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    chay = 0
    bang = []
    for i, (ten, v, loai) in enumerate(buoc):
        if i == 0:
            day, cao, mau = 0, v, MAU["xanh"]
            chay = v
        elif loai == "cuoi":
            day, cao, mau = 0, v, MAU["la"]
        else:
            day, cao, mau = chay + v, -v, MAU["do"]
            chay += v
        ax.bar(i, cao, bottom=day, color=mau, width=0.62)
        nhan = f"{v:,}" if i in (0, len(buoc) - 1) else f"{v:,}\n({-v / n_goc * 100:.1f}%)"
        ax.text(i, day + cao + n_goc * 0.012, nhan, ha="center", va="bottom", fontsize=9)
        bang.append({"buoc": ten.replace("\n", " "), "so_dong_thay_doi": int(v), "ty_le_so_voi_goc_pct": round(v / n_goc * 100, 2),
                     "so_dong_con_lai_sau_buoc": int(chay if loai != "cuoi" else v)})
    ax.set_xticks(range(len(buoc)))
    ax.set_xticklabels([b[0] for b in buoc], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_k))
    ax.set_ylim(0, n_goc * 1.12)
    ax.set_title("Từ dòng giao dịch gốc đến dữ liệu bán lẻ sạch")
    ax.set_ylabel("Số dòng giao dịch")

    ax2 = fig.add_subplot(gs[0, 1])
    v2 = [len(st), len(keep)]
    ax2.bar(["Sản phẩm có bán", "Sản phẩm đủ điều kiện"], v2, color=[MAU["nhat"], MAU["xanh"]], width=0.55)
    for i, v in enumerate(v2):
        ax2.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax2.set_ylim(0, max(v2) * 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.set_title(f"Lọc sản phẩm: ≥ {MIN_WEEKS_FILTER} tuần có bán,\ncv giá ≥ {MIN_CV}", fontsize=10)
    ax3 = fig.add_subplot(gs[1, 1])
    v3 = [len(w), len(w_keep)]
    ax3.bar(["Dòng (sản phẩm × tuần)\ntrước lọc", "Dòng (sản phẩm × tuần)\nsau lọc"], v3, color=[MAU["nhat"], MAU["xanh"]], width=0.55)
    for i, v in enumerate(v3):
        ax3.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax3.set_ylim(0, max(v3) * 1.15)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax3.set_title("Bảng tuần dùng để huấn luyện", fontsize=10)
    for a in (ax2, ax3):
        a.tick_params(axis="x", labelsize=8)

    bang += [{"buoc": "Sản phẩm có bán", "so_dong_thay_doi": len(st), "ty_le_so_voi_goc_pct": np.nan, "so_dong_con_lai_sau_buoc": len(st)},
             {"buoc": "Sản phẩm đủ điều kiện", "so_dong_thay_doi": len(keep), "ty_le_so_voi_goc_pct": np.nan, "so_dong_con_lai_sau_buoc": len(keep)},
             {"buoc": "Dòng sản phẩm×tuần trước lọc", "so_dong_thay_doi": len(w), "ty_le_so_voi_goc_pct": np.nan, "so_dong_con_lai_sau_buoc": len(w)},
             {"buoc": "Dòng sản phẩm×tuần sau lọc", "so_dong_thay_doi": len(w_keep), "ty_le_so_voi_goc_pct": np.nan, "so_dong_con_lai_sau_buoc": len(w_keep)}]
    H.luu(fig, "figA2_lam_sach_du_lieu", pd.DataFrame(bang))


# ----------------------------------------------------------------------------- B1
def ve_b1():
    print("[B1] Phân phối lượng bán theo tuần...")
    w = load_weekly()[0]
    q = w["qty"].values.astype(float)
    p = {k: float(np.percentile(q, k)) for k in (50, 75, 90, 95, 99)}
    dev2 = (q - q.mean()) ** 2
    top = q >= p[95]
    chia_dev = float(dev2[top].sum() / dev2.sum() * 100)
    chia_qty = float(q[top].sum() / q.sum() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.hist(np.log(q), bins=45, color=MAU["xanh"], alpha=0.85)
    ax.axvline(np.log(np.median(q)), color=MAU["do"], ls="--", label=f"Trung vị = {np.median(q):.0f}")
    ax.axvline(np.log(q.mean()), color=MAU["cam"], ls="--", label=f"Trung bình = {q.mean():.1f}")
    ax.set_xlabel("ln(lượng bán theo tuần)")
    ax.set_ylabel("Số dòng (sản phẩm × tuần)")
    ax.set_title("Phân phối của ln(lượng bán)")
    ax.legend()
    ax = axes[1]
    ax.hist(q[q <= p[99]], bins=50, color=MAU["xanh"], alpha=0.85)
    ax.set_yscale("log")
    for k, mau in ((50, MAU["la"]), (90, MAU["cam"]), (99, MAU["do"])):
        ax.axvline(p[k], color=mau, ls="--", label=f"P{k} = {p[k]:.0f}")
    ax.set_xlabel(f"Lượng bán theo tuần (đã cắt ở P99 = {p[99]:.0f}; lớn nhất = {q.max():.0f})")
    ax.set_ylabel("Số dòng (thang log)")
    ax.set_title("Phân phối của lượng bán (trục dọc thang log)")
    ax.legend()
    fig.text(0.5, -0.04, f"5% số tuần bán nhiều nhất chiếm {chia_qty:.0f}% tổng lượng bán và {chia_dev:.0f}% tổng bình phương độ lệch so với trung bình "
             "-> sai số ở các tuần này đóng góp nhiều vào RMSE (bình phương), ít hơn vào MAE.", ha="center", fontsize=9)
    bang = pd.DataFrame([{"chi_tieu": f"P{k}", "gia_tri": v} for k, v in p.items()] + [
        {"chi_tieu": "trung_binh", "gia_tri": float(q.mean())}, {"chi_tieu": "lon_nhat", "gia_tri": float(q.max())},
        {"chi_tieu": "so_dong", "gia_tri": len(q)},
        {"chi_tieu": "ty_le_tong_luong_ban_cua_5pct_tuan_cao_nhat_pct", "gia_tri": chia_qty},
        {"chi_tieu": "ty_le_tong_binh_phuong_do_lech_cua_5pct_tuan_cao_nhat_pct", "gia_tri": chia_dev}])
    H.luu(fig, "figB1_phan_bo_luong_ban", bang)
    cnt, edges = np.histogram(np.log(q), bins=45)
    H.luu_csv(pd.DataFrame({"ln_qty_tu": edges[:-1], "ln_qty_den": edges[1:], "so_dong": cnt}), "figB1_phan_bo_luong_ban_bin")


# ----------------------------------------------------------------------------- B3
def ve_b3():
    print("[B3] Số mức giá và số lần đổi giá của mỗi sản phẩm...")
    w = load_weekly()[0]
    ev = H.su_kien_doi_gia(w, {})
    n_ev = ev.groupby("sku").size().rename("so_lan_doi_gia")
    ct = w.groupby("sku").agg(so_muc_gia=("price", "nunique"), so_tuan=("price", "size")).join(n_ev).fillna({"so_lan_doi_gia": 0}).reset_index()
    p_p = path("products_online_retail.csv")
    if os.path.exists(p_p):
        ct = ct.merge(pd.read_csv(p_p, dtype={"sku": str})[["sku", "description"]], on="sku", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, cot, tieu_de, nhan in ((axes[0], "so_muc_gia", "Số mức giá khác nhau (giá trung vị theo tuần)", "Số mức giá của một sản phẩm"),
                                   (axes[1], "so_lan_doi_gia", "Số lần giá đổi ≥ 10% giữa hai tuần liên tiếp", "Số lần đổi giá của một sản phẩm")):
        v = ct[cot].values
        ax.hist(v, bins=range(0, int(v.max()) + 2), color=MAU["xanh"], alpha=0.85, align="left")
        ax.axvline(np.median(v), color=MAU["do"], ls="--", label=f"Trung vị = {np.median(v):.0f}")
        ax.set_xlabel(nhan)
        ax.set_ylabel("Số sản phẩm")
        ax.set_title(tieu_de, fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend()
    sw = int(ct["so_tuan"].median())
    ev_tv = float(ct["so_lan_doi_gia"].median())
    fig.text(0.5, -0.04, f"{len(ct)} sản phẩm; mỗi sản phẩm trung vị {sw} tuần có bán và {ev_tv:.0f} lần đổi giá ≥ 10% để ước lượng elasticity riêng. "
             "Ít lần đổi giá thì ước lượng riêng từng sản phẩm nhiễu (xem kiểm định ngoài mẫu ở bước 5).", ha="center", fontsize=9)
    H.luu(fig, "figB3_so_muc_gia", ct)


# ----------------------------------------------------------------------------- D3
def ve_d3():
    print("[D3] Đại diện của 20 sản phẩm...")
    w = load_weekly()[0]
    sp = pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})
    sp = sp.merge(w.groupby("sku")["price"].nunique().rename("so_muc_gia").reset_index(), on="sku", how="left")
    chon, nguon = H.nap_sku_chon()
    if chon is None:
        sap = sp.sort_values("price_median")
        chon = sap.iloc[np.linspace(0, len(sap) - 1, 20).round().astype(int)]["sku"].tolist()
        nguon = "tự chọn 20 sản phẩm cách đều theo giá (không thấy file danh sách)"
        print("  CẢNH BÁO: không thấy selected_products_online_retail.csv, dùng 20 sản phẩm cách đều theo giá")
    sp["duoc_chon"] = sp["sku"].isin(chon)
    c, k = sp[sp["duoc_chon"]], sp[~sp["duoc_chon"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.scatter(k["weeks"], k["so_muc_gia"], s=12, color=MAU["nhat"], alpha=0.6, label=f"Toàn bộ ({len(sp)} sản phẩm)")
    ax.scatter(c["weeks"], c["so_muc_gia"], s=45, color=MAU["do"], edgecolor="white", label=f"{len(c)} sản phẩm được chọn")
    ax.set_xlabel("Số tuần có bán")
    ax.set_ylabel("Số mức giá khác nhau")
    ax.set_title("Chất lượng dữ liệu của từng sản phẩm")
    ax.legend()
    ax = axes[1]
    ax.scatter(k["price_median"], k["qty_mean"], s=12, color=MAU["nhat"], alpha=0.6)
    ax.scatter(c["price_median"], c["qty_mean"], s=45, color=MAU["do"], edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Giá trung vị (thang log)")
    ax.set_ylabel("Lượng bán trung bình mỗi tuần (thang log)")
    ax.set_title("Phân khúc giá và mức bán")
    tv = lambda d, col: d[col].median()
    fig.text(0.5, -0.05, f"Trung vị của 20 sản phẩm chọn / toàn bộ: giá {tv(c, 'price_median'):.2f} / {tv(sp, 'price_median'):.2f}; "
             f"lượng bán {tv(c, 'qty_mean'):.0f} / {tv(sp, 'qty_mean'):.0f}; số mức giá {tv(c, 'so_muc_gia'):.0f} / {tv(sp, 'so_muc_gia'):.0f}. "
             f"Nguồn danh sách: {nguon}", ha="center", fontsize=9)
    H.luu(fig, "figD3_dai_dien_20_san_pham", sp[["sku", "description", "weeks", "so_muc_gia", "price_median", "qty_mean", "duoc_chon"]])


def main():
    print("=" * 70)
    print("VẼ BIỂU ĐỒ DỮ LIỆU")
    print("=" * 70)
    if not os.path.exists(path("weekly_online_retail.csv")):
        raise SystemExit("Không thấy models_2y/weekly_online_retail.csv. Chạy 1_data_processing_online_retail_2y.py trước.")
    ve_a2()
    ve_b1()
    ve_b3()
    ve_d3()
    H.cap_nhat_readme()
    print("Xong. Xem thư mục ./figures")


if __name__ == "__main__":
    main()
