"""
BƯỚC 6: HÌNH VẼ THEO TỪNG CÁCH CHIA
===================================
Cấu trúc đầu ra (đã thống nhất):
  models_2y/xen_ke/figures/       fig01, A, C, fig06, fig10
  models_2y/chieu_1_2/figures/    fig01, A, C, fig06, fig10
  models_2y/chieu_2_1/figures/    fig01, A, C, fig06, fig10
  models_2y/chung/figures/        D (elasticity thô, không phụ thuộc cách chia)

Cách làm: các hàm vẽ (ve_ket_qua, ve_them_2y, ham_chung_bieu_do) đọc đường dẫn qua y2_config.path.
Mỗi phạm vi (scope) được chạy trong một tiến trình con, với y2_config.path đã trỏ về thư mục của phạm vi
đó TRƯỚC khi nạp các module vẽ. Nhờ vậy không sửa code hàm vẽ gốc.

Hình C được vẽ lại theo từng cách chia: đọc kết quả so sánh mô hình (so_sanh_out/ket_qua_buoc4.csv),
lọc đúng cách chia, mỗi mô hình một cột. Các mô hình khác được thêm vào đúng như đã thống nhất.

Lưu ý:
  - Hình C cần so_sanh_out/ket_qua_buoc4.csv (kết quả bước so sánh, mốc XGBoost 273 cây).
  - Danh sách sản phẩm vẽ fig06, fig10 lấy từ selected_products_online_retail.csv ở thư mục gốc models_2y
    (nếu có), nếu không thì lấy 20 sản phẩm đầu của pricing_params.

Chạy:  python buoc6_hinh_ve.py
"""
import json
import os
import shutil
import subprocess
import sys

PHAM_VI = ["xen_ke", "chieu_1_2", "chieu_2_1", "chung"]
GOC = "./models_2y"
KET_QUA_SO_SANH = os.path.join("so_sanh_out", "ket_qua_buoc4.csv")


def chay_con(pham_vi):
    """Chạy một phạm vi trong tiến trình con. Trả về (mã thoát, đầu ra)."""
    moi_truong = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    ket = subprocess.run([sys.executable, __file__, "--pham-vi", pham_vi], env=moi_truong,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return ket.returncode, (ket.stdout + ket.stderr)


def cau_hinh_duong_dan(pham_vi):
    """Cấu hình y2_config TRƯỚC khi nạp module vẽ."""
    import y2_config
    y2_config.OUT_DIR = GOC                       # gốc cho khung_cach_chia
    if pham_vi == "chung":
        thu_muc = os.path.join(GOC, "chung")
    else:
        thu_muc = os.path.join(GOC, pham_vi)
    y2_config.path = lambda ten: os.path.join(thu_muc, ten)
    return thu_muc


def ve_C_theo_cach(cach, figdir, H):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    if not os.path.exists(KET_QUA_SO_SANH):
        print(f"    Thiếu {KET_QUA_SO_SANH}: bỏ qua hình C (chỉ có mốc XGBoost)")
        return
    df = pd.read_csv(KET_QUA_SO_SANH)
    df = df[df["cach_chia"] == cach].sort_values("mae")
    if df.empty:
        print(f"    Không có dòng nào của {cach} trong {KET_QUA_SO_SANH}")
        return
    ten = {"xgboost": "XGBoost (mốc)", "sku_base": "sku_base", "ridge": "Ridge", "random_forest": "Random Forest",
           "lightgbm": "LightGBM", "catboost": "CatBoost", "gam": "ExpectileGAM"}
    mau = {"xgboost": H.MAU["xanh"], "sku_base": H.MAU["xam"], "ridge": H.MAU["cam"],
           "random_forest": "#2f855a", "lightgbm": "#805ad5", "catboost": H.MAU["do"], "gam": "#319795"}
    nhan = [ten.get(m, m) for m in df["mo_hinh"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, cot, tieu_de in ((axes[0], "mae", "MAE theo số lượng"), (axes[1], "rmse", "RMSE theo số lượng")):
        vals = df[cot].values
        ax.bar(np.arange(len(vals)), vals, color=[mau.get(m, "#888") for m in df["mo_hinh"]])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(nhan, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"{tieu_de} - {H.TEN_CACH_NGAN.get(cach, cach) if hasattr(H, 'TEN_CACH_NGAN') else cach}")
    fig.text(0.5, -0.02, "Thấp hơn là tốt hơn. Chênh lệch có ý nghĩa cần đọc qua khoảng bootstrap trong bảng kết quả.",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, f"figC_mo_hinh_so_baseline_{cach}", df[["mo_hinh", "cach_chia", "mae", "rmse"]])


def chay_pham_vi(pham_vi):
    thu_muc = cau_hinh_duong_dan(pham_vi)
    import ham_chung_bieu_do as H
    figdir = os.path.join(thu_muc, "figures")
    H.FIG_DIR = figdir
    os.makedirs(figdir, exist_ok=True)
    print(f"[{pham_vi}] ghi hình vào {figdir}")

    import numpy as np
    import pandas as pd
    import ve_ket_qua as V
    import ve_them_2y as VT

    if pham_vi == "chung":
        from y2_tools import load_weekly
        w_all = load_weekly()[0]
        VT.ve_D(w_all)
        return

    model, feats = H.nap_mo_hinh()
    t = H.du_bao(H.nap_bang_mo_hinh(), model, feats)
    te = t[t["is_test"]].copy()
    q10, q90 = np.percentile(te["lq"] - te["du_bao_ln"], [10, 90])
    V.ve_01(t, q10, q90)
    VT.ve_A(t)
    ve_C_theo_cach(pham_vi, figdir, H)

    # engine cần năm file: elasticity dùng chung và mô hình của cách chia này
    from pricing_engine import PricingEngine
    tam = os.path.join(GOC, f"_tam_hinh_{pham_vi}")
    if os.path.exists(tam):
        shutil.rmtree(tam)
    os.makedirs(tam)
    try:
        for f in ["pricing_params_online_retail.csv", "pricing_model_online_retail.json"]:
            shutil.copy2(os.path.join(GOC, "chung", f), os.path.join(tam, f))
        for f in ["demand_model_online_retail.pkl", "feature_cols_online_retail.json",
                  "product_stats_online_retail.json"]:
            shutil.copy2(os.path.join(GOC, pham_vi, f), os.path.join(tam, f))
        eng = PricingEngine(tam)

        goc_sku = os.path.join(GOC, "selected_products_online_retail.csv")
        if os.path.exists(goc_sku):
            sku_chon = pd.read_csv(goc_sku, dtype={"sku": str})["sku"].tolist()
        else:
            sku_chon = pd.read_csv(os.path.join(GOC, "chung", "pricing_params_online_retail.csv"),
                                   dtype={"sku": str})["sku"].head(20).tolist()
        ds4 = V.chon_4_sku_engine(eng, sku_chon)
        V.ve_06(eng, ds4)
        V.ve_10(eng, ds4)
    finally:
        shutil.rmtree(tam, ignore_errors=True)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--pham-vi":
        for luong in (sys.stdout, sys.stderr):
            try:
                luong.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        chay_pham_vi(sys.argv[2])
        return

    print("=" * 78)
    print("BƯỚC 6: HÌNH VẼ THEO TỪNG CÁCH CHIA")
    print("=" * 78)
    if not os.path.isdir(GOC):
        raise SystemExit(f"Thiếu thư mục {GOC}. Chạy các bước 2 đến 5 trước.")
    loi = []
    for pv in PHAM_VI:
        ma, out = chay_con(pv)
        print(out.strip().splitlines()[-1] if out.strip() else "")
        if ma != 0:
            loi.append((pv, out[-600:]))
    if loi:
        print("\nLỖI:")
        for pv, msg in loi:
            print(f"--- {pv}\n{msg}")
        sys.exit(1)
    print("\nXong. Hình nằm trong models_2y/<cách chia>/figures và models_2y/chung/figures.")


if __name__ == "__main__":
    main()
