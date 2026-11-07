"""
BƯỚC 6: BIỂU ĐỒ SO SÁNH CÁC MÔ HÌNH (không có ensemble)
====================================
Vẽ bốn nhóm biểu đồ, mỗi mô hình một màu cố định, cùng một chú giải:

  S1  (theo biểu đồ 1) Ba sản phẩm, trên các tuần TEST của cách xen kẽ: thực tế và dự báo của mọi mô hình.
  SA  (theo biểu đồ A) Tổng theo tuần trên tuần test, một ô cho mỗi cách chia: thực tế và tổng dự báo của mọi mô hình.
  SC  (theo biểu đồ C) MAE và RMSE của từng mô hình, ba cách chia (cột màu theo cách chia).
  SD  Đường cầu: lượng bán dự báo trung bình theo giá, cho từng mô hình (cách xen kẽ, mô hình huấn luyện trên train).
      Dùng để so HÌNH DẠNG, không phải sai số. Baseline sku_base phẳng vì không dùng giá.

Quy tắc: chỉ đọc kết quả đã có (so_sanh_out/chi_tiet, ket_qua_buoc4.csv), và huấn luyện lại
đúng một lần cho SD. Ghi PNG và CSV vào ./figures. Không ghi đè pipeline.

Mô hình thiếu thư viện hoặc thiếu file sẽ bị bỏ qua và in cảnh báo.
Cần: các bước 1 đến 5 đã chạy.
"""
import os
import warnings

import numpy as np
import pandas as pd

import ham_chung_bieu_do as H
import khung_so_sanh_2y as K
import so_sanh_buoc4 as B4
import ve_ket_qua as V
from ham_chung_bieu_do import MAU, plt

warnings.filterwarnings("ignore")

CHI_TIET = os.path.join(K.OUT, "chi_tiet")
MO_HINH_VE = ["sku_base", "ridge", "random_forest", "lightgbm", "catboost", "gam", "xgboost"]
TEN_HIEN = {"sku_base": "sku_base (nền)", "ridge": "Ridge", "random_forest": "Random Forest",
            "lightgbm": "LightGBM", "catboost": "CatBoost", "gam": "ExpectileGAM", "xgboost": "XGBoost (mốc)"}
MAU_MH = {"sku_base": "#a0aec0", "ridge": "#dd6b20", "random_forest": "#2f855a", "lightgbm": "#805ad5",
          "catboost": "#c53030", "gam": "#319795", "xgboost": "#2b6cb0"}
MAU_CACH = {"xen_ke": "#2b6cb0", "chieu_1_2": "#dd6b20", "chieu_2_1": "#2f855a"}
MO_TA_MOI = [
    ("S1", "figS1_so_sanh_theo_tuan", "So sánh các mô hình theo tuần (3 sản phẩm)", "so_sanh_out/chi_tiet",
     "Mô hình nào bám thực tế tốt ở từng loại sản phẩm."),
    ("SA", "figSA_so_sanh_tong_theo_tuan", "So sánh tổng theo tuần giữa các mô hình", "so_sanh_out/chi_tiet",
     "Mô hình nào bám xu hướng tổng thể, theo từng cách chia."),
    ("SC", "figSC_so_sanh_mae_rmse", "MAE và RMSE các mô hình, ba cách chia", "so_sanh_out/ket_qua_buoc4.csv",
     "Thứ hạng mô hình có ổn định qua các cách chia không."),
    ("SD", "figSD_so_sanh_duong_cau", "Đường cầu theo giá của các mô hình", "khung + các mô hình (xen kẽ)",
     "Hình dạng đường cầu: trơn, đơn điệu, độ dốc."),
]


def doc_chi_tiet(ten, chia):
    p = os.path.join(CHI_TIET, f"{ten}_{chia}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["week"])


def mau_mo_hinh(ten):
    return MAU_MH.get(ten, "#000000")


# ----------------------------------------------------------------------------- S1
def ve_S1(danh_sach, t):
    print("[S1] So sánh theo tuần, 3 sản phẩm...")
    chon = V.chon_3_san_pham(t)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    bang = []
    for ax, (ten_nhom, r) in zip(axes, chon):
        sku = r["sku"]
        thuc = None
        for ten in danh_sach:
            ct = doc_chi_tiet(ten, "xen_ke")
            if ct is None:
                continue
            c = ct[ct["sku"] == sku].sort_values("week")
            if thuc is None and len(c):
                thuc = c[["week", "thuc_te"]]
                ax.plot(thuc["week"], thuc["thuc_te"], color="black", lw=1.4, marker="x", ms=5, label="Thực tế")
            ax.plot(c["week"], c["du_bao"], color=mau_mo_hinh(ten), lw=1.2, marker="o", ms=3,
                    label=TEN_HIEN.get(ten, ten))
            mae = float(np.mean(np.abs(c["du_bao"] - c["thuc_te"]))) if len(c) else np.nan
            bang.append({"san_pham": sku, "nhom": ten_nhom, "mo_hinh": ten, "mae_tuan_test": mae})
        ax.set_title(f"{sku} ({ten_nhom}; TB = {r['tb']:.1f}, CV = {r['cv']:.2f})", fontsize=10)
        ax.set_ylabel("Lượng bán / tuần")
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    fig.tight_layout()
    H.luu(fig, "figS1_so_sanh_theo_tuan", pd.DataFrame(bang))


# ----------------------------------------------------------------------------- SA
def ve_SA(danh_sach):
    print("[SA] So sánh tổng theo tuần, ba cách chia...")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    bang = []
    for ax, chia in zip(axes, K.CACH_CHIA):
        thuc_da_ve = False
        for ten in danh_sach:
            ct = doc_chi_tiet(ten, chia)
            if ct is None:
                continue
            g = ct.groupby("week").agg(thuc=("thuc_te", "sum"), du=("du_bao", "sum")).reset_index()
            if not thuc_da_ve:
                ax.plot(g["week"], g["thuc"], color="black", lw=2, marker="x", ms=4, label="Thực tế")
                thuc_da_ve = True
            ax.plot(g["week"], g["du"], color=mau_mo_hinh(ten), lw=1.4, marker="o", ms=3,
                    label=TEN_HIEN.get(ten, ten))
            lech = (g["du"].sum() / g["thuc"].sum() - 1) * 100
            bang.append({"cach_chia": chia, "mo_hinh": ten, "lech_tong_pct": lech})
        ax.set_title(K.CO.TEN_CHIA[chia], fontsize=10)
        ax.set_ylabel("Tổng lượng bán / tuần")
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.4))
    fig.tight_layout()
    H.luu(fig, "figSA_so_sanh_tong_theo_tuan", pd.DataFrame(bang))


# ----------------------------------------------------------------------------- SC
def ve_SC(ket, danh_sach):
    print("[SC] MAE và RMSE các mô hình...")
    ket = ket.copy()
    ket["mo_hinh"] = ket["mo_hinh"].astype(str)
    ds = [m for m in danh_sach if m in set(ket["mo_hinh"])]
    ket = ket[ket["mo_hinh"].isin(ds)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(ds))
    rong = 0.26
    for ax, cot, ten in ((axes[0], "mae", "MAE theo số lượng"), (axes[1], "rmse", "RMSE theo số lượng")):
        for j, chia in enumerate(K.CACH_CHIA):
            vals = []
            for m in ds:
                r = ket[(ket["mo_hinh"] == m) & (ket["cach_chia"] == chia)]
                vals.append(float(r.iloc[0][cot]) if len(r) and pd.notna(r.iloc[0][cot]) else np.nan)
            ax.bar(x + (j - 1) * rong, vals, width=rong, color=MAU_CACH[chia], label=K.CO.TEN_CHIA[chia])
        ax.set_xticks(x)
        ax.set_xticklabels([TEN_HIEN.get(m, m) for m in ds], rotation=25, ha="right", fontsize=9)
        ax.set_title(ten)
        ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "Thấp hơn là tốt hơn.",
             ha="center", fontsize=9)
    fig.tight_layout()
    H.luu(fig, "figSC_so_sanh_mae_rmse", ket[["mo_hinh", "cach_chia", "mae", "rmse", "r2_ln", "lech_tong_pct"]])


# ----------------------------------------------------------------------------- SD
def ve_SD(du, danh_sach):
    print("[SD] Đường cầu của các mô hình (huấn luyện trên train, cách xen kẽ)...")
    trn, tst, tb = K.tach(du, "xen_ke")
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    tong = []
    for ten in danh_sach:
        if ten not in K.MO_HINH:
            print(f"    bỏ qua {ten}: chưa đăng ký")
            continue
        try:
            m = K.MO_HINH[ten](trn)
        except Exception as e:
            print(f"    bỏ qua {ten}: {type(e).__name__}: {str(e)[:80]}")
            continue
        df, tang = B4.duong_hieu_ung_gia(m, tst)
        ax.plot(df["gia_tuong_doi"], df["luong_ban_tb"], color=mau_mo_hinh(ten), lw=2,
                ls="--" if ten == "sku_base" else "-", label=f"{TEN_HIEN.get(ten, ten)} (điểm tăng: {tang})")
        d = df.copy()
        d.insert(0, "mo_hinh", ten)
        tong.append(d)
    ax.set_xlabel("Giá tương đối (p / p0 trung vị của sản phẩm)")
    ax.set_ylabel("Lượng bán dự báo trung bình")
    ax.set_title("Đường cầu theo giá: hình dạng của từng mô hình (cách xen kẽ)")
    ax.legend(fontsize=9)
    fig.text(0.5, -0.02, "Trung bình trên 2000 dòng test, thay đổi giá và giữ các đặc trưng khác. "
             "Đường có bậc là cây; đường trơn là GAM hoặc Ridge; sku_base phẳng vì không dùng giá.",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    H.luu(fig, "figSD_so_sanh_duong_cau", pd.concat(tong, ignore_index=True) if tong else pd.DataFrame())


def main():
    print("=" * 78)
    print("BƯỚC 6: BIỂU ĐỒ SO SÁNH CÁC MÔ HÌNH")
    print("=" * 78)
    buoc4 = os.path.join(K.OUT, "ket_qua_buoc4.csv")
    if not os.path.exists(buoc4):
        raise SystemExit("Thiếu ket_qua_buoc4.csv. Chạy so_sanh_buoc4.py trước.")
    ket = pd.read_csv(buoc4)

    # bảng bước 6: các mô hình có chi_tiet để vẽ
    danh_sach = [m for m in MO_HINH_VE if doc_chi_tiet(m, "xen_ke") is not None]
    print(f"  Mô hình sẽ vẽ: {danh_sach}")

    t = H.nap_bang_mo_hinh()
    ve_S1(danh_sach, t)
    ve_SA(danh_sach)
    ve_SC(ket, MO_HINH_VE)

    du = K.nap_du_lieu()
    ve_SD(du, MO_HINH_VE)

    H.MO_TA.extend([(ma, ten, tieu, nguon, y) for ma, ten, tieu, nguon, y in MO_TA_MOI])
    H.cap_nhat_readme()
    print("Xong. Xem ./figures (các file figS*.png và .csv)")


if __name__ == "__main__":
    main()
