"""
BƯỚC 5: FILE EXCEL THEO LOẠI DỮ LIỆU VÀ CÁCH CHIA
==================================================
Mỗi loại dữ liệu có một file Excel cho từng cách chia (ba loại x ba cách chia = chín file):

  du_bao_<cách chia>.xlsx      Mô hình dự báo: chỉ số train/test, đặc trưng, sai số theo sản phẩm
  elasticity_<cách chia>.xlsx  Elasticity từng sản phẩm (bản dùng chung từ chung/, sao vào mỗi cách chia)
  gia_<cách chia>.xlsx         Giá đề xuất từng sản phẩm, có định dạng và tóm tắt

Lưu ý: elasticity lấy từ toàn bộ dữ liệu nên giống hệt nhau ở ba cách chia. Ba bản là để giữ đúng
cấu trúc "mỗi loại có ba cách chia" như đã thống nhất. Nội dung giống nhau, không phải lỗi.

Đầu ra: models_2y/<cách chia>/excel/ (không ghi đè file CSV hay JSON gốc).
Cần: bước 2, 3, 4 đã chạy. Cần thư viện: openpyxl (đã có khi dùng pandas Excel).

Chạy:  python buoc5_excel.py
"""
import json
import math
import os
import pickle

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import khung_cach_chia as KC

TEN_CACH = {"xen_ke": "Xen kẽ", "chieu_1_2": "Chiều 1→2", "chieu_2_1": "Chiều 2→1"}

VIEN = Border(*(Side(style="thin", color="D0D7E2"),) * 4)
NEN_TIEU_DE = PatternFill("solid", fgColor="1F4E79")
FONT_TIEU_DE = Font(bold=True, color="FFFFFF", size=10)
FONT_TIEU_DE_LON = Font(bold=True, size=13, color="1F4E79")
FONT_GHI_CHU = Font(italic=True, size=9, color="555555")
NEN_XEN = PatternFill("solid", fgColor="F7F9FC")


def ghi_bang(ws, df, bat_dau_dong, cot_so=(), dinh_dang=None):
    """Ghi DataFrame vào sheet bắt đầu từ một dòng, định dạng header, viền, số, cố định header."""
    dinh_dang = dinh_dang or {}
    for j, ten in enumerate(df.columns, 1):
        c = ws.cell(row=bat_dau_dong, column=j, value=ten)
        c.fill, c.font, c.border = NEN_TIEU_DE, FONT_TIEU_DE, VIEN
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = max(12, min(40, len(str(ten)) + 4))
    ws.row_dimensions[bat_dau_dong].height = 34
    for i, (_, row) in enumerate(df.iterrows(), 1):
        for j, ten in enumerate(df.columns, 1):
            v = row[ten]
            if isinstance(v, (float, np.floating)) and (pd.isna(v) or math.isinf(v)):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            c = ws.cell(row=bat_dau_dong + i, column=j, value=v)
            c.border = VIEN
            c.alignment = Alignment(vertical="center", horizontal="right" if ten in cot_so else "left")
            if ten in dinh_dang and v is not None:
                c.number_format = dinh_dang[ten]
            if i % 2 == 0:
                c.fill = NEN_XEN
    ws.freeze_panes = ws.cell(row=bat_dau_dong + 1, column=1)
    return bat_dau_dong + len(df)


def tieu_de(ws, text, ghi_chu):
    ws["A1"] = text
    ws["A1"].font = FONT_TIEU_DE_LON
    ws["A2"] = ghi_chu
    ws["A2"].font = FONT_GHI_CHU
    ws.sheet_view.showGridLines = False


# ----------------------------------------------------------------------------- 1. dự báo
def xay_du_bao(cach, thu_muc, duong_dan):
    with open(os.path.join(thu_muc, "model_metrics_online_retail.json"), encoding="utf-8") as f:
        met = json.load(f)
    with open(os.path.join(thu_muc, "feature_cols_online_retail.json"), encoding="utf-8") as f:
        feats = json.load(f)
    with open(os.path.join(thu_muc, "demand_model_online_retail.pkl"), "rb") as f:
        model = pickle.load(f)
    tbl = pd.read_csv(os.path.join(thu_muc, "model_table_online_retail.csv"), parse_dates=["week"],
                      dtype={"sku": str})
    X = np.load(os.path.join(thu_muc, "X_online_retail.npy"))
    pred_qty = np.exp(model.predict(X))
    tbl = tbl.assign(du_bao=pred_qty)
    te = tbl[tbl["is_test"].astype(bool)]

    chi_so = pd.DataFrame([
        ("Tập train", "R² (ln)", met.get("train_r2")), ("Tập train", "MAE (ln)", met.get("train_mae_ln")),
        ("Tập train", "RMSE (ln)", met.get("train_rmse_ln")),
        ("Tập test", "R² (ln)", met.get("test_r2")), ("Tập test", "MAE (ln)", met.get("test_mae_ln")),
        ("Tập test", "RMSE (ln)", met.get("test_rmse_ln")),
        ("Tập test", "MAE (số lượng)", met.get("test_mae_qty")),
        ("Baseline (sku_base)", "R² (ln)", met.get("baseline_r2")),
        ("Baseline (sku_base)", "MAE (ln)", met.get("baseline_mae_ln")),
        ("Baseline (sku_base)", "MAE (số lượng)", met.get("baseline_mae_qty")),
        ("Phản ứng giá", "Elasticity TB (đổi giá ±10%)", met.get("elasticity_pm10")),
        ("Phản ứng giá", "% dòng không phản ứng", met.get("pct_flat_pm10")),
    ], columns=["Nhóm", "Chỉ số", "Giá trị"])

    dac_trung = pd.DataFrame({"Đặc trưng": feats, "Độ quan trọng": model.feature_importances_})
    dac_trung = dac_trung.sort_values("Độ quan trọng", ascending=False).reset_index(drop=True)
    dac_trung.insert(0, "Hạng", range(1, len(dac_trung) + 1))

    theo_sku = te.groupby("sku").apply(
        lambda g: pd.Series({
            "Số tuần test": len(g),
            "Thực tế TB": g["qty"].mean(),
            "Dự báo TB": g["du_bao"].mean(),
            "MAE": (g["du_bao"] - g["qty"]).abs().mean(),
            "Lệch tổng %": (g["du_bao"].sum() / g["qty"].sum() - 1) * 100,
        }), include_groups=False).reset_index().rename(columns={"sku": "Mã sản phẩm"})

    wb = Workbook()
    ws = wb.active
    ws.title = "Chỉ số"
    tieu_de(ws, f"Mô hình dự báo - {TEN_CACH[cach]}", f"Cách chia: {KC.CACH_CHIA[cach]}. "
            "R² (ln): thang log. MAE (số lượng): cái/tuần.")
    ghi_bang(ws, chi_so, 4, cot_so=("Giá trị",), dinh_dang={"Giá trị": "0.000"})
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 34

    ws2 = wb.create_sheet("Đặc trưng")
    tieu_de(ws2, "Độ quan trọng đặc trưng", "Tổng các mức chia của cây. Chỉ mô tả, không phải nhân quả.")
    ghi_bang(ws2, dac_trung, 4, cot_so=("Độ quan trọng",), dinh_dang={"Độ quan trọng": "0.0000"})

    ws3 = wb.create_sheet("Theo sản phẩm")
    tieu_de(ws3, "Sai số theo sản phẩm (tập test)", "MAE và lệch tổng tính trên các tuần test của từng sản phẩm.")
    theo_sku = theo_sku.sort_values("MAE", ascending=False)
    end = ghi_bang(ws3, theo_sku, 4, cot_so=tuple(theo_sku.columns[1:]),
                   dinh_dang={"Thực tế TB": "0.0", "Dự báo TB": "0.0", "MAE": "0.00", "Lệch tổng %": "0.0"})
    if end > 4:
        ws3.conditional_formatting.add(f"D5:D{end}", ColorScaleRule(
            start_type="min", start_color="C6EFCE", mid_type="percentile", mid_value=50,
            mid_color="FFFFFF", end_type="max", end_color="FFC7CE"))
    wb.save(duong_dan)


# ----------------------------------------------------------------------------- 2. elasticity
def xay_elasticity(cach, duong_dan):
    from math import erf, sqrt
    pr = pd.read_csv(os.path.join(KC.thu_muc_chung(), "pricing_params_online_retail.csv"), dtype={"sku": str})
    with open(os.path.join(KC.thu_muc_chung(), "pricing_model_online_retail.json"), encoding="utf-8") as f:
        pm = json.load(f)

    def phi(z):
        return 0.5 * (1 + erf(z / sqrt(2)))

    pr = pr.assign(**{"Xác suất cầu co giãn": [phi((-1 - m) / s) if s > 0 else np.nan
                                             for m, s in zip(pr["eps_mean"], pr["eps_sd"])]})
    cot = {"sku": "Mã sản phẩm", "description": "Mô tả", "n_weeks": "Số tuần",
           "eps_raw": "Elasticity thô (b)", "se_raw": "se của b", "prior_mean": "Prior (mức chung)",
           "weight_own": "Trọng số riêng (w)", "eps_mean": "Elasticity dùng (m)", "eps_sd": "Độ bất định (sd)"}
    pr = pr[[c for c in cot if c in pr.columns] + ["Xác suất cầu co giãn"]].rename(columns=cot)

    thong_tin = pd.DataFrame([
        ("Phương pháp", pm.get("method")), ("τ² (tau2)", pm.get("tau2")), ("σ (sigma_extra)", pm.get("sigma_extra")),
        ("Elasticity chung (g)", pm.get("global_elasticity")), ("Hệ số scale", pm.get("scale")),
        ("Số sản phẩm", len(pr)),
    ], columns=["Thông số", "Giá trị"])

    wb = Workbook()
    ws = wb.active
    ws.title = "Elasticity"
    tieu_de(ws, f"Elasticity theo sản phẩm - {TEN_CACH[cach]}",
            "Lấy từ toàn bộ dữ liệu hai năm, giống nhau ở mọi cách chia. Xác suất = P(ε < −1).")
    dinh = {c: "0.000" for c in pr.columns if c not in ("Mã sản phẩm", "Mô tả", "Số tuần")}
    dinh["Xác suất cầu co giãn"] = "0.0%"
    dinh["Số tuần"] = "0"
    ghi_bang(ws, pr, 4, cot_so=tuple(c for c in pr.columns if c not in ("Mã sản phẩm", "Mô tả")), dinh_dang=dinh)
    ws.column_dimensions["B"].width = 34
    ws2 = wb.create_sheet("Thông số")
    tieu_de(ws2, "Thông số ước lượng elasticity", "Đọc từ pricing_model_online_retail.json.")
    ghi_bang(ws2, thong_tin, 4, cot_so=("Giá trị",))
    ws2.column_dimensions["A"].width = 26
    ws2.column_dimensions["B"].width = 20
    wb.save(duong_dan)


# ----------------------------------------------------------------------------- 3. giá đề xuất
def xay_gia(cach, thu_muc, duong_dan):
    gia = pd.read_csv(os.path.join(thu_muc, "gia_de_xuat_online_retail.csv"), dtype={"sku": str})
    cot_hien = ["sku", "mo_ta", "gia_hien_tai", "gia_de_xuat", "gia_muc_tieu", "doi_gia_pct", "hanh_dong",
                "do_manh", "xac_suat_co_loi_de_xuat", "ky_vong_thay_doi_pct", "p10_thay_doi_pct",
                "p90_thay_doi_pct", "elasticity_tb", "xac_suat_co_gian", "doanh_thu_hien_tai",
                "doanh_thu_de_xuat", "canh_bao"]
    ten_hien = {"sku": "Mã sản phẩm", "mo_ta": "Mô tả", "gia_hien_tai": "Giá hiện tại",
                "gia_de_xuat": "Giá đề xuất", "gia_muc_tieu": "Giá mục tiêu", "doi_gia_pct": "Đổi giá %",
                "hanh_dong": "Hành động", "do_manh": "Độ mạnh", "xac_suat_co_loi_de_xuat": "P có lợi",
                "ky_vong_thay_doi_pct": "Kỳ vọng DT %", "p10_thay_doi_pct": "P10 %", "p90_thay_doi_pct": "P90 %",
                "elasticity_tb": "Elasticity", "xac_suat_co_gian": "P co giãn",
                "doanh_thu_hien_tai": "DT hiện tại", "doanh_thu_de_xuat": "DT đề xuất", "canh_bao": "Cảnh báo"}
    bang = gia[[c for c in cot_hien if c in gia.columns]].rename(columns=ten_hien)
    bang = bang.sort_values("Đổi giá %")

    tom = gia.assign(hanh=gia["hanh_dong"]).groupby("hanh_dong").size().rename("Số sản phẩm").reset_index()
    dt_hien, dt_de = gia["doanh_thu_hien_tai"].sum(), gia["doanh_thu_de_xuat"].sum()
    tong = pd.DataFrame([
        ("Số sản phẩm", len(gia)),
        ("Giữ giá", int((gia["hanh_dong"] == "keep").sum())),
        ("Tăng giá", int((gia["hanh_dong"] == "raise").sum())),
        ("Giảm giá", int((gia["hanh_dong"] == "lower").sum())),
        ("Độ mạnh strong", int((gia["do_manh"] == "strong").sum())),
        ("Doanh thu hiện tại (ước lượng)", round(dt_hien, 2)),
        ("Doanh thu đề xuất (ước lượng)", round(dt_de, 2)),
        ("Chênh lệch %", round((dt_de / dt_hien - 1) * 100, 2) if dt_hien else None),
    ], columns=["Chỉ tiêu", "Giá trị"])

    wb = Workbook()
    ws = wb.active
    ws.title = "Giá đề xuất"
    ngay = gia["ngay_ap_dung"].iloc[0] if "ngay_ap_dung" in gia.columns and len(gia) else ""
    tieu_de(ws, f"Giá đề xuất - {TEN_CACH[cach]} (ngày {ngay})",
            "Doanh thu là ước lượng theo mô hình, không phải doanh thu thật. Sắp theo đổi giá %.")
    dinh = {"Giá hiện tại": "0.00", "Giá đề xuất": "0.00", "Giá mục tiêu": "0.00", "Đổi giá %": "0.0",
            "P có lợi": "0.0%", "Kỳ vọng DT %": "0.0", "P10 %": "0.0", "P90 %": "0.0",
            "Elasticity": "0.00", "P co giãn": "0.0%", "DT hiện tại": "0.00", "DT đề xuất": "0.00"}
    end = ghi_bang(ws, bang, 4, cot_so=tuple(c for c in bang.columns if c not in ("Mã sản phẩm", "Mô tả",
                                                                                   "Hành động", "Độ mạnh", "Cảnh báo")),
                   dinh_dang=dinh)
    if end > 4 and "Đổi giá %" in bang.columns:
        cot = get_column_letter(list(bang.columns).index("Đổi giá %") + 1)
        ws.conditional_formatting.add(f"{cot}5:{cot}{end}", ColorScaleRule(
            start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
            mid_color="FFFFFF", end_type="max", end_color="63BE7B"))
    ws.column_dimensions["B"].width = 34
    ws2 = wb.create_sheet("Tóm tắt")
    tieu_de(ws2, f"Tóm tắt - {TEN_CACH[cach]}", "Số liệu tổng hợp từ bảng giá đề xuất.")
    ghi_bang(ws2, tong, 4, cot_so=("Giá trị",))
    ws2.column_dimensions["A"].width = 34
    ws2.column_dimensions["B"].width = 18
    ghi_bang(ws2, tom.rename(columns={"hanh_dong": "Hành động"}), 16, cot_so=("Số sản phẩm",))
    wb.save(duong_dan)


def main():
    print("=" * 78)
    print("BƯỚC 5: FILE EXCEL THEO LOẠI DỮ LIỆU VÀ CÁCH CHIA")
    print("=" * 78)
    for cach in KC.CACH_CHIA:
        thu_muc = KC.thu_muc_cach_chia(cach)
        out = os.path.join(thu_muc, "excel")
        os.makedirs(out, exist_ok=True)
        print(f"\n--- {cach}: {KC.CACH_CHIA[cach]}")
        xay_du_bao(cach, thu_muc, os.path.join(out, f"du_bao_{cach}.xlsx"))
        print(f"    du_bao_{cach}.xlsx")
        xay_elasticity(cach, os.path.join(out, f"elasticity_{cach}.xlsx"))
        print(f"    elasticity_{cach}.xlsx")
        xay_gia(cach, thu_muc, os.path.join(out, f"gia_{cach}.xlsx"))
        print(f"    gia_{cach}.xlsx")
    print("\nXong. Các file nằm trong models_2y/<cách chia>/excel/")


if __name__ == "__main__":
    main()
