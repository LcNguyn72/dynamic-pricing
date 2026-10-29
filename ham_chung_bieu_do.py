"""
Hàm dùng chung cho các script vẽ biểu đồ
========================================
Được import bởi ve_du_lieu.py và ve_ket_qua.py (không chạy trực tiếp).

Gồm: nạp dữ liệu và mô hình, dự báo trên bảng tuần, tính chỉ số (MAE, RMSE, có và không chuẩn hóa
theo sản phẩm), tìm các lần đổi giá, lưu PNG + CSV, và tạo figures/README.md.

Chỉ ĐỌC từ ./models_2y và ./models (danh sách 20 sản phẩm), chỉ GHI vào ./figures.
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")                      # vẽ ra file, không cần cửa sổ
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Cần cài matplotlib: pip install matplotlib")

from sklearn.metrics import r2_score

from y2_config import ONE_YEAR_DIR, OUT_DIR, path

warnings.filterwarnings("ignore")

FIG_DIR = "./figures"
NGAY_MAU = "2011-11-15"          # ngày áp dụng giá cho các biểu đồ đường cầu và đề xuất
GIA_VON_GIA_DINH = 0.6           # giá vốn GIẢ ĐỊNH = 60% giá tham chiếu (chỉ để minh họa chế độ lợi nhuận)
MAU = {"xanh": "#2b6cb0", "cam": "#dd6b20", "xam": "#718096", "do": "#c53030",
       "la": "#2f855a", "den": "#1a202c", "nhat": "#a0aec0"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})

# (mã, tên file, tiêu đề, nguồn dữ liệu, ý nghĩa)
MO_TA = [
    ("A2", "figA2_lam_sach_du_lieu", "Thác nước làm sạch dữ liệu", "data/online_retail_II_sheets.csv",
     "Mỗi bước làm sạch loại bao nhiêu dòng; sau đó lọc sản phẩm."),
    ("B1", "figB1_phan_bo_luong_ban", "Phân phối lượng bán theo tuần", "models_2y/weekly_online_retail.csv",
     "Đuôi dài: vài tuần bán rất nhiều làm RMSE cao hơn MAE."),
    ("B3", "figB3_so_muc_gia", "Số mức giá và số lần đổi giá của mỗi sản phẩm", "models_2y/weekly_online_retail.csv",
     "Mỗi sản phẩm đổi giá ít, nên elasticity riêng từng sản phẩm yếu."),
    ("D3", "figD3_dai_dien_20_san_pham", "20 sản phẩm được chọn so với toàn bộ", "models_2y/products_online_retail.csv + danh sách 20 sản phẩm",
     "20 sản phẩm có đại diện cho 1306 sản phẩm không."),
    ("1", "fig01_thuc_te_du_bao_theo_tuan", "Thực tế và dự báo theo tuần (3 sản phẩm)", "model_table + demand_model",
     "Mô hình có bám xu hướng của sản phẩm ổn định, đột biến, điển hình không."),
    ("2", "fig02_thuc_te_vs_du_bao", "Thực tế so với dự báo (toàn bộ tập test)", "model_table + demand_model",
     "Điểm nằm trên đường chéo là dự báo đúng; thang thường và thang log."),
    ("3", "fig03_phan_bo_sai_so", "Phân phối sai số dự báo", "model_table + demand_model",
     "Sai số lệch về phía nào, đuôi dài đến đâu."),
    ("4", "fig04_sai_so_theo_nhom", "Sai số theo nhóm giá và nhóm mức bán", "model_table + demand_model",
     "Mô hình kém ở nhóm nào; có cả sai số chuẩn hóa theo sản phẩm."),
    ("5", "fig05_sai_so_theo_thang", "Sai số theo tháng", "model_table + demand_model",
     "Mô hình kém ở giai đoạn nào (mùa lễ, đầu năm), và dự báo thiên cao hay thấp."),
    ("6", "fig06_duong_cau", "Đường cầu dự báo của 4 sản phẩm", "pricing_engine (models_2y)",
     "Lượng bán theo giá, kèm dải 10-90% theo bất định của elasticity."),
    ("7", "fig07_phan_ung_gia", "Phản ứng giá: dự báo so với thực tế", "weekly + model_table + demand_model + pricing_params",
     "Ở các lần đổi giá thật, mô hình có dự báo đúng hướng và độ lớn không."),
    ("8", "fig08_phan_bo_elasticity", "Phân phối elasticity và xác suất cầu co giãn", "models_2y/pricing_params_online_retail.csv",
     "Phần lớn sản phẩm nằm đâu so với ngưỡng -1."),
    ("9", "fig09_de_xuat_gia", "Đề xuất giá cho 20 sản phẩm", "pricing_engine (models_2y)",
     "Đổi giá bao nhiêu phần trăm, theo doanh thu và theo lợi nhuận (giá vốn giả định)."),
    ("10", "fig10_duong_doanh_thu", "Doanh thu kỳ vọng theo giá (4 sản phẩm)", "pricing_engine (models_2y)",
     "Giá hiện tại và giá đề xuất nằm đâu trên đường doanh thu."),
]


# ----------------------------------------------------------------------------- lưu kết quả
def luu(fig, ten_file, bang=None):
    """Lưu hình thành PNG, và bảng số đã vẽ thành CSV (UTF-8 có BOM để Excel đọc đúng tiếng Việt)"""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, ten_file + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    msg = f"  {ten_file}.png"
    if bang is not None:
        bang.to_csv(os.path.join(FIG_DIR, ten_file + ".csv"), index=False, encoding="utf-8-sig")
        msg += f" + .csv ({len(bang):,} dòng)"
    print(msg)


def luu_csv(bang, ten_file):
    """Lưu thêm một bảng phụ"""
    os.makedirs(FIG_DIR, exist_ok=True)
    bang.to_csv(os.path.join(FIG_DIR, ten_file + ".csv"), index=False, encoding="utf-8-sig")
    print(f"  {ten_file}.csv ({len(bang):,} dòng)")


def cap_nhat_readme():
    """Ghi figures/README.md liệt kê các biểu đồ ĐÃ có trong thư mục"""
    os.makedirs(FIG_DIR, exist_ok=True)
    dong = ["# Biểu đồ", "",
            "Tạo bởi `ve_du_lieu.py` (biểu đồ dữ liệu) và `ve_ket_qua.py` (biểu đồ kết quả mô hình).",
            "Mỗi biểu đồ có một file CSV đi kèm chứa số liệu đã vẽ (mở bằng Excel được).", "",
            "| Mã | Biểu đồ | Nguồn dữ liệu | Ý nghĩa |", "|---|---|---|---|"]
    for ma, ten, tieu_de, nguon, y_nghia in MO_TA:
        if os.path.exists(os.path.join(FIG_DIR, ten + ".png")):
            dong.append(f"| {ma} | `{ten}.png` - {tieu_de} | {nguon} | {y_nghia} |")
    dong += ["", "Lưu ý khi dùng trong báo cáo:", "",
             "- Biểu đồ 9 và 10 phụ thuộc vào ngày áp dụng giá và (ở chế độ lợi nhuận) giá vốn GIẢ ĐỊNH 60%. "
             "Không dùng làm kết luận về lợi nhuận thật.",
             "- Biểu đồ 7: tuần xảy ra đổi giá thuộc tập test, nhưng hồi quy elasticity được ước lượng trên cả 2 năm, "
             "nên đây là so sánh mô tả, không phải kiểm định ngoài mẫu. Kiểm định ngoài mẫu nằm ở `5_fit_pricing_model_2y.py`.",
             "- Tập test là mỗi tuần thứ 5 (xen kẽ), không phải kiểm định ngoài năm."]
    with open(os.path.join(FIG_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(dong) + "\n")
    print(f"  {FIG_DIR}/README.md")


# ----------------------------------------------------------------------------- nạp dữ liệu và mô hình
def nap_sku_chon():
    """Danh sách 20 sản phẩm: ưu tiên models_2y, rồi models (bản 1 năm). Trả về (danh sách, nguồn) hoặc (None, None)"""
    for thu_muc in (OUT_DIR, ONE_YEAR_DIR):
        p = os.path.join(thu_muc, "selected_products_online_retail.csv")
        if os.path.exists(p):
            return pd.read_csv(p, dtype={"sku": str})["sku"].tolist(), p
    return None, None


def nap_mo_hinh():
    with open(path("demand_model_online_retail.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(path("feature_cols_online_retail.json")) as f:
        feats = json.load(f)
    return model, feats


def nap_bang_mo_hinh():
    return pd.read_csv(path("model_table_online_retail.csv"), parse_dates=["week"], dtype={"sku": str})


def du_bao(t, model, feats):
    """Thêm cột: du_bao_ln, du_bao (số lượng), tb_train (trung bình qty của sản phẩm trên tuần train), gia_tv (giá trung vị)"""
    t = t.copy()
    t["du_bao_ln"] = model.predict(t[feats].values.astype(np.float32))
    t["du_bao"] = np.exp(t["du_bao_ln"])
    tb = t[~t["is_test"]].groupby("sku")["qty"].mean().rename("tb_train")
    t = t.merge(tb, on="sku", how="left")
    t["gia_tv"] = np.exp(t["sku_lp_med"])
    return t


def chi_so(y, p, tb=None):
    """MAE, RMSE, R2 theo số lượng, sai lệch tổng (%); nếu có tb: MAE và RMSE chuẩn hóa (chia cho trung bình train của sản phẩm)"""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, None)
    r = {"n": int(len(y)), "mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
         "r2_qty": float(r2_score(y, p)) if len(y) > 1 else np.nan, "bias_pct": float((p.sum() / y.sum() - 1) * 100)}
    if tb is not None:
        tb = np.asarray(tb, float)
        r["mae_chuan"] = float(np.mean(np.abs(p / tb - y / tb)))
        r["rmse_chuan"] = float(np.sqrt(np.mean((p / tb - y / tb) ** 2)))
    return r


def kiem_tra_so_lieu(te):
    """Đối chiếu với model_metrics_online_retail.json để biết script đang dùng đúng mô hình và dữ liệu"""
    p = path("model_metrics_online_retail.json")
    r2_ln = float(r2_score(te["lq"], te["du_bao_ln"]))
    mae_q = float(np.mean(np.abs(te["du_bao"] - te["qty"])))
    print(f"  Tập test: {len(te):,} dòng | R2(ln) = {r2_ln:.4f} | MAE theo số lượng = {mae_q:.2f}")
    if os.path.exists(p):
        with open(p) as f:
            m = json.load(f)
        ok = abs(r2_ln - m["test_r2"]) < 5e-4 and abs(mae_q - m["test_mae_qty"]) < 0.01
        print(f"  Đối chiếu model_metrics: R2 = {m['test_r2']:.4f}, MAE = {m['test_mae_qty']:.2f} -> "
              f"{'KHỚP' if ok else 'LỆCH (kiểm tra lại: có chạy lại bước 2 và 4 trên cùng dữ liệu không?)'}")


# ----------------------------------------------------------------------------- các lần đổi giá
def su_kien_doi_gia(w, month_eff, event_min=0.10):
    """Các lần giá tuần đổi >= event_min so với tuần liền trước (tuần liên tiếp).
    Giống y2_tools.build_events nhưng giữ thêm cột week, lp_prev, lp_cur.
    w cần có: sku, week, lq, lp, month. Trả về: sku, week, lp_prev, lp_cur, dlp, dlq (dlq đã trừ mùa vụ)."""
    sea = w["month"].map(lambda m: month_eff.get(f"m_{m}", 0.0))
    d = w.assign(adj=w["lq"] - sea).sort_values(["sku", "week"])
    g = d.groupby("sku")
    d["lp_prev"] = g["lp"].shift(1)
    d["dlp"] = d["lp"] - d["lp_prev"]
    d["dlq"] = d["adj"] - g["adj"].shift(1)
    d["gap"] = g["week"].diff().dt.days
    ev = d[(d["gap"] == 7) & (d["dlp"].abs() >= event_min)]
    return ev[["sku", "week", "lp_prev", "lp", "dlp", "dlq"]].rename(columns={"lp": "lp_cur"})
