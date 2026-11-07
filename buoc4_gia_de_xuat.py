"""
BƯỚC 4: BA BẢN GIÁ ĐỀ XUẤT (MỖI CÁCH CHIA MỘT BẢN)
=================================================
Với mỗi cách chia:
  - Mô hình dự báo: models_2y/<cách chia>/demand_model_online_retail.pkl (từ bước 3)
  - Thống kê sản phẩm: models_2y/<cách chia>/product_stats_online_retail.json (từ bước 3)
  - Elasticity: models_2y/chung/pricing_params_online_retail.csv và pricing_model_online_retail.json (bước 2, dùng chung)
Engine (pricing_engine.py) đọc đúng năm file trong MỘT thư mục, nên mỗi lần chạy tạo một thư mục tạm
gom năm file đó, chạy engine, rồi xóa thư mục tạm. Không sửa engine.

Đảm bảo so sánh công bằng:
  - Cùng danh sách sản phẩm: giao của ba bộ thống kê và pricing_params (một số sản phẩm không đủ 20 tuần train
    ở một số cách chia, nên bị loại khỏi cả ba để danh sách giống nhau).
  - Cùng ngày áp dụng giá (--ngay), cùng giá hiện tại (mặc định p_ref của từng bộ thống kê, xem ghi chú).

Đầu ra: models_2y/<cách chia>/gia_de_xuat_online_retail.csv (một file cho mỗi cách chia)
Kiểm tra: elasticity (eps_mean, eps_sd) của mỗi sản phẩm phải giống nhau ở cả ba bản.

Chạy:
    python buoc4_gia_de_xuat.py                    (ngày mặc định 2011-11-15)
    python buoc4_gia_de_xuat.py --ngay 2011-11-14
Cần: pricing_engine.py, y2_config.py, khung_cach_chia.py, và bước 2, 3 đã chạy.
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

import khung_cach_chia as KC
from pricing_engine import PricingEngine
from y2_config import OUT_DIR

NGAY_MAC_DINH = "2011-11-15"
FILE_CHUNG_ENGINE = ["pricing_params_online_retail.csv", "pricing_model_online_retail.json"]
FILE_CACH_ENGINE = ["demand_model_online_retail.pkl", "feature_cols_online_retail.json",
                    "product_stats_online_retail.json"]
TEN_FILE_KET_QUA = "gia_de_xuat_online_retail.csv"


def kiem_tra_dau_vao():
    thieu = []
    for f in FILE_CHUNG_ENGINE:
        if not os.path.exists(os.path.join(KC.thu_muc_chung(), f)):
            thieu.append(f"chung/{f}")
    for c in KC.CACH_CHIA:
        for f in FILE_CACH_ENGINE:
            if not os.path.exists(os.path.join(KC.thu_muc_cach_chia(c), f)):
                thieu.append(f"{c}/{f}")
    if thieu:
        raise SystemExit("Thiếu file đầu vào: " + ", ".join(thieu) + "\nChạy bước 2 và bước 3 trước.")


def skus_chung():
    """Giao của các sản phẩm có trong product_stats của mọi cách chia và trong pricing_params."""
    tap = None
    for c in KC.CACH_CHIA:
        with open(os.path.join(KC.thu_muc_cach_chia(c), "product_stats_online_retail.json"), encoding="utf-8") as f:
            s = set(json.load(f).keys())
        tap = s if tap is None else tap & s
    params = pd.read_csv(os.path.join(KC.thu_muc_chung(), "pricing_params_online_retail.csv"), dtype={"sku": str})
    tap &= set(params["sku"].astype(str))
    return sorted(tap)


def tao_thu_muc_tam(cach):
    """Gom năm file engine cần vào một thư mục tạm."""
    tam = os.path.join(OUT_DIR, f"_tam_{cach}")
    if os.path.exists(tam):
        shutil.rmtree(tam)
    os.makedirs(tam)
    for f in FILE_CHUNG_ENGINE:
        shutil.copy2(os.path.join(KC.thu_muc_chung(), f), os.path.join(tam, f))
    for f in FILE_CACH_ENGINE:
        shutil.copy2(os.path.join(KC.thu_muc_cach_chia(cach), f), os.path.join(tam, f))
    return tam


def lam_phang(r, cach):
    """Đưa kết quả engine về một dòng (một hàng CSV)."""
    cf = r.get("confidence", {}) or {}
    el = r.get("elasticity", {}) or {}
    ex = r.get("expected", {}) or {}
    return {
        "cach_chia": cach,
        "sku": r.get("sku"),
        "mo_ta": r.get("description"),
        "ngay_ap_dung": r.get("date"),
        "gia_hien_tai": r.get("current_price"),
        "gia_de_xuat": r.get("recommended_price"),
        "gia_muc_tieu": r.get("target_price"),
        "doi_gia_pct": r.get("price_change_pct"),
        "hanh_dong": r.get("action"),
        "do_manh": r.get("strength"),
        "xac_suat_co_loi_muc_tieu": r.get("prob_gain_at_target"),
        "xac_suat_co_loi_de_xuat": cf.get("prob_gain"),
        "ky_vong_thay_doi_pct": cf.get("expected_gain_pct"),
        "p10_thay_doi_pct": cf.get("gain_p10_pct"),
        "p90_thay_doi_pct": cf.get("gain_p90_pct"),
        "elasticity_tb": el.get("mean"),
        "elasticity_sd": el.get("sd"),
        "xac_suat_co_gian": el.get("prob_elastic"),
        "luong_ban_hien_tai": ex.get("units_current"),
        "luong_ban_de_xuat": ex.get("units_recommended"),
        "doanh_thu_hien_tai": ex.get("revenue_current"),
        "doanh_thu_de_xuat": ex.get("revenue_recommended"),
        "so_canh_bao": len(r.get("warnings", []) or []),
        "canh_bao": " | ".join(r.get("warnings", []) or []),
    }


def chay_mot_cach(cach, skus, ngay):
    tam = tao_thu_muc_tam(cach)
    try:
        eng = PricingEngine(tam)
        dong, loi = [], []
        for sku in skus:
            try:
                r = eng.recommend(sku, date=ngay)
                dong.append(lam_phang(r, cach))
            except Exception as e:   # ghi lỗi từng sản phẩm, không dừng cả lô
                loi.append({"sku": sku, "loi": f"{type(e).__name__}: {str(e)[:100]}"})
        df = pd.DataFrame(dong)
        duong_dan = os.path.join(KC.thu_muc_cach_chia(cach), TEN_FILE_KET_QUA)
        df.to_csv(duong_dan, index=False, encoding="utf-8-sig")
        if loi:
            pd.DataFrame(loi).to_csv(os.path.join(KC.thu_muc_cach_chia(cach), "gia_de_xuat_loi.csv"),
                                     index=False, encoding="utf-8-sig")
        return df, len(loi), duong_dan
    finally:
        shutil.rmtree(tam, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngay", default=NGAY_MAC_DINH, help="ngày áp dụng giá, YYYY-MM-DD")
    args = ap.parse_args()

    print("=" * 78)
    print(f"BƯỚC 4: BA BẢN GIÁ ĐỀ XUẤT, ngày áp dụng {args.ngay}")
    print("=" * 78)
    kiem_tra_dau_vao()
    skus = skus_chung()
    print(f"Số sản phẩm chung của ba cách chia: {len(skus)}")

    ket = {}
    for cach in KC.CACH_CHIA:
        print(f"\n--- {cach}: {KC.CACH_CHIA[cach]}")
        df, n_loi, path = chay_mot_cach(cach, skus, args.ngay)
        ket[cach] = df
        print(f"    đã ghi {len(df)} dòng -> {path}")
        if n_loi:
            print(f"    {n_loi} sản phẩm lỗi, xem gia_de_xuat_loi.csv")

    # kiểm tra elasticity giống nhau giữa ba bản
    ref = None
    lech = 0
    for cach, df in ket.items():
        cot = df.set_index("sku")[["elasticity_tb", "elasticity_sd"]].round(6)
        if ref is None:
            ref = cot
        else:
            chung = ref.index.intersection(cot.index)
            lech += int(((ref.loc[chung] - cot.loc[chung]).abs() > 1e-6).any(axis=1).sum())
    print("\n" + "=" * 78)
    print("TÓM TẮT")
    print("=" * 78)
    print(f"{'Cách chia':<12}{'Số SP':>8}{'Giữ':>7}{'Tăng':>7}{'Giảm':>7}{'Doanh thu đề xuất / hiện tại':>32}")
    for cach, df in ket.items():
        n = len(df)
        dt_hien = df["doanh_thu_hien_tai"].sum()
        dt_de = df["doanh_thu_de_xuat"].sum()
        ty = (dt_de / dt_hien - 1) * 100 if dt_hien else float("nan")
        print(f"{cach:<12}{n:>8}{(df['hanh_dong'] == 'keep').sum():>7}"
              f"{(df['hanh_dong'] == 'raise').sum():>7}{(df['hanh_dong'] == 'lower').sum():>7}"
              f"{ty:>31.2f}%")
    if lech == 0:
        print("\nElasticity giống nhau ở cả ba bản: OK")
    else:
        print(f"\nCẢNH BÁO: {lech} sản phẩm có elasticity khác nhau giữa các bản. Kiểm tra lại bước 2.")
        sys.exit(1)
    print("Lưu ý: tổng doanh thu chỉ là ước lượng theo mô hình (không có doanh thu thật để đối chiếu).")


if __name__ == "__main__":
    main()
