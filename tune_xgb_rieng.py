"""
TUNE RIÊNG XGBOOST
==================
Chỉ tune XGBoost. Dùng để trả lời: XGBoost có thể tốt hơn cấu hình hiện tại khi được tune đúng cách không?
Kết quả được báo cáo là "XGBoost sau khi tune", không dùng để so với các mô hình khác một cách công bằng.

Lưới (1620 tổ hợp), lấy ngẫu nhiên 40 tổ hợp với seed cố định (cùng 40 tổ hợp cho cả ba cách chia):
  max_depth        : 4, 5, 6, 7, 8
  learning_rate    : 0.02, 0.03, 0.05, 0.08
  min_child_weight : 5, 10, 20
  subsample        : 0.7, 0.8, 0.9
  colsample_bytree : 0.7, 0.8, 0.9
  reg_lambda       : 1, 5, 10

Quy trình, trên từng cách chia:
  1. Với mỗi tổ hợp: huấn luyện trên sub-train (80% tuần đầu), tối đa 2000 cây, early stopping
     trên VALIDATION (20% tuần cuối) với patience 50. Lấy số cây tốt nhất.
  2. Đo MAE số lượng trên validation tại số cây tốt nhất. Đây là tiêu chí chọn duy nhất.
  3. Chọn tổ hợp có MAE validation nhỏ nhất.
  4. Huấn luyện lại trên toàn bộ train với tổ hợp và số cây đã chọn, đo TEST một lần.
  5. So với XGBoost mặc định (cấu hình hiện tại), kèm khoảng bootstrap theo sản phẩm.

Không dùng test ở bất kỳ bước chọn nào. Mọi tổ hợp và MAE validation được ghi lại để kiểm tra.
Thời gian: khoảng 30 đến 60 phút cho cả ba cách chia.

Đầu ra:
  so_sanh_out/tune_xgb_luoi.csv    : 40 tổ hợp x 3 cách chia, số cây tốt nhất, MAE validation
  so_sanh_out/tune_xgb_ket_qua.csv : tổ hợp được chọn, số cây, kết quả test, so với mặc định
  so_sanh_out/tune_xgb_tham_so.json: tham số đã chọn cho từng cách chia (dùng khi cập nhật bước 4)
Cần khung_so_sanh_2y.py trong cùng thư mục, và chi_tiet/xgboost_<cách chia>.csv (đã có từ bước 1).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

import khung_so_sanh_2y as K

warnings.filterwarnings("ignore")

N_TO_HOP = 40
SEED_LUOI = 42
MAX_CAY = 2000
PATIENCE = 50
LUOI = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [5, 10, 20],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [1, 5, 10],
}
MONO = tuple(-1 if f == "log_price_ratio" else 0 for f in K.FEATS)


def lay_to_hop(n, seed):
    """Lấy ngẫu nhiên n tổ hợp không lặp từ lưới đầy đủ."""
    import itertools
    tat_ca = [dict(zip(LUOI.keys(), v)) for v in itertools.product(*LUOI.values())]
    rng = np.random.default_rng(seed)
    chon = rng.choice(len(tat_ca), size=min(n, len(tat_ca)), replace=False)
    return [tat_ca[i] for i in sorted(chon)]


def dung_mo_hinh(p, n_cay=None, early=False):
    kw = dict(learning_rate=p["learning_rate"], max_depth=int(p["max_depth"]),
              min_child_weight=p["min_child_weight"], subsample=p["subsample"],
              colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
              monotone_constraints=MONO, random_state=42, verbosity=0)
    if early:
        return xgb.XGBRegressor(n_estimators=MAX_CAY, early_stopping_rounds=PATIENCE, eval_metric="mae", **kw)
    return xgb.XGBRegressor(n_estimators=int(n_cay), **kw)


def tune_mot_cach_chia(du, chia, to_hop):
    sub_f, val_f = K.tach_con_val(du, chia)
    Xs = sub_f[K.FEATS].values.astype(np.float32)
    ys = np.log(sub_f["qty"].values.astype(float))
    Xv = val_f[K.FEATS].values.astype(np.float32)
    yv_qty = val_f["qty"].values.astype(float)
    yv_ln = np.log(yv_qty)
    rows = []
    for i, p in enumerate(to_hop, 1):
        m = dung_mo_hinh(p, early=True)
        m.fit(Xs, ys, eval_set=[(Xv, yv_ln)], verbose=False)
        best = int(m.best_iteration) + 1
        pred = np.exp(m.predict(Xv, iteration_range=(0, best)))
        mae_v = float(np.mean(np.abs(pred - yv_qty)))
        rows.append({"cach_chia": chia, "to_hop": i, **p, "so_cay_tot_nhat": best, "mae_validation": mae_v})
        print(f"    [{i:>2}/{len(to_hop)}] depth={p['max_depth']} lr={p['learning_rate']} mcw={p['min_child_weight']} "
              f"sub={p['subsample']} col={p['colsample_bytree']} lam={p['reg_lambda']} -> "
              f"cây {best:>4} | MAE validation {mae_v:.3f}")
    bang = pd.DataFrame(rows)
    tot = bang.sort_values("mae_validation").iloc[0]
    chon = {k: tot[k] for k in LUOI.keys()}
    chon = {k: (int(v) if k == "max_depth" or k == "min_child_weight" else float(v)) for k, v in chon.items()}
    return bang, chon, int(tot["so_cay_tot_nhat"]), float(tot["mae_validation"])


def danh_gia_test(chia, p, n_cay, du):
    trn, tst, tb = K.tach(du, chia)
    m = dung_mo_hinh(p, n_cay=n_cay)
    m.fit(trn[K.FEATS].values.astype(np.float32), np.log(trn["qty"].values.astype(float)))
    X = tst[K.FEATS].values.astype(np.float32)
    qty = np.exp(m.predict(X))
    y = tst["qty"].values.astype(float)
    chi = K.chi_so(y, qty, tb)
    chi.update(K.phan_ung_gia(m.predict, X))
    ct = pd.DataFrame({"sku": tst["sku"].values, "week": tst["week"].values, "thuc_te": y,
                       "du_bao": qty, "tb": tb})
    return chi, ct


def main():
    print("=" * 78)
    print("TUNE RIÊNG XGBOOST (random search 40 tổ hợp, early stopping trên validation)")
    print("=" * 78)
    os.makedirs(K.OUT, exist_ok=True)
    du = K.nap_du_lieu()
    to_hop = lay_to_hop(N_TO_HOP, SEED_LUOI)
    print(f"  Lưới đầy đủ: 1620 tổ hợp | lấy ngẫu nhiên {len(to_hop)} tổ hợp (seed {SEED_LUOI})")
    print("  Cảnh báo: mất khoảng 30 đến 60 phút cho cả ba cách chia.\n")

    luoi_all, ket, tham_so = [], [], {}
    for chia in K.CACH_CHIA:
        print(f"--- {K.CO.TEN_CHIA[chia]}")
        bang, chon, n_cay, vmae = tune_mot_cach_chia(du, chia, to_hop)
        luoi_all.append(bang)
        tham_so[chia] = {**chon, "n_estimators": n_cay, "mae_validation": vmae}
        print(f"  => Chọn: {chon} | số cây {n_cay} | MAE validation {vmae:.3f}\n")

        chi, ct = danh_gia_test(chia, chon, n_cay, du)
        ket.append({"cach_chia": chia, "mo_hinh": "xgboost (đã tune)", "so_cay": n_cay,
                    "mae_validation": vmae, **chi})
        ct.to_csv(os.path.join(K.CHI_TIET, f"xgboost_tune_{chia}.csv"), index=False, encoding="utf-8-sig")

    so = []
    for r in ket:
        chia = r["cach_chia"]
        mac_dinh = pd.read_csv(os.path.join(K.CHI_TIET, f"xgboost_{chia}.csv"))
        tuned = pd.read_csv(os.path.join(K.CHI_TIET, f"xgboost_tune_{chia}.csv"))
        so.append(K.so_voi_moc(tuned, mac_dinh))
    ket_df = pd.concat([pd.DataFrame(ket), pd.DataFrame(so)], axis=1)
    ket_df["xgboost_mac_dinh_mae"] = [pd.read_csv(os.path.join(K.CHI_TIET, f"xgboost_{c}.csv"))
                                      .pipe(lambda d: float(np.mean(np.abs(d["du_bao"] - d["thuc_te"]))))
                                      for c in ket_df["cach_chia"]]

    pd.concat(luoi_all, ignore_index=True).to_csv(os.path.join(K.OUT, "tune_xgb_luoi.csv"),
                                                  index=False, encoding="utf-8-sig")
    ket_df.to_csv(os.path.join(K.OUT, "tune_xgb_ket_qua.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(K.OUT, "tune_xgb_tham_so.json"), "w", encoding="utf-8") as f:
        json.dump(tham_so, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("KẾT QUẢ TEST (một lần, sau khi tham số đã chọn trên validation)")
    print("=" * 78)
    print(ket_df[["cach_chia", "mae", "xgboost_mac_dinh_mae", "chenh_mae_so_voi_moc", "khoang_lo", "khoang_hi",
                  "r2_ln", "lech_tong_pct"]].round(3).to_string(index=False))
    tot_hon = int((ket_df["khoang_hi"] < 0).sum())
    kem_hon = int((ket_df["khoang_lo"] > 0).sum())
    print(f"\nXGBoost đã tune tốt hơn mặc định có ý nghĩa ở {tot_hon}/3 cách chia; kém hơn có ý nghĩa ở {kem_hon}/3.")
    print(f"Đã ghi {K.OUT}/tune_xgb_luoi.csv, tune_xgb_ket_qua.csv, tune_xgb_tham_so.json")


if __name__ == "__main__":
    main()
