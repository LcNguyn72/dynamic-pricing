"""
Mô hình giá cuối cùng (2 năm): chọn cách ước lượng elasticity bằng kiểm định NGOÀI MẪU, rồi fit
==========================================================================================
Mục tiêu: elasticity của từng sản phẩm dùng để ra giá. Đặt hai câu hỏi bằng dữ liệu:
  1. Có CÁCH NÀO dự đoán phản ứng thật ở các lần đổi giá tốt hơn "một con số chung" không?
  2. Độ bất định đúng là bao nhiêu (để engine tính XÁC SUẤT và ra đề xuất có mức độ tin cậy)?

CÁC ỨNG VIÊN (đều có dạng  m_i = prior_i + w_i * (b_i - prior_i),  w_i = tau2 / (tau2 + se_i^2)):
   global      : prior = elasticity chung của mọi sản phẩm
   covariates  : prior = g + gamma . z_i  (z_i: mức giá, mức bán, mùa vụ Giáng sinh, biến động giá/lượng của sản phẩm),
                 ước lượng GỘP trên tất cả sản phẩm -> chia sẻ thông tin, giảm nhiễu
   xgboost     : prior = độ dốc theo giá của chính mô hình XGBoost (đã học cách phản ứng phụ thuộc đặc điểm sản phẩm)
   cov+xgb     : trung bình hai prior trên
   (b_i, se_i: ước lượng thô riêng từng sản phẩm)
Đánh giá: train ở năm A, dự đoán thay đổi lượng bán ở các LẦN ĐỔI GIÁ THẬT (|giá tuần đổi| >= 10%) của năm B,
đo TƯƠNG QUAN với thay đổi thật; làm hai chiều. tau2 được chọn ở chiều này và DÙNG Ở CHIỀU KIA (cross-fit)
để không tự khen. Ứng viên tốt nhất (tương quan cross-fit trung bình cao nhất) được fit lại trên cả 2 năm.

ĐỘ BẤT ĐỊNH: sigma_extra được hiệu chỉnh BỀN VỮNG (trung vị |z| = 0.6745) giữa ước lượng năm train và ước
lượng thô năm test; đây là phần elasticity lệch ngoài sai số lấy mẫu.

Đầu vào : models_2y/weekly_online_retail.csv + (cho ứng viên xgboost) demand_model, product_stats, feature_cols
Đầu ra : models_2y/pricing_params_online_retail.csv  (eps_mean, eps_sd từng sản phẩm)
        models_2y/pricing_model_online_retail.json  (phương pháp, tau2, sigma_extra, bảng đánh giá)
"""
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.stats import norm

from y2_config import MIN_TRAIN_WEEKS, OUT_DIR, path
from y2_tools import build_events, estimate_raw, load_weekly

warnings.filterwarnings("ignore")
MIN_WEEKS = 30
TAU2S = np.geomspace(0.01, 5.0, 30)
COV_COLS = ["lp_med", "base", "q4_effect", "lp_std", "lq_std"]
FEATS = ["log_price_ratio", "sku_lp_med", "sku_base", "month", "woy",
         "sku_lq_std", "sku_lp_std", "sku_q4_effect"]
SE_GLOBAL = 0.03          # sai số chuẩn của elasticity chung (đã đo ở các bước trước: ~0.03)
CANDIDATES = ["global", "covariates", "xgboost", "cov+xgb"]


# ----------------------------------------------------------------------------- đặc điểm sản phẩm
def sku_covariates(d):
    st = d.groupby("sku").agg(lp_med=("lp", "median"), base=("lq", "mean"), lp_std=("lp", "std"),
                              lq_std=("lq", "std")).reset_index()
    q4 = d["month"].isin([11, 12])
    a = d[q4].groupby("sku")["lq"].mean().rename("q4m")
    b = d[~q4].groupby("sku")["lq"].mean().rename("restm")
    st = st.merge(a, on="sku", how="left").merge(b, on="sku", how="left")
    st["q4_effect"] = (st["q4m"] - st["restm"]).fillna(0.0)
    st[["lp_std", "lq_std"]] = st[["lp_std", "lq_std"]].fillna(0.0)
    return st[["sku"] + COV_COLS]


def standardize(cov):
    mu, sd = cov[COV_COLS].mean(), cov[COV_COLS].std().replace(0, 1.0)
    z = cov.copy()
    z[COV_COLS] = (cov[COV_COLS] - mu) / sd
    return z


def fit_cov_elasticity(d, z):
    """Elasticity phụ thuộc đặc điểm sản phẩm: gộp trên tất cả sản phẩm.
    Trả về (eps_i, var_i, beta) với eps_i = g + z_i . gamma"""
    d = d.merge(z, on="sku", how="inner").reset_index(drop=True)
    D = pd.get_dummies(d["month"].astype(str), prefix="m", drop_first=True).astype(float)
    inter = pd.DataFrame({f"lp_x_{c}": d["lp"] * d[c] for c in COV_COLS})
    Zm = pd.concat([d[["lq", "lp"]], inter, D], axis=1)
    Zd = Zm - Zm.groupby(d["sku"]).transform("mean")
    X = Zd.drop(columns="lq")
    groups = d["sku"].astype("category").cat.codes.values
    fit = sm.OLS(Zd["lq"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    names = ["lp"] + [f"lp_x_{c}" for c in COV_COLS]
    beta = fit.params[names].values
    V = fit.cov_params().loc[names, names].values
    zz = z.set_index("sku")[COV_COLS]
    A = np.column_stack([np.ones(len(zz)), zz.values])
    return (pd.Series(A @ beta, index=zz.index), pd.Series(np.einsum("ij,jk,ik->i", A, V, A), index=zz.index), beta)


# ----------------------------------------------------------------------------- XGBoost
def sku_stats_train(fit):
    st = fit.groupby("sku").agg(sku_base=("lq", "mean"), sku_lp_med=("lp", "median"),
                                sku_lq_std=("lq", "std"), sku_lp_std=("lp", "std"),
                                n=("lq", "size")).reset_index()
    q4 = fit["month"].isin([11, 12])
    a = fit[q4].groupby("sku")["lq"].mean().rename("q4m")
    b = fit[~q4].groupby("sku")["lq"].mean().rename("restm")
    st = st.merge(a, on="sku", how="left").merge(b, on="sku", how="left")
    st["sku_q4_effect"] = (st["q4m"] - st["restm"]).fillna(0.0)
    st[["sku_lq_std", "sku_lp_std"]] = st[["sku_lq_std", "sku_lp_std"]].fillna(0.0)
    return st[st["n"] >= MIN_TRAIN_WEEKS].drop(columns=["q4m", "restm", "n"])


def train_xgb(train):
    st = sku_stats_train(train)
    tr = train.merge(st, on="sku")
    tr["log_price_ratio"] = tr["lp"] - tr["sku_lp_med"]
    m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
                         monotone_constraints=tuple(-1 if f == "log_price_ratio" else 0 for f in FEATS))
    m.fit(tr[FEATS].values.astype(np.float32), tr["lq"].values)
    return m, st


def xgb_implied_eps(model, st):
    """Độ dốc theo giá của XGBoost: đổi giá ±10% tại giá tham chiếu, trung bình 12 tháng"""
    base = st.reset_index(drop=True)
    eps = np.zeros(len(base))
    span = np.log(1.1) - np.log(0.9)
    for mth in range(1, 13):
        woy = min(max(int(round(mth * 4.35)), 1), 52)

        def mk(shift):
            return pd.DataFrame({"log_price_ratio": shift, "sku_lp_med": base["sku_lp_med"],
                                 "sku_base": base["sku_base"], "month": mth, "woy": woy,
                                 "sku_lq_std": base["sku_lq_std"], "sku_lp_std": base["sku_lp_std"],
                                 "sku_q4_effect": base["sku_q4_effect"]})[FEATS].values.astype(np.float32)
        eps += (model.predict(mk(np.log(1.1))) - model.predict(mk(np.log(0.9)))) / span
    return pd.Series(eps / 12, index=base["sku"])


# ----------------------------------------------------------------------------- ước lượng
def posterior(res, prior, var_prior, tau2):
    """Trung bình và độ lệch chuẩn sau khi co về prior theo tau2"""
    pr = prior.reindex(res["sku"]).values
    vp = var_prior.reindex(res["sku"]).values
    w = tau2 / (tau2 + res["se"].values ** 2)
    m = pr + w * (res["b"].values - pr)
    s = np.sqrt(w * res["se"].values ** 2 + (1 - w) ** 2 * vp)
    return pd.Series(m, index=res["sku"].values), pd.Series(s, index=res["sku"].values)


def build_priors(train, res, g, model_xgb=None, st_xgb=None):
    """Trả về (priors, var_priors): dict tên ứng viên -> Series theo sku"""
    skus = res["sku"].values
    one = lambda v: pd.Series(v, index=skus)
    cov = sku_covariates(train)
    cov = cov[cov["sku"].isin(skus)]
    z = standardize(cov)
    e_cov, v_cov, _ = fit_cov_elasticity(train, z)
    if model_xgb is None:
        model_xgb, st_xgb = train_xgb(train)
    e_x = xgb_implied_eps(model_xgb, st_xgb).reindex(skus)
    priors = {"global": one(g), "covariates": e_cov.reindex(skus), "xgboost": e_x}
    priors["cov+xgb"] = 0.5 * (priors["covariates"] + priors["xgboost"])
    var = {"global": one(SE_GLOBAL ** 2), "covariates": v_cov.reindex(skus),
           "xgboost": one(0.0), "cov+xgb": 0.25 * v_cov.reindex(skus)}
    for k in priors:                                   # sản phẩm thiếu prior -> dùng elasticity chung
        priors[k] = priors[k].fillna(g)
        var[k] = var[k].fillna(SE_GLOBAL ** 2)
    return priors, var


def corr_curve(ev, res, prior):
    d = ev.merge(res[["sku", "b", "se"]], on="sku").merge(prior.rename("prior").rename_axis("sku"),
                                                          left_on="sku", right_index=True)
    y, dlp = d["dlq"].values, d["dlp"].values
    b, se, pr = d["b"].values, d["se"].values, d["prior"].values
    curve = np.array([np.corrcoef((pr + (t / (t + se ** 2)) * (b - pr)) * dlp, y)[0, 1] for t in TAU2S])
    return curve, float(np.corrcoef(b * dlp, y)[0, 1]), float(np.corrcoef(pr * dlp, y)[0, 1]), len(d)


def robust_sigma(m_a, s_a, b_b, se_b):
    """sigma sao cho trung vị |z| = 0.6745 (chuẩn) -> hiệu chỉnh phần giữa của phân phối"""
    best, gap = 0.0, 9.0
    for sg in np.arange(0, 2.001, 0.01):
        z = np.abs(b_b - m_a) / np.sqrt(s_a ** 2 + se_b ** 2 + sg ** 2)
        d = abs(np.median(z) - 0.6745)
        if d < gap:
            best, gap = float(sg), d
    return best


def main():
    w_all, y1, y2 = load_weekly()
    print(f"Năm 1: {len(y1):,} dòng | Năm 2: {len(y2):,} dòng | sản phẩm: {w_all['sku'].nunique()}")

    # ---------- 1. Đánh giá ngoài mẫu, hai chiều
    dirs = {}
    for name, train, test in (("nam 1 -> nam 2", y1, y2), ("nam 2 -> nam 1", y2, y1)):
        res, g, me = estimate_raw(train, MIN_WEEKS)
        res_test, _, _ = estimate_raw(test, MIN_WEEKS)
        ev = build_events(test, me)
        priors, var = build_priors(train, res, g)
        dirs[name] = {"res": res, "g": g, "res_test": res_test, "ev": ev, "priors": priors, "var": var,
                      "curves": {}}
        print(f"\n[{name}] {len(res)} sản phẩm ước lượng | {len(ev):,} lần đổi giá để kiểm tra | g = {g:.3f}")
        for c in CANDIDATES:
            cv, raw, _, _ = corr_curve(ev, res, priors[c])
            dirs[name]["curves"][c] = cv
        dirs[name]["raw"] = raw            # tương quan khi tin hoàn toàn số riêng (không phụ thuộc ứng viên)
    (n1, d1), (n2, d2) = dirs.items()
    gcorr = {n: float(np.corrcoef(d["g"] * d["ev"]["dlp"].values, d["ev"]["dlq"].values)[0, 1]) for n, d in dirs.items()}

    print("\n" + "=" * 86)
    print("KẾT QUẢ KIỂM ĐỊNH NGOÀI MẪU: tương quan với thay đổi lượng bán thật ở các lần đổi giá")
    print("=" * 86)
    print(f"Mốc so sánh:  một con số chung: {gcorr[n1]:.3f} / {gcorr[n2]:.3f} | "
          f"tin hoàn toàn số riêng: {d1['raw']:.3f} / {d2['raw']:.3f}   (1->2 / 2->1)")
    print(f"{'ứng viên':<13}{'tau2 (1->2)':>12}{'tau2 (2->1)':>12}{'cross-fit 1->2':>16}{'cross-fit 2->1':>16}{'TRUNG BINH':>12}")
    table, best = [], None
    for c in CANDIDATES:
        cA, cB = d1["curves"][c], d2["curves"][c]
        iA, iB = int(cA.argmax()), int(cB.argmax())            # tau2 tốt nhất ở từng chiều
        cross12, cross21 = float(cA[iB]), float(cB[iA])        # dùng tau2 của chiều kia
        mean = (cross12 + cross21) / 2
        table.append({"name": c, "tau2_12": float(TAU2S[iA]), "tau2_21": float(TAU2S[iB]),
                      "cross_12": cross12, "cross_21": cross21, "mean": mean})
        print(f"{c:<13}{TAU2S[iA]:>12.3f}{TAU2S[iB]:>12.3f}{cross12:>16.3f}{cross21:>16.3f}{mean:>12.3f}")
        if best is None or mean > best["mean"]:
            best = table[-1]
    base_mean = (gcorr[n1] + gcorr[n2]) / 2
    print(f"\n=> Ứng viên tốt nhất: {best['name']} (trung bình {best['mean']:.3f} so với một con số chung {base_mean:.3f}; "
          f"chênh {best['mean'] - base_mean:+.3f})")
    chosen = best["name"]
    tau2 = float(np.sqrt(best["tau2_12"] * best["tau2_21"]))

    # ---------- 2. Độ bất định (bao cả hai chiều)
    sig = []
    print(f"\nHiệu chỉnh độ bất định cho '{chosen}' (tau2 = {tau2:.3f}):")
    for name, d in dirs.items():
        m_a, s_a = posterior(d["res"], d["priors"][chosen], d["var"][chosen], tau2)
        mm = pd.DataFrame({"sku": m_a.index, "m_a": m_a.values, "s_a": s_a.values}).merge(
            d["res_test"][["sku", "b", "se"]].rename(columns={"b": "b_b", "se": "se_b"}), on="sku")
        sg = robust_sigma(mm["m_a"].values, mm["s_a"].values, mm["b_b"].values, mm["se_b"].values)
        sig.append(sg)
        cov = {lv: float((np.abs(mm["b_b"] - mm["m_a"]) <= norm.ppf(0.5 + lv / 2) *
                          np.sqrt(mm["s_a"] ** 2 + mm["se_b"] ** 2 + sg ** 2)).mean()) for lv in (0.5, 0.8, 0.9, 0.95)}
        print(f"   {name}: sigma_extra = {sg:.2f} | độ phủ thực tế: " +
              " | ".join(f"{int(lv * 100)}%: {v:.0%}" for lv, v in cov.items()) + f"  ({len(mm)} sản phẩm)")
    sigma = float(np.mean(sig))
    print(f"   sigma_extra cuối = {sigma:.2f}  (các mức 50/80/90/95% không cần khớp hết: dùng để tính XÁC SUẤT ở vùng giữa)")

    # ---------- 3. Hệ số scale (chỉ thông tin)
    scales = []
    for name, d in dirs.items():
        iT = int(np.argmin(np.abs(TAU2S - tau2)))
        e = d["ev"].merge(d["res"][["sku", "b", "se"]], on="sku")
        pr = d["priors"][chosen].reindex(e["sku"]).values
        wt = TAU2S[iT] / (TAU2S[iT] + e["se"].values ** 2)
        p = (pr + wt * (e["b"].values - pr)) * e["dlp"].values
        scales.append(float((p * e["dlq"].values).sum() / (p * p).sum()))
    scale = float(np.mean(scales))
    print(f"\nScale (hồi quy thay đổi thật ~ scale x dự báo): {scales[0]:.2f} / {scales[1]:.2f} -> {scale:.2f}  (chỉ tham khảo)")

    # ---------- 4. Fit lại trên cả 2 năm
    print("\nFit lại trên cả 2 năm...")
    res_all, g_all, _ = estimate_raw(w_all, 10)
    model_xgb = st_xgb = None
    if chosen in ("xgboost", "cov+xgb"):
        mp = path("demand_model_online_retail.pkl")
        sp = path("product_stats_online_retail.json")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            raise SystemExit("Ứng viên đã chọn cần mô hình XGBoost: chạy 5_train_final_model_online_retail_2y.py trước.")
        with open(mp, "rb") as f:
            model_xgb = pickle.load(f)
        with open(sp) as f:
            stats = json.load(f)
        st_xgb = pd.DataFrame([{"sku": k, "sku_base": v["sku_base"], "sku_lp_med": v["sku_lp_med"],
                                "sku_lq_std": v["sku_lq_std"], "sku_lp_std": v["sku_lp_std"],
                                "sku_q4_effect": v["sku_q4_effect"]} for k, v in stats.items()])
    priors, var = build_priors(w_all, res_all, g_all, model_xgb, st_xgb)
    m_all, s_all = posterior(res_all, priors[chosen], var[chosen], tau2)
    wt = tau2 / (tau2 + res_all["se"].values ** 2)
    out = pd.DataFrame({"sku": res_all["sku"].values, "n_weeks": res_all["n"].values,
                        "eps_raw": res_all["b"].values, "se_raw": res_all["se"].values,
                        "prior_mean": priors[chosen].reindex(res_all["sku"]).values,
                        "weight_own": wt, "eps_mean": m_all.values,
                        "eps_sd": np.sqrt(s_all.values ** 2 + sigma ** 2)})
    prod = pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})[["sku", "description"]]
    out = out.merge(prod, on="sku", how="left")
    out.to_csv(path("pricing_params_online_retail.csv"), index=False)

    meta = {"method": chosen, "tau2": tau2, "sigma_extra": sigma, "global_elasticity": g_all,
            "scale": scale, "candidates": table, "baseline_global_corr": base_mean,
            "created": datetime.now().isoformat(timespec="seconds"),
            "note": "eps_sd đã gồm sigma_extra. Dùng eps_mean/eps_sd để tính xác suất, không phải khoảng tin cậy 95%."}
    with open(path("pricing_model_online_retail.json"), "w") as f:
        json.dump(meta, f, indent=2)

    p_el = pd.Series(norm.cdf((-1 - out["eps_mean"]) / out["eps_sd"]))
    print(f"\nĐã lưu {path('pricing_params_online_retail.csv')} ({len(out)} sản phẩm) và pricing_model_online_retail.json")
    print(f"Elasticity trung bình theo sản phẩm: phân vị 10/50/90% = "
          f"{out['eps_mean'].quantile(.1):.2f} / {out['eps_mean'].quantile(.5):.2f} / {out['eps_mean'].quantile(.9):.2f} "
          f"| sd trung vị = {out['eps_sd'].median():.2f}")
    print(f"Xác suất cầu co giãn (elasticity < -1): phân vị 10/50/90% = "
          f"{p_el.quantile(.1):.2f} / {p_el.quantile(.5):.2f} / {p_el.quantile(.9):.2f}")


if __name__ == "__main__":
    main()
