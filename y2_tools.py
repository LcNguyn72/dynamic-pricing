"""Hàm dùng chung cho các bước hiệu chỉnh elasticity (2 năm)."""
import numpy as np
import pandas as pd

from y2_config import OOT_CUT_WEEK, path


def load_weekly():
    """Trả về (tất cả, năm 1, năm 2) với các cột lq, lp, month"""
    w = pd.read_csv(path("weekly_online_retail.csv"), parse_dates=["week"], dtype={"sku": str})
    w = w[(w["qty"] > 0) & (w["price"] > 0)].copy()
    w["lq"], w["lp"] = np.log(w["qty"]), np.log(w["price"])
    w["month"] = w["week"].dt.month
    w["woy"] = w["week"].dt.isocalendar().week.astype(int)
    cut = pd.Timestamp(OOT_CUT_WEEK)
    return w, w[w["week"] < cut].copy(), w[w["week"] >= cut].copy()


def estimate_raw(d, min_n=30):
    """Elasticity chung (fixed effect sản phẩm + tháng) và elasticity THÔ riêng từng sản phẩm.
    Trả về (res[sku, n, b, se], b_global, month_eff)"""
    D = pd.get_dummies(d["month"].astype(str), prefix="m", drop_first=True).astype(float)
    Z = pd.concat([d[["lq", "lp"]], D], axis=1)
    Zd = Z - Z.groupby(d["sku"]).transform("mean")
    X = Zd.drop(columns="lq")
    beta, *_ = np.linalg.lstsq(X.values, Zd["lq"].values, rcond=None)
    coef = pd.Series(beta, index=X.columns)
    b_global, month_eff = float(coef["lp"]), coef.drop("lp")
    d = d.assign(lq_adj=d["lq"] - (D @ month_eff).values)
    rows = []
    for sku, g in d.groupby("sku"):
        x, y, n = g["lp"].values, g["lq_adj"].values, len(g)
        sxx = float(((x - x.mean()) ** 2).sum())
        if n < min_n or sxx < 1e-9:
            continue
        b = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
        resid = y - y.mean() - b * (x - x.mean())
        se = max(float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx)), 1e-6)
        rows.append((sku, n, b, se))
    return pd.DataFrame(rows, columns=["sku", "n", "b", "se"]), b_global, month_eff


def tau2_mom(res, floor=0.05):
    """Cách tính tau2 (phân tán thật giữa các sản phẩm) đang dùng trong pipeline: phương pháp moment có sẵn"""
    return max(float(res["b"].var() - (res["se"] ** 2).mean()), floor)


def shrink(res, g, tau2):
    """Co về mức chung g với tau2 cho trước. Trả về (trung bình, độ lệch chuẩn sau co)"""
    w = tau2 / (tau2 + res["se"] ** 2)
    return g + w * (res["b"] - g), res["se"] * np.sqrt(w)


def build_events(test, month_eff, persistent=False, event_min=0.10, keep_tol=0.03):
    """Các lần giá tuần đổi >= event_min so với tuần trước (tuần liên tiếp).
    Trả về sku, dlp (thay đổi ln giá), dlq (thay đổi ln lượng đã trừ mùa vụ)"""
    sea = test["month"].map(lambda m: month_eff.get(f"m_{m}", 0.0))
    d = test.assign(adj=test["lq"] - sea).sort_values(["sku", "week"])
    g = d.groupby("sku")
    d["dlp"] = g["lp"].diff()
    d["dlq"] = d["adj"] - g["adj"].shift(1)
    d["gap"] = g["week"].diff().dt.days
    d["next_dlp"] = g["lp"].shift(-1) - d["lp"]
    d["next_gap"] = (g["week"].shift(-1) - d["week"]).dt.days
    ev = d[(d["gap"] == 7) & (d["dlp"].abs() >= event_min)]
    if persistent:
        ev = ev[(ev["next_gap"] == 7) & (ev["next_dlp"].abs() < keep_tol)]
    return ev[["sku", "dlp", "dlq"]]
