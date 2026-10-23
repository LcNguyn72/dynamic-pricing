"""
Pricing Engine (TV2) - thành phần dùng để đóng gói thành API
============================================================
Nhận: mã sản phẩm, giá hiện tại, (tùy chọn) giá vốn, ngày.
Trả về: giá đề xuất, hành động (lower / raise / keep), độ mạnh, xác suất có lợi, khoảng kỳ vọng, cảnh báo.

Mô hình:
  - MỨC CẦU tại giá hiện tại: XGBoost (sản phẩm, mùa vụ) - làm mượt theo giá.
  - PHẢN ỨNG THEO GIÁ: elasticity ước lượng từng sản phẩm, kèm độ bất định (eps_mean, eps_sd) -
    do 11_fit_pricing_model_2y.py chọn bằng kiểm định ngoài mẫu.
  - Cầu:  Q(p) = Q0 * (p / p0)^eps,  eps ~ Normal(eps_mean, eps_sd)

Quy tắc ra quyết định (thay cho "hold nếu khoảng tin cậy 95% chứa -1"):
  1. Tìm giá MỤC TIÊU tối đa hóa GIÁ TRỊ KỲ VỌNG (doanh thu, hoặc LỢI NHUẬN nếu có unit_cost),
     trong khoảng ±max_change quanh giá hiện tại và trong vùng giá đã quan sát.
  2. Tính XÁC SUẤT việc chuyển sang giá mục tiêu cho kết quả tốt hơn giá hiện tại (theo phân phối của eps).
  3. Độ mạnh f = clip((P - prob_floor) / (prob_full - prob_floor), 0, 1);  giá đề xuất = p0 + f * (mục_tiêu - p0).
     P thấp -> đổi giá rất nhỏ hoặc giữ nguyên; P cao -> đổi giá đầy đủ. Giữ nguyên giá chỉ khi mô hình
     không thấy có lợi hoặc xác suất có lợi không vượt prob_floor.

LƯU Ý KHI SỬ DỤNG:
  - Thay đổi đề xuất nhỏ hơn min_move (mặc định 1%) bị coi là không đáng kể -> giữ nguyên giá.
  - Không truyền unit_cost -> tối ưu DOANH THU (không nói gì về lợi nhuận). Truyền unit_cost để tối ưu LỢI NHUẬN.
  - Đây là dữ liệu quan sát (1 cửa hàng, 2 năm), không chứng minh nhân quả. Nên thử nghiệm A/B trên các đề xuất mạnh.
"""
import datetime as dt
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from y2_config import OUT_DIR as DEFAULT_DIR
except ImportError:                     # engine có thể chạy độc lập, không cần y2_config
    DEFAULT_DIR = "./models_2y"

MODEL_VERSION = "2y-final-1"
_ZQ = norm.ppf((np.arange(401) + 0.5) / 401)


class PricingEngine:
    def __init__(self, model_dir=DEFAULT_DIR, max_change=0.10, prob_floor=0.55, prob_full=0.90, n_grid=41,
                 min_move=0.01):
        self.dir = model_dir
        self.max_change, self.prob_floor, self.prob_full, self.n_grid = max_change, prob_floor, prob_full, n_grid
        self.min_move = min_move          # đổi giá nhỏ hơn ngưỡng này (tương đối) coi là không đáng kể -> keep
        need = ["demand_model_online_retail.pkl", "feature_cols_online_retail.json",
                "product_stats_online_retail.json", "pricing_params_online_retail.csv",
                "pricing_model_online_retail.json"]
        miss = [f for f in need if not os.path.exists(os.path.join(model_dir, f))]
        if miss:
            raise FileNotFoundError(f"Thiếu file trong {model_dir}: {miss}. "
                                    "Chạy 5_train_final_model_online_retail_2y.py và 11_fit_pricing_model_2y.py.")
        with open(os.path.join(model_dir, need[0]), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, need[1])) as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(model_dir, need[2])) as f:
            self.stats = json.load(f)
        self.params = pd.read_csv(os.path.join(model_dir, need[3]), dtype={"sku": str}).set_index("sku")
        with open(os.path.join(model_dir, need[4])) as f:
            self.meta = json.load(f)

    # ------------------------------------------------------------------ tiện ích
    @staticmethod
    def _date(d):
        if d is None:
            return dt.datetime.now()
        if isinstance(d, str):
            return dt.datetime.strptime(d[:10], "%Y-%m-%d")
        if isinstance(d, dt.datetime):
            return d
        return dt.datetime(d.year, d.month, d.day)

    def _get(self, sku):
        sku = str(sku)
        if sku not in self.stats or sku not in self.params.index:
            raise KeyError(f"Không có sản phẩm {sku}")
        return self.stats[sku], self.params.loc[sku]

    def _units(self, sku, prices, date):
        """Mức cầu dự báo (số lượng/tuần) của XGBoost tại các mức giá"""
        s = self.stats[str(sku)]
        prices = np.atleast_1d(np.asarray(prices, dtype=float))
        n = len(prices)
        values = {"log_price_ratio": np.log(prices) - s["sku_lp_med"],
                  "sku_lp_med": np.full(n, s["sku_lp_med"]), "sku_base": np.full(n, s["sku_base"]),
                  "month": np.full(n, date.month), "woy": np.full(n, int(date.isocalendar()[1])),
                  "sku_lq_std": np.full(n, s["sku_lq_std"]), "sku_lp_std": np.full(n, s["sku_lp_std"]),
                  "sku_q4_effect": np.full(n, s["sku_q4_effect"])}
        X = np.column_stack([values[c] for c in self.feature_cols]).astype(np.float32)
        return np.exp(self.model.predict(X))

    def _level(self, sku, p0, date):
        """Mức cầu tại giá hiện tại, làm mượt đường cầu bậc thang của cây bằng hồi quy log-log trên lưới giá"""
        grid = np.linspace(0.8 * p0, 1.2 * p0, self.n_grid)
        b, a = np.polyfit(np.log(grid), np.log(self._units(sku, grid, date)), 1)
        return float(np.exp(a + b * np.log(p0)))

    @staticmethod
    def _bounds(s, p0, mc):
        lo, hi = p0 * (1 - mc), p0 * (1 + mc)
        pmin, pmax = s.get("price_min"), s.get("price_max")
        if pmin is not None and pmax is not None and pmin <= p0 <= pmax:
            lo, hi = max(lo, pmin), min(hi, pmax)
        return lo, hi

    # ------------------------------------------------------------------ danh mục sản phẩm
    def products(self):
        out = []
        for sku, s in self.stats.items():
            if sku in self.params.index:
                p = self.params.loc[sku]
                out.append({"sku": sku, "description": s["description"], "reference_price": s["p_ref"],
                            "price_min": s.get("price_min"), "price_max": s.get("price_max"),
                            "elasticity_mean": round(float(p["eps_mean"]), 3),
                            "elasticity_sd": round(float(p["eps_sd"]), 3)})
        return out

    def model_info(self):
        return {"model_version": MODEL_VERSION, "elasticity_method": self.meta["method"],
                "tau2": self.meta["tau2"], "sigma_extra": self.meta["sigma_extra"],
                "global_elasticity": self.meta["global_elasticity"], "trained_at": self.meta["created"],
                "n_products": len(self.params), "max_change_default": self.max_change,
                "prob_floor": self.prob_floor, "prob_full": self.prob_full, "min_move": self.min_move}

    # ------------------------------------------------------------------ đề xuất giá
    def recommend(self, sku, current_price=None, unit_cost=None, date=None, max_change=None):
        sku = str(sku)
        s, p = self._get(sku)
        d = self._date(date)
        p0 = float(current_price) if current_price is not None else float(s["p_ref"])
        if p0 <= 0:
            raise ValueError("current_price phải > 0")
        mc = self.max_change if max_change is None else float(max_change)
        m, sd = float(p["eps_mean"]), max(float(p["eps_sd"]), 1e-6)
        has_cost = unit_cost is not None
        c = float(unit_cost) if has_cost else 0.0
        objective = "profit" if has_cost else "revenue"
        warns = []

        lo, hi = self._bounds(s, p0, mc)
        u0 = self._level(sku, p0, d)
        prices = np.unique(np.append(np.linspace(lo, hi, 81), p0))
        ln_r = np.log(prices / p0)
        exp_obj = (prices - c) * u0 * np.exp(m * ln_r + 0.5 * sd ** 2 * ln_r ** 2)
        p_t = float(prices[int(np.argmax(exp_obj))])

        eps = m + sd * _ZQ
        obj0 = (p0 - c) * u0
        base = abs(obj0) if abs(obj0) > 1e-9 else p0 * u0

        def outcome(price):
            q = u0 * (price / p0) ** eps
            return ((price - c) * q - obj0) / base, q

        g_t, _ = outcome(p_t)
        prob_t = float((g_t > 0).mean()) if p_t != p0 else 0.0
        f = 0.0 if p_t == p0 else float(np.clip((prob_t - self.prob_floor) / (self.prob_full - self.prob_floor), 0, 1))
        p_rec = round(p0 + f * (p_t - p0), 2)
        suppressed = False
        if abs(p_rec / p0 - 1) < self.min_move:
            suppressed = f > 0
            p_rec = p0
        g_r, q_r = outcome(p_rec)
        moved = p_rec != p0
        action = "keep" if not moved else ("raise" if p_rec > p0 else "lower")
        strength = "none" if not moved else ("strong" if f >= 0.75 else "moderate" if f >= 0.4 else "weak")
        prob_gain = float((g_r > 0).mean()) if moved else None
        p_el = float(norm.cdf((-1 - m) / sd))

        # ---- cảnh báo
        if not has_cost:
            warns.append("Không có unit_cost: tối ưu DOANH THU, không kết luận được về lợi nhuận.")
        elif c >= p0:
            warns.append("unit_cost >= giá hiện tại: biên lợi nhuận không dương.")
        if s.get("price_min") is not None and not (s["price_min"] <= p0 <= s["price_max"]):
            warns.append("Giá hiện tại nằm ngoài vùng giá đã quan sát; mô hình ít đáng tin.")
        if sd > 0.7:
            warns.append("Elasticity của sản phẩm này rất bất định (ít dữ liệu đổi giá).")
        if not has_cost and abs(m + 1) < 0.15:
            warns.append("Elasticity gần -1: đổi giá gần như không đổi doanh thu.")
        if p_t in (float(prices[0]), float(prices[-1])) and p_t != p0:
            warns.append(f"Giá mục tiêu chạm giới hạn ±{mc:.0%}; mức tối ưu thật có thể xa hơn.")

        change = (p_rec / p0 - 1) * 100
        if moved:
            reason = (f"Elasticity ước lượng {m:.2f} (sd {sd:.2f}), xác suất cầu co giãn {p_el:.0%}. "
                      f"Đề xuất {'tăng' if action == 'raise' else 'giảm'} giá {abs(change):.1f}% (mức {strength}) để "
                      f"tối đa hóa {'lợi nhuận' if has_cost else 'doanh thu'}; xác suất có lợi {prob_gain:.0%}, "
                      f"kỳ vọng {np.mean(g_r) * 100:+.1f}% (10-90%: {np.percentile(g_r, 10) * 100:+.1f}% .. "
                      f"{np.percentile(g_r, 90) * 100:+.1f}%).")
        elif suppressed:
            reason = (f"Thay đổi mô hình đề xuất nhỏ hơn ngưỡng đáng kể ({self.min_move:.0%}); "
                      "giữ nguyên giá.")
        elif p_t == p0:
            reason = "Theo mô hình, giá hiện tại đã gần mức tối ưu trong giới hạn cho phép."
        else:
            reason = (f"Mô hình nghiêng về {'tăng' if p_t > p0 else 'giảm'} giá đến {p_t:.2f} nhưng xác suất có lợi chỉ "
                      f"{prob_t:.0%} (ngưỡng {self.prob_floor:.0%}); giữ nguyên giá.")

        exp_units = float(np.mean(q_r))
        res = {
            "sku": sku, "description": s["description"], "date": d.strftime("%Y-%m-%d"),
            "objective": objective, "current_price": round(p0, 2), "recommended_price": round(p_rec, 2),
            "price_change_pct": round(change, 1), "action": action, "strength": strength,
            "target_price": round(p_t, 2), "prob_gain_at_target": round(prob_t, 3) if p_t != p0 else None,
            "confidence": {"prob_gain": None if prob_gain is None else round(prob_gain, 3),
                           "expected_gain_pct": round(float(np.mean(g_r)) * 100, 1),
                           "gain_p10_pct": round(float(np.percentile(g_r, 10)) * 100, 1),
                           "gain_p90_pct": round(float(np.percentile(g_r, 90)) * 100, 1)},
            "elasticity": {"mean": round(m, 3), "sd": round(sd, 3), "prob_elastic": round(p_el, 3)},
            "expected": {"units_current": round(u0, 1), "units_recommended": round(exp_units, 1),
                         "revenue_current": round(p0 * u0, 2), "revenue_recommended": round(p_rec * exp_units, 2)},
            "price_range_used": [round(lo, 2), round(hi, 2)],
            "reason": reason, "warnings": warns, "model_version": MODEL_VERSION,
            "priority_score": round(float(np.mean(g_r)) * 100 * (prob_gain or 0.0), 2),
        }
        if has_cost:
            res["expected"]["profit_current"] = round((p0 - c) * u0, 2)
            res["expected"]["profit_recommended"] = round((p_rec - c) * exp_units, 2)
            res["margin_pct"] = round((p0 - c) / p0 * 100, 1)
            res["breakeven_elasticity"] = round(-p0 / (p0 - c), 2) if p0 > c else None
        return res

    def recommend_many(self, items):
        """items: danh sách dict các tham số của recommend(). Lỗi của từng mục không làm hỏng cả lô."""
        out = []
        for it in items:
            try:
                out.append(self.recommend(**it))
            except (KeyError, ValueError) as e:
                out.append({"sku": str(it.get("sku")), "error": str(e).strip("'\"")})
        return out

    # ------------------------------------------------------------------ đường cầu cho dashboard
    def demand_curve(self, sku, current_price=None, unit_cost=None, date=None, max_change=None, points=21):
        s, p = self._get(sku)
        d = self._date(date)
        p0 = float(current_price) if current_price is not None else float(s["p_ref"])
        mc = self.max_change if max_change is None else float(max_change)
        m, sd = float(p["eps_mean"]), max(float(p["eps_sd"]), 1e-6)
        lo, hi = self._bounds(s, p0, mc)
        u0 = self._level(str(sku), p0, d)
        prices = np.linspace(lo, hi, points)
        c = float(unit_cost) if unit_cost is not None else 0.0
        out = []
        for pr in prices:
            r = pr / p0
            qs = [u0 * r ** (m + k * sd) for k in (-1.2816, 0.0, 1.2816)]
            q_lo, q_mid, q_hi = min(qs), qs[1], max(qs)
            row = {"price": round(float(pr), 2), "units": round(q_mid, 1),
                   "units_low": round(q_lo, 1), "units_high": round(q_hi, 1),
                   "revenue": round(float(pr) * q_mid, 2),
                   "revenue_low": round(float(pr) * q_lo, 2), "revenue_high": round(float(pr) * q_hi, 2)}
            if unit_cost is not None:
                row["profit"] = round((float(pr) - c) * q_mid, 2)
            out.append(row)
        return {"sku": str(sku), "current_price": round(p0, 2), "band": "10%-90% theo bất định của elasticity",
                "points": out}


def _demo():
    eng = PricingEngine()
    info = eng.model_info()
    print(f"Mô hình {info['model_version']} | elasticity: {info['elasticity_method']} | "
          f"tau2 = {info['tau2']:.2f} | sigma_extra = {info['sigma_extra']:.2f} | {info['n_products']} sản phẩm")
    sel_path = os.path.join(DEFAULT_DIR, "selected_products_online_retail.csv")
    skus = (pd.read_csv(sel_path, dtype={"sku": str})["sku"].tolist() if os.path.exists(sel_path)
            else list(eng.params.index[:20]))
    skus = [s for s in skus if s in eng.params.index]
    when = "2011-11-15"
    for title, cost_ratio in (("MỤC TIÊU DOANH THU (không có giá vốn)", None),
                              ("MỤC TIÊU LỢI NHUẬN (giá vốn giả định = 60% giá tham chiếu)", 0.6)):
        print(f"\n=== {title} | ngày {when} ===")
        print(f"{'sku':<8}{'sản phẩm':<28}{'giá hiện tại':>9}{'giá đề xuất':>9}{'đổi':>7}{'hành động':>10}{'độ mạnh':>10}"
              f"{'P(có lợi)':>10}{'kỳ vọng':>9}")
        n = {"lower": 0, "raise": 0, "keep": 0}
        for sku in skus:
            s = eng.stats[sku]
            r = eng.recommend(sku, unit_cost=None if cost_ratio is None else cost_ratio * s["p_ref"], date=when)
            n[r["action"]] += 1
            pg = r["confidence"]["prob_gain"]
            print(f"{sku:<8}{str(r['description'])[:26]:<28}{r['current_price']:>9.2f}{r['recommended_price']:>9.2f}"
                  f"{r['price_change_pct']:>6.1f}%{r['action']:>10}{r['strength']:>10}"
                  f"{'-' if pg is None else format(pg, '.0%'):>10}{r['confidence']['expected_gain_pct']:>8.1f}%")
        print(f"Tổng: lower = {n['lower']}, raise = {n['raise']}, keep = {n['keep']}")
    print("\nVí dụ JSON đầy đủ (sản phẩm đầu tiên):")
    print(json.dumps(eng.recommend(skus[0], date=when), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()
