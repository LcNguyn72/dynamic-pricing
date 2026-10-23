"""
Chọn sản phẩm cho hệ thống - 2 NĂM
==================================
Để SO SÁNH CÔNG BẰNG, mặc định dùng CHUNG danh sách 20 sản phẩm của bạn 1 năm
(./models/selected_products_online_retail.csv). Nếu không có, tự chọn ~20 sản phẩm theo chất lượng
dữ liệu (không theo kết quả elasticity).

Đầu ra: models_2y/selected_products_online_retail.csv
"""
import os

import pandas as pd

from y2_config import ONE_YEAR_DIR, path

N_PER_TIER = 5
MIN_WEEKS = 90
ONE_SEL = os.path.join(ONE_YEAR_DIR, "selected_products_online_retail.csv")

p = pd.read_csv(path("products_online_retail.csv"), dtype={"sku": str})
w = pd.read_csv(path("weekly_online_retail.csv"), dtype={"sku": str})
p = p.merge(w.groupby("sku").agg(n_levels=("price", "nunique")).reset_index(), on="sku")

if os.path.exists(ONE_SEL):
    skus = pd.read_csv(ONE_SEL, dtype={"sku": str})["sku"].tolist()
    sel = p[p["sku"].isin(skus)].copy()
    sel["tier"] = ""
    print(f"Dùng CHUNG danh sách {len(skus)} sản phẩm của bạn 1 năm ({len(sel)} có trong dữ liệu 2 năm)")
else:
    p = p[p["weeks"] >= MIN_WEEKS].copy()
    p["score"] = p["weeks"].rank() + p["n_levels"].rank() + p["qty_mean"].rank()
    p = p.sort_values("score", ascending=False).drop_duplicates("description")
    p["tier"] = pd.qcut(p["price_median"], 4, labels=["re", "vua_thap", "vua_cao", "dat"])
    sel = pd.concat([g.nlargest(N_PER_TIER, "score") for _, g in p.groupby("tier", observed=True)])
    print("Không thấy danh sách của bạn 1 năm -> tự chọn theo chất lượng dữ liệu")

e_path = path("elasticity_online_retail.csv")
if os.path.exists(e_path):
    e = pd.read_csv(e_path, dtype={"sku": str})
    keep = [c for c in ["sku", "b_raw", "se", "b_shrunk", "se_shrunk", "p_value"] if c in e.columns]
    sel = sel.merge(e[keep], on="sku", how="left")

sel.to_csv(path("selected_products_online_retail.csv"), index=False)
cols = [c for c in ["sku", "description", "tier", "weeks", "n_levels", "price_median",
                    "qty_mean", "b_shrunk", "se_shrunk"] if c in sel.columns]
print(f"Đã chọn {len(sel)} sản phẩm -> {path('selected_products_online_retail.csv')}")
print(sel[cols].round(2).to_string(index=False))
