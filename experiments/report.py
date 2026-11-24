"""Build a compact Markdown comparison from saved experiment artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FORECAST_RUNS = (
    ("20 SKU demo · 70/15/15", ARTIFACTS / "online_retail_ii_20" / "forecast"),
    ("20 SKU demo · 80/20", ARTIFACTS / "online_retail_ii_20" / "retail2_80_20" / "forecast"),
    ("20 SKU demo · 12 tuần cuối", ARTIFACTS / "online_retail_ii_20" / "retail2_last12" / "forecast"),
    ("20 SKU chọn từ train · 70/15/15", ARTIFACTS / "retail_ii_paper_20" / "forecast"),
)
PRICING_DIR = ARTIFACTS / "online_retail_ii_20" / "pricing_verified"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def build_report(output: str | Path = ARTIFACTS / "comparison_report.md") -> Path:
    output = Path(output)
    lines = [
        "# Đối chiếu thuật toán trên Online Retail II",
        "",
        f"Tạo lúc {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. ",
        "Các bảng được tính từ artifact đã lưu; mỗi bảng dự báo chấm các mô hình trên cùng SKU–tuần của chính bảng đó.",
        "",
        "## Thiết kế đối chiếu nguồn nghiên cứu",
        "",
        _table(
            ["Nguồn", "Dữ liệu và mục tiêu", "Chia test", "Tình trạng trong dự án"],
            [
                ["[Song (2026)](https://link.springer.com/article/10.1007/s10791-026-10745-2)",
                 "Online Retail một năm dự báo; M5 định giá", "70/15/15 theo thời gian",
                 "Có cấu hình tương ứng trên Online Retail II; chưa phải tái lập cùng dữ liệu"],
                ["[Harighanesh & Senthilkumar (2026)](https://www.ijert.org/research/price-elasticity-of-demand-and-dynamic-pricing-in-online-retail-a-machine-learning-approach-to-reven-IJERTV15IS100885.pdf)",
                 "Online Retail II, 150 SKU–tuần", "80/20 tuần theo thời gian",
                 "Đã chạy cùng cách chia trên 20 SKU; khác cohort và làm sạch"],
                ["[K C & Dangol (2026)](https://osuva.uwasa.fi/items/06715eee-317a-4538-a18d-9f7c3e8bc713)",
                 "Online Retail II, tổng lượng bán tuần, chân trời 12 tuần", "12 tuần cuối sau 4 tuần đệm",
                 "Đã thử 12 tuần cuối trên 20 SKU, dự báo từng tuần; không so RMSE trực tiếp"],
                ["[Qiao và cộng sự (2026)](https://arxiv.org/html/2602.12147v4)",
                 "Online Retail II, tổng lượng bán ngày", "180 ngày cuối, 6 cửa sổ × 30 ngày",
                 "Chưa tái tạo bài toán tổng theo ngày"],
            ],
        ),
        "",
        "## Dự báo nhu cầu",
        "",
        "RMSE và MAE là số sản phẩm/tuần. WAPE là phần trăm. Dòng test giữa các cách chia và cohort khác nhau; chỉ xếp hạng thuật toán **trong từng bảng**.",
    ]
    for label, directory in FORECAST_RUNS:
        file = directory / "forecast_metrics.csv"
        if not file.exists():
            continue
        metrics = pd.read_csv(file)
        test = metrics.loc[metrics["split"].eq("test")].sort_values("rmse")
        if test.empty:
            continue
        if test["n"].nunique() != 1:
            raise ValueError(f"Unequal test counts in {file}")
        rows = [[r.model, str(int(r.n)), f"{r.rmse:.2f}", f"{r.mae:.2f}", f"{r.wape_pct:.1f}%"]
                for r in test.itertuples(index=False)]
        lines.extend(["", f"### {label}", "", _table(["Mô hình", "N test", "RMSE", "MAE", "WAPE"], rows)])
        by_name = test.set_index("model")
        comparisons = []
        for base, price in (("xgboost", "xgboost_price"), ("lstm", "lstm_price"),
                            ("transformer", "transformer_price")):
            if base in by_name.index and price in by_name.index:
                change = 100 * (by_name.loc[price, "rmse"] / by_name.loc[base, "rmse"] - 1)
                comparisons.append(f"{base}: {change:+.1f}%")
        if comparisons:
            lines.extend(["", "Đổi RMSE khi thêm giá (âm là cải thiện): " + "; ".join(comparisons) + "."])
        if "80/20" in label and "xgboost_price" in by_name.index:
            paper_wape = 49.3
            own_wape = float(by_name.loc["xgboost_price", "wape_pct"])
            lines.extend([
                "",
                f"Bài IJERT báo cáo WAPE {paper_wape:.1f}% cho gradient boosting trên 150 SKU; "
                f"XGBoost có giá ở đây đạt {own_wape:.1f}% trên 20 SKU. "
                "Khác tập sản phẩm và cách làm sạch nên không thể xếp hạng hai kết quả bằng các số này.",
            ])
        by_sku_file = directory / "forecast_metrics_by_sku.csv"
        if by_sku_file.exists():
            per_sku = pd.read_csv(by_sku_file)
            per_sku = per_sku.loc[per_sku["split"].eq("test")]
            winners = per_sku.loc[per_sku.groupby("sku")["rmse"].idxmin(), "model"].value_counts()
            lines.append("Mô hình tốt nhất theo từng SKU: " +
                         "; ".join(f"{name} {count}/{per_sku['sku'].nunique()}" for name, count in winners.items()) + ".")
    lines.extend([
        "", "**Giới hạn:** Các mô hình dự báo chỉ được kiểm tra tại mức giá từng quan sát. Sai số nhỏ chưa chứng minh dự báo đúng lượng bán tại giá chưa từng áp dụng. 20 SKU demo được chọn từ toàn bộ hai năm; cohort nghiên cứu ở bảng cuối được chọn chỉ từ train.",
    ])
    sensitivity_file = ARTIFACTS / "online_retail_ii_20" / "price_diagnostics" / "price_sensitivity_summary.json"
    if sensitivity_file.exists():
        sensitivity = json.loads(sensitivity_file.read_text(encoding="utf-8"))
        lines.extend([
            "", "## Độ nhạy theo giá của mô hình dự báo", "",
            f"Trên {sensitivity['test_rows']} dòng test, {100 * sensitivity['negative_gradient_fraction']:.1f}% đạo hàm theo giá là âm; "
            f"trung vị độ co giãn cục bộ suy từ mô hình là {sensitivity['median_local_elasticity_proxy']:.2f}. "
            f"Có {sensitivity['price_change_events_abs_log_ge_0_10']} tuần đổi giá đáng kể. "
            "Đây là đạo hàm của hàm dự báo trên dữ liệu quan sát, không phải tác động nhân quả của việc đổi giá.",
        ])
        if sensitivity.get("median_simulator_local_elasticity") is not None:
            lines.append(
                f"Trung vị độ co giãn cục bộ của môi trường kernel là "
                f"{sensitivity['median_simulator_local_elasticity']:.2f}; dấu đạo hàm của hai mô hình trùng nhau "
                f"ở {100 * sensitivity['forecaster_simulator_gradient_sign_agreement_fraction']:.1f}% dòng. "
                "Hai ước lượng khác nhau đáng kể về độ lớn, nên tín hiệu gradient đưa vào PPO cần được diễn giải thận trọng."
            )

    policy_file = PRICING_DIR / "policy_runs.csv"
    if policy_file.exists():
        runs = pd.read_csv(policy_file)
        if runs.groupby("policy")["seed"].nunique().nunique() != 1:
            raise ValueError("Policies have unequal seed counts")
        aggregation = runs.groupby("policy").agg(
            seeds=("seed", "nunique"), revenue_mean=("simulated_revenue", "mean"),
            revenue_sd=("simulated_revenue", "std"), uplift_mean=("uplift_vs_fixed_pct", "mean"),
            gap_mean=("expected_regret", "mean"),
        ).sort_values("revenue_mean", ascending=False)
        rows = [[name, str(int(r.seeds)), f"{r.revenue_mean:,.0f} ± {r.revenue_sd:,.0f}",
                 f"{r.uplift_mean:+.1f}%", f"{r.gap_mean:,.0f}"]
                for name, r in aggregation.iterrows()]
        lines.extend([
            "", "## Chính sách giá trong môi trường mô phỏng", "",
            _table(["Chính sách", "Seed", "Doanh thu mô phỏng", "So với giá cố định", "Khoảng cách oracle"], rows),
        ])
        if "fixed_mean" in aggregation.index and "ppo_feedback_gradient" in aggregation.index:
            full = aggregation.loc["ppo_feedback_gradient"]
            winner = str(aggregation.index[0])
            lines.extend([
                "",
                f"Chính sách có doanh thu mô phỏng trung bình cao nhất là `{winner}`. "
                f"PPO có cập nhật dự báo và gradient đạt {full.uplift_mean:+.1f}% so với giá cố định "
                "trên 5 seed; thí nghiệm hiện tại chưa tái tạo mức tăng doanh thu của bài Song trên M5.",
            ])
        if {"ppo_frozen", "ppo_feedback_gradient"}.issubset(aggregation.index):
            paired = runs.pivot(index="seed", columns="policy", values="simulated_revenue")
            delta = 100 * (paired["ppo_feedback_gradient"] / paired["ppo_frozen"] - 1)
            lines.extend([
                "",
                f"So với PPO dự báo đóng băng, PPO cập nhật đạt {delta.mean():+.1f}% "
                f"± {delta.std():.1f} điểm phần trăm qua {len(delta)} seed; "
                f"{int((delta < 0).sum())}/{len(delta)} seed giảm. "
                "Song báo cáo +5,0% trên M5; dữ liệu và môi trường khác nhau nên đây chỉ là đối chiếu hướng tác động.",
            ])
        episode_file = PRICING_DIR / "policy_episodes.csv"
        if episode_file.exists():
            episodes = pd.read_csv(episode_file)
            per_sku_seed = episodes.groupby(["policy", "seed", "sku"], as_index=False)["simulated_revenue"].sum()
            fixed_sku = per_sku_seed.loc[per_sku_seed["policy"].eq("fixed_mean"),
                                         ["seed", "sku", "simulated_revenue"]].rename(
                columns={"simulated_revenue": "fixed_revenue"}
            )
            per_sku_seed = per_sku_seed.merge(fixed_sku, on=["seed", "sku"], validate="many_to_one")
            per_sku_seed["uplift_pct"] = 100 * (per_sku_seed["simulated_revenue"] / per_sku_seed["fixed_revenue"] - 1)
            by_sku = per_sku_seed.groupby(["policy", "sku"], as_index=False)["uplift_pct"].mean()
            by_policy = by_sku.groupby("policy").agg(
                median_sku_uplift=("uplift_pct", "median"),
                winning_skus=("uplift_pct", lambda values: int((values > 0).sum())),
            )
            per_sku_rows = [[name, f"{r.median_sku_uplift:+.1f}%", f"{int(r.winning_skus)}/{by_sku['sku'].nunique()}"]
                            for name, r in by_policy.iterrows()]
            lines.extend([
                "", "### Độ phủ theo sản phẩm", "",
                _table(["Chính sách", "Trung vị thay đổi của SKU", "SKU tăng doanh thu mô phỏng"], per_sku_rows),
            ])
        observed_check = PRICING_DIR / "simulator_observed_price_check.csv"
        if observed_check.exists():
            check = pd.read_csv(observed_check)
            for split, group in check.groupby("split"):
                wape = 100 * group["absolute_error"].sum() / group["qty"].sum()
                lines.append(f"\nMôi trường mô phỏng tại giá đã quan sát, {split}: MAE {group['absolute_error'].mean():.2f} sản phẩm/tuần, WAPE {wape:.1f}% (n={len(group)}).")
        lines.extend([
            "", "**Cách đọc:** Đây là kết quả của cùng một mô hình cầu ước lượng từ train. Doanh thu không phải tiền bán thật. Oracle chọn trong toàn bộ vùng giá đã có ở train, nên khoảng cách với oracle có thể gồm những mức giá không đạt được chỉ sau một bước thay đổi ±10%. Môi trường cố định có điều kiện theo giá/lịch; nó không chứng minh nhu cầu thị trường thực sự trôi theo chính sách. Bảng RMSE ở trên chấm mô hình trước cập nhật online; chưa có RMSE sau phản hồi PPO.",
        ])
    lines.extend([
        "", "## Kết luận triển khai", "",
        "- Giữ riêng bảng dự báo, bảng phản ứng giá và bảng chính sách giá.",
        "- Dùng cùng SKU–tuần, seed và quy tắc chia cho các mô hình trong một bảng.",
        "- Muốn đối chiếu số học trực tiếp với Song cần dữ liệu Online Retail một năm và M5 thật cùng giao thức xử lý; adapter M5 đã có nhưng repo chưa có các CSV M5.",
        "- Muốn đối chiếu số học trực tiếp với bài IJERT cần tái tạo tập 150 SKU và cách xử lý của bài; trùng tỷ lệ 80/20 một mình chưa đủ.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_report.md")
    args = parser.parse_args()
    print(build_report(args.output))


if __name__ == "__main__":
    main()
