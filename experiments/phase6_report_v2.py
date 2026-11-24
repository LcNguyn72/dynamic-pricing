"""Render the V2 test report from independently verified Phase 5 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "experiments" / "artifacts"
DATASETS = {"uci": ("UCI Online Retail", "uci_online_retail", "GBP"), "m5": ("M5", "m5", "USD")}
MODEL_NAMES = {
    "naive_last": "Tuần trước", "arima_sparse_fallback": "ARIMA + fallback",
    "xgboost": "XGBoost", "xgboost_price": "XGBoost + giá", "lstm": "LSTM",
    "lstm_price": "LSTM + giá", "transformer": "Transformer",
    "transformer_price": "Transformer + giá", "song_transformer": "Transformer theo Song",
}
PRICE_MODELS = {"xgboost_price", "lstm_price", "transformer_price", "song_transformer"}
FIXED_MAX = "fixed_max_observed_train_price"
PPO_FULL = "ppo_feedback_gradient"
PPO_VARIANTS = (PPO_FULL, "ppo_feedback_no_gradient", "ppo_frozen", "ppo_frozen_no_gradient")
POLICY_NAMES = {
    "fixed_mean": "Giá trung bình train", FIXED_MAX: "Giá cận trên train",
    "heuristic": "Heuristic", "ddpg": "DDPG", "sac": "SAC", "grid_song": "Lưới giá Song",
    "grid_xgboost_price": "Lưới giá XGBoost", PPO_FULL: "PPO có cập nhật + gradient",
    "ppo_feedback_no_gradient": "PPO có cập nhật, bỏ gradient",
    "ppo_frozen": "PPO đóng băng + gradient",
    "ppo_frozen_no_gradient": "PPO đóng băng, bỏ gradient",
}
SIMULATOR_NAMES = {"gaussian_song": "Gaussian theo Song", "lag_0.75": "Lag-adjusted (0,75)"}
TIER_NAMES = {"low": "Thấp", "mid": "Trung bình", "high": "Cao"}
COLORS = ["#006858", "#2857a4", "#b34e17", "#684996", "#6c7b85", "#9d4e70"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1e7:
        return f"{value:.{digits}e}"
    return f"{value:,.{digits}f}"


def table(headers, rows) -> str:
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def load_forecast(source, dataset, lock, lock_hash):
    path = source / "audit" / dataset
    audit = read_json(path / "audit.json")
    models = set(lock["forecast"][dataset]["settings_by_model"])
    frozen = lock["frozen_test_keys"][dataset]
    expected = 2 + (len(models) - 2) * len(lock["phase5_seeds"])
    if (audit["status"] != "complete" or audit["verified_model_seed_runs"] != expected
            or set(audit["complete_models"]) != models or audit["lock_sha256"] != lock_hash
            or audit["phase1_sha256"] != lock["selection_evidence"]["phase1"]["sha256"]):
        raise ValueError(f"{dataset}: audit Phase 5 chưa đủ hoặc dùng cấu hình khác")
    scores = pd.read_csv(path / "forecast_test.csv")
    if (set(scores["model"]) != models or scores["model"].duplicated().any()
            or not scores["dataset"].eq(dataset).all() or not scores["n"].eq(frozen["count"]).all()
            or not scores["test_keys_sha256"].eq(frozen["sha256"]).all()
            or not scores["seeds"].eq(scores["model"].map(
                lambda model: 1 if model in {"naive_last", "arima_sparse_fallback"} else len(lock["phase5_seeds"])
            )).all()):
        raise ValueError(f"{dataset}: bảng dự báo lệch khóa, mô hình hoặc seed")
    tiers = pd.read_csv(path / "forecast_test_by_price_range.csv")
    tier_counts = tiers.pivot(index="model", columns="tier", values="n")
    if set(tier_counts.index) != models or not tier_counts.sum(axis=1).eq(frozen["count"]).all():
        raise ValueError(f"{dataset}: phân nhóm giá không bao phủ đủ khóa test")
    return scores.sort_values("rmse"), tiers


def load_pricing(source, lock_hash):
    path = source / "pricing"
    audit = read_json(source / "audit_pricing.json")["pricing"]
    run = read_json(path / "pricing_run.json")
    scores = pd.read_csv(path / "pricing_test.csv")
    runs = pd.read_csv(path / "pricing_test_runs.csv")
    decisions = pd.read_csv(path / "pricing_test_decisions.csv", parse_dates=["week"])
    expected_runs = pd.MultiIndex.from_product(
        [run["simulators"], run["policies"], run["seeds"]], names=["simulator", "policy", "seed"]
    )
    index = list(expected_runs.names)
    weekly = (decisions.groupby([*index, "week"], sort=True)["simulated_revenue"]
              .sum().reset_index(name="weekly_revenue"))
    counts = decisions.groupby(index).size().reindex(expected_runs)
    if (audit["status"] != "complete" or run["status"] != "complete" or run["split"] != "test"
            or run["lock_sha256"] != lock_hash or len(decisions) != audit["verified_decisions"]
            or not counts.eq(run["decisions_per_policy_per_seed"]).all()
            or set(counts.index) != set(runs.set_index(index).index) or runs.duplicated(index).any()
            or decisions.duplicated([*index, "sku", "week"]).any()):
        raise ValueError("Bộ kết quả định giá chưa đủ hoặc không khớp audit Phase 5")
    weekly["cumulative_revenue"] = weekly.groupby(index)["weekly_revenue"].cumsum()
    totals = weekly.groupby(index)["cumulative_revenue"].last()
    np.testing.assert_allclose(totals, runs.set_index(index)["simulated_revenue"].reindex(totals.index))
    return scores, runs, decisions, weekly, run


def save_plot(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=170, facecolor="white")
    plt.close(fig)


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, alpha=0.25)
    ax.set_axisbelow(True)


def plot_forecast(scores, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    for dataset, ax in zip(DATASETS, axes, strict=True):
        rows = scores[dataset].iloc[::-1]
        colors = [COLORS[0] if rank == len(rows) - 1 else COLORS[1] for rank in range(len(rows))]
        ax.barh(rows["model"].map(MODEL_NAMES), rows["rmse"], color=colors, alpha=0.88)
        if dataset == "uci":
            ax.set_xscale("log")
            ax.set_xlim(80, rows["rmse"].max() * 20)
        else:
            ax.set_xlim(0, rows["rmse"].max() * 1.26)
        for y, value in enumerate(rows["rmse"]):
            ax.annotate(number(value), (value, y), xytext=(5, 0), textcoords="offset points", va="center")
        ax.set(title=DATASETS[dataset][0],
               xlabel="RMSE (sản phẩm/tuần, thang log)" if dataset == "uci" else "RMSE (sản phẩm/tuần)")
        style(ax, "x")
    fig.suptitle("Sai số dự báo lượng bán qua toàn bộ khóa test V2", x=0.03, ha="left", weight="bold")
    save_plot(fig, directory, "forecast_rmse")


def plot_tiers(scores, tiers, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout="constrained")
    for dataset, ax in zip(DATASETS, axes, strict=True):
        selected = dict.fromkeys([scores[dataset].iloc[0]["model"], "arima_sparse_fallback", "song_transformer"])
        pivot = tiers[dataset].pivot(index="model", columns="tier", values="rmse")
        counts = tiers[dataset].query("model == 'song_transformer'").set_index("tier")["n"]
        for model, color in zip(selected, COLORS):
            ax.plot(list(TIER_NAMES), pivot.loc[model, list(TIER_NAMES)], "o-", color=color,
                    linewidth=2.2, label=MODEL_NAMES[model])
        ax.set_xticks(range(3), [f"{label}\n{number(counts[tier], 0)} khóa" for tier, label in TIER_NAMES.items()])
        ax.set(title=DATASETS[dataset][0], ylabel="RMSE (sản phẩm/tuần)")
        ax.legend(loc="best", frameon=False)
        style(ax)
    fig.suptitle("Sai số theo vùng giá xác định từ tập train", x=0.03, ha="left", weight="bold")
    save_plot(fig, directory, "forecast_price_tiers")


def plot_example(source, directory, sku):
    store = sku.rsplit("@", 1)[1]
    sample = None
    for model in ("arima_sparse_fallback", "naive_last", "song_transformer", "xgboost_price"):
        path = source / "forecast" / "m5" / model / "seed42" / f"predictions_{store}.csv"
        rows = pd.read_csv(path, parse_dates=["week"]).loc[lambda frame: frame["sku"].eq(sku)]
        if sample is None:
            sample = rows[["week", "qty", "price"]].copy()
        sample = sample.merge(rows[["week", "predicted_qty"]].rename(columns={"predicted_qty": model}),
                              on="week", how="left", validate="one_to_one")
    if sample.empty or sample.isna().any().any():
        raise ValueError(f"Thiếu dự báo mẫu của {sku}")
    sample.sort_values("week").to_csv(directory / "forecast_example_m5.csv", index=False)
    fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
    ax.plot(sample["week"], sample["qty"], color="#202a35", label="Lượng bán ghi nhận", linewidth=2.6)
    for model, color in zip(sample.columns[3:], COLORS):
        ax.plot(sample["week"], sample[model], label=MODEL_NAMES[model], color=color, linewidth=1.7)
    dates(ax)
    ax.set(title=f"Dự báo tại giá lịch sử của {sku} · Seed 42", ylabel="Sản phẩm/tuần")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)
    style(ax)
    save_plot(fig, directory, "forecast_example_m5")


def dates(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_pricing(weekly, runs, decisions, directory, sku):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    policies = ("ddpg", FIXED_MAX, "grid_song", "fixed_mean", PPO_FULL)
    for (sim, label), ax in zip(SIMULATOR_NAMES.items(), axes, strict=True):
        for policy, color in zip(policies, COLORS):
            data = weekly.loc[weekly["simulator"].eq(sim) & weekly["policy"].eq(policy)]
            stats = data.groupby("week")["cumulative_revenue"].agg(["mean", "min", "max"])
            ax.plot(stats.index, stats["mean"], label=POLICY_NAMES[policy], color=color, linewidth=2)
            if policy == "ddpg":
                ax.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.13,
                                label="DDPG: min–max theo seed")
        ax.set(title=label, ylabel="Doanh thu mô phỏng tích lũy (USD)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        dates(ax)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Doanh thu tích lũy theo tuần, gộp 10 chuỗi M5", x=0.03, ha="left", weight="bold")
    save_plot(fig, directory, "pricing_cumulative")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), layout="constrained")
    comparisons = ["ddpg", "grid_song", *PPO_VARIANTS]
    for (sim, label), ax in zip(SIMULATOR_NAMES.items(), axes, strict=True):
        for y, policy in enumerate(comparisons):
            rows = runs.loc[runs["simulator"].eq(sim) & runs["policy"].eq(policy)]
            values = rows["uplift_vs_fixed_max_pct"]
            color = COLORS[0] if policy == "ddpg" else COLORS[1]
            ax.scatter(values, np.full(len(rows), y), color=color, alpha=0.56, s=33)
            ax.scatter(values.mean(), y, marker="D", color="#202a35", s=24, zorder=3)
        ax.set_yticks(range(len(comparisons)), [POLICY_NAMES[policy] for policy in comparisons])
        ax.axvline(0, color="#67727e", linewidth=1)
        ax.invert_yaxis()
        ax.set(title=label, xlabel="Chênh lệch doanh thu so với cận trên train (%)")
        style(ax, "x")
    fig.suptitle("So sánh ghép cùng seed · Hình thoi là trung bình 5 lượt", x=0.03, ha="left", weight="bold")
    save_plot(fig, directory, "pricing_paired")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained", sharey=True)
    example_seed = 123
    sample = decisions.loc[decisions["sku"].eq(sku) & decisions["seed"].eq(example_seed)].sort_values("week")
    sample.to_csv(directory / "pricing_example_m5.csv", index=False)
    for (sim, label), ax in zip(SIMULATOR_NAMES.items(), axes, strict=True):
        rows = sample.loc[sample["simulator"].eq(sim)]
        limits = rows.loc[rows["policy"].eq(FIXED_MAX)]
        ax.fill_between(limits["week"], limits["train_price_min"], limits["train_price_max"],
                        color="#dce4e8", alpha=0.7, label="Miền giá train")
        for policy, color in zip((FIXED_MAX, "ddpg", "grid_song", PPO_FULL), COLORS):
            series = rows.loc[rows["policy"].eq(policy)]
            ax.plot(series["week"], series["chosen_price"], label=POLICY_NAMES[policy], color=color,
                    linewidth=1.7, marker="o", markersize=2.5)
        ax.set(title=label, ylabel="Giá đề xuất (USD)")
        dates(ax)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Quyết định giá mô phỏng của {sku} · Seed {example_seed}", x=0.03, ha="left", weight="bold")
    save_plot(fig, directory, "pricing_example_m5")


def forecast_report(dataset, scores, tiers, protocol, source, phase4, output):
    label, dataset_key, currency = DATASETS[dataset]
    spec = protocol["datasets"][dataset_key]
    count = spec["currently_prepared_skus" if dataset == "uci" else "currently_prepared_series"]
    audit_path = source / "audit" / dataset
    reference = read_json(audit_path / "reference" / "reference.json")
    arima_run = read_json(source / "forecast" / dataset / "arima_sparse_fallback" / "seed42" / "run.json")
    fallback = sum(shard["fallback"]["rows"] for shard in arima_run["shards"].values())
    scope = (
        f"Chuỗi đủ điều kiện: {number(count, 0)}. Train/validation/test: {spec['train_weeks']}/"
        f"{spec['validation_weeks']}/{spec['test_weeks']} tuần. Test bắt đầu {spec['test_start']} "
        f"với {number(scores['n'].iloc[0], 0)} khóa, trong đó {number(scores['mape_positive_target_count'].iloc[0], 0)} "
        f"khóa có lượng bán dương. SHA256 khóa test: `{scores['test_keys_sha256'].iloc[0]}`."
    )
    rows = (
        (MODEL_NAMES[row.model], "Có" if row.model in PRICE_MODELS else "—", row.seeds, number(row.rmse),
         number(row.rmse_std_sample), number(row.mae), number(row.r2, 3), number(row.wape_pct),
         number(row.mape_pct_positive_targets), number(row.rmse_train_mean_relative, 3), number(row.rmse_train_z, 3))
        for row in scores.itertuples(index=False)
    )
    by_tier = tiers.pivot(index="model", columns="tier", values="rmse").loc[scores["model"]]
    tier_n = tiers.loc[tiers["model"].eq("song_transformer")].set_index("tier")["n"]
    sections = [
        f"### {label}", scope,
        table(("Mô hình", "Dùng giá", "Seed", "RMSE", "Std RMSE", "MAE", "R²", "WAPE %",
               "MAPE dương %", "RMSE / TB train", "RMSE / SD train"), rows),
        f"ARIMA dùng lượng bán đã biết gần nhất trên {number(fallback, 0)} khóa test.",
        f"Ngưỡng giá theo phân vị 1/3 và 2/3 của tập train: {number(reference['price_q1'], 4)} và "
        f"{number(reference['price_q2'], 4)} {currency}. Giá bằng ngưỡng được xếp vào nhóm phía dưới.",
        table(("Mô hình", *(f"{name} (n={number(tier_n[tier], 0)})" for tier, name in TIER_NAMES.items())),
              ((MODEL_NAMES[model], *(number(by_tier.loc[model, tier]) for tier in TIER_NAMES))
               for model in by_tier.index)),
    ]
    for name in ("forecast_test", "forecast_test_by_seed", "forecast_test_by_price_range"):
        shutil.copyfile(audit_path / f"{name}.csv", output / f"{dataset}_{name}.csv")
    validation = phase4 / f"forecast_{dataset}"
    winner = pd.read_csv(validation / "forecast_validation.csv").sort_values("rmse").iloc[0]
    trials = pd.read_csv(validation / "forecast_trials.csv")
    sections.append(f"Sàng lọc validation: **{MODEL_NAMES[winner['model']]}**, RMSE "
                    f"**{number(winner['rmse'])}** trên {number(winner['n'], 0)} khóa, "
                    f"với {len(trials)} lượt thử siêu tham số.")
    for name in ("forecast_validation", "forecast_trials"):
        shutil.copyfile(validation / f"{name}.csv", output / f"{dataset}_{name}.csv")
    return "\n\n".join(sections)


def pricing_report(scores, runs, run, source, output):
    sections = [
        f"10 SKU/cửa hàng, 20 tuần test, 200 quyết định mỗi phương pháp/seed. Năm seed: "
        f"{', '.join(map(str, run['seeds']))}. Doanh thu và regret dùng thang mô phỏng thô "
        "(giá USD × lượng bán mô phỏng). Regret sử dụng doanh thu kỳ vọng của oracle trong cùng simulator. "
        "Uplift là trung bình chênh lệch phần trăm của năm cặp cùng seed.",
        "Cohort: " + ", ".join(f"`{sku}`" for sku in run["pricing_training_config"]["cohort_skus"]) + ".",
    ]
    for sim, label in SIMULATOR_NAMES.items():
        data = scores.loc[scores["simulator"].eq(sim)].sort_values("mean_revenue", ascending=False)
        sections.extend([
            f"### {label}",
            table(("Chính sách", "TB", "Std mẫu", "Min", "Max", "Regret TB", "So giá TB %",
                   "So cận trên %", "So PPO đóng băng %", "Chọn cận trên %"),
                  ((POLICY_NAMES[row.policy], number(row.mean_revenue), number(row.revenue_std),
                    number(row.revenue_min), number(row.revenue_max), number(row.mean_regret),
                    number(row.uplift_vs_fixed_mean_pct), number(row.uplift_vs_fixed_max_pct),
                    number(row.uplift_vs_ppo_frozen_pct), number(row.upper_price_pct))
                   for row in data.itertuples(index=False))),
            "#### Ablation PPO theo seed",
            table(("Cấu hình", *run["seeds"]),
                  ((POLICY_NAMES[policy], *(number(value) for value in runs.loc[
                      runs["simulator"].eq(sim) & runs["policy"].eq(policy)
                  ].set_index("seed").loc[run["seeds"], "simulated_revenue"])) for policy in PPO_VARIANTS)),
        ])
    paired_path = source / "audit" / "pricing_ddpg_check.csv"
    paired = pd.read_csv(paired_path)
    sections.extend([
        "### DDPG và mốc giá cận trên, ghép cùng seed",
        table(("Simulator", "Seed", "Số giá trùng cận trên", "Chênh doanh thu (%)", "SKU đổi giá > 0,01 USD"),
              ((SIMULATOR_NAMES[row.simulator], row.seed, row.n_prices_equal_upper_bound,
                number(row.uplift_vs_fixed_max_pct), row.skus_with_weekly_change_gt_one_cent)
               for row in paired.itertuples(index=False))),
    ])
    for name in ("pricing_test", "pricing_test_runs"):
        shutil.copyfile(source / "pricing" / f"{name}.csv", output / f"{name}.csv")
    shutil.copyfile(paired_path, output / "pricing_ddpg_pairs.csv")
    return "\n\n".join(sections)


def build_report(source: Path, phase4: Path, output: Path):
    lock_path = ROOT / "experiments" / "phase5_settings_v2.json"
    lock = read_json(lock_path)
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    protocol = read_json(ROOT / "experiments" / "protocol_v2.json")
    forecasts = {dataset: load_forecast(source, dataset, lock, lock_hash) for dataset in DATASETS}
    scores, runs, decisions, weekly, run = load_pricing(source, lock_hash)
    output.mkdir(parents=True, exist_ok=True)
    forecast_scores = {dataset: data[0] for dataset, data in forecasts.items()}
    tier_scores = {dataset: data[1] for dataset, data in forecasts.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.labelcolor": "#243647", "text.color": "#192a37"})
    sku = run["pricing_training_config"]["cohort_skus"][0]
    plot_forecast(forecast_scores, output)
    plot_tiers(forecast_scores, tier_scores, output)
    plot_example(source, output, sku)
    plot_pricing(weekly, runs, decisions, output, sku)
    weekly.to_csv(output / "pricing_weekly_cumulative.csv", index=False)
    guide = ROOT / "docs" / "phase6-report-v2.md"
    guide_label = "Phân tích, lựa chọn mô hình và đối chiếu nghiên cứu"
    guide_link = f"{guide_label} nằm trong `docs/phase6-report-v2.md`."
    if guide.drive == output.drive:
        relative_guide = Path(os.path.relpath(guide, output)).as_posix()
        guide_link = f"[{guide_label}](<{relative_guide}>)."
    report = [
        "# Kết quả đánh giá V2 · Phase 6",
        f"Giao thức `{lock['protocol_id']}`. SHA256 cấu hình khóa: `{lock_hash}`. "
        + guide_link,
        "## Dự báo lượng bán", "Điểm dự báo một tuần tại giá ghi nhận. RMSE và MAE tính theo sản phẩm/tuần. "
        "Mỗi điểm ngẫu nhiên là trung bình theo seed. Std dùng độ lệch chuẩn mẫu, áp dụng từ hai seed trở lên. "
        "RMSE chuẩn hóa của nhóm dùng trung bình hoặc độ lệch chuẩn lượng bán chung trên train.",
        "![RMSE dự báo](forecast_rmse.png)",
        *(forecast_report(dataset, *forecasts[dataset], protocol, source, phase4, output) for dataset in DATASETS),
        "![Sai số theo nhóm giá](forecast_price_tiers.png)",
        "![Dự báo trên một chuỗi M5](forecast_example_m5.png)",
        "## Định giá M5", pricing_report(scores, runs, run, source, output),
        "![Doanh thu tích lũy](pricing_cumulative.png)",
        "![So sánh ghép theo seed](pricing_paired.png)",
        "![Giá từng tuần trên một chuỗi M5](pricing_example_m5.png)",
    ]
    path = output / "report.md"
    path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-dir", type=Path, default=ARTIFACTS / "phase5_v2")
    parser.add_argument("--phase4-dir", type=Path, default=ARTIFACTS / "phase4_v2")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "phase6_v2")
    args = parser.parse_args()
    print(build_report(args.phase5_dir, args.phase4_dir, args.output_dir.resolve()))


if __name__ == "__main__":
    main()
