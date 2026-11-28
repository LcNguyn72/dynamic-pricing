"""Build normalized M5 revenue tables from verified, seed-paired simulations."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .phase6_report_v2 import ARTIFACTS, POLICY_NAMES, PPO_VARIANTS, SIMULATOR_NAMES, read_json, table


OUTPUT = ARTIFACTS / "pricing_normalized_v2"
PHASE5 = ARTIFACTS / "phase5_v2"
FOLLOWUP = ARTIFACTS / "pricing_arimax_feedback_v2"
SONG_URL = "https://link.springer.com/article/10.1007/s10791-026-10745-2"
SONG_SEEDS = (42, 123, 456, 789, 1024)
# Published data in Song (2026), Tables 3 and 6. Table 3 only provides summaries.
SONG_TABLE3 = {
    "fixed_mean": 610, "heuristic": 755, "ddpg": 798, "sac": 821, "ppo_frozen": 841,
    "ppo_feedback_gradient": 883,
}
SONG_TABLE6 = {
    "ppo_feedback_gradient": (874, 879, 883, 887, 892),
    "ppo_feedback_no_gradient": (850, 854, 858, 862, 866),
    "ppo_frozen": (830, 835, 841, 847, 852),
    "ppo_frozen_no_gradient": (826, 838, 843, 846, 857),
}
NAMES = {
    **POLICY_NAMES, "fixed_validation_sku": "Giá cố định từng SKU từ validation",
    "grid_arimax_step3": "Lưới ARIMAX Bước 3", "grid_arimax_tuned": "Lưới ARIMAX đã tune",
    "ppo_arimax_frozen": "ARIMAX–PPO giữ hệ số", "ppo_arimax_feedback": "ARIMAX–PPO có phản hồi",
}
KEYS = ["simulator", "seed", "policy"]


def normalize_runs(runs, baseline, methods, seeds, simulators):
    rows = runs[[*KEYS, "raw_revenue"]].copy()
    expected = set(product(simulators, seeds, methods))
    observed = set(rows[KEYS].itertuples(index=False, name=None))
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("Pricing runs are missing, duplicate, or unexpected")
    if baseline not in methods:
        raise ValueError("Baseline is absent")
    if not np.isfinite(rows["raw_revenue"]).all() or rows["raw_revenue"].lt(0).any():
        raise ValueError("Revenue must be finite and nonnegative")
    anchor = rows.loc[rows["policy"].eq(baseline), ["simulator", "seed", "raw_revenue"]]
    if not anchor["raw_revenue"].gt(0).all():
        raise ValueError("Baseline revenue must be positive in every simulator and seed")
    output = rows.merge(anchor.rename(columns={"raw_revenue": "baseline_revenue"}),
                        on=["simulator", "seed"], validate="many_to_one")
    output["baseline_policy"] = baseline
    output["revenue_index"] = 100 * (output["raw_revenue"] / output["baseline_revenue"])
    output["uplift_pct"] = output["revenue_index"] - 100
    return output.sort_values(["simulator", "policy", "seed"]).reset_index(drop=True)


def summarize(rows):
    return rows.groupby(["simulator", "baseline_policy", "policy"], as_index=False).agg(
        seeds=("seed", "size"), raw_mean=("raw_revenue", "mean"), raw_std=("raw_revenue", "std"),
        raw_min=("raw_revenue", "min"), raw_max=("raw_revenue", "max"),
        index_mean=("revenue_index", "mean"), index_std=("revenue_index", "std"),
        index_min=("revenue_index", "min"), index_max=("revenue_index", "max"),
        mean_paired_uplift_pct=("uplift_pct", "mean"),
    )


def verified_csv(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Revenue file differs from the verified artifact: {path}")
    return pd.read_csv(path)


def phase5_runs(root):
    pricing = root / "pricing"
    run = read_json(pricing / "pricing_run.json")
    audit = read_json(root / "audit_pricing.json")["pricing"]
    count = run["decisions_per_policy_per_seed"]
    if (run["status"] != "complete" or run["split"] != "test" or audit["status"] != "complete"
            or run["n_rows"] != audit["verified_decisions"]
            or run["n_rows"] != count * len(run["policies"]) * len(run["seeds"]) * len(run["simulators"])
            or sorted(run["seeds"]) != sorted(audit["seeds"])
            or sorted(run["simulators"]) != sorted(audit["simulators"])
            or len(run["policies"]) != audit["policies"]):
        raise ValueError("Phase 5 pricing must be fully verified before reporting")
    rows = verified_csv(pricing / "pricing_test_runs.csv", run["files"]["pricing_test_runs.csv"])
    if not rows["n_sku_weeks"].eq(count).all():
        raise ValueError("Phase 5 pricing uses inconsistent decision counts")
    return rows.rename(columns={"simulated_revenue": "raw_revenue"}), run


def followup_runs(root, phase5):
    run = read_json(root / "run.json")
    simulators = phase5["simulators"]
    if (run["status"] != "complete" or set(run["scopes"]) != {"core", "expanded"}
            or sorted(run["seeds"]) != sorted(phase5["seeds"])):
        raise ValueError("Both ARIMAX follow-up cohorts must be complete")
    outputs = {}
    for scope, meta in run["scopes"].items():
        if meta["decisions"] != meta["n_keys"] * len(run["methods"]) * len(run["seeds"]) * len(simulators):
            raise ValueError(f"Unexpected ARIMAX follow-up decision count in {scope}")
        rows = verified_csv(root / scope / "runs.csv", meta["files"]["runs.csv"])
        outputs[scope] = normalize_runs(rows.rename(columns={"revenue": "raw_revenue"}),
                                        "fixed_validation_sku", run["methods"], run["seeds"], simulators)
    return outputs, run


def song_table3(summary):
    frame = summary[summary["policy"].isin(SONG_TABLE3)].copy()
    fixed = frame.loc[frame["policy"].eq("fixed_mean")].set_index("simulator")["raw_mean"]
    frame["v2_index_ratio_of_means"] = 100 * frame["raw_mean"] / frame["simulator"].map(fixed)
    frame["song_reported_mean"] = frame["policy"].map(SONG_TABLE3)
    frame["song_index_ratio_of_means"] = 100 * frame["song_reported_mean"] / SONG_TABLE3["fixed_mean"]
    return frame[["simulator", "policy", "index_mean", "v2_index_ratio_of_means",
                  "song_reported_mean", "song_index_ratio_of_means"]]


def song_table6():
    rows = pd.DataFrame(("song_reported", seed, policy, float(revenue))
                        for policy, scores in SONG_TABLE6.items()
                        for seed, revenue in zip(SONG_SEEDS, scores, strict=True))
    rows.columns = [*KEYS, "raw_revenue"]
    return normalize_runs(rows, "ppo_frozen", PPO_VARIANTS, SONG_SEEDS, ["song_reported"])


def value(number):
    return f"{number:,.2f}"


def summary_table(rows):
    rows = rows.sort_values("index_mean", ascending=False)
    return table(("Phương pháp", "Doanh thu TB", "SD doanh thu", "Min doanh thu", "Max doanh thu",
                  "Điểm TB", "SD điểm", "Min điểm", "Max điểm"),
                 ((NAMES[row.policy], *(value(getattr(row, column)) for column in (
                     "raw_mean", "raw_std", "raw_min", "raw_max", "index_mean", "index_std",
                     "index_min", "index_max"))) for row in rows.itertuples(index=False)))


def ablation_table(rows):
    seeds = sorted(rows["seed"].unique())
    wide = rows.pivot(index="policy", columns="seed", values="revenue_index").loc[list(PPO_VARIANTS), seeds]
    return table(("Cấu hình", *seeds, "Điểm TB", "SD điểm", "Min điểm", "Max điểm"),
                 ((NAMES[policy], *(value(x) for x in series),
                   value(series.mean()), value(series.std()), value(series.min()), value(series.max()))
                  for policy, series in wide.iterrows()))


def build_report(phase5, output, followup):
    rows, run = phase5_runs(phase5)
    full = normalize_runs(rows, "fixed_mean", run["policies"], run["seeds"], run["simulators"])
    selected = rows[rows["policy"].isin(PPO_VARIANTS)]
    ablation = normalize_runs(selected, "ppo_frozen", PPO_VARIANTS, run["seeds"], run["simulators"])
    summary = summarize(full)
    paper3 = song_table3(summary)
    paper6 = song_table6()
    extra, follow = followup_runs(followup, run)
    output.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Doanh thu mô phỏng M5 trên thang điểm chuẩn hóa",
        "Trong cùng simulator và seed, điểm = 100 × doanh thu của phương pháp / doanh thu của mốc. "
        "Điểm 102 tương ứng tăng 2% so với mốc. Bảng dùng trung bình của năm chỉ số theo seed "
        "và độ lệch chuẩn mẫu. Doanh thu thô là tổng giá nhân lượng bán mô phỏng.",
        "## Bảng 3 của nhóm · Giá trung bình lịch sử train = 100",
        f"Phase 5 dùng {run['decisions_per_policy_per_seed']} quyết định mỗi phương pháp và seed, "
        f"từ {run['decision_first_week']} đến {run['decision_last_week']}.",
    ]
    for simulator, group in summary.groupby("simulator"):
        sections.extend([f"### {SIMULATOR_NAMES[simulator]}", summary_table(group)])
    sections.extend([
        "### Đối chiếu với Bảng 3 của Song",
        "Để dùng cùng cách tính với các giá trị trung bình đã công bố, dòng dưới chia trung bình "
        "doanh thu của từng phương pháp cho trung bình của mốc giá cố định, rồi nhân 100. "
        "Cột này dùng tỷ lệ của hai trung bình. Các bảng chính dùng trung bình theo từng cặp seed. "
        "Điểm của mỗi nghiên cứu lấy mốc trong chính simulator, cohort và lịch của nghiên cứu đó. "
        f"[Bài Song, Bảng 3]({SONG_URL}#Tab3)",
        table(("Phương pháp", "V2 Gaussian", "V2 Lag-adjusted", "Song"),
              ((NAMES[policy], value(part.set_index("simulator").loc["gaussian_song", "v2_index_ratio_of_means"]),
                value(part.set_index("simulator").loc["lag_0.75", "v2_index_ratio_of_means"]),
                value(part["song_index_ratio_of_means"].iloc[0]))
               for policy, part in paper3.groupby("policy", sort=False))),
        "## Bảng 6 của nhóm · PPO đóng băng Transformer = 100",
        "Cùng bốn cấu hình và năm seed đã dùng trong Phase 5. Các cột theo seed là điểm đã chia "
        "cho kết quả PPO đóng băng tương ứng.",
    ])
    for simulator, group in ablation.groupby("simulator"):
        sections.extend([f"### {SIMULATOR_NAMES[simulator]}", ablation_table(group)])
    sections.extend([
        "### Điểm quy về PPO đóng băng từ Bảng 6 của Song",
        f"Chia lại các số đã công bố theo từng seed. Công thức tạo thang doanh thu nguồn, "
        f"cohort và cửa sổ chấm giá của Song vẫn cần xác nhận để so trực tiếp. [Bài Song, Bảng 6]({SONG_URL}#Tab6)",
        ablation_table(paper6),
        "## Kiểm thử ARIMAX bổ sung · Giá cố định theo SKU từ validation = 100",
        f"Cửa sổ {follow['identity']['test_start']} đến {follow['test_end']}. Các bảng dùng mốc giá cố định "
        "riêng theo SKU và kỳ test kế tiếp của nhóm.",
    ])
    for scope, data in extra.items():
        n_sku = follow["scopes"][scope]["n_sku"]
        for simulator, group in summarize(data).groupby("simulator"):
            sections.extend([f"### {n_sku} chuỗi · {SIMULATOR_NAMES[simulator]}", summary_table(group)])

    tables = {
        "table3_by_seed": full, "table3_summary": summary, "table3_song_reference": paper3,
        "table6_by_seed": ablation, "table6_summary": summarize(ablation),
        "table6_song_reference": paper6.rename(columns={"raw_revenue": "song_reported_score",
                                                        "baseline_revenue": "song_reported_baseline_score"}),
        "arimax_followup_by_seed": pd.concat([data.assign(cohort=scope) for scope, data in extra.items()]),
        "arimax_followup_summary": pd.concat([summarize(data).assign(cohort=scope) for scope, data in extra.items()]),
    }
    sections.append("## Dữ liệu để chạy lại và kiểm tra")
    sections.append("\n".join(f"- [{name}.csv]({name}.csv)" for name in tables))
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    path = output / "report.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, default=PHASE5)
    parser.add_argument("--followup-root", type=Path, default=FOLLOWUP)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    options = parser.parse_args()
    print(build_report(options.phase5_root, options.output_dir, options.followup_root))


if __name__ == "__main__":
    main()
