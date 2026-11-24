"""Read-only API for the completed forecast and pricing experiments.

These routes serve recorded test results. They do not train a model, set a
business price, or turn simulated sales into observed sales.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd


router = APIRouter(prefix="/research", tags=["research"])
Dataset = Literal["uci", "m5"]
Simulator = Literal["lag_adjusted_kernel", "song_kernel"]

DATASETS = {
    "uci": {"label": "UCI Online Retail", "cohort": "100 SKU", "folder": "song_online_retail_100"},
    "m5": {"label": "M5 · cửa hàng CA_1", "cohort": "20 chuỗi", "folder": "song_m5_pilot"},
}
MODEL_LABELS = {
    "naive_last": "Tuần trước", "arima": "ARIMA", "xgboost": "XGBoost",
    "xgboost_price": "XGBoost + giá", "lstm": "LSTM",
    "lstm_price": "LSTM + giá", "transformer": "Transformer",
    "transformer_price": "Transformer + giá",
    "song_transformer_price": "Transformer theo Song",
}
POLICY_LABELS = {
    "fixed_mean": "Giá cố định", "heuristic": "Quy tắc đơn giản",
    "kernel_grid": "Kernel grid", "ddpg": "DDPG", "sac": "SAC",
    "ppo_frozen": "PPO · dự báo cố định",
    "ppo_feedback_gradient": "PPO · cập nhật + đạo hàm",
    "ppo_feedback_no_gradient": "PPO · cập nhật, bỏ đạo hàm",
    "ppo_frozen_no_gradient": "PPO · cố định, bỏ đạo hàm",
}
SIMULATOR_LABELS = {
    "lag_adjusted_kernel": "Kernel hiệu chỉnh",
    "song_kernel": "Kernel gốc",
}
OLDER_PREDICTION_MODELS = {"naive_last", "xgboost", "xgboost_price"}


def _artifact_root() -> Path:
    configured = os.environ.get("PRICING_RESEARCH_ARTIFACTS")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parent / "artifacts"


def _data_root(dataset: Dataset) -> Path:
    return _artifact_root() / DATASETS[dataset]["folder"]


def _file(path: Path) -> Path:
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Thiếu kết quả thí nghiệm. Hãy chạy lại Phase 3C/4 theo README.")
    return path


@lru_cache(maxsize=36)
def _cached_csv(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path, dtype={"sku": str})


@lru_cache(maxsize=20)
def _cached_json(path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path: Path) -> pd.DataFrame:
    path = _file(path)
    return _cached_csv(str(path), path.stat().st_mtime_ns)


def _json(path: Path) -> dict:
    path = _file(path)
    return _cached_json(str(path), path.stat().st_mtime_ns)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


@router.get("/forecast/overview")
def forecast_overview(dataset: Dataset = "uci") -> dict:
    root = _data_root(dataset)
    phase = root / "phase3c"
    aggregate = _csv(phase / "aggregate.csv")
    all_runs = _csv(phase / "all_runs.csv")
    per_sku = _csv(phase / "per_sku.csv")
    effects = _csv(phase / "price_feature_effect.csv")
    manifest = _json(phase / "run_manifest.json")
    model_seeds = {str(model): sorted(group["seed"].astype(int).unique().tolist())
                   for model, group in all_runs.groupby("model")}
    sku_sizes = (per_sku.loc[per_sku["model"].eq("naive_last"), ["sku", "n"]]
                 .sort_values("sku"))
    models = aggregate.copy()
    models["label"] = models["model"].map(MODEL_LABELS)
    models["seeds"] = models["model"].map(model_seeds)
    return {
        "dataset": dataset, "label": DATASETS[dataset]["label"],
        "cohort": DATASETS[dataset]["cohort"],
        "source": str(aggregate.iloc[0]["dataset"]),
        "test_rows": int(aggregate.iloc[0]["test_rows"]),
        "sku_count": int(sku_sizes["sku"].nunique()),
        "models": _records(models[["model", "label", "runs", "rmse_mean", "rmse_sd",
                                    "mae_mean", "mae_sd", "seeds"]]),
        "skus": _records(sku_sizes.rename(columns={"n": "test_weeks"})),
        "price_effects": _records(effects),
        "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
        "scope": "Dự báo một tuần trên cùng SKU–tuần test; RMSE tính bằng sản phẩm/tuần.",
    }


def _prediction_path(root: Path, model: str, seed: int) -> Path:
    folder = "multiseed" if model in OLDER_PREDICTION_MODELS else "phase3c"
    return root / folder / f"seed{seed}" / "comparison" / "comparison_predictions.csv"


@router.get("/forecast/series")
def forecast_series(dataset: Dataset = "uci", sku: str = Query(min_length=1),
                    seed: int = 42, models: str = "song_transformer_price,naive_last") -> dict:
    root = _data_root(dataset)
    phase = root / "phase3c"
    available = _csv(phase / "aggregate.csv")["model"].astype(str).tolist()
    chosen = [name.strip() for name in models.split(",") if name.strip()]
    if not 1 <= len(chosen) <= 2 or len(chosen) != len(set(chosen)) or not set(chosen).issubset(available):
        raise HTTPException(status_code=422, detail="Chọn một hoặc hai mô hình trong bảng so sánh.")
    per_sku = _csv(phase / "per_sku.csv")
    if sku not in set(per_sku["sku"].astype(str)):
        raise HTTPException(status_code=404, detail="Sản phẩm không thuộc tập test này.")
    all_runs = _csv(phase / "all_runs.csv")
    rows = None
    for model in chosen:
        valid_seeds = set(all_runs.loc[all_runs["model"].eq(model), "seed"].astype(int))
        if seed not in valid_seeds:
            raise HTTPException(status_code=422, detail=f"{MODEL_LABELS[model]} không có kết quả ở seed {seed}.")
        predictions = _csv(_prediction_path(root, model, seed))
        selected = predictions.loc[predictions["split"].eq("test")
                                   & predictions["model"].eq(model)
                                   & predictions["sku"].astype(str).eq(sku),
                                   ["week", "price", "y_true", "y_pred"]].copy()
        if selected.empty or selected["week"].duplicated().any():
            raise HTTPException(status_code=503, detail="Dự báo SKU–tuần không đầy đủ.")
        selected = selected.rename(columns={"y_pred": model})
        if rows is None:
            rows = selected
        else:
            rows = rows.merge(selected, on="week", how="outer", validate="one_to_one",
                              suffixes=("", "_other"), indicator=True)
            matching = (rows["_merge"].eq("both").all()
                        and np.allclose(rows["price"], rows["price_other"])
                        and np.allclose(rows["y_true"], rows["y_true_other"]))
            if not matching:
                raise HTTPException(status_code=503, detail="Hai mô hình không dùng cùng hàng test.")
            rows = rows.drop(columns=["price_other", "y_true_other", "_merge"])
    assert rows is not None
    rows = rows.sort_values("week").rename(columns={"y_true": "actual"})
    metrics = per_sku.loc[per_sku["sku"].astype(str).eq(sku)
                          & per_sku["model"].isin(chosen),
                          ["model", "n", "rmse_mean", "rmse_sd", "mae_mean"]]
    if len(metrics) != len(chosen) or not (metrics["n"].astype(int) == len(rows)).all():
        raise HTTPException(status_code=503, detail="Số tuần test không khớp bảng điểm từng SKU.")
    return {"dataset": dataset, "sku": sku, "seed": seed,
            "models": chosen, "rows": _records(rows), "metrics": _records(metrics)}


def _pricing_root(simulator: Simulator) -> Path:
    folder = ("pricing_lag_512_5seeds" if simulator == "lag_adjusted_kernel"
              else "pricing_song_kernel_512_5seeds")
    return _data_root("m5") / folder


@router.get("/pricing/overview")
def pricing_overview() -> dict:
    root = _data_root("m5")
    compare = root / "phase4_comparison"
    summary = _csv(compare / "policy_summary.csv")
    ablations = _csv(compare / "ppo_ablations.csv")
    manifest = _json(compare / "comparison_manifest.json")
    selection = _json(root / "simulator_selection" / "selected_simulator.json")
    checks = _csv(root / "simulator_selection" / "observed_price_metrics.csv")
    checks = checks.loc[checks["segment"].eq("in_support")
                        & checks["model"].isin(["song_kernel", "selected_lag_kernel", "naive_last"]),
                        ["split", "model", "n", "rmse"]]
    return {
        "simulators": [{"id": key, "label": value} for key, value in SIMULATOR_LABELS.items()],
        "policies": [{"id": key, "label": value} for key, value in POLICY_LABELS.items()],
        "summary": _records(summary), "ablations": _records(ablations),
        "observed_price_checks": _records(checks),
        "positive_price_response_skus": selection["price_response_positive_skus"],
        "price_response_checked_skus": selection["price_response_checked_skus"],
        "seeds": manifest["seeds"], "skus": manifest["evaluation_skus"],
        "cycles": manifest["evaluation_cycles_per_seed_policy"],
        "training_timesteps": manifest["training_timesteps_requested"],
        "scope": "Doanh thu và lượng bán đều được tạo bởi simulator; không phải kết quả kinh doanh thực tế.",
    }


@router.get("/pricing/replay")
def pricing_replay(simulator: Simulator = "lag_adjusted_kernel",
                   policy: str = "ppo_feedback_gradient", seed: int = 42) -> dict:
    root = _pricing_root(simulator)
    manifest = _json(root / "pricing_manifest.json")
    if seed not in manifest["seeds"]:
        raise HTTPException(status_code=422, detail="Seed không thuộc lượt chạy Phase 4.")
    if policy not in manifest["policies"]:
        raise HTTPException(status_code=422, detail="Chính sách không thuộc lượt chạy Phase 4.")
    episodes = _csv(root / "policy_episodes.csv")
    chosen = episodes.loc[episodes["seed"].eq(seed) & episodes["policy"].eq(policy)].copy()
    fixed = episodes.loc[episodes["seed"].eq(seed) & episodes["policy"].eq("fixed_mean"),
                         ["sku", "week", "chosen_price", "simulated_units", "simulated_revenue"]]
    fixed = fixed.rename(columns={"chosen_price": "fixed_price",
                                  "simulated_units": "fixed_units",
                                  "simulated_revenue": "fixed_revenue"})
    if (len(chosen) != manifest["evaluation_cycles"]
            or len(fixed) != manifest["evaluation_cycles"]
            or chosen.duplicated(["sku", "week"]).any()
            or fixed.duplicated(["sku", "week"]).any()):
        raise HTTPException(status_code=503, detail="Nhật ký mô phỏng không đầy đủ.")
    paired = chosen.merge(fixed, on=["sku", "week"], how="left", validate="one_to_one")
    if len(paired) != manifest["evaluation_cycles"] or paired["fixed_revenue"].isna().any():
        raise HTTPException(status_code=503, detail="Không ghép được quyết định với giá cố định cùng tuần.")
    paired = paired.sort_values(["sku", "week"])
    totals = paired.groupby("sku", as_index=False).agg(
        simulated_revenue=("simulated_revenue", "sum"),
        fixed_revenue=("fixed_revenue", "sum"), weeks=("week", "size"))
    totals["uplift_pct"] = 100 * (totals["simulated_revenue"] / totals["fixed_revenue"] - 1)
    all_revenue = float(paired["simulated_revenue"].sum())
    all_fixed = float(paired["fixed_revenue"].sum())
    return {
        "simulator": simulator, "policy": policy, "seed": seed,
        "rows": _records(paired[["sku", "week", "chosen_price", "simulated_units",
                                 "simulated_revenue", "reference_price", "forecast_update_count",
                                 "fixed_price", "fixed_units", "fixed_revenue"]]),
        "per_sku": _records(totals),
        "totals": {"simulated_revenue": all_revenue, "fixed_revenue": all_fixed,
                   "uplift_pct": 100 * (all_revenue / all_fixed - 1)},
        "scope": "Phát lại quyết định test đã lưu; không chạy mô hình để đề xuất giá mới.",
    }
