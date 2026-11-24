"""Interactive M5 price inference and read-only Phase 6/7 evidence.

Recommendations are one-step research scenarios. No endpoint writes a price,
sale, or recommendation to SQL Server.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from stable_baselines3 import DDPG, PPO, SAC
import torch

from .phase6_optimize import ForecastGridOptimizer, _policy_features, _policy_observations
from .pricing_simulator import LagAdjustedKernelSimulator
from .research_api import _artifact_root
from .song_pricing_pilot import load_forecaster


router = APIRouter(prefix="/research", tags=["research"])
_infer_lock = Lock()
METHODS = ("hold_current", "fixed_mean", "heuristic", "forecast_grid_ensemble",
           "ppo_frozen", "ddpg", "sac")
LEARNED = {"ppo_frozen": PPO, "ddpg": DDPG, "sac": SAC}
METHOD_LABELS = {
    "hold_current": "Giữ giá hiện tại",
    "fixed_mean": "Giá trung bình train",
    "heuristic": "Quy tắc ±5%",
    "forecast_grid_ensemble": "Lưới 5 Transformer",
    "ppo_frozen": "PPO dự báo cố định",
    "ddpg": "DDPG",
    "sac": "SAC",
}


class PriceRequest(BaseModel):
    sku: str = Field(min_length=1)
    week: date
    current_price: float | None = Field(default=None, gt=0)
    method: Literal["validation_rule", "hold_current", "fixed_mean", "heuristic",
                    "forecast_grid_ensemble", "ppo_frozen", "ddpg", "sac"] = "validation_rule"
    seed: int | None = None


def _root() -> Path:
    return _artifact_root() / "song_m5_pilot"


def _required(path: Path) -> Path:
    if not path.is_file():
        raise HTTPException(status_code=503, detail=(
            "Thiếu artifact M5 Phase 6/7. Hãy chạy lại theo README."))
    return path


def _json(path: Path):
    return json.loads(_required(path).read_text(encoding="utf-8"))


def _csv(path: Path):
    return pd.read_csv(_required(path), dtype={"sku": str})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _required(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata() -> dict:
    root = _root()
    phase6 = root / "phase6_price_selection"
    phase7_demand = root / "phase7_demand_validation"
    phase7_robust = root / "phase7_policy_robustness"
    price_manifest = _json(phase6 / "manifest.json")
    demand_manifest = _json(phase7_demand / "manifest.json")
    robust_manifest = _json(phase7_robust / "manifest.json")
    expected = (
        (_sha256(phase6 / "manifest.json"), robust_manifest["phase6_manifest_sha256"]),
        (_sha256(phase6 / "selected_price_decisions.csv"),
         robust_manifest["phase6_selected_decisions_sha256"]),
        (_sha256(phase6 / "one_step_decisions.csv"),
         robust_manifest["phase6_decisions_sha256"]),
        (_sha256(phase7_demand / "manifest.json"),
         robust_manifest["train_cv_demand_manifest_sha256"]),
        (_sha256(root / "weekly.csv"), price_manifest["weekly_sha256"]),
        (_sha256(root / "windows.csv"), price_manifest["windows_sha256"]),
        (demand_manifest["prepared_weekly_sha256"], price_manifest["weekly_sha256"]),
    )
    if any(actual != saved for actual, saved in expected):
        raise HTTPException(status_code=503, detail="Artifact Phase 6/7 không cùng lượt dữ liệu.")
    if (price_manifest["business_deployment_supported"]
            or demand_manifest["business_deployment_supported"]
            or robust_manifest["business_deployment_supported"]):
        raise HTTPException(status_code=503, detail="Trạng thái kiểm định đã thay đổi; cần cập nhật giao thức.")
    selected_policy = _csv(phase6 / "selected_policy_by_sku.csv")
    selected_decisions = _csv(phase6 / "selected_price_decisions.csv")
    all_decisions = _csv(phase6 / "one_step_decisions.csv")
    if selected_policy["sku"].duplicated().any() or selected_decisions.duplicated(
            ["split", "sku", "week"]).any():
        raise HTTPException(status_code=503, detail="Quy tắc chọn giá có dòng trùng.")
    return {
        "root": root, "price_manifest": price_manifest,
        "demand_manifest": demand_manifest, "robust_manifest": robust_manifest,
        "selected_policy": selected_policy,
        "selected_decisions": selected_decisions,
        "all_decisions": all_decisions,
    }


def _bundle_versions(root: Path, seeds: list[int]) -> tuple[int, ...]:
    paths = [root / "phase6_price_selection" / "manifest.json",
             root / "weekly.csv",
             root / "simulator_selection" / "selected_simulator.json"]
    paths.extend(root / "multiseed" / f"seed{seed}" / "song" /
                 f"song_transformer_seed{seed}.pt" for seed in seeds)
    return tuple(_required(path).stat().st_mtime_ns for path in paths)


@lru_cache(maxsize=2)
def _load_bundle(root_text: str, versions: tuple[int, ...]):
    del versions
    root = Path(root_text)
    manifest = _json(root / "phase6_price_selection" / "manifest.json")
    config_path = root / "simulator_selection" / "selected_simulator.json"
    config = _json(config_path)
    if (_sha256(config_path) != manifest["simulator_config_sha256"]
            or _sha256(root / "weekly.csv") != config["prepared_weekly_sha256"]):
        raise HTTPException(status_code=503, detail="Simulator không khớp Phase 6.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weekly = _csv(root / "weekly.csv")
    weekly["week"] = pd.to_datetime(weekly["week"])
    simulator = LagAdjustedKernelSimulator(
        float(config["recent_weight"]), ratio_limit=float(config["ratio_limit"])).fit(weekly)
    forecasters = {}
    for seed in manifest["seeds"]:
        checkpoint = (root / "multiseed" / f"seed{seed}" / "song" /
                      f"song_transformer_seed{seed}.pt")
        if _sha256(checkpoint) != manifest["forecaster_checkpoint_sha256_by_seed"][str(seed)]:
            raise HTTPException(status_code=503, detail="Checkpoint Transformer khác Phase 6.")
        forecasters[seed] = load_forecaster(checkpoint)
    return simulator, forecasters, config


@lru_cache(maxsize=15)
def _load_policy(path_text: str, modified_ns: int, expected_hash: str, method: str):
    del modified_ns
    path = Path(path_text)
    if _sha256(path) != expected_hash:
        raise HTTPException(status_code=503, detail="Checkpoint chính sách khác Phase 6.")
    return LEARNED[method].load(str(path), device="cpu")


@router.get("/price/options")
def price_options() -> dict:
    data = _metadata()
    policies = data["selected_policy"].set_index("sku")
    train = _csv(data["root"] / "weekly.csv")
    support = train.loc[train["split"].eq("train")
                        & train["observed_sales"].astype(bool)].groupby("sku")["price"].agg(["min", "max"])
    choices = data["selected_decisions"]
    choices = choices.loc[choices["split"].eq("test")].sort_values(["sku", "week"])
    items = []
    for sku, rows in choices.groupby("sku", sort=False):
        rule = policies.loc[sku]
        items.append({
            "sku": sku,
            "selected_method": str(rule["method"]),
            "selected_method_label": METHOD_LABELS[str(rule["method"])],
            "selected_seed": int(rule["seed"]),
            "validation_uplift_pct": float(rule["validation_mean_uplift_pct"]),
            "train_min": float(support.loc[sku, "min"]),
            "train_max": float(support.loc[sku, "max"]),
            "weeks": [{"week": str(row.week), "reference_price": float(row.reference_price)}
                      for row in rows.itertuples(index=False)],
        })
    manifest = data["price_manifest"]
    return {
        "dataset": "M5", "store": "CA_1", "items": items,
        "pricing_evaluation_skus": len(manifest["evaluation_skus"]),
        "eligible_test_skus": len(items),
        "eligible_test_weeks": int(manifest["cohort"]["test"]["eligible_rows"]),
        "excluded_test_weeks": int(manifest["cohort"]["test"]["excluded_outside_train_support"]),
        "methods": [{"id": key, "label": value} for key, value in METHOD_LABELS.items()],
        "seeds": manifest["seeds"],
        "max_change_pct": 100 * float(manifest["price_rule"]["max_change_fraction"]),
        "scope": "Thử một quyết định trên 12 tuần lịch sử quan sát; không đặt giá bán thật.",
    }


def _target_row(root: Path, sku: str, week: date) -> pd.Series:
    windows = _csv(root / "windows.csv")
    rows = windows.loc[windows["sku"].eq(sku) & windows["split"].eq("test")
                       & windows["week"].eq(week.isoformat())]
    if len(rows) != 1:
        raise HTTPException(status_code=503, detail="Không tìm thấy lịch sử cho SKU–tuần đã chọn.")
    return _policy_features(rows.iloc[0])


def _chosen_price(method: str, seed: int, row: pd.Series, reference: float,
                  grid: dict, simulator, forecasters: dict, manifest: dict,
                  root: Path) -> float:
    lo, hi = float(grid["local_min"]), float(grid["local_max"])
    if method == "hold_current":
        return reference
    if method == "fixed_mean":
        return float(np.clip(np.mean(np.exp(simulator.surfaces[str(row["sku"])].log_prices)), lo, hi))
    if method == "heuristic":
        recent = np.asarray([float(row[f"qty_lag_{lag}"]) for lag in range(1, 5)])
        direction = 1 if recent[0] > recent.mean() else -1
        return float(np.clip(reference * (1 + .05 * direction), lo, hi))
    if method == "forecast_grid_ensemble":
        return float(grid["recommended_price"])
    (demand_env, demand_obs), (bare_env, bare_obs) = _policy_observations(
        row, simulator, forecasters[42], reference_price=reference)
    env, observation = ((demand_env, demand_obs) if method == "ppo_frozen"
                        else (bare_env, bare_obs))
    path = root / "pricing_lag_512_5seeds" / "models" / f"{method}_seed{seed}.zip"
    model = _load_policy(str(_required(path)), path.stat().st_mtime_ns,
                         manifest["policy_checkpoint_sha256_by_method_seed"][f"{method}_seed{seed}"],
                         method)
    if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
        raise HTTPException(status_code=503, detail="Đầu vào chính sách không khớp checkpoint.")
    action = float(np.asarray(model.predict(observation, deterministic=True)[0]).reshape(-1)[0])
    return float(np.clip(env.price_from_action(action), lo, hi))


@router.post("/price/recommend")
def price_recommend(request: PriceRequest) -> dict:
    return _price_recommend(request, include_comparison=False)


@router.post("/price/compare")
def price_compare(request: PriceRequest) -> dict:
    """Compute all Phase 6 methods for one shared SKU, week and reference price."""
    return _price_recommend(request, include_comparison=True)


def _comparison_methods(data: dict, sku: str) -> list[dict]:
    validation = data["all_decisions"]
    validation = validation.loc[validation["split"].eq("validation")
                                & validation["sku"].eq(sku)
                                & validation["method"].isin(METHODS)]
    runs = validation.groupby(["method", "seed"], as_index=False).agg(
        expected_revenue=("simulator_expected_revenue", "sum"),
        hold_revenue=("hold_expected_revenue", "sum"))
    expected_count = len(METHODS) * len(data["price_manifest"]["seeds"])
    if len(runs) != expected_count or (runs["hold_revenue"] <= 0).any():
        raise HTTPException(status_code=503, detail="Thiếu lượt validation của phương pháp giá.")
    runs["uplift_pct"] = 100 * (runs["expected_revenue"] / runs["hold_revenue"] - 1)
    methods = []
    for method in METHODS:
        choices = runs.loc[runs["method"].eq(method)].sort_values(["uplift_pct", "seed"])
        seed = int(choices.iloc[len(choices) // 2]["seed"]) if method in LEARNED else 42
        methods.append({"method": method, "seed": seed,
                        "validation_mean_uplift_pct": float(choices["uplift_pct"].mean())})
    return sorted(methods, key=lambda item: (
        -item["validation_mean_uplift_pct"], item["method"] != "hold_current",
        item["method"]))


def _price_recommend(request: PriceRequest, include_comparison: bool) -> dict:
    data = _metadata()
    manifest = data["price_manifest"]
    selected = data["selected_decisions"]
    historical = selected.loc[selected["split"].eq("test")
                              & selected["sku"].eq(request.sku)
                              & selected["week"].eq(request.week.isoformat())]
    if len(historical) != 1:
        raise HTTPException(status_code=422, detail=(
            "Chọn SKU và tuần test có giá tham chiếu trong miền train ở danh sách."))
    rules = data["selected_policy"].loc[data["selected_policy"]["sku"].eq(request.sku)]
    if len(rules) != 1:
        raise HTTPException(status_code=503, detail="Thiếu quy tắc validation của sản phẩm.")
    rule = rules.iloc[0]
    selected_method = str(rule["method"])
    method = selected_method if request.method == "validation_rule" else request.method
    if method not in METHODS:
        raise HTTPException(status_code=422, detail="Phương pháp giá không hợp lệ.")
    if request.method == "validation_rule" and request.seed is not None:
        raise HTTPException(status_code=422, detail="Quy tắc validation đã cố định seed; chọn trực tiếp thuật toán để đổi seed.")
    if method not in LEARNED and request.seed is not None:
        raise HTTPException(status_code=422, detail="Phương pháp này không có seed chính sách.")
    seed = (int(rule["seed"]) if request.method == "validation_rule" and method in LEARNED
            else request.seed if request.seed is not None else 42)
    if seed not in manifest["seeds"]:
        raise HTTPException(status_code=422, detail="Seed không thuộc năm lượt huấn luyện.")
    root = data["root"]
    row = _target_row(root, request.sku, request.week)
    with _infer_lock:
        simulator, forecasters, config = _load_bundle(
            str(root), _bundle_versions(root, manifest["seeds"]))
        surface = simulator.surfaces[request.sku]
        reference = (float(row["price_lag_1"]) if request.current_price is None
                     else float(request.current_price))
        if not np.isfinite(reference) or not surface.price_min <= reference <= surface.price_max:
            raise HTTPException(status_code=422, detail=(
                f"Giá hiện tại phải nằm trong miền train {surface.price_min:.2f}–{surface.price_max:.2f} USD."))
        optimizer = ForecastGridOptimizer(forecasters, simulator)
        grid = optimizer.recommend(row, reference_price=reference)
        chosen = _chosen_price(method, seed, row, reference, grid, simulator,
                               forecasters, manifest, root)
        prices = grid["candidate_prices"]
        seed_grid_revenue = optimizer.forecast_revenue(row, prices)
        reference_index = int(np.argmin(np.abs(prices - reference)))
        current_by_seed = seed_grid_revenue[:, reference_index]
        chosen_by_seed = optimizer.forecast_revenue(row, np.asarray([chosen]))[:, 0]
        forecast_current = float(current_by_seed.mean())
        forecast_chosen = float(chosen_by_seed.mean())
        context = {name: row[name] for name in
                   ("event_days", "snap_days", "qty_lag_1", "price_lag_1") if name in row.index}
        sim_current_units = float(simulator.predict_mean(request.sku, reference,
                                                         request.week, context))
        sim_chosen_units = float(simulator.predict_mean(request.sku, chosen,
                                                        request.week, context))
        cv_name = str(data["demand_manifest"]["selection"]["model"])
        if not cv_name.startswith("flat_response_"):
            raise HTTPException(status_code=503, detail="Mô hình cầu Phase 7 đã thay đổi.")
        cv_model = LagAdjustedKernelSimulator(
            float(cv_name.rsplit("_", 1)[1]), ratio_limit=float(config["ratio_limit"]))
        cv_model.surfaces = simulator.surfaces
        flat_units = float(cv_model.predict_mean(request.sku, float(row["price_lag_1"]),
                                                 request.week, context))
        curve_sim = [float(price * simulator.predict_mean(request.sku, float(price),
                                                           request.week, context))
                     for price in prices]
        matches_history = abs(reference - float(historical.iloc[0]["reference_price"])) <= 1e-8
        if request.method == "validation_rule" and matches_history:
            if abs(chosen - float(historical.iloc[0]["chosen_price"])) > 1e-5:
                raise HTTPException(status_code=503, detail="Suy luận không tái lập giá Phase 6 đã lưu.")
        historical_reproduction = bool(request.method == "validation_rule" and matches_history)
        comparison = None
        if include_comparison:
            choices = _comparison_methods(data, request.sku)
            decisions = data["all_decisions"]
            saved = decisions.loc[decisions["split"].eq("test")
                                  & decisions["sku"].eq(request.sku)
                                  & decisions["week"].eq(request.week.isoformat())]
            if matches_history and len(saved) != len(METHODS) * len(manifest["seeds"]) + len(manifest["seeds"]):
                raise HTTPException(status_code=503, detail="Thiếu quyết định test Phase 6 để đối chiếu.")
            for item in choices:
                item["seed_source"] = "validation_median" if item["method"] in LEARNED else None
                if item["method"] == method:
                    item["seed"] = seed
                    if item["method"] in LEARNED and request.method != "validation_rule":
                        item["seed_source"] = "user_choice"
                item["price"] = (chosen if item["method"] == method else _chosen_price(
                    item["method"], item["seed"], row, reference, grid, simulator,
                    forecasters, manifest, root))
                if matches_history:
                    prior = saved.loc[saved["method"].eq(item["method"])
                                      & saved["seed"].eq(item["seed"])]
                    if len(prior) != 1 or abs(item["price"] - float(prior.iloc[0]["chosen_price"])) > 1e-5:
                        raise HTTPException(status_code=503, detail="So sánh không tái lập Phase 6 đã lưu.")
            all_prices = np.asarray([item["price"] for item in choices], dtype=float)
            all_forecasts = optimizer.forecast_revenue(row, all_prices).mean(axis=0)
            comparison_rows = []
            for index, item in enumerate(choices):
                price_at_method = float(item["price"])
                forecast_revenue = (forecast_chosen if item["method"] == method
                                    else forecast_current if item["method"] == "hold_current"
                                    else float(all_forecasts[index]))
                simulator_revenue = (chosen * sim_chosen_units if item["method"] == method
                                     else reference * sim_current_units if item["method"] == "hold_current"
                                     else price_at_method * float(simulator.predict_mean(
                                         request.sku, price_at_method, request.week, context)))
                comparison_rows.append({
                    "method": item["method"], "method_label": METHOD_LABELS[item["method"]],
                    "seed": item["seed"] if item["method"] in LEARNED else None,
                    "seed_source": item["seed_source"],
                    "recommended_price": price_at_method,
                    "price_change_pct": 100 * (price_at_method / reference - 1),
                    "forecast_revenue": forecast_revenue,
                    "forecast_uplift_pct": (100 * (forecast_revenue / forecast_current - 1)
                                            if forecast_current > 0 else None),
                    "simulator_revenue": simulator_revenue,
                    "simulator_uplift_pct": (100 * (simulator_revenue /
                        (reference * sim_current_units) - 1) if sim_current_units > 0 else None),
                    "validation_mean_uplift_pct": item["validation_mean_uplift_pct"],
                    "is_active": item["method"] == method,
                    "is_selected_rule": item["method"] == selected_method,
                })
            comparison = {
                "validation_scores_use_historical_reference": matches_history,
                "rows": comparison_rows,
            }
    validation_uplift = float(rule["validation_mean_uplift_pct"])
    if request.method == "validation_rule":
        if method == "hold_current":
            reason = "Giữ giá đứng đầu trong các phương pháp của SKU này trên validation mô phỏng."
        else:
            reason = (f"{METHOD_LABELS[method]} đứng đầu cho SKU này trên validation mô phỏng "
                      f"({validation_uplift:+.2f}% so với giữ giá).")
    else:
        reason = f"Bạn đang thử {METHOD_LABELS[method]}; quy tắc chọn theo validation của SKU này là {METHOD_LABELS[selected_method]}."
    if not matches_history:
        reason += " Giá đầu vào khác lịch sử, nên điểm validation/test đã lưu không áp dụng trực tiếp cho kịch bản này."
    quality_check = data["demand_manifest"]["selected_model_observed_price_audit"]["test:in_support"]
    result = {
        "dataset": "M5", "store": "CA_1", "sku": request.sku,
        "week": request.week.isoformat(), "method_requested": request.method,
        "method_used": method, "method_label": METHOD_LABELS[method],
        "seed": seed if method in LEARNED else None,
        "selected_rule": {"method": selected_method,
                          "method_label": METHOD_LABELS[selected_method],
                          "seed": int(rule["seed"]) if selected_method in LEARNED else None,
                          "validation_simulator_uplift_pct": validation_uplift,
                          "exploratory_test_simulator_uplift_pct": (
                              None if pd.isna(rule["exploratory_test_uplift_pct"])
                              else float(rule["exploratory_test_uplift_pct"]))},
        "reference_price": reference, "reference_source": (
            "observed_previous_week" if request.current_price is None else "user_input"),
        "historical_reproduction": historical_reproduction,
        "recommended_price": chosen,
        "price_change_pct": 100 * (chosen / reference - 1),
        "price_limits": {"train_min": float(surface.price_min),
                         "train_max": float(surface.price_max),
                         "local_min": float(grid["local_min"]),
                         "local_max": float(grid["local_max"]),
                         "max_change_pct": 100 * float(manifest["price_rule"]["max_change_fraction"])},
        "forecast": {
            "units_current": forecast_current / reference,
            "units_recommended": forecast_chosen / chosen,
            "revenue_current": forecast_current,
            "revenue_recommended": forecast_chosen,
            "revenue_change_pct": (100 * (forecast_chosen / forecast_current - 1)
                                   if forecast_current > 0 else None),
            "positive_seed_count": int(np.sum(chosen_by_seed > current_by_seed + 1e-8)),
            "seed_count": len(forecasters),
        },
        "simulator": {
            "units_current": sim_current_units, "units_recommended": sim_chosen_units,
            "revenue_current": reference * sim_current_units,
            "revenue_recommended": chosen * sim_chosen_units,
        },
        "flat_response_stress": {
            "units_at_both_prices": flat_units,
            "revenue_current": reference * flat_units,
            "revenue_recommended": chosen * flat_units,
            "model": cv_name,
        },
        "curve": [{"price": float(price),
                   "forecast_revenue": float(seed_grid_revenue[:, index].mean()),
                   "simulator_revenue": curve_sim[index]}
                  for index, price in enumerate(prices)],
        "context": {"previous_week_price": float(row["price_lag_1"]),
                    "previous_week_units": float(row["qty_lag_1"]),
                    "event_days": float(row.get("event_days", 0)),
                    "snap_days": float(row.get("snap_days", 0)),
                    "history": [{"weeks_before": lag,
                                 "price": float(row[f"price_lag_{lag}"]),
                                 "units": float(row[f"qty_lag_{lag}"])}
                                for lag in range(1, 13)]},
        "selection_reason": reason,
        "quality": {
            "deployment_supported": False,
            "test_in_support_rows": int(quality_check["n"]),
            "selected_demand_rmse": float(quality_check["selected_model_rmse"]),
            "naive_last_rmse": float(quality_check["naive_last_rmse"]),
            "meaning": "Doanh thu ở giá mới là dự báo hoặc mô phỏng, chưa được quan sát.",
        },
    }
    if comparison is not None:
        result["comparison"] = comparison
    return result


@router.get("/method/overview")
def method_overview() -> dict:
    data = _metadata()
    root = data["root"]
    p6 = root / "phase6_price_selection"
    p7d = root / "phase7_demand_validation"
    p7r = root / "phase7_policy_robustness"
    demand = _csv(p7d / "observed_price_metrics.csv")
    demand = demand.loc[demand["model"].isin(["naive_last",
                                               data["demand_manifest"]["selection"]["model"]])
                        & demand["segment"].isin(["in_support",
                                                   "phase6_pricing_and_target_in_support"])]
    cv = _csv(p7d / "train_cv_metrics.csv")
    cv = cv.loc[cv["model"].isin(["naive_last",
                                    data["demand_manifest"]["selection"]["model"]])
                & cv["segment"].eq("in_support")]
    forecast = _csv(root / "phase3c" / "aggregate.csv")
    forecast = forecast.loc[forecast["split"].eq("test")
                            & forecast["model"].isin(["song_transformer_price", "naive_last"])]
    policies = _csv(p6 / "policy_summary.csv")
    policies = policies.loc[policies["method"].isin(METHODS)]
    robustness = _csv(p7r / "ranking_stability.csv")
    robustness = robustness.loc[robustness["split"].eq("test")]
    sources = _json(Path(__file__).resolve().parent / "phase8_papers.json")
    if any(item["comparison_level"] != "method_only" for item in sources):
        raise HTTPException(status_code=503, detail="Mức so sánh bài báo chưa được kiểm định.")
    return {
        "dataset": "M5", "store": "CA_1",
        "series_count": 20,
        "pricing_evaluation_skus": len(data["price_manifest"]["evaluation_skus"]),
        "pricing_test_eligible_rows": int(data["price_manifest"]["cohort"]["test"]["eligible_rows"]),
        "training_protocol": {"split": "70/15/15 theo thời gian",
                              "history_weeks": 12,
                              "forecast_model": "Transformer có giá theo Song, 5 seed train-only",
                              "pricing_objective": "Giá × lượng bán dự báo; chưa có giá vốn",
                              "price_limit": "Miền giá train và ±10% so với giá tham chiếu",
                              "demand_selection": "Ba fold cuốn chiếu nằm trong train; chọn theo RMSE tại giá đã bán"},
        "forecast_test": json.loads(forecast.to_json(orient="records", force_ascii=False)),
        "demand_cv": json.loads(cv.to_json(orient="records", force_ascii=False)),
        "demand_holdout": json.loads(demand.to_json(orient="records", force_ascii=False)),
        "demand_selected_model": data["demand_manifest"]["selection"]["model"],
        "policy_summary": json.loads(policies.to_json(orient="records", force_ascii=False)),
        "global_policy": data["price_manifest"]["global_validation_selection"],
        "per_sku_policy": data["price_manifest"]["per_sku_validation_selection"],
        "robustness": json.loads(robustness.to_json(orient="records", force_ascii=False)),
        "simulator_gate": data["price_manifest"]["simulator_test_observed_price_gate"],
        "papers": sources,
        "comparison_rule": "Chỉ so số trực tiếp khi cohort, độ hạt, split, horizon, chỉ số và simulator cùng giao thức.",
        "deployment_supported": False,
    }
