"""One-step M5 scenarios for frozen PPO and the paired online-feedback PPO.

The online mode first reproduces simulated decisions and forecast updates
before the selected test week. Neither mode claims measured revenue uplift.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
import hashlib
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from stable_baselines3 import PPO
import torch

from .pricing_simulator import LagAdjustedKernelSimulator
from .research_api import _artifact_root, _csv, _file, _json
from .song_forecast import SongDemandForecaster, SongTransformer, feature_arrays
from .song_pricing_pilot import _environment, load_forecaster


router = APIRouter(prefix="/research/pricing/scenario", tags=["research"])
_infer_lock = Lock()


class ScenarioRequest(BaseModel):
    sku: str = Field(min_length=1)
    week: date
    current_price: float | None = Field(default=None, gt=0)
    seed: int = 42
    policy_variant: Literal["ppo_frozen", "ppo_feedback_gradient"] = "ppo_frozen"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root() -> Path:
    return _artifact_root() / "song_m5_pilot"


def _paths(root: Path) -> dict[str, Path]:
    paths = {
        "pricing_manifest": root / "pricing_lag_512_5seeds" / "pricing_manifest.json",
        "prepared_manifest": root / "manifest.json",
        "simulator_config": root / "simulator_selection" / "selected_simulator.json",
        "forecast_checkpoint": root / "multiseed" / "seed42" / "song" / "song_transformer_seed42.pt",
        "weekly": root / "weekly.csv",
        "windows": root / "windows.csv",
    }
    return {name: _file(path) for name, path in paths.items()}


@lru_cache(maxsize=2)
def _load_bundle(root_text: str, versions: tuple[int, ...]):
    del versions
    root = Path(root_text)
    paths = _paths(root)
    manifest = _json(paths["pricing_manifest"])
    config = _json(paths["simulator_config"])
    if manifest.get("simulator_variant") != "lag_adjusted_kernel":
        raise HTTPException(status_code=503, detail="Checkpoint PPO không thuộc simulator hiệu chỉnh.")
    checks = (
        (paths["prepared_manifest"], manifest["source_manifest_sha256"]),
        (paths["forecast_checkpoint"], manifest["forecast_checkpoint_sha256"]),
        (paths["simulator_config"], manifest["simulator_config_sha256"]),
        (paths["weekly"], config["prepared_weekly_sha256"]),
    )
    if any(_sha256(path) != expected for path, expected in checks):
        raise HTTPException(status_code=503, detail="Dữ liệu hoặc checkpoint không khớp lượt chạy Phase 4.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weekly = _csv(paths["weekly"])
    weekly = weekly.copy()
    weekly["week"] = pd.to_datetime(weekly["week"])
    simulator = LagAdjustedKernelSimulator(
        float(config["recent_weight"]), ratio_limit=float(config["ratio_limit"])).fit(weekly)
    forecaster = load_forecaster(paths["forecast_checkpoint"])
    return manifest, simulator, forecaster


def _bundle():
    root = _root()
    paths = _paths(root)
    versions = tuple(path.stat().st_mtime_ns for path in paths.values())
    return _load_bundle(str(root), versions)


@lru_cache(maxsize=6)
def _policy(path_text: str, modified_ns: int) -> PPO:
    del modified_ns
    return PPO.load(path_text, device="cpu")


def _evaluation_rows(sku: str, manifest: dict) -> pd.DataFrame:
    if sku not in manifest["evaluation_skus"]:
        raise HTTPException(status_code=404, detail="Sản phẩm không thuộc 10 chuỗi M5 đã đánh giá.")
    windows = _csv(_root() / "windows.csv")
    rows = windows.loc[windows["sku"].astype(str).eq(sku)
                       & windows["split"].eq("test")
                       & windows["observed_sales"].astype(bool)].copy()
    rows["week"] = pd.to_datetime(rows["week"])
    rows = rows.sort_values("week").head(20).reset_index(drop=True)
    if len(rows) != 20:
        raise HTTPException(status_code=503, detail="Thiếu 20 tuần test của sản phẩm.")
    return rows


def _online_root() -> Path:
    return _root() / "phase5c_online_state"


@lru_cache(maxsize=2)
def _online_manifest(modified_ns: int) -> dict:
    del modified_ns
    path = _file(_online_root() / "manifest.json")
    saved = _json(path)
    checks = (
        (_root() / "pricing_lag_512_5seeds" / "pricing_manifest.json", "pricing_manifest_sha256"),
        (_root() / "manifest.json", "prepared_manifest_sha256"),
        (_root() / "simulator_selection" / "selected_simulator.json", "simulator_config_sha256"),
        (_root() / "weekly.csv", "weekly_sha256"),
        (_root() / "windows.csv", "windows_sha256"),
        (_root() / "pricing_lag_512_5seeds" / "policy_episodes.csv", "phase4_episodes_sha256"),
    )
    if any(_sha256(_file(path)) != saved[key] for path, key in checks):
        raise HTTPException(status_code=503, detail="Trạng thái PPO online không khớp dữ liệu Phase 4.")
    return saved


def _online_record(seed: int) -> tuple[dict, Path, Path]:
    manifest_path = _file(_online_root() / "manifest.json")
    manifest = _online_manifest(manifest_path.stat().st_mtime_ns)
    record = next((item for item in manifest["records"] if item["seed"] == seed), None)
    if record is None or record["replay_audit"]["rows"] != 200:
        raise HTTPException(status_code=503, detail="Seed chưa có trạng thái online đã đối chiếu.")
    model_path = _file(_online_root() / "models" / f"ppo_feedback_gradient_seed{seed}.zip")
    state_path = _file(_online_root() / "forecasters" / f"ppo_feedback_gradient_seed{seed}.pt")
    if (_sha256(model_path) != record["policy_zip_sha256"]
            or _sha256(state_path) != record["forecaster_checkpoint_sha256"]):
        raise HTTPException(status_code=503, detail="Checkpoint PPO online bị thay đổi sau khi đối chiếu.")
    return record, model_path, state_path


@lru_cache(maxsize=6)
def _online_state(path_text: str, modified_ns: int) -> dict:
    del modified_ns
    return torch.load(path_text, map_location="cpu", weights_only=False)


def _online_forecaster(saved: dict) -> SongDemandForecaster:
    model = SongTransformer()
    model.load_state_dict(saved["state_dict"])
    return SongDemandForecaster(model, saved["scalers"])


@router.get("/options")
def scenario_options() -> dict:
    manifest, simulator, _ = _bundle()
    online_ready = False
    try:
        online_path = _file(_online_root() / "manifest.json")
        online = _online_manifest(online_path.stat().st_mtime_ns)
        online_ready = set(manifest["seeds"]) == {record["seed"] for record in online["records"]}
    except HTTPException:
        pass
    items = []
    for sku in manifest["evaluation_skus"]:
        rows = _evaluation_rows(sku, manifest)
        surface = simulator.surfaces[sku]
        items.append({
            "sku": sku, "price_min": float(surface.price_min),
            "price_max": float(surface.price_max),
            "weeks": [{"week": row.week.date().isoformat(),
                       "previous_price": float(row.price_lag_1),
                       "previous_units": float(row.qty_lag_1)}
                      for row in rows.itertuples(index=False)],
        })
    return {
        "policy_variants": ([{"id": "ppo_feedback_gradient", "label": "PPO có cập nhật dự báo"}]
                            if online_ready else []) + [
            {"id": "ppo_frozen", "label": "PPO dự báo cố định"}],
        "simulator": "lag_adjusted_kernel", "simulator_label": "Kernel hiệu chỉnh",
        "seeds": manifest["seeds"], "items": items,
        "scope": ("PPO online phát lại các tuần mô phỏng trước tuần chọn; PPO cố định dùng 12 tuần lịch sử quan sát."
                  if online_ready else "Chạy Phase 5C để mở PPO online; PPO cố định dùng lịch sử quan sát."),
    }


def _forecast_curve(forecaster, row: pd.Series, prices: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame([row.to_dict()] * len(prices))
    history, current = feature_arrays(frame, forecaster.scalers, prices.astype(np.float32))
    with torch.no_grad():
        z = forecaster.model(torch.from_numpy(history), torch.from_numpy(current)).numpy()
    scaler = forecaster.scalers[str(row["sku"])]
    return np.maximum(0.0, z * float(scaler["qty_std"]) + float(scaler["qty_mean"]))


def _decision(request: ScenarioRequest, env, observation: np.ndarray, policy,
              simulator: LagAdjustedKernelSimulator, *, prior_decisions: int,
              current_price_source: str) -> dict:
    forecaster = env.forecaster
    assert forecaster is not None
    surface = simulator.surfaces[request.sku]
    current_price = float(env.current_price)
    if tuple(policy.observation_space.shape) != tuple(env.observation_space.shape):
        raise HTTPException(status_code=503, detail="Kích thước đầu vào PPO không khớp môi trường.")
    action = float(np.asarray(policy.predict(observation, deterministic=True)[0]).reshape(-1)[0])
    recommended = env.price_from_action(action)
    row = env._current_row()
    forecast_current = float(forecaster.predict_q(row, current_price))
    forecast_recommended = float(forecaster.predict_q(row, recommended))
    gradient = float(forecaster.price_gradient(row, current_price))
    context = {key: row[key] for key in ("event_days", "snap_days", "qty_lag_1", "price_lag_1")
               if key in row.index}
    simulator_current = float(simulator.predict_mean(request.sku, current_price, row["week"], context))
    simulator_recommended = float(simulator.predict_mean(request.sku, recommended, row["week"], context))
    prices = np.linspace(surface.price_min, surface.price_max, 41)
    forecast_units = _forecast_curve(forecaster, row, prices)
    simulator_units = [float(simulator.predict_mean(request.sku, float(price), row["week"], context))
                       for price in prices]
    current_revenue = current_price * forecast_current
    recommended_revenue = recommended * forecast_recommended
    simulated_current_revenue = current_price * simulator_current
    simulated_recommended_revenue = recommended * simulator_recommended
    online = request.policy_variant == "ppo_feedback_gradient"
    return {
        "sku": request.sku, "week": request.week.isoformat(), "seed": request.seed,
        "policy": request.policy_variant,
        "policy_label": ("PPO có cập nhật dự báo" if online else "PPO dự báo cố định"),
        "simulator": "lag_adjusted_kernel",
        "price_min": float(surface.price_min), "price_max": float(surface.price_max),
        "current_price": current_price, "current_price_source": current_price_source,
        "recommended_price": recommended,
        "price_change_pct": 100 * (recommended / current_price - 1),
        "policy_action": action,
        "prior_simulated_decisions": prior_decisions,
        "forecast_updates_before_week": env.feedback_updates,
        "forecast": {
            "units_current": forecast_current, "units_recommended": forecast_recommended,
            "revenue_current": current_revenue, "revenue_recommended": recommended_revenue,
            "revenue_change": recommended_revenue - current_revenue,
            "revenue_change_pct": (100 * (recommended_revenue / current_revenue - 1)
                                   if current_revenue > 0 else None),
            "demand_gradient_at_current_price": gradient,
        },
        "simulator_expected": {
            "units_current": simulator_current, "units_recommended": simulator_recommended,
            "revenue_current": simulated_current_revenue,
            "revenue_recommended": simulated_recommended_revenue,
            "revenue_change": simulated_recommended_revenue - simulated_current_revenue,
        },
        "context": {
            "previous_week_price": float(row["price_lag_1"]),
            "previous_week_units": float(row["qty_lag_1"]),
            "simulated_history_weeks": min(prior_decisions, 12) if online else 0,
            "event_days": float(row.get("event_days", 0)),
            "snap_days": float(row.get("snap_days", 0)),
            "history": [{"weeks_before": lag,
                         "units": float(row[f"qty_lag_{lag}"]),
                         "price": float(row[f"price_lag_{lag}"])}
                        for lag in range(1, 13)],
        },
        "curve": [{"price": float(price), "forecast_units": float(q),
                   "forecast_revenue": float(price * q),
                   "simulator_units": float(sq),
                   "simulator_revenue": float(price * sq)}
                  for price, q, sq in zip(prices, forecast_units, simulator_units)],
        "interpretation": (
            "The online policy replays simulated decisions before this week and updates the "
            "forecaster after each ten simulated sales. The selected week's actual demand is hidden."
            if online else
            "The frozen policy uses twelve observed previous weeks without online updates."
        ),
    }


@router.post("")
def scenario_infer(request: ScenarioRequest) -> dict:
    manifest, simulator, base_forecaster = _bundle()
    if request.seed not in manifest["seeds"]:
        raise HTTPException(status_code=422, detail="Seed không thuộc lượt chạy Phase 4.")
    rows = _evaluation_rows(request.sku, manifest)
    target = rows.loc[rows["week"].dt.date.eq(request.week)]
    if len(target) != 1:
        raise HTTPException(status_code=422, detail="Chọn một trong 20 tuần test của sản phẩm.")
    surface = simulator.surfaces[request.sku]
    if request.current_price is not None and not (
            np.isfinite(request.current_price)
            and surface.price_min <= request.current_price <= surface.price_max):
        raise HTTPException(status_code=422, detail=(
            f"Giá hiện tại phải trong miền giá train {surface.price_min:.2f}–{surface.price_max:.2f}."))
    with _infer_lock:
        if request.policy_variant == "ppo_frozen":
            model_path = _file(_root() / "pricing_lag_512_5seeds" / "models" /
                               f"ppo_frozen_seed{request.seed}.zip")
            policy = _policy(str(model_path), model_path.stat().st_mtime_ns)
            env = _environment(target.drop(columns=["qty", "revenue"], errors="ignore"),
                               simulator, "test", base_forecaster,
                               gradient=True, feedback=False)
            env.reset(seed=request.seed, options={"sku": request.sku})
            env.current_price = (float(request.current_price) if request.current_price is not None
                                 else float(target.iloc[0]["price_lag_1"]))
            observation = env._observation()
            return _decision(request, env, observation, policy, simulator,
                             prior_decisions=0,
                             current_price_source=("user_input" if request.current_price is not None
                                                   else "observed_previous_week"))

        _, model_path, state_path = _online_record(request.seed)
        policy = _policy(str(model_path), model_path.stat().st_mtime_ns)
        saved = _online_state(str(state_path), state_path.stat().st_mtime_ns)
        if saved["seed"] != request.seed or request.sku not in saved["rng_by_sku"]:
            raise HTTPException(status_code=503, detail="Trạng thái online không khớp seed hoặc sản phẩm.")
        with torch.random.fork_rng(devices=[]):
            forecaster = _online_forecaster(saved)
            all_rows = pd.concat([_evaluation_rows(sku, manifest)
                                  for sku in manifest["evaluation_skus"]], ignore_index=True)
            all_rows = all_rows.drop(columns=["qty", "revenue"], errors="ignore")
            env = _environment(all_rows, simulator, "test", forecaster,
                               gradient=True, feedback=True)
            sku_index = env.available_skus.index(request.sku)
            week_index = int(rows.index[rows["week"].dt.date.eq(request.week)][0])
            torch.set_rng_state(saved["rng_by_sku"][request.sku].clone())
            observation, _ = env.reset(seed=request.seed + sku_index,
                                       options={"sku": request.sku})
            for _ in range(week_index):
                action = policy.predict(observation, deterministic=True)[0]
                observation, _, done, _, _ = env.step(action)
                if done:
                    raise HTTPException(status_code=503, detail="Chuỗi mô phỏng kết thúc trước tuần chọn.")
            if request.current_price is not None:
                env.current_price = float(request.current_price)
                observation = env._observation()
            return _decision(request, env, observation, policy, simulator,
                             prior_decisions=week_index,
                             current_price_source=("user_input" if request.current_price is not None
                                                   else "train_mean_initial" if week_index == 0
                                                   else "simulated_rollout"))
