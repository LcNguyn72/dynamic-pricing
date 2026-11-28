"""One-step quantity forecasts and pricing from observed weekly history."""

from __future__ import annotations

from datetime import date, timedelta
import math
from statistics import mean

from .bundle import ModelBundle
from .feedback import update_arimax
from .schemas import ForecastRequest, PriceRequest, WeekObservation


class InferenceError(ValueError):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code = code
        self.status = status


def _history(bundle, dataset, week, observations):
    if week.weekday() != 0 or any(row.week.weekday() != 0 for row in observations):
        raise InferenceError("invalid_week", "Tuần dự báo và lịch sử phải bắt đầu vào thứ Hai.")
    if any(row.week >= week for row in observations):
        raise InferenceError("future_history", "Lịch sử chỉ được chứa tuần trước tuần dự báo.")
    if len({row.week for row in observations}) != len(observations):
        raise InferenceError("duplicate_week", "Lịch sử có tuần trùng.")
    dates = bundle.manifest["config"].get("datasets", {}).get(dataset, {})
    if "train_end" in dates and week <= date.fromisoformat(dates["train_end"]):
        raise InferenceError("week_in_training_period", "Chọn tuần sau kỳ huấn luyện của mô hình.")
    return max(observations, key=lambda row: row.week)


def _series(bundle, sku, store):
    if not store:
        raise InferenceError("store_required", "M5 cần mã cửa hàng.")
    item, separator, qualified_store = sku.partition("@")
    if not item or separator and qualified_store != store:
        raise InferenceError("store_mismatch", "Mã SKU và cửa hàng không khớp.")
    key = f"{item}@{store}"
    if key not in bundle.data["m5"]:
        raise InferenceError("unknown_series", "Không tìm thấy chuỗi M5 trong bộ mô hình.", 404)
    return key, bundle.data["m5"][key]


def _arimax_state(series, week, history):
    state = series["arimax"]
    if state is None:
        raise InferenceError("price_model_unavailable", "SKU và cửa hàng này chưa có mô hình ARIMAX đủ điều kiện.")
    if history.week != week - timedelta(days=7) or history.price is None:
        raise InferenceError("missing_previous_week", "ARIMAX cần giá và lượng bán quan sát của tuần liền trước.")
    return state


def _shared_demand(bundle, series, request, previous):
    state = _arimax_state(series, request.week, previous)
    feedback = request.feedback
    if any(row.week.weekday() != 0 for row in feedback):
        raise InferenceError("invalid_week", "Tuần phản hồi phải bắt đầu vào thứ Hai.")
    if any(row.week >= request.week for row in feedback):
        raise InferenceError("future_feedback", "Phản hồi phải đến từ tuần trước tuần dự báo.")
    if len({row.week for row in feedback}) != len(feedback):
        raise InferenceError("duplicate_feedback_week", "Phản hồi có tuần trùng.")
    dates = bundle.manifest["config"].get("datasets", {}).get("m5", {})
    if "train_end" in dates:
        train_end = date.fromisoformat(dates["train_end"])
        feedback = [row for row in feedback if row.week > train_end]
    return update_arimax(state, feedback)


def _price_point(state, history: WeekObservation, price):
    intercept, beta, phi, _ = state["params"]
    quantity_lag_z = (history.qty - state["qty_mean"]) / state["qty_sd"]
    price_lag_z = (history.price - state["price_mean"]) / state["price_sd"]
    price_z = (price - state["price_mean"]) / state["price_sd"]
    z = intercept + phi * (quantity_lag_z - beta * price_lag_z) + beta * price_z
    prediction = state["qty_mean"] + z * state["qty_sd"]
    if not math.isfinite(prediction):
        raise InferenceError("forecast_not_finite", "Lịch sử đầu vào làm dự báo vượt phạm vi số hợp lệ.")
    quantity = max(0.0, prediction)
    revenue = price * quantity
    if not math.isfinite(revenue):
        raise InferenceError("forecast_not_finite", "Lịch sử đầu vào làm dự báo vượt phạm vi số hợp lệ.")
    return {"price": price, "predicted_qty": quantity, "expected_revenue": revenue}


def forecast(bundle: ModelBundle, request: ForecastRequest):
    week = request.week
    previous = _history(bundle, request.dataset, week, request.history)
    response = {"dataset": request.dataset, "sku": request.sku, "store_id": request.store_id, "week": week,
                "source_week": previous.week, "model_version": bundle.manifest["version"], "unit": "items_per_week"}
    if request.dataset == "uci":
        if request.store_id is not None or request.candidate_price is not None or request.feedback:
            raise InferenceError("uci_quantity_only", "UCI chỉ nhận yêu cầu dự báo lượng bán theo SKU.")
        if request.sku not in bundle.data["uci_skus"]:
            raise InferenceError("unknown_series", "Không tìm thấy SKU UCI trong bộ mô hình.", 404)
        if previous.week != week - timedelta(days=7):
            raise InferenceError("missing_previous_week", "Dự báo UCI cần lượng bán của tuần liền trước.")
        uci = bundle.manifest["config"].get("datasets", {}).get("uci", {})
        if uci.get("forecast") == "rolling_mean_4":
            quantities = {row.week: row.qty for row in request.history}
            lag_weeks = [week - timedelta(weeks=lag) for lag in range(1, 5)]
            if all(lag_week in quantities for lag_week in lag_weeks):
                prediction = mean(quantities[lag_week] for lag_week in lag_weeks)
                return {**response, "method": "rolling_mean_4", "predicted_qty": prediction}
        return {**response, "method": "previous_week", "predicted_qty": previous.qty}

    key, series = _series(bundle, request.sku, request.store_id)
    response["series_id"] = key
    m5 = bundle.manifest["config"].get("datasets", {}).get("m5", {})
    if request.candidate_price is not None or request.feedback or m5.get("forecast") == "arimax_1_0_0_shared":
        state, feedback_count = _shared_demand(bundle, series, request, previous)
        price = previous.price if request.candidate_price is None else request.candidate_price
        if not state["price_min"] <= price <= state["price_max"]:
            raise InferenceError("price_outside_train_range", "Chọn giá trong miền giá huấn luyện của SKU.")
        point = _price_point(state, previous, price)
        return {**response, "method": "arimax_1_0_0", "candidate_price": point["price"],
                "predicted_qty": point["predicted_qty"], "feedback_count": feedback_count}

    state = series["ar1"]
    if state is None:
        return {**response, "method": "last_observed", "predicted_qty": previous.qty}
    intercept, phi, _ = state["params"]
    steps = (week - previous.week).days // 7
    decay = phi ** steps
    intercept_steps = steps if phi == 1 else (1 - decay) / (1 - phi)
    prediction = (state["qty_mean"] + intercept_steps * intercept * state["qty_sd"]
                  + decay * (previous.qty - state["qty_mean"]))
    if not math.isfinite(prediction):
        raise InferenceError("forecast_not_finite", "Lịch sử đầu vào làm dự báo vượt phạm vi số hợp lệ.")
    return {**response, "method": "arima_1_0_0", "predicted_qty": max(0.0, prediction)}


def recommend_price(bundle: ModelBundle, request: PriceRequest, policy=None):
    week = request.week
    previous = _history(bundle, "m5", week, request.history)
    key, series = _series(bundle, request.sku, request.store_id)
    state, feedback_count = _shared_demand(bundle, series, request, previous)
    low, high = state["price_min"], state["price_max"]
    current = previous.price if request.current_price is None else request.current_price
    points = bundle.manifest["config"]["pricing"]["grid_points"]
    grid = [low + (high - low) * i / (points - 1) for i in range(points - 1)] + [high]
    in_support = low <= current <= high
    if in_support and not any(abs(current - price) <= 1e-12 for price in grid):
        grid.append(current)
    candidates = [_price_point(state, previous, price) for price in sorted(grid)]
    max_revenue = max(point["expected_revenue"] for point in candidates)
    ties = [point for point in candidates if max_revenue - point["expected_revenue"] <= 1e-8]
    best = min(ties, key=lambda point: (abs(point["price"] - current), point["price"]))

    recommendation, policy_version = best, None
    if request.strategy == "ppo_feedback":
        if policy is None:
            raise InferenceError("ppo_policy_unavailable", "Policy PPO chưa được nạp vào API mô hình.", 503)
        try:
            policy_price = policy.recommend(sku=request.sku, store_id=request.store_id,
                                            previous_qty=previous.qty, previous_price=previous.price,
                                            current_price=current, week=week, state=state)
        except LookupError as exc:
            raise InferenceError("ppo_series_unavailable", "PPO chưa hỗ trợ cặp sản phẩm và cửa hàng này.") from exc
        if not low <= policy_price <= high:
            raise InferenceError("invalid_ppo_price", "Policy PPO trả giá ngoài miền đã huấn luyện.", 503)
        recommendation = _price_point(state, previous, policy_price)
        policy_version = policy.version

    reference = series["fixed_reference_price"]
    dates = bundle.manifest["config"].get("datasets", {}).get("m5", {})
    if reference is None:
        status = "no_validated_reference"
    elif "validation_end" in dates and week <= date.fromisoformat(dates["validation_end"]):
        reference, status = None, "available_after_validation"
    else:
        status = "available"
    return {"dataset": "m5", "series_id": key, "week": week, "model_version": bundle.manifest["version"],
            "method": f"arimax_1_0_0_{request.strategy}", "strategy": request.strategy,
            "policy_version": policy_version, "feedback_count": feedback_count,
            "currency": "USD", "source_week": previous.week,
            "current_price": current, "current_price_in_train_range": in_support,
            "current_price_point": _price_point(state, previous, current) if in_support else None,
            "train_price_range": {"min": low, "max": high},
            "grid_result": best, "recommendation": recommendation, "fixed_reference_status": status,
            "fixed_reference": None if reference is None else _price_point(state, previous, reference),
            "candidates": candidates}
