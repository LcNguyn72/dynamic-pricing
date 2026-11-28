"""Request schemas shared by the Python engine and model HTTP API."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Quantity = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=100)]
PriceStrategy = Literal["grid", "ppo_feedback"]


class WeekObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week: date
    qty: Quantity
    price: Positive | None = None


class FeedbackObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week: date
    qty: Quantity
    price: Positive
    previous_qty: Quantity
    previous_price: Positive


class HistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Identifier
    week: date
    history: Annotated[list[WeekObservation], Field(min_length=1, max_length=277)]
    feedback: Annotated[list[FeedbackObservation], Field(max_length=32)] = Field(default_factory=list)


class ForecastRequest(HistoryRequest):
    dataset: Literal["uci", "m5"]
    store_id: Identifier | None = None
    candidate_price: Positive | None = None


class PriceRequest(HistoryRequest):
    dataset: Literal["m5"] = "m5"
    store_id: Identifier
    current_price: Positive | None = None
    strategy: PriceStrategy = "grid"


class ForecastResponse(BaseModel):
    dataset: Literal["uci", "m5"]
    sku: str
    store_id: str | None
    week: date
    source_week: date
    model_version: str
    unit: Literal["items_per_week"]
    method: Literal["rolling_mean_4", "previous_week", "last_observed", "arima_1_0_0", "arimax_1_0_0"]
    predicted_qty: Quantity
    series_id: str | None = None
    candidate_price: Positive | None = None
    feedback_count: int | None = None


class PricePoint(BaseModel):
    price: Positive
    predicted_qty: Quantity
    expected_revenue: Quantity


class PriceRange(BaseModel):
    min: Positive
    max: Positive


class PriceResponse(BaseModel):
    dataset: Literal["m5"]
    series_id: str
    week: date
    model_version: str
    method: Literal["arimax_1_0_0_grid", "arimax_1_0_0_ppo_feedback"]
    strategy: PriceStrategy
    policy_version: str | None = None
    feedback_count: int
    currency: Literal["USD"]
    source_week: date
    current_price: Positive
    current_price_in_train_range: bool
    current_price_point: PricePoint | None
    train_price_range: PriceRange
    grid_result: PricePoint
    recommendation: PricePoint
    fixed_reference_status: Literal["available", "no_validated_reference", "available_after_validation"]
    fixed_reference: PricePoint | None
    candidates: list[PricePoint]
