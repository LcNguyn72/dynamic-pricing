"""Request schemas shared by the Python engine and model HTTP API."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Quantity = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=100)]


class WeekObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week: date
    qty: Quantity
    price: Positive | None = None


class HistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Identifier
    week: date
    history: Annotated[list[WeekObservation], Field(min_length=1, max_length=277)]


class ForecastRequest(HistoryRequest):
    dataset: Literal["uci", "m5"]
    store_id: Identifier | None = None
    candidate_price: Positive | None = None


class PriceRequest(HistoryRequest):
    store_id: Identifier
    current_price: Positive | None = None


class ForecastResponse(BaseModel):
    dataset: Literal["uci", "m5"]
    sku: str
    store_id: str | None
    week: date
    source_week: date
    model_version: str
    unit: Literal["items_per_week"]
    method: Literal["previous_week", "last_observed", "arima_1_0_0", "arimax_1_0_0"]
    predicted_qty: Quantity
    series_id: str | None = None
    candidate_price: Positive | None = None


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
    method: Literal["arimax_1_0_0_grid"]
    currency: Literal["USD"]
    source_week: date
    current_price: Positive
    current_price_in_train_range: bool
    train_price_range: PriceRange
    grid_result: PricePoint
    fixed_reference_status: Literal["available", "no_validated_reference", "available_after_validation"]
    fixed_reference: PricePoint | None
    candidates: list[PricePoint]
