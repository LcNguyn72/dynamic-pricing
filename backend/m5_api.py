"""Serve the M5 shop UI, SQL Server data, and pricing model gateway."""

from __future__ import annotations

from contextlib import closing
from datetime import date, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import Depends, FastAPI, Query, Request as ClientRequest
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
import pyodbc

from database.m5_connection import connect_m5
from database.m5_repository import M5DataError, M5Repository


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
M5_ASSETS = {"m5.css", "m5.js", "m5-basket.webp", "m5-products.webp"}
LOGGER = logging.getLogger(__name__)
POSITIVE_PRICE = Annotated[float, Field(gt=0, le=1_000_000, allow_inf_nan=False)]
STORE_KEY = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9_]{1,16}$")]
PRODUCT_KEY = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9_]{1,32}$")]


class WeekInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store_id: STORE_KEY
    sku: PRODUCT_KEY
    week: date

    @field_validator("week")
    @classmethod
    def monday(cls, value: date) -> date:
        if value.weekday() != 0:
            raise ValueError("Tuần bắt đầu vào thứ Hai.")
        return value


class ForecastInput(WeekInput):
    candidate_price: POSITIVE_PRICE | None = None


class PriceInput(WeekInput):
    current_price: POSITIVE_PRICE | None = None
    strategy: Literal["grid", "ppo_feedback"] = "grid"


class CatalogPriceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price: Annotated[Decimal, Field(gt=0, le=1_000_000, decimal_places=2)]


class CatalogBatchPriceInput(CatalogPriceInput):
    store_ids: Annotated[list[STORE_KEY], Field(min_length=1, max_length=10)]

    @field_validator("store_ids")
    @classmethod
    def unique_stores(cls, stores: list[str]) -> list[str]:
        if len(stores) != len(set(stores)):
            raise ValueError("Mỗi cửa hàng chỉ được chọn một lần.")
        return stores


class FeedbackInput(WeekInput):
    source: Literal["actual", "m5_history"] = "actual"
    applied_price: POSITIVE_PRICE | None = None
    observed_qty: Annotated[int, Field(ge=0, le=2_147_483_647)] | None = None

    @model_validator(mode="after")
    def require_outcome(self):
        if self.source == "actual" and (self.applied_price is None or self.observed_qty is None):
            raise ValueError("Cần giá đã áp dụng và số lượng bán thực tế.")
        return self


app = FastAPI(title="M5 Shop API")


def get_repository():
    with closing(connect_m5()) as db:
        yield M5Repository(db)


Repository = Annotated[M5Repository, Depends(get_repository)]


@app.exception_handler(M5DataError)
def data_error(_request: ClientRequest, exc: M5DataError):
    return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": str(exc)})


@app.exception_handler(RequestValidationError)
def request_error(_request: ClientRequest, exc: RequestValidationError):
    detail = [{key: item[key] for key in ("loc", "msg", "type")} for item in exc.errors()]
    return JSONResponse(status_code=422, content={"code": "invalid_request", "detail": detail})


@app.exception_handler(pyodbc.Error)
def database_error(_request: ClientRequest, exc: pyodbc.Error):
    LOGGER.error("M5 SQL Server request failed", exc_info=exc)
    return JSONResponse(status_code=503, content={
        "code": "m5_database_unavailable", "message": "Không truy vấn được dữ liệu M5 trong SQL Server.",
    })


def call_model(path: str, payload: dict | None = None) -> JSONResponse:
    base_url = os.environ.get("M5_MODEL_URL", "http://127.0.0.1:8766").rstrip("/")
    body = json.dumps(jsonable_encoder(payload)).encode() if payload is not None else None
    request = Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            return JSONResponse(status_code=response.status, content=json.load(response))
    except HTTPError as exc:
        try:
            status, content = exc.code, json.load(exc)
        except ValueError:
            raise M5DataError("invalid_model_response", "Dịch vụ mô hình trả về dữ liệu không hợp lệ.", 502) from exc
        finally:
            exc.close()
        return JSONResponse(status_code=status, content=content)
    except (URLError, TimeoutError) as exc:
        raise M5DataError("model_unavailable", "Dịch vụ mô hình M5 chưa sẵn sàng.", 503) from exc
    except ValueError as exc:
        raise M5DataError("invalid_model_response", "Dịch vụ mô hình trả về dữ liệu không hợp lệ.", 502) from exc


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "m5.html")


@app.get("/m5-assets/{name}", include_in_schema=False)
def assets(name: str):
    if name not in M5_ASSETS:
        return JSONResponse(status_code=404, content={"code": "asset_not_found"})
    return FileResponse(FRONTEND / name)


@app.get("/api/m5/stores")
def stores(repository: Repository):
    return {"stores": repository.stores()}


@app.get("/api/m5/catalog")
def catalog(repository: Repository):
    return {"items": repository.catalog()}


@app.get("/api/m5/home-highlights")
def home_highlights(repository: Repository):
    return repository.home_highlights()


@app.get("/api/m5/catalog/{sku}")
def catalog_item(sku: PRODUCT_KEY, repository: Repository):
    return repository.catalog_item(sku)


@app.put("/api/m5/catalog/{sku}/stores/{store_id}/price")
def set_catalog_price(sku: PRODUCT_KEY, store_id: STORE_KEY, data: CatalogPriceInput, repository: Repository):
    result = repository.set_catalog_price(sku, store_id, data.price)
    repository.db.commit()
    return result


@app.put("/api/m5/catalog/{sku}/price")
def set_catalog_prices(sku: PRODUCT_KEY, data: CatalogBatchPriceInput, repository: Repository):
    results = repository.set_catalog_prices(sku, data.store_ids, data.price)
    repository.db.commit()
    return {"sku": sku, "price": data.price, "stores": results}


@app.get("/api/m5/products")
def products(store_id: STORE_KEY, repository: Repository, q: Annotated[str, Query(max_length=32)] = "",
             limit: Annotated[int, Query(ge=1, le=100)] = 24, offset: Annotated[int, Query(ge=0)] = 0):
    return repository.products(store_id, q.strip(), limit, offset)


@app.get("/api/m5/products/{sku}/weeks")
def product_weeks(sku: PRODUCT_KEY, store_id: STORE_KEY, repository: Repository,
                  limit: Annotated[int, Query(ge=1, le=277)] = 52, before: date | None = None):
    return repository.product_weeks(store_id, sku, limit, before)


@app.get("/api/m5/stats")
def stats(store_id: STORE_KEY, repository: Repository):
    return repository.stats(store_id)


@app.get("/api/m5/model-info")
def model_info():
    return call_model("/model-info")


def model_payload(data: WeekInput, repository: M5Repository) -> dict:
    history, feedback = repository.model_inputs(data.store_id, data.sku, data.week)
    if not history or history[-1]["week"] != data.week - timedelta(weeks=1):
        raise M5DataError("missing_previous_week", "Cần giá và lượng bán quan sát của tuần liền trước.")
    return {**data.model_dump(exclude_none=True), "dataset": "m5", "history": history, "feedback": feedback}


@app.post("/api/m5/forecast")
def forecast(data: ForecastInput, repository: Repository):
    payload = model_payload(data, repository)
    payload.setdefault("candidate_price", payload["history"][-1]["price"])
    return call_model("/forecast", payload)


@app.post("/api/m5/recommend-price")
def recommend_price(data: PriceInput, repository: Repository):
    return call_model("/recommend-price", model_payload(data, repository))


@app.post("/api/m5/feedback")
def save_feedback(data: FeedbackInput, repository: Repository):
    result = repository.save_feedback(data.store_id, data.sku, data.week, source=data.source,
                                      applied_price=data.applied_price, observed_qty=data.observed_qty)
    repository.db.commit()
    return result
