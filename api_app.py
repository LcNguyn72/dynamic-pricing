"""
API định giá (TV2) - FastAPI
============================
Cài đặt:   pip install fastapi uvicorn
Chạy:      uvicorn api_app:app --port 8000
Tài liệu:  http://localhost:8000/docs   (Swagger tự động)

Endpoint:
  GET  /health                     tình trạng + phiên bản mô hình
  GET  /model_info                 thông tin mô hình (phương pháp, tau2, sigma_extra...)
  GET  /products                   danh mục sản phẩm có thể định giá (kèm elasticity + bất định)
  POST /recommend                  đề xuất giá cho 1 sản phẩm
  POST /recommend/batch            đề xuất giá cho nhiều sản phẩm (lỗi từng mục không làm hỏng cả lô)
  GET  /demand_curve/{sku}         đường giá - lượng bán - doanh thu (kèm khoảng) cho dashboard
  GET  /demo/products              20 sản phẩm demo và tổng hợp dữ liệu lịch sử từ SQL Server
  GET  /demo/products/{sku}        chi tiết một sản phẩm demo
  GET  /demo/products/{sku}/weeks  giá và lượng bán theo tuần
  GET  /demo/products/{sku}/recommendations  nhật ký thử giá
  POST /demo/simulations           tính và lưu một lần thử giá của người dùng
  GET  /research/forecast/*        bảng điểm và dự báo test của Phase 3C
  GET  /research/pricing/*         bảng điểm và quyết định mô phỏng của Phase 4
  GET  /research/pricing/scenario/options  SKU, tuần test, miền giá của Phase 5B
  POST /research/pricing/scenario  suy luận một quyết định PPO cố định hoặc có phản hồi online trên M5
  GET  /research/price/options     SKU và tuần M5 hợp lệ cho giá Phase 8
  POST /research/price/recommend   tính giá M5 một bước từ checkpoint Phase 6
  POST /research/price/compare     tính giá và so sánh 7 phương pháp trên cùng kịch bản M5
  GET  /research/method/overview   chỉ số Phase 6/7 và đối chiếu bài báo

Các thư mục/tệp mô hình đọc từ ./models_2y (đặt biến môi trường PRICING_MODEL_DIR để đổi).
"""
import datetime as dt
import os
from typing import List, Optional

import pyodbc
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from database import repository as pricing_db
from experiments.pricing_inference_api import router as pricing_scenario_router
from experiments.phase8_api import router as phase8_router
from experiments.research_api import router as research_router
from pricing_engine import DEFAULT_DIR, MODEL_VERSION, PricingEngine

app = FastAPI(title="Dynamic Pricing API (TV2)", version="1.0",
              description="Đề xuất giá dựa trên elasticity từng sản phẩm có kèm độ bất định.")
app.include_router(research_router)
app.include_router(pricing_scenario_router)
app.include_router(phase8_router)
engine = PricingEngine(os.environ.get("PRICING_MODEL_DIR", DEFAULT_DIR))


class RecommendRequest(BaseModel):
    sku: str = Field(..., description="Mã sản phẩm (StockCode)")
    current_price: Optional[float] = Field(None, gt=0, description="Giá hiện tại; bỏ trống = giá tham chiếu")
    unit_cost: Optional[float] = Field(None, ge=0, description="Giá vốn; có -> tối ưu lợi nhuận, không -> doanh thu")
    date: Optional[dt.date] = Field(None, description="Ngày áp dụng giá (mùa vụ); bỏ trống = hôm nay")
    max_change: Optional[float] = Field(None, gt=0, le=0.5, description="Mức đổi giá tối đa (mặc định 0.10)")


class BatchRequest(BaseModel):
    items: List[RecommendRequest] = Field(..., max_length=500)


def _kwargs(r: RecommendRequest):
    return {"sku": r.sku, "current_price": r.current_price, "unit_cost": r.unit_cost,
            "date": r.date, "max_change": r.max_change}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": engine.model_info()["model_version"],
            "n_products": len(engine.params)}


@app.get("/model_info")
def model_info():
    return engine.model_info()


@app.get("/products")
def products():
    return engine.products()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        return engine.recommend(**_kwargs(req))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/recommend/batch")
def recommend_batch(req: BatchRequest):
    return {"results": engine.recommend_many([_kwargs(i) for i in req.items])}


def _database_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except pyodbc.Error as error:
        raise HTTPException(status_code=503, detail="SQL Server chưa sẵn sàng.") from error


def _selected_product(sku):
    item = _database_call(pricing_db.product, sku, MODEL_VERSION)
    if item is None:
        raise HTTPException(status_code=404, detail=f"SKU {sku} không thuộc 20 sản phẩm demo.")
    return item


@app.get("/demo/products")
def demo_products():
    """Only the 20 selected products, with historical summaries from SQL Server."""
    return _database_call(pricing_db.products, MODEL_VERSION)


@app.get("/demo/products/{sku}")
def demo_product(sku: str):
    return _selected_product(sku)


@app.get("/demo/products/{sku}/weeks")
def demo_product_weeks(sku: str):
    _selected_product(sku)
    return _database_call(pricing_db.weekly_sales, sku)


@app.get("/demo/products/{sku}/recommendations")
def demo_product_recommendations(sku: str, limit: int = Query(50, ge=1, le=100)):
    _selected_product(sku)
    return _database_call(pricing_db.recommendations, limit, sku)


@app.post("/demo/simulations")
def demo_simulation(req: RecommendRequest):
    """Calculate and record an explicit user simulation; batch previews remain read-only."""
    item = _selected_product(req.sku)
    try:
        result = engine.recommend(**_kwargs(req))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'\"")) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    reference = round(float(item["reference_price"]), 2)
    price_source = "model_reference" if req.current_price is None or abs(req.current_price - reference) < 0.005 else "user_input"
    recommendation_id = _database_call(
        pricing_db.save_recommendation, result, price_source,
        req.max_change if req.max_change is not None else engine.max_change
    )
    return {**result, "recommendation_id": recommendation_id}


@app.get("/demand_curve/{sku}")
def demand_curve(sku: str, current_price: Optional[float] = Query(None, gt=0),
                 unit_cost: Optional[float] = Query(None, ge=0), date: Optional[dt.date] = None,
                 max_change: Optional[float] = Query(None, gt=0, le=0.5), points: int = Query(21, ge=5, le=101)):
    try:
        return engine.demand_curve(sku, current_price, unit_cost, date, max_change, points)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
