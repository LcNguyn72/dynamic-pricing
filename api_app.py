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

Các thư mục/tệp mô hình đọc từ ./models_2y (đặt biến môi trường PRICING_MODEL_DIR để đổi).
"""
import datetime as dt
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from pricing_engine import DEFAULT_DIR, PricingEngine

app = FastAPI(title="Dynamic Pricing API (TV2)", version="1.0",
              description="Đề xuất giá dựa trên elasticity từng sản phẩm có kèm độ bất định.")
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


@app.get("/demand_curve/{sku}")
def demand_curve(sku: str, current_price: Optional[float] = Query(None, gt=0),
                 unit_cost: Optional[float] = Query(None, ge=0), date: Optional[dt.date] = None,
                 max_change: Optional[float] = Query(None, gt=0, le=0.5), points: int = Query(21, ge=5, le=101)):
    try:
        return engine.demand_curve(sku, current_price, unit_cost, date, max_change, points)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
