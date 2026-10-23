# Định giá động cho TMĐT bán lẻ (TV2 - AI Model)

Hệ thống gồm 3 phần: xử lý dữ liệu, huấn luyện mô hình nhu cầu và ước lượng elasticity, và API đề xuất giá.
README này hướng dẫn cách tải dữ liệu, cài đặt và chạy từ đầu.

## 1. Cấu trúc thư mục

Tất cả file nằm ở thư mục gốc của dự án:

```
<thư mục dự án>/
├── data/
│   └── online_retail_II.xlsx                    <- TẢI THỦ CÔNG (xem mục 2)
├── 1_data_processing_online_retail_2y.py
├── 2_data_processing_online_retail_ml_2y.py
├── 3_select_products_2y.py                      <- tùy chọn (xem mục 6)
├── 4_train_final_model_online_retail_2y.py
├── 5_fit_pricing_model_2y.py
├── pricing_engine.py
├── api_app.py
├── y2_config.py
├── y2_tools.py
├── test.py                                      <- xem mục 6
└── README.md
```

Thư mục `models_2y/` và file cache `data/online_retail_II_sheets.csv` sẽ được tạo tự động khi chạy. Không cần tạo tay.

## 2. Cần tải những gì

**Dữ liệu (bắt buộc, không đưa lên git):**

- Bộ dữ liệu: _Online Retail II_ của UCI Machine Learning Repository
  - Trang: https://archive.ics.uci.edu/dataset/502/online+retail+ii
  - Tải về, giải nén, lấy file `online_retail_II.xlsx`
  - Đặt vào `data/online_retail_II.xlsx`
  - Giấy phép CC BY 4.0. Khi báo cáo, nhớ trích dẫn: Chen, D. (2012), DOI 10.24432/C5CG6D

**Thư viện Python (bắt buộc):**

- Python 3.10 trở lên
- pandas, numpy, scipy, statsmodels, scikit-learn
- xgboost phiên bản 2.x
- openpyxl (để đọc file Excel)
- fastapi, uvicorn (chỉ cần khi chạy API)

Cài bằng một lệnh:

```
pip install pandas numpy scipy statsmodels scikit-learn "xgboost>=2.0" openpyxl fastapi uvicorn
```

Kiểm tra phiên bản xgboost:

```
python -c "import xgboost; print(xgboost.__version__)"
```

**Lưu ý phiên bản:** file mô hình `demand_model_online_retail.pkl` được lưu bằng xgboost. Khi chạy trên máy khác, nên dùng cùng phiên bản xgboost, không thì có thể lỗi khi nạp mô hình. Nếu gặp lỗi, chạy lại bước 4 để tạo lại file mô hình trên máy đó.

## 3. Cách chạy từ đầu

Mở terminal tại thư mục dự án (nơi có `data/`), rồi chạy theo đúng thứ tự:

```
python 1_data_processing_online_retail_2y.py
python 2_data_processing_online_retail_ml_2y.py
python 4_train_final_model_online_retail_2y.py
python 5_fit_pricing_model_2y.py
python pricing_engine.py
uvicorn api_app:app --port 8000
```

| Bước | File                                       | Làm gì                                                             | Tạo ra (trong `models_2y/`)                                                                                                                                                       |
| ---- | ------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `1_data_processing_online_retail_2y.py`    | Đọc Excel, bỏ phần chồng lấn giữa 2 sheet, làm sạch, gộp theo tuần | `weekly_online_retail.csv`, `products_online_retail.csv`, và cache `data/online_retail_II_sheets.csv`                                                                             |
| 2    | `2_data_processing_online_retail_ml_2y.py` | Tạo X, y và thống kê từng sản phẩm                                 | `X_online_retail.npy`, `y_online_retail.npy`, `is_test_online_retail.npy`, `feature_cols_online_retail.json`, `product_stats_online_retail.json`, `model_table_online_retail.csv` |
| 4    | `4_train_final_model_online_retail_2y.py`  | Huấn luyện XGBoost dự báo lượng bán                                | `demand_model_online_retail.pkl`, `model_metrics_online_retail.json`                                                                                                              |
| 5    | `5_fit_pricing_model_2y.py`                | Ước lượng elasticity từng sản phẩm, chọn cách co, tính độ bất định | `pricing_params_online_retail.csv`, `pricing_model_online_retail.json`                                                                                                            |
| —    | `pricing_engine.py`                        | Kiểm tra engine, in bảng đề xuất mẫu                               | (không tạo file)                                                                                                                                                                  |
| —    | `api_app.py`                               | Chạy API                                                           | (không tạo file)                                                                                                                                                                  |

Bước 1 mất vài phút ở lần đầu vì đọc Excel. Các lần sau dùng cache nên nhanh hơn.

## 4. Kiểm tra kết quả

Sau bước 4, các con số sau phải khớp (nếu dữ liệu và môi trường giống nhau):

- TEST R² (ln qty): khoảng **0.587**
- MAE theo số lượng: khoảng **21.17** (baseline khoảng 23.37)
- Số sản phẩm: **1306**

Sau bước 5, phân vị elasticity trung bình khoảng **-1.5** (cầu co giãn vừa phải).

## 5. Chạy API và thử

Sau khi `uvicorn api_app:app --port 8000` chạy được:

- Giao diện thử nghiệm: http://localhost:8000/docs
- Kiểm tra tình trạng: http://localhost:8000/health

Ví dụ gọi đề xuất giá (Windows CMD):

```
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"sku\":\"22112\",\"current_price\":4.95,\"unit_cost\":3.0,\"date\":\"2011-11-15\"}"
```

Các endpoint: `/health`, `/model_info`, `/products`, `POST /recommend`, `POST /recommend/batch`, `GET /demand_curve/{sku}`.

## 6. Các file tùy chọn

- `3_select_products_2y.py`: chọn 20 sản phẩm làm demo. Không cần để chạy API. Chạy sau bước 1.
- `test.py`: không nằm trên đường chạy chính. Chạy riêng khi cần.
