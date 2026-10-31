# Chạy giao diện với API gốc

Từ thư mục gốc dự án, mở hai terminal:

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

```powershell
python frontend/serve.py
```

Mở <http://127.0.0.1:8765/>. `serve.py` chỉ phục vụ frontend trên localhost,
chuyển tiếp các route API hiện có tới cổng 8000 và đọc trực tiếp danh sách 20 SKU
từ `models_2y/selected_products_online_retail.csv`. Vì frontend và API có cùng
origin ở cổng 8765, không cần đổi CORS hoặc mã backend.

Giao diện hiện dùng `/health`, `/model_info`, `/products`, `POST /recommend`
và `/demand_curve/{sku}`. Giá hiện tại luôn được gửi trong yêu cầu đề xuất.
API gốc không lưu kết quả vào SQL Server; mỗi lần tải lại trang, kết quả vừa tính
sẽ không còn trên giao diện.
