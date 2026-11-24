# Chạy giao diện

Từ thư mục gốc dự án, mở hai terminal:

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

```powershell
python frontend/serve.py
```

Mở <http://127.0.0.1:8765/>. `serve.py` phục vụ frontend trên localhost và
chuyển tiếp API tới cổng 8000. Trang tổng quan đọc danh sách 20 SKU từ
`models_2y/selected_products_online_retail.csv`. Trang danh sách và trang chi
tiết đọc dữ liệu đã nhập vào SQL Server. Cần có database `DynamicPricingDemo`
với 20 sản phẩm và 2.072 tuần bán; nếu chưa có, chạy `database/setup.ps1` từ
PowerShell. Có thể cấu hình kết nối bằng `PRICING_SQL_SERVER`, `PRICING_DB_NAME`
hoặc `PRICING_DB_CONNECTION_STRING`.

## Phase 8: đề xuất giá M5 và trang phương pháp

Sau khi chạy các Phase 1–7 và có artifact ở
`experiments/artifacts/song_m5_pilot/`, mở
<http://127.0.0.1:8765/price-m5.html>. Trang tự chọn SKU có điểm validation
cao nhất và tuần test đầu tiên;
đổi SKU, tuần, giá hiện tại hoặc thuật toán rồi bấm **Tính giá đề xuất** để
chạy lại suy luận. Để trống giá hiện tại nghĩa là dùng giá quan sát của tuần
trước. Giá nhập phải thuộc miền giá train ghi dưới ô. Trang gọi
`POST /research/price/compare` để tính 7 phương pháp trên cùng SKU, tuần,
giá đầu vào và lịch sử trước quyết định. Bảng cho biết giá, doanh thu dự báo
từ 5 Transformer, doanh thu mô phỏng và điểm validation trung bình 5 seed so
với giữ giá.
Các chính sách học dùng seed đại diện chọn trên validation; nếu bạn chọn
thuật toán và seed trực tiếp, hàng đó dùng seed bạn chọn. Đồ thị và 12 tuần
lịch sử nằm trong phần mở rộng. Doanh thu ở giá chưa bán chưa được quan sát;
giá này chỉ là phép thử một bước và không được lưu thành giá bán.

Mở <http://127.0.0.1:8765/method.html> để xem kết quả trên đúng cohort M5,
chất lượng mô hình cầu, thứ hạng giá theo validation/test, độ nhạy với giả
định cầu và liên kết bài báo gốc. Số trong bài báo chỉ được đối chiếu ở mức
phương pháp khi giao thức của họ khác nhóm. Hai trang này **không phụ thuộc
SQL Server**; cần API Python chạy trên cổng 8000 và artifact Phase 6–7. Nếu
trang báo 404 ở `/api/research/price/options`, khởi động lại backend cũ để
nạp các route Phase 8. Lệnh chạy nhanh nằm trong README dự án, còn lệnh tái tạo
các phase nằm trong PROJECT_HISTORY.md.

Trang [thí nghiệm mô hình](http://127.0.0.1:8765/research.html) đọc **kết quả
đã lưu** của Phase 3C/4 qua bốn API chỉ đọc `/research/forecast/overview`,
`/research/forecast/series`, `/research/pricing/overview` và
`/research/pricing/replay`. Chọn UCI hoặc M5, mô hình, SKU và seed để xem
dự báo trên đúng các hàng **test**; chọn simulator, chính sách và seed để
phát lại 200 quyết định M5 đã lưu. Trang này dùng các CSV trong
`experiments/artifacts/`, không đọc bảng SQL Server của demo Online Retail
II. Nếu thiếu kết quả, chạy lại các lệnh Phase 3C/4 ở PROJECT_HISTORY.md. Có thể
đổi thư mục artifact bằng biến môi trường `PRICING_RESEARCH_ARTIFACTS` trước
khi khởi động API.

Tab [Thử giá M5](http://127.0.0.1:8765/research.html#scenario) của Phase 5B
cho chọn 1 trong 10 SKU × 20 tuần test, giá hiện tại trong miền giá train và
1 trong 5 seed. `GET /research/pricing/scenario/options` tải lựa chọn;
`POST /research/pricing/scenario` dùng Transformer và checkpoint `ppo_frozen`
đã lưu để tính **một giá mới** từ 12 tuần lịch sử đã quan sát trước tuần chọn.
Giá nhập mặc định là giá quan sát của tuần trước. Kết quả có doanh thu dự báo
từ Transformer, doanh thu kỳ vọng từ simulator và đồ thị 41 mức giá. Với tuần
test sau tuần đầu, đây là kịch bản một bước trên lịch sử thực, không phải
phát lại chuỗi giá mô phỏng. Trang này không cập nhật mô hình online hay ghi
một lần đổi giá của doanh nghiệp.

Các số doanh thu trong thí nghiệm là **dự báo hoặc mô phỏng**, không phải
doanh thu thực tế ở giá mới. Simulator hiện chưa vượt mốc dự báo tuần trước
trên test; chỉ dùng kết quả này để kiểm thử và giải thích thuật toán.

Phase 5C thêm lựa chọn **PPO có cập nhật dự báo** trong cùng tab. Chạy
`python -m experiments.phase5c_online_state` trước khi khởi động API để tạo
cặp checkpoint PPO–Transformer đã đối chiếu với nhật ký Phase 4. Nếu thiếu
artifact này, tab vẫn cho thử **PPO dự báo cố định**. Ở chế độ online, để
trống ô giá để phát lại giá đầu vào đúng chuỗi mô phỏng; nhập giá khác sẽ
tính một quyết định giả định tại tuần chọn sau khi phát lại các tuần trước.
Lượng bán phản hồi và các lần cập nhật Transformer đều đến từ simulator,
không phải doanh thu thật hay đơn hàng của doanh nghiệp.

Giao diện dùng `/health`, `/products`, `POST /recommend/batch`,
`POST /demo/simulations`, `/demand_curve/{sku}` và các route
`GET /demo/products/{sku}` cùng `/weeks`, `/recommendations`. Bảng so sánh gửi
20 SKU trong một
yêu cầu batch, lấy giá tham chiếu của mô hình làm đầu vào cho từng SKU, rồi xếp
theo `priority_score` (mức tăng doanh thu dự kiến nhân xác suất tăng doanh thu).
Ngày mô phỏng và giới hạn điều chỉnh áp dụng cho cả bảng; thay đổi hai thông số
này sẽ tính lại batch. Khi chọn một hàng, kết quả batch được mở trong phần mô
phỏng. Khi bấm “Tính giá đề xuất”, `POST /demo/simulations` tính và lưu một lần
thử giá vào `RecommendationHistory`; bảng batch không tạo lịch sử.

Trang [danh sách](http://127.0.0.1:8765/products.html) hiển thị 20 SKU được chọn.
Trang chi tiết (`product.html?sku=...`) hiển thị giá trung vị, lượng bán từng tuần
và các lần thử giá đã lưu. Doanh thu quy đổi của dữ liệu lịch sử được tính bằng
`qty × price` từ `WeeklySales`: đây là ước tính từ dữ liệu gộp, không phải doanh
thu hóa đơn. Doanh thu trong nhật ký là dự báo của mô hình; giá đề xuất chưa phải
giá bán đã áp dụng.
