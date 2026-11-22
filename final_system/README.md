# Final system · Dịch vụ dự báo lượng bán và đề xuất giá

`final_system/` chạy độc lập toàn bộ luồng **dữ liệu gốc → dữ liệu tuần → huấn luyện cấu hình đã chốt → lưu mô hình → API mô hình**. Mã dựng dữ liệu và huấn luyện nằm trong `pipeline/`, giao thức và cấu hình trong `configs/`. Khi tích hợp, backend gửi lịch sử tuần cho API. Mã benchmark, tune và kết quả đối chiếu Song dùng riêng ở thư mục `experiments/` của repo.

## 1. Chạy lại từ dữ liệu gốc

Mở PowerShell tại thư mục cha có chứa `final_system/`, dùng Python 3.11 trở lên. Có thể chép riêng thư mục `final_system/` sang một thư mục cha mới và dùng các lệnh dưới đây. Phiên bản kiểm thử trên máy dự án là Python 3.14.4.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r final_system/requirements.txt
.\.venv\Scripts\python.exe -m final_system.data_pipeline download
.\.venv\Scripts\python.exe -m final_system.data_pipeline prepare
.\.venv\Scripts\python.exe -m final_system.training train --workers 6
```

| Bước | Kết quả |
| --- | --- |
| `download` | Lấy UCI Online Retail 352 và ba CSV M5 (`sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv`). Dùng lại tệp đang có tại `final_system/runtime/raw/` và kiểm tra SHA-256. [Nguồn UCI](https://archive.ics.uci.edu/dataset/352/online+retail) · [Nguồn CSV M5](https://zenodo.org/records/10203108) |
| `prepare` | Tạo panel tuần thứ Hai trong `final_system/runtime/prepared/` theo giao thức V2. Dùng ranh giới thời gian riêng cho mỗi dataset. UCI áp dụng quy tắc loại ngoại lệ đã học trên train cho chính train, giữ lượng bán hợp lệ ở validation và test. M5 dựng panel đủ 10 cửa hàng và chỉ xem tuần đủ giá là tuần quan sát. Kiểm tra mã băm tập khóa test trước khi chuyển bước. |
| `train` | UCI lưu danh mục SKU dùng quy tắc tuần trước. M5 fit AR(1) và ARIMAX từ 193 tuần train cho từng chuỗi đủ điều kiện. Tái tạo giá cố định trên 20 tuần cuối của validation cho cohort 60 chuỗi đã khóa. Ghi `manifest.json` và tệp tham số nén có mã SHA-256 vào `final_system/runtime/models/`. |

| Dataset | Tuần train / validation / test | Tuần train cuối | Tuần test đầu |
| --- | --- | --- | --- |
| UCI Online Retail 352 | 37 / 8 / 9 | 2011-08-08 | 2011-10-10 |
| M5 | 193 / 42 / 42 | 2014-10-06 | 2015-08-03 |

Nếu dữ liệu gốc đang ở nơi khác, truyền `--raw-root` cho `download` và `prepare`. Nếu đã dựng panel thành công, `prepare` kiểm tra lại bản đang có. Nếu cần dựng một bản khác, truyền `--prepared-root` đến thư mục mới cho cả `prepare` và `train`.

**Nhập lại tham số từ kho thí nghiệm cũ.** Lệnh tùy chọn sau dùng các kết quả Phase 1, Bước 3 ARIMAX và thí nghiệm ARIMAX–PPO trong repo. `train` dựng mô hình trực tiếp từ dữ liệu đã chuẩn bị của gói.

```powershell
python -m final_system.training import-verified
```

`runtime/` được Git bỏ qua. Khi chuyển máy, chép `runtime/models/` gồm manifest và tệp `.json.gz` để chạy API ngay sau khi cài thư viện. Chép thêm `runtime/raw/` nếu muốn dựng và huấn luyện lại từ dữ liệu đã tải sẵn. `runtime/prepared/` và `runtime/cache/` được tạo khi chạy. API nạp bộ mô hình một lần lúc khởi động. Đặt biến môi trường `FINAL_SYSTEM_MODEL_DIR` để trỏ sang một thư mục mô hình khác và khởi động lại API để nạp phiên bản đó.

Nếu commit riêng bản bàn giao mô hình từ một repo có nhiều phần, xem danh sách sẽ thêm bằng `git add -n -- final_system/`, rồi dùng `git add -- final_system/` để đưa riêng mã nguồn và cấu hình của gói vào vùng stage. Các tệp đã theo dõi ở nhánh hiện tại vẫn nằm trong lịch sử nhánh.

## 2. Khởi động và gọi API

```powershell
.\.venv\Scripts\python.exe -m uvicorn final_system.api:app --host 127.0.0.1 --port 18765
```

Mở [Swagger của API mô hình](http://127.0.0.1:18765/docs) để gửi thử các yêu cầu.

| Endpoint | Kết quả và dữ liệu bắt buộc |
| --- | --- |
| `GET /health` | Trạng thái sẵn sàng và phiên bản mô hình đã nạp. |
| `GET /model-info` | Độ phủ, nguồn dữ liệu, mốc thời gian, cấu hình và nguồn gốc bộ tham số. |
| `POST /forecast` | Dự báo số sản phẩm cho một tuần. Gửi `dataset`, `sku`, `week`, `history` và `store_id` với M5. Gửi thêm `candidate_price` để dự báo lượng M5 theo một mức giá cụ thể. |
| `POST /recommend-price` | Đề xuất giá M5. Gửi `sku`, `store_id`, `week` và `history`. Có thể gửi `current_price`; mặc định lấy giá của tuần liền trước. |

Trong `history`, mỗi hàng gồm `week`, `qty` và `price` nếu có. `week` là thứ Hai và luôn đứng trước tuần dự báo. Với UCI, gửi lượng của tuần ngay trước đó theo panel tuần. Với M5, dùng các tuần quan sát hợp lệ; ARIMAX cần giá và lượng của tuần liền trước. SKU hoặc chuỗi ngoài danh mục trả `404`. Lịch sử sai, giá ngoài miền train khi dự báo với giá ứng viên, hoặc chuỗi thiếu mô hình giá trả `422`.

Ví dụ từ M5. Tuần trước ghi nhận 454 sản phẩm với giá 1,48 USD tại cửa hàng `CA_1`.

```powershell
$modelApi = 'http://127.0.0.1:18765'
Invoke-RestMethod "$modelApi/model-info"

$body = @'
{
  "sku": "FOODS_3_090",
  "store_id": "CA_1",
  "week": "2015-08-03",
  "history": [{"week": "2015-07-27", "qty": 454, "price": 1.48}]
}
'@
$result = Invoke-RestMethod -Uri "$modelApi/recommend-price" -Method Post `
    -ContentType 'application/json' -Body $body
$result.grid_result
$result.fixed_reference
```

Với bộ tham số đã kiểm tra, `grid_result` trả giá **1,00 USD**, lượng dự báo **984,80** và doanh thu dự kiến **984,80 USD**. `fixed_reference` trả giá đối chiếu **1,012 USD**. Để kiểm thử UCI, gửi payload dưới đây đến `/forecast`. Kết quả là 5 sản phẩm.

```json
{
  "dataset": "uci",
  "sku": "10120",
  "week": "2011-10-10",
  "history": [{"week": "2011-10-03", "qty": 5}]
}
```

## 3. Đọc kết quả và phạm vi mô hình

| Luồng | Quy tắc và phạm vi |
| --- | --- |
| UCI · Lượng bán | Dùng lượng của tuần trước cho 3.387 SKU. Quy tắc này không cần ước lượng tham số. |
| M5 · Lượng bán | AR(1), tức ARIMA(1,0,0), fit trên train cho 28.506 trong 29.069 chuỗi. Chuỗi thiếu tham số dùng lượng quan sát gần nhất. Đây là cấu hình Bước 3. Kết quả ARIMA(1,0,1) fit trên train + validation thuộc benchmark Phase 5. |
| M5 · Thử giá | ARIMAX(1,0,0) có giá làm biến đầu vào, khả dụng cho 18.244 chuỗi trong bộ đã kiểm tra. Engine dò 41 mức trong miền giá train, thêm giá hiện tại nếu hợp lệ và khác lưới. Mỗi ứng viên có `predicted_qty` và `expected_revenue = price × predicted_qty`. `grid_result` là ứng viên đạt doanh thu dự kiến cao nhất; khi bằng nhau ưu tiên giá gần giá hiện tại rồi đến giá thấp hơn. |
| M5 · Giá đối chiếu | `fixed_reference` có cho 60 chuỗi đã khóa và các tuần sau validation. Trong thí nghiệm theo dõi, chính sách giá cố định theo SKU dẫn đầu trên hai nhóm và hai môi trường mô phỏng. Trường này cung cấp mốc đối chiếu bên cạnh kết quả động. |

`expected_revenue` là dự báo một tuần dưới giả định của mô hình cầu. Các kết quả doanh thu tích lũy theo seed thuộc thí nghiệm và dùng môi trường mô phỏng riêng.

Chạy kiểm thử engine và lưu/nạp mô hình từ thư mục cha có chứa `final_system/`.

```powershell
python -m unittest discover -s final_system/tests -v
```
