# Dynamic Pricing · Hướng dẫn chạy nhánh Thong

Nhánh `Thong` có gói [`final_system/`](final_system/) để dựng dữ liệu theo tuần, huấn luyện và chạy API mô hình. API dự báo lượng bán cho **UCI Online Retail 352** và **M5**, đồng thời đề xuất giá cho các chuỗi sản phẩm–cửa hàng M5 có mô hình ARIMAX.

Thực hiện các lệnh bên dưới bằng **PowerShell** trong terminal của Visual Studio hoặc VS Code. Luồng này chạy bằng CPU và dùng API mô hình trực tiếp. Có thể thử bằng Swagger trước khi tích hợp backend, frontend và SQL Server.

## 1. Clone nhánh và cài thư viện

Cần Git, Python từ **3.11** và vài GB trống để lưu thư viện, dữ liệu gốc, dữ liệu tuần và cache. Các bước đã được kiểm tra trên Windows với Python 3.14.4.

```powershell
git clone --branch Thong --single-branch https://github.com/LcNguyn72/dynamic-pricing.git
Set-Location dynamic-pricing
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r final_system/requirements.txt
```

Nếu đã clone nhánh `Thong`, mở terminal tại thư mục có `final_system/` và bắt đầu từ `python --version`. Các lệnh dùng trực tiếp Python trong `.venv`, nên có thể dán vào PowerShell mà chưa kích hoạt môi trường ảo.

## 2. Chuẩn bị bộ mô hình

Chọn **một** trong hai cách sau. Sau khi có `manifest.json` và tệp `.json.gz` được ghi trong manifest, chuyển sang bước 3.

### Cách A · Dùng bộ mô hình đã lưu

Nhận cả thư mục `runtime/models/` từ người đã huấn luyện và đặt vào `final_system/runtime/models/`. Kiểm tra tên tệp trong manifest khớp với tệp đã chép.

```powershell
$manifest = Get-Content final_system/runtime/models/manifest.json -Raw | ConvertFrom-Json
$manifest.version
Test-Path (Join-Path 'final_system/runtime/models' $manifest.file)
```

Dòng cuối cần trả về `True`. Git bỏ qua `runtime/`, nên một bản clone mới sẽ tạo bộ mô hình bằng Cách B hoặc nhận từ thành viên trong nhóm theo Cách A.

### Cách B · Tải dữ liệu và huấn luyện từ đầu

Chạy lần lượt từ thư mục gốc và chờ mỗi lệnh hoàn tất trước khi chạy lệnh tiếp theo.

```powershell
.\.venv\Scripts\python.exe -u -m final_system.data_pipeline download
.\.venv\Scripts\python.exe -u -m final_system.data_pipeline prepare
.\.venv\Scripts\python.exe -u -m final_system.training train --workers 6
```

| Bước | Việc thực hiện | Thư mục đầu ra |
| --- | --- | --- |
| `download` | Tải UCI Online Retail 352 và ba tệp M5 gồm `sales_train_evaluation.csv`, `sell_prices.csv` và `calendar.csv`. Kiểm tra mã băm dữ liệu. | `final_system/runtime/raw/` |
| `prepare` | Gộp theo tuần bắt đầu vào thứ Hai, áp dụng quy tắc xử lý theo giao thức V2, dựng đủ 10 cửa hàng M5 và đối chiếu khóa test. | `final_system/runtime/prepared/` |
| `train` | Lưu danh mục SKU UCI dùng dự báo tuần trước. Huấn luyện AR(1) và ARIMAX cho M5, rồi ghi manifest và tham số mô hình. | `final_system/runtime/models/` |

Nếu đã tải dữ liệu, đặt tệp `Online Retail.xlsx` vào thư mục con `online_retail/` và ba CSV vào thư mục con `m5/` của `runtime/raw/`. Lệnh `download` dùng lại các tệp khớp mã băm. Có thể truyền cùng `--raw-root "<đường dẫn>"` cho `download` và `prepare` nếu dữ liệu nằm ở nơi khác. Máy ít tài nguyên có thể dùng `--workers 2`. Khi hoàn tất, lệnh `train` in `version` và `coverage` của bộ mô hình.

[Nguồn UCI Online Retail 352](https://archive.ics.uci.edu/dataset/352/online+retail) · [Nguồn tệp M5](https://zenodo.org/records/10203108)

## 3. Khởi động API và kiểm tra trạng thái

Giữ terminal này mở trong khi thử API.

```powershell
.\.venv\Scripts\python.exe -m uvicorn final_system.api:app --host 127.0.0.1 --port 18765
```

Mở [Swagger](http://127.0.0.1:18765/docs) hoặc dùng một terminal PowerShell thứ hai.

```powershell
$modelApi = 'http://127.0.0.1:18765'
Invoke-RestMethod "$modelApi/health"
(Invoke-RestMethod "$modelApi/model-info").coverage
```

`/health` trả `status = ready` và phiên bản mô hình. Với bộ mô hình đã kiểm tra, `/model-info` ghi nhận **3.387 SKU UCI**, **29.069 chuỗi SKU–cửa hàng M5** và **18.244 chuỗi có ARIMAX để thử giá**.

## 4. Gửi hai yêu cầu mẫu

Dán hai khối lệnh vào terminal PowerShell đang có biến `$modelApi`. Ngày trong `week` là thứ Hai, theo định dạng `YYYY-MM-DD`. Trong các ví dụ, `history` chứa lượng bán của tuần liền trước và thêm giá bán khi đề xuất giá M5.

**Dự báo lượng bán UCI cho SKU `10120`**

```powershell
$forecastBody = @{
    dataset = 'uci'
    sku = '10120'
    week = '2011-10-10'
    history = @(@{ week = '2011-10-03'; qty = 5 })
} | ConvertTo-Json -Depth 5

$forecastResult = Invoke-RestMethod -Uri "$modelApi/forecast" -Method Post -ContentType 'application/json' -Body $forecastBody
$forecastResult | Select-Object method, predicted_qty, source_week, model_version
```

Kết quả mẫu có `method = previous_week` và `predicted_qty = 5` sản phẩm.

**Đề xuất giá M5 cho `FOODS_3_090` tại cửa hàng `CA_1`**

```powershell
$priceBody = @{
    sku = 'FOODS_3_090'
    store_id = 'CA_1'
    week = '2015-08-03'
    history = @(@{ week = '2015-07-27'; qty = 454; price = 1.48 })
} | ConvertTo-Json -Depth 5

$priceResult = Invoke-RestMethod -Uri "$modelApi/recommend-price" -Method Post -ContentType 'application/json' -Body $priceBody
$priceResult.grid_result
$priceResult.fixed_reference
$priceResult.candidates.Count
```

- `grid_result` là mức giá có doanh thu dự kiến cao nhất trong các ứng viên. Bộ mô hình đã kiểm tra trả khoảng **1,00 USD**, với **984,80 USD doanh thu dự kiến**.
- `fixed_reference` là giá cố định đối chiếu khi chuỗi có mốc từ validation. Ví dụ này trả **1,012 USD**.
- `candidates` chứa danh sách giá, lượng bán dự báo và doanh thu dự kiến cho từng mức. Ví dụ có **41 ứng viên**.

## 5. Kiểm thử và lỗi thường gặp

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s final_system/tests -v
```

| Hiện tượng | Cách xử lý |
| --- | --- |
| `No module named final_system` | Mở terminal tại thư mục cha của `final_system/`. |
| API dừng khi báo `No model bundle` | Kiểm tra `final_system/runtime/models/manifest.json` và tệp `.json.gz` khớp với manifest, hoặc chạy lại mục 2. |
| `Address already in use` | Chọn cổng khác trong lệnh Uvicorn rồi đổi `$modelApi` sang cùng cổng. |
| HTTP `422` khi đề xuất giá M5 | Xem `detail` trong phản hồi. Kiểm tra mã cửa hàng, ngày thứ Hai và lịch sử tuần liền trước có cả lượng bán lẫn giá. Những chuỗi thiếu ARIMAX sẽ trả lỗi `price_model_unavailable`. |

[Xem thêm hợp đồng API và cấu hình mô hình](final_system/README.md). Hướng dẫn cho demo Online Retail II và nhóm script `*_2y.py` nằm trong [README của bản demo cũ](https://github.com/LcNguyn72/dynamic-pricing/blob/2917712af3d8b73b169e36e68c0063facbe08537/README.md).
