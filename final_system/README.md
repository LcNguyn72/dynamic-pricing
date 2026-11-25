# Final system · API dự báo lượng bán và đề xuất giá

`final_system/` chuẩn bị dữ liệu, huấn luyện ARIMAX, lưu mô hình và phục vụ API. Luồng định giá dùng dữ liệu M5. API lấy lịch sử từ nội dung yêu cầu, nên có thể thử trực tiếp qua Swagger hoặc PowerShell. Để chạy giao diện cửa hàng với backend và SQL Server, dùng [hướng dẫn ứng dụng M5](../README.md#ứng-dụng-cửa-hàng-m5).

## 1. Cài đặt

Cần **Python 3.11 trở lên**. Mở PowerShell tại thư mục **cha chứa `final_system/`**. Ví dụ, nếu có `D:\dynamic-pricing\final_system`, mở PowerShell tại `D:\dynamic-pricing`.

```powershell
python --version
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r final_system/requirements.txt
```

Nếu mở PowerShell mới để chạy lệnh Python ở các bước sau, đặt lại `$python = '.\.venv\Scripts\python.exe'`.

## 2. Đặt tệp mô hình đã huấn luyện

Khi chuyển máy, chép các tệp sau vào đúng cấu trúc này. Git bỏ qua thư mục `runtime/`, nên các tệp mô hình cần được gửi kèm riêng hoặc tạo lại theo mục 5.

```text
final_system/
└── runtime/
    ├── models/
    │   ├── manifest.json
    │   └── models-<sha256>.json.gz
    └── ppo/
        └── m5-policy.json
```

- **Dự báo lượng bán và lưới giá** cần `manifest.json` và đúng tệp `.json.gz` có tên ghi trong trường `file` của manifest. Giữ nguyên tên tệp.
- **PPO có phản hồi** cần thêm `m5-policy.json`. Lấy policy và bộ ARIMAX từ cùng bản bàn giao vì API kiểm tra mã SHA-256 của bộ tham số khi nạp PPO.

Với các tệp đã huấn luyện ở trên, có thể chuyển thẳng sang mục 3. Nếu đang lưu mô hình ở đường dẫn khác, đặt biến môi trường `FINAL_SYSTEM_MODEL_DIR` và `FINAL_SYSTEM_PPO_PATH` trong cửa sổ chạy API.

## 3. Bật API và kiểm tra mô hình

Từ thư mục cha chứa `final_system/`, chạy lệnh sau và để cửa sổ này mở.

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn final_system.api:app --host 127.0.0.1 --port 8766
```

Mở [Swagger](http://127.0.0.1:8766/docs) hoặc mở một PowerShell khác và chạy:

```powershell
$modelApi = 'http://127.0.0.1:8766'
Invoke-RestMethod "$modelApi/health"
$info = Invoke-RestMethod "$modelApi/model-info"
$info.m5_pricing_policies.grid
$info.m5_pricing_policies.ppo_feedback | ConvertTo-Json -Depth 4
```

`/health` trả `status = ready` và phiên bản mô hình. Trong `/model-info`, `ppo_feedback.available = true` xác nhận policy đã nạp. `ppo_feedback.scope` liệt kê các cặp SKU–cửa hàng có PPO.

## 4. Thử lưới, PPO và dự báo lượng bán

Ví dụ dùng M5, sản phẩm `FOODS_3_090` ở cửa hàng `CA_1`. Tuần `2015-07-27` ghi nhận 454 sản phẩm ở giá 1,48 USD. Chạy khối sau trong PowerShell dùng để kiểm tra API.

```powershell
$modelApi = 'http://127.0.0.1:8766'
$info = Invoke-RestMethod "$modelApi/model-info"
$inputData = @{
    dataset = 'm5'
    sku = 'FOODS_3_090'
    store_id = 'CA_1'
    week = '2015-08-03'
    history = @(@{ week = '2015-07-27'; qty = 454; price = 1.48 })
}

$gridRequest = $inputData.Clone()
$gridRequest['strategy'] = 'grid'
$grid = Invoke-RestMethod "$modelApi/recommend-price" -Method Post -ContentType 'application/json' -Body ($gridRequest | ConvertTo-Json -Depth 5)
$grid.recommendation | Format-List price, predicted_qty, expected_revenue

$selected = $grid.recommendation
if ($info.m5_pricing_policies.ppo_feedback.available) {
    $ppoRequest = $inputData.Clone()
    $ppoRequest['strategy'] = 'ppo_feedback'
    $ppo = Invoke-RestMethod "$modelApi/recommend-price" -Method Post -ContentType 'application/json' -Body ($ppoRequest | ConvertTo-Json -Depth 5)
    $ppo.recommendation | Format-List price, predicted_qty, expected_revenue
    $selected = $ppo.recommendation
}

$forecastRequest = $inputData.Clone()
$forecastRequest['candidate_price'] = $selected.price
$forecast = Invoke-RestMethod "$modelApi/forecast" -Method Post -ContentType 'application/json' -Body ($forecastRequest | ConvertTo-Json -Depth 5)
$forecast | Format-List method, candidate_price, predicted_qty, feedback_count
```

`recommendation` trả về `price`, `predicted_qty` và `expected_revenue`. Với bộ tham số và policy đang bàn giao, ví dụ chưa có phản hồi này cho kết quả sau:

| Phương pháp | Giá đề xuất | Lượng bán dự báo | Doanh thu dự kiến |
| --- | ---: | ---: | ---: |
| Lưới | 1,00 USD | 984,80 | 984,80 USD |
| PPO | 1,224688 USD | 727,44 | 890,89 USD |

Kết quả `/forecast` tại giá được chọn khớp với lượng bán dự báo trong kết quả định giá. Mỗi phương pháp là một yêu cầu tới cùng endpoint `/recommend-price`.

Bạn cũng có thể dán yêu cầu sau vào `POST /recommend-price` trên Swagger, chọn **Try it out** rồi **Execute**. Đổi `strategy` thành `"ppo_feedback"` để thử PPO.

```json
{
  "dataset": "m5",
  "sku": "FOODS_3_090",
  "store_id": "CA_1",
  "week": "2015-08-03",
  "history": [{"week": "2015-07-27", "qty": 454, "price": 1.48}],
  "strategy": "grid"
}
```

### Quy tắc đầu vào M5

- `week` là ngày thứ Hai đầu tuần và phải sau giai đoạn huấn luyện. `history` chứa tuần đã quan sát, gồm `week`, `qty`, `price`. Cần có tuần liền trước tuần mục tiêu.
- `POST /forecast` cho phép truyền `candidate_price` trong miền giá train. Khi bỏ trống, bản M5 hiện tại dùng giá tuần liền trước.
- `POST /recommend-price` nhận `strategy = grid` hoặc `ppo_feedback`. Có thể truyền `current_price` làm giá so sánh. Giá này mặc định là giá tuần liền trước.
- `feedback` là tùy chọn cho cả hai endpoint. Mỗi dòng là số liệu đã quan sát gồm `week`, `qty`, `price`, `previous_qty` và `previous_price`. Engine dùng tối đa 32 tuần đủ điều kiện sau train và trước tuần mục tiêu. Cần ít nhất 4 quan sát với giá thay đổi trong miền train để cập nhật hệ số giá của ARIMAX. `feedback_count` cho biết số dòng đã dùng. PPO nhận hệ số đã cập nhật trong khi trọng số policy giữ nguyên.

### Nếu gặp lỗi

| Hiện tượng | Cách xử lý |
| --- | --- |
| API báo thiếu model bundle khi bật | Kiểm tra `runtime/models/manifest.json` và tệp `.json.gz` mà manifest tham chiếu. |
| `ppo_feedback.available = false` | Chép `runtime/ppo/m5-policy.json` đúng bản ARIMAX rồi khởi động lại API. Lưới vẫn dùng được. |
| `ppo_series_unavailable` | Chọn cặp có trong `ppo_feedback.scope` hoặc dùng lưới. |
| `missing_previous_week` | Thêm giá và lượng bán quan sát của tuần liền trước vào `history`. |
| `price_outside_train_range` | Khi gửi `candidate_price`, chọn giá trong `train_price_range` mà endpoint đề xuất giá trả về. |
| `price_model_unavailable` | Chọn cặp SKU–cửa hàng có mô hình ARIMAX. Ví dụ trong mục 4 có mô hình. |

## 5. Dựng lại khi chỉ có mã nguồn

### Tạo bộ mô hình dùng cho dự báo và lưới

Chạy từ thư mục cha chứa `final_system/` sau khi cài đặt ở mục 1.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m final_system.data_pipeline download
& $python -m final_system.data_pipeline prepare
& $python -m final_system.training train --workers 6
```

| Bước | Kết quả |
| --- | --- |
| `download` | Tải và kiểm tra UCI Online Retail 352 cùng ba CSV M5 (`sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv`) vào `runtime/raw/`. Script dùng lại nguồn đã tải nếu khớp mã kiểm tra. [Nguồn UCI](https://archive.ics.uci.edu/dataset/352/online+retail) · [Nguồn M5](https://zenodo.org/records/10203108) |
| `prepare` | Dựng panel theo tuần cho hai tập riêng vào `runtime/prepared/`. M5 có 10 cửa hàng. UCI áp dụng quy tắc loại ngoại lệ học từ train trên chính tập train và giữ lượng bán hợp lệ ở validation/test. Script xác minh tập khóa test đã chốt. |
| `train` | Huấn luyện tham số M5 từ 193 tuần train, tạo giá đối chiếu cho 60 chuỗi từ validation và lưu bộ mô hình vào `runtime/models/`. Với UCI, bộ mô hình lưu danh mục SKU và cấu hình trung bình 4 tuần. |

Có thể giảm `--workers` nếu máy thiếu RAM. Nếu đã có dữ liệu gốc ở nơi khác, dùng `--raw-root` cho `download` và `prepare`. Nếu muốn đặt panel ở nơi khác, dùng cùng `--prepared-root` cho `prepare` và `train`. Lệnh `final_system.training import-verified` dành cho repo đã có các artifact thí nghiệm Phase 1, ARIMAX và ARIMAX–PPO dùng để đối chiếu.

### Chọn nguồn riêng để chuẩn hóa, train và test

Luồng `source-train` nằm trong `final_system/` và chạy độc lập với protocol/bundle UCI–M5 ở trên. Mỗi lần chạy chọn đúng một nguồn, chia train/test theo thời gian, rồi lưu model, panel, dự đoán test, metrics và manifest mới trong thư mục run riêng. Luồng này không ghi đè `runtime/models/` hoặc artifacts của nguồn còn lại.

Chọn nguồn trong `final_system/.env` (có thể tạo từ `.env.example`):

```env
FINAL_SYSTEM_DATA_SOURCE=lazada
```

Đặt thành `uci` để dùng panel Online Retail đã chuẩn bị. Chạy từ thư mục cha của `final_system/`:

```powershell
& '.\.venv\Scripts\python.exe' -m final_system.training source-train
```

Có thể truyền `--source uci` hoặc `--source lazada` để ghi đè env cho một lần chạy. UCI cần `runtime/prepared/uci/weekly.csv` cùng manifest được tạo ở bước `prepare`. Lazada đọc `crawler/data/daily_product_info.csv` và `crawler/data/ref_products.csv`, rồi chuẩn hóa thành kỳ 3 ngày trong `final_system/pipeline/lazada.py`. Sau khi gộp kỳ, Lazada luôn gắn cờ và cap spike theo SKU bằng ngưỡng log-MAD tính trên các kỳ train; quantity gốc vẫn được lưu trong panel và metrics test được tính theo quantity gốc. Không cần biến env riêng để bật/tắt bước này.

Mỗi run nằm dưới `final_system/runtime/source_runs/<source>/<run-id>/`, gồm `panel.csv`, `test_predictions.csv`, `metrics.json`, `model.json` và `manifest.json`. Model OLS theo SKU dùng `log1p(qty)`, log giá và xu hướng thời gian. Hệ số giá là quan hệ quan sát, không phải ước lượng nhân quả; do lịch sử Lazada hiện còn ngắn, metrics hiện chỉ để kiểm tra pipeline.

### Huấn luyện lại PPO khi có đủ repo thí nghiệm

Cần `experiments/` và `requirements-experiments.txt` ở thư mục gốc dự án. Sau khi hoàn tất phần tạo bộ mô hình, chạy:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-experiments.txt
& $python -m experiments.train_m5_serving_ppo
```

Lệnh lưu `final_system/runtime/ppo/m5-policy.json` và `m5-policy.evaluation.json`. Khởi động lại API rồi kiểm tra `ppo_feedback.available` theo mục 3.

## 6. Đọc kết quả và kiểm thử

- **M5:** cả lưới và PPO dùng chung ARIMAX(1,0,0) để dự báo lượng bán tại mỗi giá. Bộ đã kiểm tra có ARIMAX cho **18.244 cặp SKU–cửa hàng** và PPO cho **10 cặp**. Lưới dò 41 mức và thử thêm giá hiện tại nếu giá đó hợp lệ và khác các mức trên lưới.
- **Doanh thu:** `expected_revenue = price × predicted_qty` là doanh thu dự kiến một tuần. `fixed_reference` cung cấp giá đối chiếu cho 60 chuỗi đã chọn và các tuần sau validation.
- **UCI:** `/forecast` hỗ trợ trung bình 4 tuần làm mốc đối chiếu với `dataset = uci`. Gửi 4 tuần liền trước, gồm cả dòng có `qty = 0` trong panel UCI. Benchmark ARIMAX UCI nằm trong `experiments/`. Ứng dụng cửa hàng và `/recommend-price` dùng M5.

Chạy kiểm thử engine và lưu/nạp mô hình:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s final_system/tests -v
```
