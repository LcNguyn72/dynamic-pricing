# Biên bản Phase 0 — khóa giao thức trước thí nghiệm toàn phạm vi

Ngày kiểm tra: 2026-11-20. **Trạng thái:** đã hoàn thành kiểm tra nguồn và lập giao thức; sau đó nhóm yêu cầu tiếp tục Phase 1. Giao thức nằm ở [PROTOCOL_V2.md](../experiments/PROTOCOL_V2.md), giá trị có cấu trúc ở [protocol_v2.json](../experiments/protocol_v2.json), [mẫu báo cáo V2](report-template-v2.md) dành cho các kết quả sau, và thiết kế tổng thể ở [dynamic-pricing-final-design.md](dynamic-pricing-final-design.md).

## 1. Nguồn đã kiểm tra

Đã đối chiếu SHA256 tính lại từ tệp trên máy với `data/song_replication/download_manifest.json`:

| Dataset | Tệp cần cho giao thức chính | SHA256 khớp manifest? |
| --- | --- | --- |
| UCI Online Retail ID 352 | `data/song_replication/online_retail/Online Retail.xlsx` | Có — `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d` |
| M5 | `data/song_replication/m5/sales_train_evaluation.csv` | Có — `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| M5 | `data/song_replication/m5/sell_prices.csv` | Có — `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |
| M5 | `data/song_replication/m5/calendar.csv` | Có — `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |

`Online Retail II` của demo gốc và `sales_train_validation.csv` của pilot M5 không thuộc bảng toàn phạm vi mới.

## 2. Mốc chia và phạm vi đã nhìn thấy trên artifact hiện có

| Dataset | Nguồn | Tuần | Train | Validation | Test | Phạm vi hiện chuẩn bị |
| --- | --- | ---: | --- | --- | --- | --- |
| UCI ID 352 | 541.909 dòng giao dịch gốc | 54 | 37 tuần, đến 2011-08-08 | 8 tuần, 2011-08-15 đến 2011-10-03 | 9 tuần, từ 2011-10-10 | 3.387 SKU đủ điều kiện từ train |
| M5 đủ 10 cửa hàng | 30.490 chuỗi sản phẩm–cửa hàng gốc | 277 | 193 tuần, đến 2014-10-06 | 42 tuần, 2014-10-13 đến 2015-07-27 | 42 tuần, từ 2015-08-03 | 29.069 chuỗi đủ điều kiện từ train |

Cả 10 tệp `manifest_<store>.json` và 10 panel `weekly_<store>.csv` của M5 hiện tồn tại và có cùng ranh giới tuần. **Tệp tổng hợp `full_m5_preparation.json` chưa tồn tại**, nên chưa gọi gate chuẩn bị toàn M5 của runner là hoàn thành. Benchmark mô hình toàn M5 cũng chưa hoàn tất.

**Không khóa số hàng/hash test UCI hiện tại:** panel hiện có đã xóa giao dịch ngoại lệ ở validation/test bằng ngưỡng IQR fit trên train. Điều đó không dùng test để fit ngưỡng nhưng làm thay đổi nhãn chấm. Phase 1 phải tạo lại panel test chính giữ lượng bán hợp lệ, rồi ghi số hàng và SHA256 của khóa `(dataset, sku, week)` đã sắp xếp. Tương tự, danh sách cohort giá M5 10 chuỗi theo quy tắc train-only sẽ được tạo ở Phase 1; Phase 0 chỉ khóa quy tắc.

## 3. Quyết định và khác biệt so với kết quả cũ

1. Bảng chính UCI và M5 là **hai bảng riêng**. Mỗi thuật toán trong một bảng phải chấm cùng khóa SKU–tuần test trên phạm vi đủ điều kiện, không trộn RMSE từ pilot 100 SKU UCI hoặc 20 chuỗi M5 với kết quả toàn phạm vi.
2. Nguồn M5 chính là `sales_train_evaluation.csv`; pilot CA_1 từ `sales_train_validation.csv` có mốc test khác và là kết quả khám phá, không được gộp.
3. Điểm test chính gộp theo tuần nhưng **không lọc lượng bán thật chỉ vì IQR**. Quy tắc hợp lệ cố định vẫn áp dụng cho mọi split. Ngưỡng, scaler, SKU và miền giá chỉ fit trên train.
4. Có hai vai trò checkpoint: dự báo test có thể refit train+validation sau khi khóa siêu tham số; Transformer dùng cho chính sách giá chỉ học train, validation chọn epoch. Mọi báo cáo và API phải chỉ đúng phiên bản.
5. ARIMA cũ yêu cầu 12 tuần train quan sát trong khi UCI nhận SKU từ hai tuần. Trên panel hiện tại, 1.176/3.387 SKU có dưới 12 tuần train quan sát; con số này là **chẩn đoán tạm** trước Phase 1. Bảng chính toàn phạm vi dùng nhãn `ARIMA + fallback chuỗi thưa` với quy tắc cố định trước test và ghi số lượt fallback. ARIMA thuần chỉ được so ở bảng phụ trên một cohort chung đủ điều kiện.
6. Bản chính định giá tái dựng Song dùng kernel Gaussian/Silverman và giá trong miền train, 200 quyết định × 5 seed. Quy tắc chọn 10 chuỗi M5 cân bằng 10 cửa hàng được cố định bằng train; không chọn SKU bằng lượng bán hoặc kết quả test. `lag_adjusted_kernel` và giới hạn đổi giá ±10% là phân tích khác, không trộn vào bảng Song chính.
7. Bài Song không công bố đủ SKU, cách chuẩn hóa doanh thu và chi tiết UCI `Ours` có phản hồi để bảo đảm số của nhóm trùng bài. Bảng đối chiếu ghi rõ đơn vị, phạm vi và sai khác.

## 4. Cách kiểm tra lại Phase 0 (chỉ đọc)

Từ thư mục gốc dự án, đối chiếu tệp trong checkout hiện tại với mã băm trong manifest. Manifest có thể chứa đường dẫn tuyệt đối của máy đã tải; lệnh dưới đây không phụ thuộc các đường dẫn đó:

```powershell
$manifest = Get-Content data/song_replication/download_manifest.json -Raw | ConvertFrom-Json
$localFiles = [ordered]@{
    'Online Retail.xlsx' = 'data/song_replication/online_retail/Online Retail.xlsx'
    'sales_train_evaluation.csv' = 'data/song_replication/m5/sales_train_evaluation.csv'
    'sell_prices.csv' = 'data/song_replication/m5/sell_prices.csv'
    'calendar.csv' = 'data/song_replication/m5/calendar.csv'
}
foreach ($name in $localFiles.Keys) {
    $record = $manifest.files | Where-Object { (Split-Path $_.path -Leaf) -eq $name } | Select-Object -First 1
    $actual = (Get-FileHash -LiteralPath $localFiles[$name] -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{ name = $name; matches = ($actual -eq $record.sha256) }
}
```

Kết quả quan sát trong Phase 0: **cả bốn `matches` đều `True`**. Mốc chia UCI nằm trong `experiments/artifacts/song_online_retail/paper_70_15_15/manifest.json`; mốc M5 nằm trong `experiments/artifacts/song_m5_full/manifest_<store>.json`. Các artifact dữ liệu lớn bị Git bỏ qua, thành viên khác cần tải và tạo lại.

## 5. Điều kiện để bắt đầu Phase 1 sau khi nhóm duyệt

- Nhóm chấp nhận nguồn UCI ID 352, M5 `sales_train_evaluation.csv`, mốc chia và định nghĩa bảng chính trong `PROTOCOL_V2.md`.
- Phase 1 sửa tiền xử lý UCI để giữ nhãn test hợp lệ, tạo lại số hàng/hash test, kiểm tra cohort ARIMA, kiểm tra `promotion_proxy`/`promotion_flag`, tạo danh sách 10 chuỗi định giá M5 bằng train và xác minh đủ 20 tuần đánh giá mỗi chuỗi.
- Chưa train, tune, chạy PPO hay nối API trong Phase 0. Sau khi lập biên bản này, dừng để nhóm kiểm tra.
