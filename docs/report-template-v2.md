# Mẫu báo cáo kết quả theo giao thức V2

**Trạng thái:** mẫu Phase 0, chưa có điểm benchmark mới. Chỉ điền các ô sau khi hoàn thành các phase dữ liệu, huấn luyện và đánh giá. Đọc cùng [giao thức V2](../experiments/PROTOCOL_V2.md) và [cấu hình cố định](../experiments/protocol_v2.json). Không chép điểm pilot 100 SKU UCI hoặc 20 chuỗi M5 vào bảng toàn phạm vi.

## 1. Danh tính lượt chạy

| Trường | Giá trị cần ghi |
| --- | --- |
| Dataset, phiên bản nguồn, SHA256 | UCI ID 352 hoặc M5 evaluation; mã băm từng tệp |
| Git commit, phiên bản giao thức | Mã commit; `song_full_scope_v2` |
| Máy, thư viện, thời gian chạy | CPU/GPU, RAM, Python, PyTorch/XGBoost, ngày giờ |
| Split và panel | Tuần train/validation/test, số chuỗi đủ điều kiện, số khóa test, SHA256 khóa test |
| Chính sách nhãn | Quy tắc hợp lệ cố định, ngoại lệ train, tuần không bán, tuần thiếu giá |
| Đầu vào/đầu ra | Đường dẫn artifact panel, dự báo từng hàng, checkpoint, scaler, log |

## 2. Dự báo lượng bán, hai bảng độc lập

Lập **một bảng UCI và một bảng M5** với cùng các cột. Trong từng bảng, mọi dòng chính phải có cùng số khóa `(dataset, sku, week)` và cùng SHA256 khóa test. Nếu thiếu khóa hoặc seed, ghi “chưa đầy đủ” và chuyển sang bảng phụ; không xếp hạng cùng bảng chính.

| Mô hình | Có giá đầu vào? | Chuỗi train đủ điều kiện | Khóa test | SHA256 khóa test | RMSE trung bình ± std | MAE | R² | WAPE % | MAPE % (n dương) | Seed hoàn tất | Trạng thái |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARIMA + fallback chuỗi thưa | Không | — | — | — | — | — | — | — | — | Xác định | Chưa chạy V2 |
| XGBoost | Ghi rõ | — | — | — | — | — | — | — | — | — | Chưa chạy V2 |
| LSTM / LSTM + giá | Ghi rõ | — | — | — | — | — | — | — | — | — | Chưa chạy V2 |
| Transformer / Transformer + giá | Ghi rõ | — | — | — | — | — | — | — | — | — | Chưa chạy V2 |
| Song Transformer + giá | Có | — | — | — | — | — | — | — | — | — | Chưa chạy V2 |
| TFT / Informer và biến thể giá | Ghi rõ | — | — | — | — | — | — | — | — | — | Chưa chạy V2 |

Ghi thêm RMSE chuẩn hóa bằng trung bình train và độ lệch chuẩn train, thống kê theo SKU, số lần ARIMA dùng fallback, và phạm vi fit checkpoint. Mô hình xác định ghi một giá trị và “không áp dụng” cho std; không tạo độ lệch chuẩn bằng cách lặp cùng kết quả.

### Theo nhóm giá

Ấn định ngưỡng thấp/trung bình/cao bằng phân vị 1/3 và 2/3 của **giá tuần train** trong từng dataset, rồi giữ nguyên khi phân loại test theo quy tắc ở giao thức V2. Báo cả ngưỡng, số khóa test và RMSE của từng nhóm. Bảng này tương ứng ý tưởng Table 2 của Song, nhưng không gọi là cùng tập test nếu bài không công bố danh sách SKU và ngưỡng.

| Dataset | Mô hình | Ngưỡng giá train | Nhóm thấp: n/RMSE | Nhóm giữa: n/RMSE | Nhóm cao: n/RMSE |
| --- | --- | --- | --- | --- | --- |
| UCI hoặc M5 | — | — | — | — | — |

## 3. Chọn siêu tham số trên validation

| Dataset | Mô hình | Không gian thử và ngân sách | Tiêu chí validation | Cấu hình được chọn | RMSE validation | Thời gian | Checkpoint test |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| — | — | — | Micro RMSE | — | — | — | — |

Ghi danh sách **mọi cấu hình đã thử**, seed và thời gian, kể cả cấu hình không thắng. Bản Song cố định theo bài được đánh dấu riêng, không gọi là kết quả sau tune. Không dùng test để chọn tham số.

## 4. Định giá M5 trên cùng simulator và cohort

Ghi trước danh sách 10 chuỗi chọn từ train (một chuỗi mỗi cửa hàng), 20 tuần test mỗi chuỗi, miền giá train, phiên bản simulator, ngân sách tương tác và năm seed `42, 123, 456, 789, 1024`. Nếu không đủ 200 quyết định hợp lệ, ghi thiếu hụt và không điền bảng chính.

| Chính sách | Seed đủ | Doanh thu mô phỏng: mean | Std mẫu | Min | Max | Uplift so giá cố định % | Regret simulator | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Giá cố định | — | — | — | — | — | 0 | — | — |
| Heuristic | — | — | — | — | — | — | — | — |
| DDPG / SAC | — | — | — | — | — | — | — | — |
| PPO đóng băng Transformer | — | — | — | — | — | — | — | — |
| PPO + phản hồi Transformer | — | — | — | — | — | — | — | — | — |
| Ablation bỏ gradient / bỏ cập nhật | — | — | — | — | — | — | — | — | — |

Giữ doanh thu mô phỏng ở đơn vị `giá × số lượng` và ghi riêng thang reward huấn luyện. Số 883 trong Table 3 Song là thang simulator của bài, không so trực tiếp với số thô của nhóm. Lưu giá, lượng bán mô phỏng và doanh thu từng quyết định để vẽ đường doanh thu tích lũy theo 200 bước.

## 5. Đối chiếu bài Song và kết luận

| Bảng/chỉ số trong Song | Giá trị bài báo | Giá trị nhóm | Cùng nguồn/tuần/cohort/đơn vị? | Có thể so trực tiếp? | Lý do sai khác |
| --- | ---: | ---: | --- | --- | --- |
| Table 1: RMSE, RMSE Std., MAPE trên UCI/M5 | — | — | — | Chưa xác định | — |
| Table 2: RMSE theo ba nhóm giá | — | — | — | Chưa xác định | — |
| Table 3: doanh thu tích lũy năm lần chạy | — | — | — | Không nếu thang simulator khác | — |
| Table 6: ablation theo seed | — | — | — | Chưa xác định | — |

Kết luận cần nêu: mô hình dự báo thắng theo validation/test, kiểm tra đáp ứng giá, chính sách giá thắng trong **cùng simulator**, chi phí tính toán, phần tái dựng giống Song, phần khác do nhóm đặt quy tắc, và giới hạn không có quan sát phản thực tế/doanh thu thực ở giá đề xuất.
