# Thí nghiệm dự báo nhu cầu và định giá

Thư mục này xây một luồng **nghiên cứu độc lập** bên cạnh API và giao diện demo đang có. Mục tiêu là kiểm tra từng phần của kiến trúc trong [Song (2026)](https://link.springer.com/article/10.1007/s10791-026-10745-2): chuỗi sản phẩm–tuần → dự báo nhu cầu có giá làm đầu vào → đạo hàm theo giá → PPO → lượng bán mô phỏng → cập nhật mô hình dự báo. Không thay thế các mô hình mà API hiện đang dùng.

Giao thức mới cho lượt toàn phạm vi nằm trong [PROTOCOL_V2.md](PROTOCOL_V2.md), [biên bản Phase 0](../docs/phase0-audit.md) và [mẫu báo cáo](../docs/report-template-v2.md). [PROTOCOL.md](PROTOCOL.md) chỉ ghi phạm vi pilot lịch sử. Lệnh dựng và kiểm tra dữ liệu Phase 1 V2 nằm ở [README gốc](../README.md#phase-1--dựng-và-xác-minh-dữ-liệu-tuần-v2) và số liệu kiểm tra ở [biên bản Phase 1](../docs/phase1-audit.md). Các lệnh pilot bên dưới chưa tạo kết quả theo giao thức V2. Nhật ký hoàn thành từng phase, ý nghĩa và lệnh chạy lại thủ công nằm ở [PROJECT_HISTORY.md](../PROJECT_HISTORY.md#9-nhật-ký-xây-dựng-thí-nghiệm-theo-phase).

## Thử feature dự báo M5 và làm trơn lượng bán UCI

Chạy tại thư mục gốc sau khi có panel Phase 1 V2, kết quả Bước 3 ARIMAX và bộ đối chiếu định giá ARIMAX–PPO. Lệnh cuối đọc lại các mô hình XGBoost đã lưu để kiểm tra đề xuất giá.

```powershell
python -m unittest experiments.test_forecast_feature_ablation experiments.test_price_feature_ablation -v
python -u -m experiments.forecast_feature_ablation --threads 8
python -u -m experiments.uci_forecast_smoothing
python -u -m experiments.price_feature_ablation
```

Có thể thêm `--stores CA_1` để thử nhanh trước. Lần chạy đủ 10 cửa hàng dùng lại kết quả từng cửa hàng đã xác minh. Nếu thay số luồng hoặc cấu hình cho một lần chạy khác, dùng `--output-dir` mới và truyền thư mục đó bằng `--ablation-dir` cho lệnh thử giá. Kết quả đầy đủ và tham số từng cửa hàng nằm trong `artifacts/forecast_feature_ablation_v2/`.

**Dự báo M5.** Bốn biến thể XGBoost huấn luyện riêng theo cửa hàng với cùng siêu tham số. Mỗi biến thể dùng cùng các nhãn train có quan sát. Cột lượng bán và giá lịch sử chỉ đọc các tuần đã qua. Biến thể có giá được chấm tại giá đã ghi nhận của tuần dự báo. So sánh chính đối chiếu AR(1) và ARIMAX Bước 3 trên cùng 1.220.898 khóa test V2. Mô hình được chọn bằng RMSE gộp trên validation.

| Mô hình và feature cộng dồn | RMSE validation | RMSE test | MAE test | R² test |
| --- | ---: | ---: | ---: | ---: |
| AR(1) Bước 3 + fallback | 9,164 | 8,790 | 4,001 | 0,841 |
| ARIMAX Bước 3 + fallback | 9,274 | 9,337 | 4,122 | 0,821 |
| XGBoost với lượng bán tuần trước và quan sát gần nhất | 9,916 | 9,417 | 3,967 | 0,818 |
| Thêm lượng bán trễ 2, 4, 52 tuần và trung bình 4, 12 tuần | 8,998 | 8,358 | 3,407 | 0,857 |
| Thêm mùa vụ theo tuần trong năm, số ngày sự kiện và SNAP | 9,009 | 8,506 | 3,400 | 0,852 |
| Thêm giá tuần dự báo, giá tuần trước và tỷ lệ đổi giá | **8,718** | **8,324** | **3,383** | **0,858** |

XGBoost có giá giảm RMSE so với AR(1) tại 8/10 cửa hàng test và giảm MAE tại 10/10. Phần lớn cải thiện của XGBoost đến từ nhóm lịch sử bán. Phần thêm lịch và SNAP giảm nhẹ MAE nhưng tăng RMSE trong cấu hình thử. Mốc ARIMA(1,0,1) của Phase 5 đạt RMSE 8,15 trong giao thức fit trên train + validation. Bảng mới đối chiếu các mô hình fit trên train. [Kết quả tổng hợp](artifacts/forecast_feature_ablation_v2/metrics.csv) · [Theo cửa hàng](artifacts/forecast_feature_ablation_v2/by_store.csv)

**Thử giá M5.** Dùng cùng 1.200 khóa test của hai nhóm đã khóa (10 và 50 chuỗi). Lưới XGBoost và ARIMAX thử giá trong cùng miền train. Khi đổi giá ứng viên, lưới XGBoost tính lại cả tỷ lệ đổi giá. Hai bộ mô phỏng Gaussian và điều chỉnh theo lượng tuần trước cùng lấy lịch sử quan sát làm điểm xuất phát cho từng quyết định một tuần. Bảng dưới tính doanh thu kỳ vọng trong bộ mô phỏng và lấy giá cố định theo SKU đã chọn trên validation làm mốc.

| Nhóm | Bộ mô phỏng | Lưới ARIMAX so với giá cố định | Lưới XGBoost so với giá cố định |
| --- | --- | ---: | ---: |
| 10 chuỗi | Gaussian | −1,83% | −11,24% |
| 10 chuỗi | Điều chỉnh theo tuần trước | −2,18% | −9,36% |
| 50 chuỗi | Gaussian | −2,64% | −10,99% |
| 50 chuỗi | Điều chỉnh theo tuần trước | −3,01% | −10,08% |

Lưới XGBoost kém lưới ARIMAX ở cả bốn phép so sánh. Trên 53,25% đường cầu XGBoost, có ít nhất một đoạn lượng bán dự báo tăng khi giá tăng và lịch sử được giữ nguyên. Tỷ lệ tương ứng của ARIMAX là 41,67%. Đây là kiểm tra mô phỏng một bước với lịch sử quan sát được đặt lại mỗi tuần. Kết quả ủng hộ giữ ARIMAX cho bước đề xuất giá hiện tại. [Kết quả thử giá](artifacts/forecast_feature_ablation_v2/price_one_step_summary.csv)

**Dự báo UCI.** Kiểm tra trên cùng 18.542 khóa test V2. Với mốc tuần trước, 186 dòng có sai số lớn nhất đóng góp 68,74% tổng sai số bình phương. Theo RMSE validation, trung bình 4 tuần gần nhất là lựa chọn tốt nhất trong các quy tắc đã thử. Theo MAE validation, trung vị các tuần có quan sát trong 8 tuần gần nhất là lựa chọn tốt nhất.

| Quy tắc UCI | RMSE test | MAE test | R² test |
| --- | ---: | ---: | ---: |
| Lượng bán tuần trước | 140,84 | 48,27 | 0,315 |
| Trung bình 4 tuần gần nhất | **115,82** | 41,62 | **0,537** |
| Trung vị các tuần quan sát trong 8 tuần gần nhất | 122,63 | **38,57** | 0,481 |

Các giá trị trên tính bằng sản phẩm mỗi tuần. Bảng 1 của Song báo RMSE 15,7 và MAPE cho UCI. Mục 4.1.3 mô tả RMSE UCI sau lọc ngoại lệ theo đơn vị sản phẩm mỗi tuần, còn mục 4.2 mô tả điều chỉnh theo trung bình lịch sử từng SKU ở cả hai bộ dữ liệu. Cần thống nhất cách xử lý ngoại lệ, tập chấm và thang đo trước khi kết luận chênh lệch giữa hai nghiên cứu. [Kết quả UCI](artifacts/forecast_feature_ablation_v2/uci_baseline_exploration.csv) · [Song, mục 4.1.3 và 4.2](https://link.springer.com/article/10.1007/s10791-026-10745-2)

## Tune XGBoost UCI và kiểm tra độ phủ mô hình cầu M5

Chạy từ thư mục gốc với panel Phase 1 V2 và bộ mô hình M5 fit chỉ trên train. Thử nghiệm dùng bộ tham số đã lưu trong `final_system/runtime/models/` để lấy các hệ số M5 theo SKU–cửa hàng. Thay đường dẫn bằng `--model-dir` nếu bộ tham số nằm ở nơi khác.

```powershell
python -m unittest experiments.test_forecast_reliability -v
python -u -m experiments.uci_xgb_reliability
python -u -m experiments.m5_demand_coverage
```

**UCI.** XGBoost học phần sai lệch so với trung bình 4 tuần. Nhóm lịch sử gồm lượng bán trễ 1, 2, 3, 4, 8 và 12 tuần, cùng trung bình, trung vị tuần có bán, số tuần có bán và mức bán gần đây cao nhất. Nhóm lịch thêm tháng, tuần trong năm, cờ tháng 11–12 và tuần Giáng sinh. Mỗi nhóm thử 12 tổ hợp độ sâu cây, số quan sát tối thiểu ở lá và L2, với ba trọng số áp dụng phần hiệu chỉnh. Số cây dừng sớm và cấu hình được chọn theo validation ở seed 42. Hai lượt seed 123 và 456 kiểm tra lại cấu hình đã chọn. Các phần lịch sử chỉ đọc tuần đứng trước tuần cần dự báo.

| Phương pháp UCI | RMSE validation | RMSE test | MAE test | R² test |
| --- | ---: | ---: | ---: | ---: |
| Trung bình 4 tuần đang dùng | 105,95 | 115,82 | 41,62 | 0,537 |
| XGBoost hiệu chỉnh từ lịch sử | 105,43 | **115,20** | **41,48** | **0,542** |
| XGBoost hiệu chỉnh từ lịch sử + lịch | **105,27** | 115,37 | 41,54 | 0,540 |

Điểm XGBoost là trung bình ba seed, trên cùng 14.574 bản ghi validation và 18.542 bản ghi test. Cấu hình có lịch được chọn qua validation. Cây sâu tối đa 4, trọng số tối thiểu ở lá 12, L2 bằng 5 và dừng ở 11 cây tại seed 42. Mức giảm RMSE test so với trung bình 4 tuần chỉ khoảng 0,4%. Điểm test của cả ba seed dùng lịch đều cao hơn bản XGBoost chỉ dùng lịch sử. Đề xuất hiện tại là giữ trung bình 4 tuần cho API và giữ kết quả XGBoost làm ứng viên. [Điểm theo seed](artifacts/uci_xgb_reliability_v2/scores.csv) · [Kết quả tune](artifacts/uci_xgb_reliability_v2/validation_trials.csv)

**M5.** Giữ mô hình cầu ARIMAX ở 18.244 chuỗi có hệ số giá riêng. Với chuỗi còn thiếu hệ số giá, thử giữ động lực lượng bán tại cửa hàng và lấy hệ số giá từ cùng SKU ở ít nhất hai cửa hàng khác có miền giá train phủ mức tham chiếu. Mỗi giá thử nằm trong ±5% quanh trung vị giá train của cửa hàng và thuộc miền giá train của ít nhất hai cửa hàng tham chiếu. Hệ số lấy trung vị của các hệ số co giãn quy đổi và chọn mức co trên validation.

- Có **1.452 chuỗi** đủ điều kiện để thử chuyển hệ số. Hệ số nhân **0,25** được chọn trên các tuần validation có đổi giá trong vùng hỗ trợ.
- Trên **197 tuần test có đổi giá** thuộc 132 chuỗi trong vùng hỗ trợ, RMSE của cùng phương trình khi đặt hệ số giá bằng 0 là **18,039** và khi dùng hệ số mượn là **18,070**. Kết quả chưa ủng hộ mở quyền đề xuất giá cho nhóm này.
- Với các chuỗi thiếu hệ số giá riêng, **8.984 chuỗi** có tổng **345.989 tuần test** mà giá lịch sử và giá tuần dự báo đều khớp mốc train. Dùng cùng phương trình với hệ số giá bằng 0 đạt RMSE **8,438** và MAE **3,495** trên nhóm đó. Nhóm này có thể mở rộng dự báo ở giá đã quan sát. Miền giá thử mới vẫn cần bằng chứng riêng.

[Độ phủ theo SKU–cửa hàng](artifacts/m5_demand_coverage_v2/series_coverage.csv) · [Sai số trên các nhóm tuần](artifacts/m5_demand_coverage_v2/transfer_scores.csv)

**Lưới giá và PPO có phản hồi.** Trong benchmark bổ sung trước đó, lưới ARIMAX đã tune và PPO có phản hồi bắt đầu từ cùng cấu hình ARIMAX. Lưới có doanh thu mô phỏng cao hơn PPO phản hồi ở cả 20 cặp gồm hai nhóm SKU, hai bộ mô phỏng và năm seed. Trung bình từng nhóm và bộ mô phỏng, khi lấy giá cố định theo SKU làm mốc 100, lưới đạt 97,50–97,75 và PPO phản hồi đạt 79,62–89,30. Kết luận này áp dụng cho các mô phỏng đã chạy. Phản hồi là cách cập nhật mô hình cầu từ kết quả bán và có thể được dùng cùng lưới khi có dữ liệu kết quả phù hợp. [Báo cáo doanh thu chuẩn hóa](artifacts/pricing_normalized_v2/report.md)

**Nhánh kết quả pilot cũ:** `phase3c.py` đối chiếu ARIMA/LSTM/Transformer có
và không có giá trên panel UCI 100 SKU và M5 20 chuỗi đã khóa. Trên M5,
`select_simulator.py` chọn hiệu chỉnh mức cầu bằng validation;
`song_pricing_pilot.py` chạy PPO, DDPG, SAC và các phép bỏ thành phần;
`phase4_compare.py` so mức tăng doanh thu *mô phỏng* theo seed giữa hai
simulator. Các lệnh và số liệu cụ thể nằm trong PROJECT_HISTORY.md.

Để chạy các baseline trên đúng panel UCI/M5 đã chuẩn bị, dùng `python -m experiments.common_forecast` thay vì để `experiments.run` tự dựng lại panel demo. Sau đó chạy `experiments.song_forecast --refit-train-validation` và `experiments.compare_forecasts` như các lệnh Phase 2–3A trong PROJECT_HISTORY.md. Checkpoint `song_transformer_seed42.pt` vẫn chỉ học train để dùng cho chính sách giá; checkpoint có hậu tố `_refit` chỉ dùng để chấm dự báo test.

## Bước ưu tiên: dữ liệu và luồng theo Song (2026)

Tải **UCI Online Retail một năm** từ UCI và **đủ năm tệp M5** từ bản sao công khai; mã MD5 và kích thước của M5 phải khớp [bản lưu Zenodo](https://zenodo.org/records/10203108). File `data/song_replication/download_manifest.json` lưu URL, kích thước và SHA256; dữ liệu gốc không commit vào Git. Các thí nghiệm hiện tại dùng `sales_train_validation.csv` (1.913 ngày), `sell_prices.csv` và `calendar.csv`; hai tệp M5 còn lại được giữ để có bộ nguồn đầy đủ, chưa dùng làm nhãn huấn luyện/test.

Link chính thức Kaggle/UCI, mirror tải trực tiếp từng file M5 và bộ Online Retail II 2009–2011 được liệt kê trong mục **Link tải dữ liệu** của [README gốc](../README.md). Script `download_paper_data` tải tự động UCI Online Retail và các file M5 vào `data/song_replication/`; với Online Retail II, tải ZIP từ UCI, giải nén rồi đặt `online_retail_II.xlsx` vào `data/`.

```powershell
python -m pip install -r requirements-experiments.txt
python -m experiments.download_paper_data --dataset all
python -m experiments.song_online_retail
python -m experiments.run --stage data --weekly-path experiments/artifacts/song_online_retail/weekly.csv --selected-path experiments/artifacts/song_online_retail/selected_products.csv --output-root experiments/artifacts/song_online_retail/paper_70_15_15
python -m experiments.song_forecast --epochs 50
```

`song_forecast` dùng toàn bộ SKU đủ điều kiện từ **chỉ tập train**, 12 tuần quá khứ, Transformer 4 lớp/8 heads/128, Adam 1e-3 và MSE trên lượng bán chuẩn hóa. Kết quả lưu ở `experiments/artifacts/song_online_retail/song_forecast/`. Manifest tiền xử lý liệt kê các lựa chọn mà bài báo không nêu đủ chi tiết, gồm loại mã hàng, xử lý tuần không giao dịch và quy tắc 3×IQR. UCI không kèm cờ khuyến mãi hay lịch lễ, nên các đầu vào đó bằng 0. Các thang RMSE thô, theo trung bình train và z-score đều được xuất vì mô tả chuẩn hóa RMSE trong bài không nhất quán.

Luồng định giá trên M5 đã có **pilot 20 sản phẩm tại CA_1**, chọn theo lượng bán ở train và yêu cầu ít nhất hai giá quan sát trong train. Lệnh này kiểm tra được 200 chu kỳ test mô phỏng, gồm 10 sản phẩm × 20 tuần; phản hồi mỗi 10 quyết định có thể ảnh hưởng 10 quyết định tiếp theo của cùng sản phẩm. `--timesteps 64 --seeds 42` chỉ dùng để kiểm tra kết nối; tăng ngân sách và chạy năm seed mới là thí nghiệm đánh giá.

```powershell
python -m experiments.song_m5_pilot
python -m experiments.song_forecast --prepared-root experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/song_forecast --epochs 30
python -m experiments.song_pricing_pilot --seeds 42 --timesteps 64 --policies fixed_mean heuristic ppo_feedback_gradient
```

Pilot này **không tương đương toàn bộ M5** của Song và doanh thu của nó không được đối chiếu trực tiếp với con số 883 trong bài. Tác giả không công bố code, cách chọn chuỗi M5, công thức chuẩn hóa doanh thu và chi tiết simulator đầy đủ; `pricing_manifest.json` ghi rõ giả định của dự án. API/frontend và dữ liệu 20 SKU Online Retail II cũ không tham gia nhánh Song.
Nhánh `song_pricing_pilot.py` chọn hành động trên **toàn miền giá train**,
còn `experiments.run` dùng giới hạn thay đổi ±10% mỗi bước; hai luồng này
không nên gộp kết quả. Đường giá–cầu của simulator và doanh thu ở giá chưa
quan sát chưa được xác nhận từ dữ liệu thực.

## Dữ liệu và cách chia

### Tối ưu bộ nhớ cho benchmark M5 toàn phạm vi

`forecast_torch.py` dùng embedding SKU thay cho one-hot dày khi panel có hơn
256 chuỗi; panel nhỏ vẫn giữ cách mã hóa cũ. `forecast_benchmark.py` dùng
one-hot sparse cho XGBoost ở cùng ngưỡng. Dự báo validation/test được chia lô,
và mảng chuỗi lịch sử được cấp phát trực tiếp để giảm các bản sao tạm.
`make_windows(..., forecast_only=True)` tạo bảng gọn chỉ gồm các cột cần cho
benchmark dự báo. Các thay đổi này chỉ tối ưu biểu diễn/tài nguyên; không thay
đổi nhãn, mốc chia train/validation/test hay quy tắc tính chỉ số.

Runner duyệt lần lượt các panel cửa hàng để giới hạn RAM; mỗi cửa hàng lưu dự
báo và chỉ số riêng, đồng thời cập nhật bảng chỉ số tổng hợp. Benchmark dùng
8 phương pháp hiện có và chấm test trên cùng SKU–tuần quan sát trong mỗi cửa
hàng. Hướng dẫn từ clone mới: tải M5, tạo panel toàn phạm vi, rồi chạy benchmark.

```powershell
python -m experiments.download_paper_data --dataset m5
python -m experiments.full_m5_prepare --data-dir data/song_replication/m5 --output-root experiments/artifacts/song_m5_full
python -u -m experiments.full_m5_benchmark --prepared-root experiments/artifacts/song_m5_full --output-root experiments/artifacts/song_m5_full/full_benchmark
```

`full_m5_prepare` dùng `sales_train_evaluation.csv`, đủ 10 cửa hàng, chia thời
gian 70/15/15 và lookback 12 tuần. Nó lưu `weekly_<store>.csv` và manifest
riêng từng cửa hàng. `full_m5_benchmark` dùng seed 42, tối đa 25 epoch và có
thể chạy lại cùng lệnh sau khi bị gián đoạn; kết quả cửa hàng đã hoàn tất được
tái sử dụng. Thêm `--stores CA_1` nếu chỉ muốn chạy thử một cửa hàng.

Kết quả tổng hợp là
`experiments/artifacts/song_m5_full/full_benchmark/forecast_metrics_full_scope.csv`;
thư mục con từng cửa hàng chứa dự báo, chỉ số theo SKU và manifest lần chạy.
Các artifact không được Git theo dõi. Bộ test này là holdout theo thời gian của
dự án, không phải test chính thức 28 ngày của cuộc thi M5 và chưa thể khẳng
định trùng hoàn toàn phạm vi/test của Song.

Trong lượt chạy toàn phạm vi hiện tại, ARIMA không có điểm validation nếu một
cửa hàng có chuỗi thiếu 12 tuần train quan sát; ARIMA vẫn phải dự báo đủ test
chung. Việc huấn luyện ARIMA/Transformer trên CPU có thể mất nhiều giờ.

```powershell
python -m experiments.full_m5_benchmark --prepared-root experiments/artifacts/song_m5_full --output-root experiments/artifacts/song_m5_full/full_benchmark
```

Mặc định `experiments.data` dùng **20 SKU đang hiển thị trên hệ thống**, tạo đủ các tuần trong giai đoạn 2009–2011 và cửa sổ 12 tuần. Tuần không có giao dịch được đánh dấu `observed_sales=False`; giá được điền tiến từ quá khứ để tạo ngữ cảnh, nhưng tuần đó **không được tính là nhãn nhu cầu thật** trong bảng dự báo. Mọi tham số chuẩn hóa được học từ train.

Có ba cấu hình chia theo **tuần lịch, không xáo trộn**:

| Cấu hình | Train / validation / test | Nguồn phương pháp | Phạm vi so sánh hợp lệ |
|---|---|---|---|
| `paper_70_15_15` | 70% / 15% / 15% | Song (2026), mục 4.2 | So sánh các thuật toán của dự án trên cùng 20 SKU; dữ liệu của Song là Online Retail một năm và M5 |
| `retail2_80_20` | 80% / 0% / 20% | [Harighanesh & Senthilkumar (2026)](https://www.ijert.org/research/price-elasticity-of-demand-and-dynamic-pricing-in-online-retail-a-machine-learning-approach-to-reven-IJERTV15IS100885.pdf), mục III.C | Cùng Online Retail II và mức sản phẩm–tuần, nhưng bài dùng 150 SKU và cách chọn khác |
| `retail2_last12` | Giữ 12 tuần cuối làm test | [K C & Dangol (2026)](https://osuva.uwasa.fi/items/06715eee-317a-4538-a18d-9f7c3e8bc713), mục 3.7 | Chỉ mượn độ dài holdout. Luận văn dùng 90 tuần train sau buffer 4 tuần và dự báo lượng bán tổng 12 tuần; dự án dùng 20 SKU và dự báo từng tuần |

[Qiao et al. (2026)](https://arxiv.org/html/2602.12147v4) cũng dùng Online Retail II nhưng gộp thành một chuỗi **theo ngày**, giữ 180 ngày cuối và sáu cửa sổ dự báo 30 ngày. Không so RMSE của bài đó với RMSE sản phẩm–tuần ở đây.

Manifest của hai cấu hình sau liệt kê ba fold cuốn chiếu hoàn toàn nằm trong train để dùng cho lần điều chỉnh siêu tham số tiếp theo; benchmark hiện tại giữ siêu tham số cố định và **chưa chạy cross-validation trên các fold này**.

Lưu ý: file 20 SKU hiện hữu đã được chọn bằng thống kê của **toàn bộ hai năm**, nên có thiên lệch lựa chọn sản phẩm từ tương lai. Lựa chọn chặt chẽ hơn nằm trong bước tiền xử lý riêng bên dưới; nó thay đổi tập 20 SKU, không tự động thay dữ liệu demo.

## Cài đặt và chạy

Tại thư mục gốc dự án, cài thư viện nghiên cứu:

```powershell
python -m pip install -r requirements-experiments.txt
```

Chạy bảng dự báo trên 20 SKU demo với cùng tập test:

```powershell
python -m experiments.run --stage forecast
```

Chạy cấu hình 80/20 từ bài cùng Online Retail II:

```powershell
python -m experiments.run --stage forecast --split-profile retail2_80_20
```

Chạy môi trường giá cùng các chính sách; lệnh sau gồm 5 seed và các phép bỏ thành phần:

```powershell
python -m experiments.run --stage pricing --seeds 42 123 456 789 1024 --timesteps 1024 --ablations
```

`--stage all` chạy cả dự báo và định giá. `--no-learned-policies` chỉ chạy giá cố định và quy tắc đơn giản, hữu ích để kiểm tra nhanh dữ liệu/môi trường. Kết quả mặc định nằm dưới `experiments/artifacts/online_retail_ii_20/`; khi chọn cách chia khác, CLI tạo thư mục con tên `retail2_80_20` hoặc `retail2_last12`. Các artifact không được commit; manifest ghi nguồn, SHA256, mốc chia, seed và phương pháp đo.

### Tiền xử lý gần phương pháp bài báo hơn

Script sau đọc lại giao dịch gốc, giải quyết đoạn trùng giữa hai sheet, coi hủy/hoàn là không đóng góp lượng bán dương, lọc ngoại lệ 3×IQR với ngưỡng học từ train, tính **giá trung bình** và tổng doanh thu từng dòng hóa đơn. Nó chọn 20 SKU bằng thống kê ở **train** rồi xuất một cohort nghiên cứu riêng:

```powershell
python -m experiments.retail_ii_preprocess
python -m experiments.run --stage forecast --weekly-path experiments/artifacts/retail_ii_paper/weekly.csv --selected-path experiments/artifacts/retail_ii_paper/selected_products.csv --output-root experiments/artifacts/retail_ii_paper_20
```

Để dùng đúng 20 SKU demo trong bộ tiền xử lý mới, thêm `--selection-mode existing` ở lệnh đầu. Manifest ghi mọi sai khác với Song (2026). Dữ liệu nhóm là **Online Retail II 2009–2011**; bài báo chính dùng **Online Retail 2010–2011** để dự báo.

### M5

`experiments.m5` chuyển các CSV chính thức của [M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) thành sản phẩm–cửa hàng theo tuần. Cần cung cấp `sales_train_validation.csv`, `sell_prices.csv`, `calendar.csv` trong một thư mục. Ví dụ kiểm thử adapter mà không cần tải dữ liệu:

```powershell
python -m experiments.m5 --self-test
```

M5 đã tải vào `data/song_replication/m5/` nhưng không nằm trong Git. Dữ liệu sự kiện và SNAP có trong adapter; nhãn `promotion_proxy` chỉ suy từ giảm giá và **không phải cờ khuyến mãi được quan sát độc lập**. Cần chạy toàn bộ M5 và xác định chuẩn hóa của bài báo trước khi đối chiếu bảng doanh thu PPO của Song.

Sau khi tải ba CSV chính thức, có thể chạy một tập con có giới hạn bộ nhớ:

```powershell
python -m experiments.run --dataset m5 --m5-dir "E:\duong-dan\m5" --m5-item-id FOODS_1_001 --m5-store-id CA_1 --stage data
```

Các đặc trưng sự kiện và SNAP đã biết từ lịch được đưa vào mô hình dự báo, trạng thái chính sách và môi trường mô phỏng. Khi đã chốt một tập sản phẩm/cửa hàng đủ lớn, đổi `--stage data` thành `--stage all` và chỉ định ngân sách huấn luyện. Ví dụ một sản phẩm–cửa hàng chỉ kiểm tra luồng đọc dữ liệu; kết quả của nó không đại diện toàn bộ M5 trong bài báo.

## Các thuật toán và chỉ số hiện có

`forecast_benchmark.py` chạy naive, ARIMA, XGBoost, LSTM, Transformer, cùng biến thể có giá của ba mô hình sau. Mọi dự báo được chấm trên cùng SKU–tuần quan sát ở đơn vị **sản phẩm/tuần** bằng RMSE, MAE, WAPE và MAPE. ARIMA/naive sử dụng các quan sát đã đến ở tuần trước; đây là **dự báo cuốn chiếu một bước**, không phải dự báo 12 tuần từ một thời điểm duy nhất. TFT và Informer trong bài báo chưa được cài trong nhánh này.

`pricing_simulator.py` fit hồi quy kernel Gaussian bằng **train**, dùng giá và tuần trong năm để tạo nhu cầu ở giá giả định. `pricing_benchmark.py` giữ cùng vùng giá và chuỗi tuần test cho giá cố định, quy tắc tăng/giảm, DDPG, SAC, PPO dự báo đóng băng và PPO cập nhật dự báo/đạo hàm. Có các phép bỏ đạo hàm và ngừng cập nhật. PPO dùng forecast Transformer có giá; cứ 10 tương tác mô phỏng sẽ fine-tune từ tối đa 200 phản hồi gần nhất.

Transformer có 4 lớp, 8 heads, kích thước 128 như bài báo; huấn luyện tối đa 25 epochs cho tập 20 SKU. PPO dùng actor/critic ba lớp 256, `gamma=0.99`, `gae_lambda=0.95`, `clip=0.2`, batch 64. Giá mỗi bước nằm trong vùng train và tối đa ±10% so với giá đang dùng. Checkpoint dự báo dùng cho định giá chỉ học **train**; validation chỉ chọn epoch. Khi đánh giá test, phản hồi online được dùng theo thứ tự tuần **trong từng SKU** rồi trạng thái mô hình được khôi phục trước SKU kế tiếp, tránh nhìn trước thời gian giữa các sản phẩm.

Kiểm tra riêng đạo hàm giá của Transformer trên test:

```powershell
python -m experiments.price_diagnostics
```

Script xuất từng gradient, độ co giãn cục bộ suy từ mô hình và kết quả tại các tuần có đổi giá. Đây chỉ là kiểm tra độ nhạy của **mô hình dự báo**; các thay đổi giá quan sát được không phải can thiệp ngẫu nhiên.

**Doanh thu từ môi trường này là số mô phỏng.** Kiểm tra sai số tại giá đã bán không xác nhận nhu cầu tại một mức giá chưa từng bán, và dữ liệu không cho phép kết luận tác động nhân quả hay mức tăng doanh thu thật. Bài báo cũng nêu giới hạn này. Các con số RMSE 15,7 và doanh thu 883 của Song không cùng bộ dữ liệu/thang đo với kết quả của nhánh 20 SKU.

Môi trường kernel hiện có quan hệ giá–cầu cố định theo sản phẩm và lịch; nó không tạo ra thay đổi thật của thị trường sau khi chính sách giá đổi. PPO được huấn luyện với cùng số tương tác mô phỏng giữa các thuật toán, nhưng kiến trúc và chi phí tính toán của DDPG/SAC/PPO khác nhau. Doanh thu của các chính sách được so trong cùng môi trường, còn diễn giải về "demand drift" của bài báo cần một thiết kế mô phỏng khác hoặc thử nghiệm thực tế.

Bảng RMSE là đánh giá **mô hình dự báo trước khi cập nhật online** tại giá quan sát; nhánh hiện tại chưa đo RMSE sau phản hồi PPO. Dữ liệu bán thật cũng không cho nhãn lượng cầu tại mức giá do PPO chọn, vì vậy không thể dùng nó để xác nhận độ chính xác phản thực tế.

## Tệp kết quả

- `forecast/forecast_metrics.csv`: chỉ số theo thuật toán và split.
- `forecast/forecast_metrics_by_sku.csv`: chỉ số từng sản phẩm.
- `forecast/forecast_predictions.csv`: một dòng cho mỗi thuật toán, SKU và tuần.
- `forecast/models/transformer_price.pt`: checkpoint train+validation để chấm dự báo trên test.
- `forecast/models/transformer_price_train_only.pt`: checkpoint dành cho thí nghiệm định giá, không học nhãn validation/test.
- `price_diagnostics/price_sensitivity.csv`: độ nhạy giá của mô hình trên các dòng test.
- `pricing_verified/simulator_observed_price_check.csv`: sai số môi trường ở các giá từng quan sát.
- `pricing_verified/policy_runs.csv` và `policy_aggregate.csv`: kết quả theo seed và trung bình.
- `pricing_verified/policy_episodes.csv`: giá, cầu, doanh thu và khoảng cách so với oracle giá toàn vùng quan sát của từng chu kỳ mô phỏng. Oracle có thể nằm ngoài giới hạn đổi giá ±10% của một bước.

Sau khi có kết quả các cách chia, tổng hợp thành một báo cáo Markdown:

```powershell
python -m experiments.report
```

Báo cáo được tạo ở `experiments/artifacts/comparison_report.md`.

Kiểm tra chức năng tập trung:

```powershell
python -m unittest discover -s experiments -p 'test_*.py' -v
```

### Phase 6: chọn giá một bước trên pilot M5

Sau khi tạo các artifact M5 Phase 3C/4, chạy:

```powershell
python -m experiments.phase6_optimize
python -m unittest experiments.test_phase6_optimize -v
```

Script dùng 5 Transformer train-only để tối đa hóa giá × lượng bán dự báo
trên lưới giá trong miền train và ±10% từ giá quan sát tuần trước. Nó so
cùng 200 tuần validation, 174 tuần test hợp lệ với giữ giá, quy tắc đơn
giản, PPO cố định, DDPG và SAC; oracle biết simulator chỉ là mốc chẩn đoán.
`phase6_price_selection/selected_price_decisions.csv` chứa từng giá được
quy tắc chọn theo validation. Chi tiết kết quả và giới hạn diễn giải nằm
trong [nhật ký dự án](../PROJECT_HISTORY.md).

### Phase 7: kiểm định cầu và độ bền của giá

Chạy theo thứ tự sau từ thư mục gốc, sau Phase 6:

```powershell
python -m experiments.phase7_demand_validation
python -m experiments.phase7_policy_robustness
python -m unittest discover -s experiments -p 'test_*.py' -q
```

Lệnh đầu tạo ba fold cuốn chiếu chỉ nằm trong train, chọn mô hình cầu bằng
RMSE tại giá đã bán và xuất riêng sai số validation/test, kể cả cohort Phase
6. Lệnh sau giữ nguyên giá của Phase 6 để chấm dưới các giả định cầu khác
nhau; không cập nhật hay chọn lại chính sách. Artifact nằm trong
`artifacts/song_m5_pilot/phase7_demand_validation/` và
`artifacts/song_m5_pilot/phase7_policy_robustness/`. Xem [nhật ký dự án](../PROJECT_HISTORY.md)
để đọc kết quả và [bảng đối chiếu bài báo](phase7_paper_comparison.md) để
phân biệt chỉ số so sánh trực tiếp với đối chiếu phương pháp.

### Phase 8: thử giá và so sánh trực tiếp

Không huấn luyện lại. Sau khi có artifact Phase 6–7, chạy backend và frontend
theo [README gốc](../README.md), rồi mở `price-m5.html`. Trang gửi một yêu cầu
`POST /research/price/compare` để tính 7 phương pháp trên cùng SKU, tuần,
giá hiện tại và lịch sử trước quyết định. Seed đại diện của các chính sách
học được chọn từ validation; nếu thử trực tiếp một seed, chỉ hàng phương pháp
đó dùng seed bạn nhập. Cột validation là điểm trung bình trên các tuần
validation lịch sử, còn doanh thu của tuần đang xem là dự báo/mô phỏng. API
`POST /research/price/recommend` vẫn dùng được cho một phương pháp. Kiểm tra
bằng `python -B -m unittest experiments.test_phase8_api -v`.
