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

## 7. Giao diện sản phẩm và lịch sử giá

Sau khi chạy pipeline và có mô hình trong `models_2y/`, nhập 20 sản phẩm cùng
2.072 dòng theo tuần vào SQL Server (chỉ cần làm khi database chưa có dữ liệu):

```powershell
.\database\setup.ps1
```

Mở hai terminal tại thư mục gốc dự án và chạy:

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

```powershell
python frontend/serve.py
```

Mở `http://127.0.0.1:8765/` để thử giá, `/products.html` để xem danh sách 20
sản phẩm, rồi chọn một SKU để xem giá bán theo tuần, doanh thu quy đổi và nhật
ký các lần thử giá. Bảng so sánh batch không tạo bản ghi lịch sử; khi bấm
“Tính giá đề xuất”, kết quả được lưu vào `RecommendationHistory`.

Doanh thu quy đổi của dữ liệu lịch sử là `qty × price` từ `WeeklySales`, với
`price` là giá trung vị tuần. Đây không phải tổng tiền chính xác trên hóa đơn.
Nhật ký lưu giá nhập và giá mô hình đề xuất, không ghi nhận việc áp dụng giá bán.
Chi tiết cách chạy giao diện ở `frontend/README.md`.

## 8. Thí nghiệm so sánh thuật toán

Luồng nghiên cứu theo bài báo Song (2026), các cách chia tập theo thời gian,
benchmark dự báo nhu cầu và mô phỏng chính sách giá nằm ở
[`experiments/README.md`](experiments/README.md). Các script này ghi kết quả
vào `experiments/artifacts/` và không thay đổi mô hình/API demo hiện tại.

## 9. Nhật ký xây dựng thí nghiệm theo phase

Thực hiện các phase theo thứ tự dưới đây. Mỗi phase ghi ý nghĩa, kết quả và
cách chạy lại ngay tại đây; giao thức chi tiết nằm ở
[`experiments/PROTOCOL.md`](experiments/PROTOCOL.md). Các lệnh chạy từ thư mục
gốc dự án trong PowerShell.

### Phase 0 — Chốt nguồn và giao thức (hoàn thành 2026-11-15)

**Ý nghĩa:** xác định đúng Online Retail 2010–2011 và M5 của Song, tách dữ liệu
Online Retail II của demo cũ, và quy định khi nào có thể so điểm số giữa các
bài báo. Bốn nguồn và các giới hạn tái dựng được ghi trong `experiments/PROTOCOL.md`.

**Đã kiểm tra:** lệnh dưới đây xác nhận Excel UCI và đủ năm tệp M5 đang có;
kích thước/MD5 M5 khớp bản Zenodo. Manifest nguồn nằm ở
`data/song_replication/download_manifest.json`.

```powershell
python -m experiments.download_paper_data --dataset all
Get-Content experiments/PROTOCOL.md
Get-Content data/song_replication/download_manifest.json
```

Lệnh đầu kiểm tra tệp sẵn có và chỉ tải khi thiếu. Chưa có phép so sánh chất
lượng mô hình nào ở Phase 0.

### Phase 1 — Chuẩn hóa nguồn và tập dữ liệu (hoàn thành 2026-11-15)

**Ý nghĩa:** tạo một bộ dữ liệu có nguồn gốc rõ ràng cho các thuật toán chạy
trên cùng SKU, tuần và cách chia thời gian. Chọn cohort UCI bằng thống kê
**chỉ trong train**; không dùng danh sách 20 SKU Online Retail II cũ.

**Đã thay đổi:** `experiments.song_online_retail` có tùy chọn `--max-skus` để
chọn cohort theo số tuần bán và lượng bán trong train. Manifest ghi danh sách
SKU, quy tắc chọn và mã băm tệp. `experiments.data` xác nhận danh sách/mã băm
trước khi truyền tên dataset xuống kết quả; nếu manifest không khớp thì báo
lỗi thay vì gắn nhầm nhãn Online Retail II. Pilot M5 cũng ghi mã băm ba tệp
nguồn và hai tệp đầu ra. `experiments.audit_prepared` kiểm tra nguồn, ranh
giới thời gian, scaler train và lag một tuần.

Chạy lại thủ công từ dữ liệu gốc (lệnh UCI đầu tạo toàn bộ SKU đủ điều kiện;
lệnh thứ hai tạo cohort 100 SKU cho benchmark trên CPU):

```powershell
python -m experiments.song_online_retail
python -m experiments.data --weekly experiments/artifacts/song_online_retail/weekly.csv --selected experiments/artifacts/song_online_retail/selected_products.csv --output-dir experiments/artifacts/song_online_retail/paper_70_15_15
python -m experiments.audit_prepared experiments/artifacts/song_online_retail/paper_70_15_15

python -m experiments.song_online_retail --max-skus 100 --output-dir experiments/artifacts/song_online_retail_100
python -m experiments.data --weekly experiments/artifacts/song_online_retail_100/weekly.csv --selected experiments/artifacts/song_online_retail_100/selected_products.csv --output-dir experiments/artifacts/song_online_retail_100/paper_70_15_15
python -m experiments.audit_prepared experiments/artifacts/song_online_retail_100/paper_70_15_15

python -m experiments.song_m5_pilot --count 20 --store-id CA_1
python -m experiments.audit_prepared experiments/artifacts/song_m5_pilot
python -m unittest experiments.test_data experiments.test_retail_ii_preprocess -v
```

**Kết quả đã kiểm tra:** UCI đầy đủ có 3.387 SKU và 129.305 cửa sổ dự báo;
cohort 100 SKU có 4.200 cửa sổ. Cả hai có train kết thúc 2011-08-08,
validation 2011-08-15 đến 2011-10-03, test từ 2011-10-10. M5 pilot có 20
chuỗi sản phẩm–cửa hàng, 273 tuần và 5.220 cửa sổ. Cả ba lệnh audit đều
thành công; 8 kiểm tra dữ liệu tự động đạt.

Lượt dự báo UCI `song_forecast/run_seed42.json` tạo **trước Phase 1** còn
nhãn Online Retail II cũ; không dùng nhãn/kết quả đó trong bảng so sánh mới.
Phase 3 sẽ chạy lại mô hình trên panel đã kiểm tra và tạo run manifest mới.

### Phase 2 — Cùng panel, cùng đầu ra dự báo (hoàn thành 2026-11-15)

**Ý nghĩa:** các baseline chạy trên dữ liệu đã kiểm tra, chấm đúng các
SKU–tuần test giống nhau. Mỗi lượt lưu tên dataset, mã băm manifest/panel,
seed, dự đoán từng tuần, điểm validation/test và các siêu tham số XGBoost
đã thử. Giá hiện tại là đầu vào của biến thể có giá; đây là dự báo nhu cầu
**có điều kiện theo giá**, không phải dự báo trước giá bán.

Chạy lại thủ công:

```powershell
python -m experiments.common_forecast --prepared-root experiments/artifacts/song_online_retail_100/paper_70_15_15 --output-dir experiments/artifacts/song_online_retail_100/common_forecast --models naive_last xgboost xgboost_price
python -m experiments.common_forecast --prepared-root experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/common_forecast --models naive_last xgboost xgboost_price
```

Mỗi thư mục đầu ra có `forecast_metrics.csv`, `forecast_predictions.csv`,
`forecast_metrics_by_sku.csv`, `forecast_run.json` và mô hình XGBoost. Lệnh
chạy sẽ audit dữ liệu trước và dừng nếu nguồn/split không khớp. XGBoost chọn
một trong bốn cấu hình bằng validation rồi fit lại bằng train+validation cho
test; lịch sử thử các cấu hình nằm trong `forecast_run.json`.

**Kết quả bước này, seed 42, RMSE sản phẩm/tuần:** UCI 100 SKU có 887 hàng
test chung: naive 113,90; XGBoost 109,12; XGBoost có giá 111,06. M5 20
chuỗi có 820 hàng test chung: naive 43,58; XGBoost 49,26; XGBoost có giá
49,27. Đây là baseline trong tập con của nhóm; chưa đối chiếu số công bố
của Song và chưa có kết quả đa seed. Việc đưa giá vào XGBoost chưa cải thiện
RMSE ở hai lượt này, nên sẽ được phân tích thay vì mặc định coi là tốt hơn.

### Phase 3A — Transformer theo Song trên cùng panel (lượt đầu hoàn thành 2026-11-15)

**Ý nghĩa:** dựng mốc Transformer 12 tuần có đầu vào giá và đạo hàm theo giá,
đồng thời so dự báo với baseline trên đúng những SKU–tuần test. Số epoch được
chọn bằng validation. Để so dự báo, mô hình test được fit lại bằng
train+validation như XGBoost; checkpoint **chỉ học train** được lưu riêng để
định giá không thấy nhãn validation.

Chạy lại thủ công cho seed 42:

```powershell
python -m experiments.song_forecast --prepared-root experiments/artifacts/song_online_retail_100/paper_70_15_15 --output-dir experiments/artifacts/song_online_retail_100/song_forecast_refit --epochs 50 --refit-train-validation
python -m experiments.compare_forecasts --common-dir experiments/artifacts/song_online_retail_100/common_forecast --song-dir experiments/artifacts/song_online_retail_100/song_forecast_refit --output-dir experiments/artifacts/song_online_retail_100/comparison

python -m experiments.song_forecast --prepared-root experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/song_forecast_refit --epochs 50 --refit-train-validation
python -m experiments.compare_forecasts --common-dir experiments/artifacts/song_m5_pilot/common_forecast --song-dir experiments/artifacts/song_m5_pilot/song_forecast_refit --output-dir experiments/artifacts/song_m5_pilot/comparison

python -m unittest experiments.test_song_forecast experiments.test_forecast_benchmark -v
```

`comparison/comparison_metrics.csv` chứa điểm cùng thang đo và
`comparison/comparison_predictions.csv` chứa từng dự đoán. Lệnh so sánh
kiểm tra mã băm panel, ấn định đúng cùng seed, cùng giá quan sát, cùng nhãn
và cùng hàng test; nếu khác sẽ dừng. `song_forecast_refit/run_seed42.json`
ghi số epoch tốt nhất, phạm vi fit và hai checkpoint.

**Kết quả tạm, seed 42, RMSE sản phẩm/tuần:** UCI 100 SKU/887 hàng test:
Transformer Song 97,04; XGBoost 109,12; XGBoost có giá 111,06; naive 113,90.
M5 20 chuỗi/820 hàng test: naive 43,58; XGBoost 49,26; XGBoost có giá
49,27; Transformer Song 50,02. Sáu kiểm tra dự báo/gradient đạt. Đây chưa
phải kết quả năm lần chạy và không thể so số học với RMSE công bố của Song
vì cohort và chi tiết chuẩn hóa khác.

**Tại thời điểm Phase 3A:** đây là mốc kỹ thuật một seed. Phase 3B bên dưới
chạy đa seed; các baseline còn thiếu và phân tích theo chuỗi vẫn cần làm.

### Phase 3B — Lặp lại dự báo trên năm seed (hoàn thành 2026-11-15)

**Ý nghĩa:** đo mức biến động do khởi tạo/huấn luyện và tránh kết luận dựa
trên một lần chạy. Mỗi seed có thư mục riêng cho baseline, Transformer và
bảng đối chiếu; tất cả dùng cùng panel đã khóa ở Phase 1. Siêu tham số
XGBoost chọn trên validation theo cùng lưới bốn cấu hình ở mỗi seed; số epoch
Transformer cũng chọn trên validation.

Chạy lại thủ công (mỗi lệnh có thể mất vài phút trên CPU):

```powershell
python -m experiments.multiseed_forecast --prepared-root experiments/artifacts/song_online_retail_100/paper_70_15_15 --output-root experiments/artifacts/song_online_retail_100/multiseed --seeds 42 123 456 789 1024 --epochs 50
python -m experiments.multiseed_forecast --prepared-root experiments/artifacts/song_m5_pilot --output-root experiments/artifacts/song_m5_pilot/multiseed --seeds 42 123 456 789 1024 --epochs 50
```

Đọc `multiseed/all_runs.csv` để xem từng seed và `multiseed/aggregate.csv`
để xem trung bình ± độ lệch chuẩn mẫu. `multiseed/run_manifest.json` lưu
mã băm panel, danh sách seed, mô hình và ngân sách epoch.

| Dataset/cohort | Mô hình | RMSE test trung bình ± độ lệch chuẩn (sản phẩm/tuần) |
| --- | --- | ---: |
| UCI 100 SKU, 887 hàng test | Transformer theo Song, có giá | 101,23 ± 5,52 |
| UCI 100 SKU, 887 hàng test | XGBoost | 109,28 ± 1,20 |
| UCI 100 SKU, 887 hàng test | XGBoost có giá | 111,88 ± 1,39 |
| UCI 100 SKU, 887 hàng test | Naive tuần trước | 113,90 ± 0,00 |
| M5 CA_1, 20 chuỗi, 820 hàng test | Naive tuần trước | 43,58 ± 0,00 |
| M5 CA_1, 20 chuỗi, 820 hàng test | Transformer theo Song, có giá | 48,46 ± 7,16 |
| M5 CA_1, 20 chuỗi, 820 hàng test | XGBoost có giá | 49,34 ± 1,19 |
| M5 CA_1, 20 chuỗi, 820 hàng test | XGBoost | 50,17 ± 1,36 |

**Diễn giải tại Phase 3B:** Transformer tốt hơn các baseline nhóm lúc đó trên cohort
UCI 100 SKU theo RMSE trung bình; trên cohort M5 này, naive có RMSE thấp
nhất. Đây là so sánh **trong dự án**, không phải tái tạo số RMSE của bài Song.
Phase 3C bên dưới bổ sung ARIMA/LSTM/Transformer không giá và phân tích
theo SKU, nên thứ hạng sau đó thay đổi.

### Phase 3C — Đối chứng dự báo mở rộng (hoàn thành 2026-11-15)

**Ý nghĩa:** chạy thêm ARIMA, LSTM và Transformer có/không có giá trên đúng
hai panel đã khóa; ghép với Song, XGBoost và naive đã chạy. ARIMA là mô hình
xác định nên chạy một lần; bốn mô hình học sâu chạy năm seed với tối đa 25
epoch, chọn epoch bằng validation. TFT/Informer chưa có triển khai đã kiểm
chứng trong môi trường CPU 16,8 GB RAM, không được gán tên cho một kiến trúc
thay thế. Đây là điểm chưa tái dựng đủ bài Song.

Chạy lại thủ công (lệnh tự dùng kết quả seed hợp lệ đã có; thiếu seed nào sẽ
chạy đúng seed đó):

```powershell
python -m experiments.phase3c --prepared-root experiments/artifacts/song_online_retail_100/paper_70_15_15 --previous-multiseed-root experiments/artifacts/song_online_retail_100/multiseed --output-root experiments/artifacts/song_online_retail_100/phase3c --torch-epochs 25
python -m experiments.phase3c --prepared-root experiments/artifacts/song_m5_pilot --previous-multiseed-root experiments/artifacts/song_m5_pilot/multiseed --output-root experiments/artifacts/song_m5_pilot/phase3c --torch-epochs 25
```

`phase3c/aggregate.csv` có điểm tổng và số lượt chạy từng mô hình;
`all_runs.csv` có từng seed; `per_sku.csv` có sai số từng chuỗi;
`price_feature_effect.csv` cho biết thêm giá giúp bao nhiêu SKU và đổi RMSE
ra sao. Các mô hình đều được kiểm tra trên 887 hàng test UCI hoặc 820 hàng
test M5 tương ứng.

| Panel | Kết quả RMSE test đáng chú ý (sản phẩm/tuần) |
| --- | --- |
| UCI 100 SKU | LSTM có giá **99,03 ± 2,34**; LSTM không giá 99,59 ± 4,14; Transformer theo Song 101,23 ± 5,52; ARIMA 104,32 (một lượt) |
| M5 CA_1, 20 chuỗi | ARIMA **42,31** (một lượt); naive 43,58; Transformer theo Song 48,46 ± 7,16; XGBoost có giá 49,34 ± 1,19 |

**Tác động của giá không đồng nhất:** trên UCI, thêm giá cải thiện RMSE tổng
của LSTM khoảng 0,56 và Transformer khoảng 2,26, nhưng làm XGBoost kém hơn
khoảng 2,60. Trên M5, biến thể LSTM có giá dao động rất mạnh và có RMSE
trung bình cao hơn LSTM không giá; cần xem theo SKU trước khi kết luận.
Các số ở bảng này vẫn là tập con của nhóm, không so trực tiếp với điểm
công bố của Song.

### Phase 4A — Kiểm tra simulator và luồng định giá (hoàn thành kiểm tra 2026-11-15)

**Ý nghĩa:** trước khi diễn giải doanh thu của PPO, kiểm tra kernel simulator
ở các mức giá thật đã quan sát trong validation/test, so với dự báo lượng
bán tuần trước và trung bình train. Lệnh audit tách các giá nằm trong/ngoài
khoảng giá train. Đây là điều kiện cần; ngay cả khi qua kiểm tra này vẫn
chưa chứng minh được cầu ở giá chưa bán.

Chạy lại thủ công:

```powershell
python -m experiments.audit_simulator --prepared-root experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/simulator_audit
python -m experiments.song_pricing_pilot --pilot-dir experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/song_pricing_phase4_smoke --forecast-checkpoint experiments/artifacts/song_m5_pilot/multiseed/seed42/song/song_transformer_seed42.pt --seeds 42 --timesteps 64 --eval-cycles 200 --cycles-per-sku 20 --policies fixed_mean heuristic ppo_frozen ppo_feedback_gradient ppo_feedback_no_gradient
```

`simulator_audit/observed_price_metrics.csv` lưu RMSE/MAE theo split và
phạm vi giá; `audit_manifest.json` ghi trạng thái kiểm tra. Lượt PPO smoke
ghi `policy_runs.csv`, `policy_episodes.csv`, `policy_aggregate.csv` và
`pricing_manifest.json` có mã băm checkpoint, dataset và trạng thái audit.

**Kết quả kiểm tra:** ở validation, 778/820 hàng có giá trong khoảng train;
kernel RMSE **98,06**, naive **64,06** trên chính 778 hàng đó. Trên toàn
bộ test, chỉ 603/820 hàng nằm trong khoảng giá train; kernel RMSE **86,25**,
naive **43,58**. Điều kiện cần của simulator **không đạt**. Lượt smoke
64 timesteps/1 seed vẫn hoàn thành 200 quyết định test cho giá cố định,
heuristic và ba biến thể PPO, chứng tỏ luồng kỹ thuật chạy được. Các số
doanh thu của lượt smoke chỉ là kết quả trong simulator yếu này, không dùng
để kết luận chính sách nào tạo doanh thu thật cao hơn.

### Phase 4B — Chọn simulator bằng validation (hoàn thành 2026-11-15)

**Ý nghĩa:** kiểm tra giả thuyết kernel gốc bỏ lỡ mức cầu của tuần gần nhất.
Biến thể `LagAdjustedKernelSimulator` trộn dự báo kernel với lượng bán tuần
trước; phản ứng theo giá vẫn lấy từ kernel và tỉ lệ điều chỉnh bị chặn trong
`[0,5; 2]`. Chỉ dùng **train** để fit kernel, chọn trọng số trong
`{0; 0,25; 0,5; 0,75; 1}` bằng RMSE **validation** ở những giá thuộc miền
train, rồi mới xem test. Mã băm panel và cấu hình được ghi để tránh dùng nhầm
simulator ở lượt huấn luyện chính sách.

Chạy lại thủ công:

```powershell
python -m experiments.select_simulator --prepared-root experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/simulator_selection
```

Đọc `simulator_selection/validation_candidates.csv` để xem từng trọng số,
`observed_price_metrics.csv` để so với kernel gốc và naive, và
`price_response.csv` để kiểm tra hình dạng đường giá–cầu.
`selected_simulator.json` lưu trọng số được chọn **0,75** cùng mã băm nguồn.

| Đo trên giá đã quan sát | Kernel gốc | Kernel hiệu chỉnh | Naive tuần trước |
| --- | ---: | ---: | ---: |
| Validation, 778 hàng giá trong miền train, RMSE | 98,06 | **60,54** | 64,06 |
| Test, 603 hàng giá trong miền train, RMSE | 85,81 | 34,94 | **31,41** |
| Test, toàn bộ 820 hàng, RMSE | 86,25 | 43,64 | **43,58** |

Biến thể đạt ngưỡng cần thiết trên validation nhưng **không vượt naive trên
test**. Sáu trong 20 chuỗi còn có cầu mô phỏng *tăng* khi tăng giá trong
khoảng ±1% quanh giá tham chiếu. Test ở đây là kiểm tra thăm dò vì sai số
test của kernel gốc đã được xem ở Phase 4A; không dùng test để chọn trọng số
trong lệnh này. Kiểm tra này là dự báo **một tuần** với lượng bán tuần trước
đã biết; khi chạy chính sách nhiều tuần, lượng bán mô phỏng được đưa ngược
vào lịch sử nên sai số có thể tích lũy. Giá đã bán cũng không xác nhận cầu ở
giá giả định, nên doanh thu của chính sách sau đây vẫn chỉ là doanh thu mô
phỏng.

### Phase 4C — So chính sách và kiểm tra độ nhạy simulator (2026-11-15)

**Ý nghĩa:** huấn luyện cùng ngân sách 512 tương tác/seed cho PPO, DDPG,
SAC trên 20 chuỗi M5 train. Episode train dài 12 tuần, bắt đầu tại một đoạn
train ngẫu nhiên để các seed gặp nhiều sản phẩm. Chấm trên cùng 200 quyết
định test (10 chuỗi × 20 tuần) cho năm seed, với cùng nhiễu cầu theo cặp.
Chạy hai môi trường: kernel gốc theo cấu trúc Song và kernel hiệu chỉnh đã
chọn ở Phase 4B. Các phép bỏ cập nhật online và đạo hàm giá cho thấy phần
nào của PPO đóng góp trong **môi trường mô phỏng**.

Chạy lại thủ công (mỗi lệnh huấn luyện nhiều chính sách trên CPU):

```powershell
python -m experiments.song_pricing_pilot --pilot-dir experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/pricing_lag_512_5seeds --forecast-checkpoint experiments/artifacts/song_m5_pilot/multiseed/seed42/song/song_transformer_seed42.pt --simulator-config experiments/artifacts/song_m5_pilot/simulator_selection/selected_simulator.json --seeds 42 123 456 789 1024 --timesteps 512 --eval-cycles 200 --cycles-per-sku 20 --policies fixed_mean heuristic kernel_grid ddpg sac ppo_frozen ppo_feedback_gradient ppo_feedback_no_gradient ppo_frozen_no_gradient
python -m experiments.song_pricing_pilot --pilot-dir experiments/artifacts/song_m5_pilot --output-dir experiments/artifacts/song_m5_pilot/pricing_song_kernel_512_5seeds --forecast-checkpoint experiments/artifacts/song_m5_pilot/multiseed/seed42/song/song_transformer_seed42.pt --seeds 42 123 456 789 1024 --timesteps 512 --eval-cycles 200 --cycles-per-sku 20 --policies fixed_mean heuristic kernel_grid ddpg sac ppo_frozen ppo_feedback_gradient ppo_feedback_no_gradient ppo_frozen_no_gradient
python -m experiments.phase4_compare --lag-dir experiments/artifacts/song_m5_pilot/pricing_lag_512_5seeds --song-dir experiments/artifacts/song_m5_pilot/pricing_song_kernel_512_5seeds --output-dir experiments/artifacts/song_m5_pilot/phase4_comparison
```

`phase4_comparison/policy_summary.csv` ghi trung bình ± độ lệch chuẩn theo
seed của doanh thu mô phỏng, mức tăng phần trăm so với giá cố định **trong
từng simulator**, số SKU train đã gặp và tỉ lệ chọn giá biên.
`ppo_ablations.csv` ghi chênh lệch theo cặp giữa PPO đầy đủ và từng đối
chứng; `simulator_sensitivity.csv` cho thấy kết luận đổi thế nào khi đổi
simulator. `comparison_manifest.json` kiểm tra cùng panel, checkpoint,
seed, ngân sách, SKU–tuần test và miền giá train. `kernel_grid` được phép
đọc trực tiếp công thức cầu của simulator để thử giá trong khoảng ±10% so
với giá hiện tại; đây là **mốc chẩn đoán có đặc quyền**, không phải chính
sách triển khai được với nhu cầu thật chưa biết.

Mức tăng doanh thu **mô phỏng** so với giá cố định theo từng seed, trung
bình ± độ lệch chuẩn của năm seed:

| Chính sách | Kernel hiệu chỉnh | Kernel gốc |
| --- | ---: | ---: |
| Giá cố định | 0% | 0% |
| Quy tắc đơn giản | 2,04 ± 3,42% | 1,63 ± 1,83% |
| DDPG | **6,72 ± 4,55%** | **8,58 ± 5,43%** |
| SAC | −1,81 ± 5,03% | −0,48 ± 1,71% |
| PPO, cập nhật online + đạo hàm | 4,10 ± 4,24% | 3,72 ± 6,84% |
| PPO, cập nhật online, bỏ đạo hàm | 2,02 ± 2,88% | 3,74 ± 4,33% |
| PPO, dự báo đóng băng + đạo hàm | 1,23 ± 4,24% | 1,92 ± 3,04% |
| PPO, dự báo đóng băng, bỏ đạo hàm | 0,52 ± 3,28% | 1,51 ± 2,30% |
| Kernel grid có đặc quyền | 8,79 ± 1,57% | 8,90 ± 1,34% |

Trên kernel hiệu chỉnh, PPO đầy đủ hơn PPO đóng băng **2,87 điểm phần trăm**
và hơn PPO bỏ đạo hàm **2,08 điểm phần trăm** khi lấy chênh lệch cùng seed;
độ lệch chuẩn của các chênh lệch lần lượt là 7,73 và 5,40 điểm. Trên kernel
gốc, tác dụng của đạo hàm gần như bằng 0 (**−0,02 điểm**) theo cùng cách
đo. Với năm seed và độ dao động này, không khẳng định thành phần nào luôn
tốt hơn. DDPG có mức tăng mô phỏng cao nhất trong các chính sách học được
ở cả hai simulator, nhưng chọn **giá thấp nhất hoặc cao nhất miền train ở
99,9% quyết định** trên kernel hiệu chỉnh (95,7% trên kernel gốc); đây là
dấu hiệu nhạy với biên giá, không phải bằng chứng khuyến nghị giá thực tế.
Mỗi chính sách học được gặp ít nhất 16/20 chuỗi train ở mỗi seed. Ngân sách
512 tương tác/seed còn nhỏ, chưa kiểm tra hội tụ hay độ ổn định khi đổi
ngân sách.

**Kết luận Phase 4:** luồng Transformer → gradient → PPO → nhu cầu mô phỏng →
cập nhật online đã chạy được cùng đối chứng DDPG/SAC và các phép bỏ thành
phần. Chất lượng mô phỏng **không qua kiểm tra test so với naive**, và đường
giá–cầu không được xác thực nhân quả. Do đó các con số trên chỉ là phép
so sánh thuật toán *bên trong simulator*; chưa được dùng để tuyên bố tăng
doanh thu doanh nghiệp hay tái tạo số doanh thu trong bài Song.

### Phase 5A — Nối kết quả thí nghiệm vào frontend (2026-11-15)

**Ý nghĩa:** đưa kết quả Phase 3C/4 đã lưu lên một trang chỉ đọc để nhóm và
giảng viên tự chọn bộ dữ liệu, mô hình, SKU, seed và kiểm tra dự báo trên
**cùng các hàng test**. Phần định giá cho chọn simulator/chính sách/seed và
phát lại từng tuần của 200 quyết định M5 (10 chuỗi × 20 tuần), kèm giá cố
định cùng tuần để đối chiếu. API kiểm tra mô hình và seed có kết quả, số
hàng test của từng SKU, và ghép từng quyết định với đúng tuần giá cố định.

Chạy lại thủ công từ thư mục gốc dự án, mỗi lệnh trong một terminal:

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
python frontend/serve.py
```

Mở `http://127.0.0.1:8765/research.html`. Các API mới nằm dưới
`/research/forecast/{overview,series}` và
`/research/pricing/{overview,replay}`; frontend chuyển tiếp qua `/api`.
Nguồn của trang là CSV và manifest ở `experiments/artifacts/`, vì vậy các
thao tác chọn/lật tuần **không huấn luyện lại**. Trang demo 20 SKU Online
Retail II tiếp tục dùng mô hình cũ và SQL Server; hai nguồn dữ liệu được
hiển thị riêng. Phần phát lại chỉ xem những giá đã chạy trong thí nghiệm,
chưa suy luận một giá mới từ PPO khi người dùng nhập kịch bản khác.

Kiểm tra kết nối: 27 bài test đạt, gồm trường hợp không để hàng validation
lọt vào đồ thị test và ghép giá cố định đúng SKU–tuần. Đã đọc thử đủ 82 tổ
hợp mô hình–seed trên hai dataset và 90 tổ hợp simulator–chính sách–seed;
mỗi lượt phát lại có đúng 200 quyết định. Trong trình duyệt ở 390 px, đã
kiểm tra đổi UCI→M5, chọn ARIMA (chỉ seed 42), chuyển simulator, chọn DDPG,
đổi seed và tiến tới tuần kế tiếp; trang không tràn ngang.

### Phase 5B — Thử một quyết định giá PPO trên M5 (2026-11-15)

**Ý nghĩa:** từ trang thí nghiệm, người dùng chọn một trong 10 SKU M5, một
trong 20 tuần test, giá hiện tại trong miền giá **train** và một trong 5
seed. Nút “Tính giá bằng PPO” lấy 12 tuần lịch sử **đã quan sát trước tuần
chọn**, dự báo lượng bán và đạo hàm theo giá bằng Transformer Phase 3C, rồi
đưa các đặc trưng đó vào checkpoint `ppo_frozen` Phase 4 để suy luận **một
quyết định**. Giá đề xuất là ánh xạ hành động PPO sang miền giá train. Trang
hiển thị dự báo lượng bán/doanh thu ở giá hiện tại và giá PPO, doanh thu kỳ
vọng từ simulator tại hai giá, cùng đường 41 mức giá để giải thích.

Phase này dùng `ppo_frozen` với Transformer cố định và kernel hiệu chỉnh
đã đánh giá trong Phase 4. Tệp chính sách PPO của biến thể cập nhật online
không lưu trạng thái Transformer sau cập nhật, nên không ghép tệp đó với
checkpoint gốc để suy luận. Khi chọn tuần test thứ hai trở đi, 12 tuần đầu
vào vẫn là lịch sử quan sát trong M5; đây là **kịch bản một bước**, không
phải phát lại chuỗi quyết định giả lập hay cập nhật online. API loại cột
lượng bán và doanh thu của chính tuần được chọn trước khi tạo đầu vào PPO.

Chạy lại thủ công từ thư mục gốc (hai terminal):

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
python frontend/serve.py
```

Mở `http://127.0.0.1:8765/research.html#scenario`, chọn SKU/tuần/giá/seed
rồi bấm “Tính giá bằng PPO”. `GET /research/pricing/scenario/options` cho
biết lựa chọn và biên giá; `POST /research/pricing/scenario` tính kết quả.
Frontend gọi hai route qua `/api`. API kiểm tra hash của manifest, dữ liệu
tuần, cấu hình simulator và checkpoint dự báo theo kết quả Phase 4. Cần có
artifact Phase 3C/4 ở `experiments/artifacts/song_m5_pilot/`; nếu thiếu,
chạy lại các lệnh tạo artifact ở các phase trên. Không cần SQL Server cho
trang này. Lần thử không huấn luyện lại, không ghi giá vào SQL Server và
không thay giá bán thực tế.

Kiểm tra lại:

```powershell
python -m unittest experiments.test_pricing_inference_api -v
python -m unittest discover -s experiments -p 'test_*.py' -q
node --check frontend/research.js
```

Bốn bài test mới kiểm tra quyết định đầu tiên trùng giá `ppo_frozen` đã lưu
trong Phase 4 khi bắt đầu từ giá trung bình train, nhãn bán của tuần test
không ảnh hưởng suy luận, và từ chối tuần/seed/giá ngoài phạm vi. Qua
frontend proxy, API trả 10 SKU × 20 tuần, 5 seed và 41 điểm đường giá.
Toàn bộ 31 bài test hiện đạt; Chrome ở 390 px đã đổi SKU, tuần, seed và giá
thành công mà không tràn ngang. Doanh thu hai đường đều là **ước tính**;
simulator hiệu chỉnh thua mốc dự báo tuần trước trên test và cầu phản thực
tế chưa được xác nhận, nên chưa thể coi giá PPO là khuyến nghị kinh doanh.

### Phase 5C — Tái lập PPO có phản hồi online trên giao diện (2026-11-15)

**Ý nghĩa:** Phase 4 đã chạy luồng Transformer → đạo hàm giá → PPO → lượng
bán mô phỏng → cập nhật Transformer sau mỗi 10 quyết định, nhưng mới lưu
checkpoint PPO. Để suy luận đúng nhánh đó, cần cả trạng thái Transformer
**sau huấn luyện PPO** và trạng thái ngẫu nhiên dùng khi cập nhật trong
chuỗi test. `experiments.phase5c_online_state` chạy lại riêng chính sách
`ppo_feedback_gradient` trên cùng panel, simulator, siêu tham số và 5 seed;
lưu hai trạng thái cùng checkpoint PPO thành một cặp. Manifest ghi mã băm
nguồn, mã băm checkpoint và độ lệch khi phát lại.

Chạy lại thủ công từ thư mục gốc dự án:

```powershell
python -m experiments.phase5c_online_state
python -m unittest experiments.test_pricing_inference_api -v
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
python frontend/serve.py
```

Hai lệnh chạy server cần mở ở hai terminal riêng. Mở
`http://127.0.0.1:8765/research.html#scenario`, chọn **PPO có cập nhật dự
báo**, SKU, tuần và seed. Để trống giá đầu vào sẽ phát lại đúng chuỗi:
tuần đầu bắt đầu ở giá trung bình train, các tuần sau dùng giá từ quyết
định mô phỏng trước. Nếu nhập một giá khác trong miền train, backend giữ
nguyên các tuần mô phỏng trước và tính lại **một quyết định** cho tuần chọn.
Màn hình cho biết đã phát lại bao nhiêu tuần, Transformer được cập nhật
bao nhiêu lần và phần nào của 12 tuần lịch sử là lượng bán mô phỏng. Có thể
chuyển sang **PPO dự báo cố định** của Phase 5B để đối chiếu; chế độ này
dùng 12 tuần lịch sử quan sát và giá tuần trước làm mặc định.

**Kiểm tra:** 5 seed × 200 quyết định test của chính sách online đã được
tái lập đúng từ đầu đến cuối; sai khác lớn nhất ở giá, lượng bán mô phỏng,
doanh thu mô phỏng và số lần cập nhật đều bằng **0** so với nhật ký Phase 4.
API kiểm tra checkpoint bằng mã băm và loại nhãn lượng bán/doanh thu của
tuần đang thử khỏi đầu vào. Các bài test chọn tuần trước/sau lần cập nhật,
kiểm tra không lọt nhãn test và kiểm tra trạng thái ngẫu nhiên của backend.
Toàn bộ **33 bài test** đạt; qua frontend proxy, chế độ online trả đường
41 mức giá. Trình duyệt ở 390 px đã thử tuần đầu, tuần sau lần cập nhật và
chuyển giữa hai chính sách mà không tràn ngang.

Đây vẫn là **pilot M5 20 chuỗi tại CA_1**, đánh giá giá trên 10 chuỗi × 20
tuần. Phản hồi dùng để cập nhật Transformer là **lượng bán từ simulator**,
không phải đơn hàng thực. Giá PPO là hành động của chính sách đã học, chưa
được chứng minh là mức tối đa doanh thu ở từng tuần. Simulator hiện chưa
vượt mốc naive trên test; Phase 6 dưới đây kiểm tra quy tắc chọn giá cuối.

### Phase 6 — Chọn giá theo mục tiêu doanh thu và so sánh chính sách (2026-11-15)

**Ý nghĩa:** chuyển từ việc chỉ xem giá PPO sang một quyết định có mục tiêu
rõ ràng: tối đa hóa **doanh thu kỳ vọng = giá × lượng bán dự báo**. Chưa có
giá vốn nên đây không phải bài toán tối đa lợi nhuận. Bộ tối ưu thử 81 mức
giá đều nhau và cả giá giữ nguyên, dùng trung bình dự báo của 5 Transformer
chỉ huấn luyện trên train. Giá hợp lệ phải nằm đồng thời trong miền giá
train của SKU và khoảng ±10% so với **giá quan sát tuần trước**. Khi hai
giá có cùng dự báo doanh thu, ưu tiên giá gần giá đang dùng.

Phép thử dùng 10 SKU đánh giá giá của pilot M5 tại cửa hàng CA_1. Mỗi SKU
có 20 tuần validation cuối và 20 tuần test đầu làm ứng viên. Có 200 tuần
validation hợp lệ; trong 200 tuần test, 26 tuần có giá tham chiếu ngoài
miền train nên được loại theo quy tắc đã công bố, còn 174 tuần thuộc 9 SKU.
Mỗi quyết định dùng đúng 12 tuần lịch sử đã quan sát cùng sự kiện/SNAP đã
biết; không đưa lượng bán, giá hay các cột suy từ giá của chính tuần đích
vào chính sách. Đây là phép thử **một quyết định mỗi tuần với lịch sử quan
sát**, khác chuỗi Phase 4 vốn đưa phản hồi mô phỏng tuần trước vào bước sau.

Trên cùng SKU–tuần và giới hạn giá, Phase 6 so sánh giữ giá, giá trung bình
train, quy tắc ±5%, tối ưu lưới dự báo, PPO dự báo cố định, DDPG và SAC.
`simulator_oracle` được tính riêng bằng cách nhìn trực tiếp hàm cầu của
simulator; nó chỉ là mốc chẩn đoán, **không được phép chọn làm chính sách**.
Điểm so sánh là doanh thu **kỳ vọng trong simulator đã chọn ở Phase 4**,
không phải doanh thu thực tại giá chưa từng bán. PPO có cập nhật online của
Phase 4/5C không nằm trong bảng một bước này vì trạng thái phản hồi tuần
trước khác; không đối chiếu trực tiếp phần trăm của hai giao thức.

| Chính sách | Validation so với giữ giá | Test thăm dò so với giữ giá |
|---|---:|---:|
| Giữ giá tuần trước | 0,00% | 0,00% |
| Lưới dự báo 5 Transformer | −2,52% | −0,94% |
| Quy tắc ±5% | −2,79% | −2,99% |
| PPO dự báo cố định | −4,02 ± 1,66% | −3,18 ± 1,13% |
| DDPG | −4,83 ± 0,67% | −3,84 ± 0,37% |
| SAC | −5,35 ± 0,85% | −4,02 ± 0,85% |
| Giá trung bình train | −8,24% | −7,17% |
| Oracle biết simulator (chỉ chẩn đoán) | +2,05% | +2,08% |

Các dấu ± là độ lệch chuẩn qua 5 seed của chính sách học máy; các quy tắc
cố định được tính trên cùng SKU–tuần nên không có biến thiên theo seed.
Giá tối ưu theo dự báo thay đổi khi siết giới hạn từ ±10% xuống ±5% ở
**95/200 tuần validation** và **69/174 tuần test** (chọn trong cùng lưới
81 giá). Trong 5 Transformer, mức giá tốt nhất theo từng seed khác nhau ở
**31,0%** tuần validation và **26,4%** tuần test. Các cột này nằm trong
`forecast_grid_prices.csv` để kiểm tra độ nhạy; chúng cho thấy quyết định
phụ thuộc đáng kể vào giới hạn vận hành và mô hình dự báo.
Lưới Transformer tự dự báo tăng doanh thu trung bình **1,99%** trên
validation và **1,43%** trên test, nhưng simulator lại chấm chính sách này
**−2,52%** và **−0,94%**. Sự bất đồng giữa hai mô hình là lý do cần kiểm
định cầu theo giá trước khi đưa giá đề xuất vào vận hành.

Với **một chính sách chung cho mọi SKU**, chọn theo doanh thu kỳ vọng trung
bình trên validation cho kết quả **giữ giá hiện tại**. Bảng kiểm tra thêm
một quy tắc riêng từng SKU: chọn phương pháp có doanh thu mô phỏng validation
cao nhất; nếu hòa thì giữ giá; với chính sách học máy chọn seed ở giữa theo
điểm validation. Quy tắc này chọn giữ giá cho 5 SKU, PPO cố định cho 3 SKU,
quy tắc ±5% cho 1 SKU và lưới Transformer cho 1 SKU. Nó đổi giá ở 88/200
quyết định validation và 89/174 quyết định test; doanh thu mô phỏng tương
ứng là **+1,58% và +1,69%** so với giữ giá. Đây là phân tích thăm dò:
simulator đã được chọn trên cùng validation và kỳ test này đã được xem ở
Phase 4. Không xem +1,69% là ước lượng tăng doanh thu kinh doanh độc lập.
Ví dụ ở `FOODS_3_587@CA_1`, tuần test 2015-07-13: giá tham chiếu **2,48**,
giá lưới được chọn **2,337143**; doanh thu kỳ vọng trong simulator là
**597,05** so với **561,09** khi giữ giá. Đây là một dòng minh họa cách
đọc kết quả, không phải doanh thu đã quan sát ở giá mới.

**Chạy lại thủ công** từ thư mục gốc, sau khi đã có artifact M5 Phase 3C/4:

```powershell
python -m experiments.phase6_optimize
python -m unittest experiments.test_phase6_optimize -v
```

Kết quả ở `experiments/artifacts/song_m5_pilot/phase6_price_selection/`:
`forecast_grid_prices.csv` chứa giá tối ưu theo dự báo và lợi ích dự báo;
`policy_summary.csv` chứa bảng so sánh trên; `one_step_decisions.csv` chứa
giá/doanh thu mô phỏng của từng phương pháp, SKU, tuần và seed;
`selected_policy_by_sku.csv` là quy tắc chọn riêng theo validation;
`selected_price_decisions.csv` là giá được quy tắc đó chọn ở từng tuần.
`manifest.json` lưu hash đầu vào, phạm vi, giới hạn giá, số dòng bị loại và
các giới hạn diễn giải. Việc chạy lại không huấn luyện mô hình, không thay
API/frontend và không ghi giá vào SQL Server.

**Giới hạn bắt buộc trước triển khai:** kiểm tra cầu tại giá đã quan sát
cho simulator trên test có RMSE **34,94**, kém dự báo dùng lượng bán tuần
trước **31,41**. M5 không quan sát lượng cầu ở giá thay thế, nên cả giá lưới,
PPO và quy tắc riêng SKU đều chưa được xác thực phản thực tế. Vì vậy
`business_deployment_supported=false` trong manifest. Phase sau cần cải
thiện và kiểm tra mô hình cầu độc lập trước khi coi bất kỳ giá nào là tối ưu
cho doanh nghiệp. Giá trong thí nghiệm còn là số liên tục; bước làm tròn
giá bán, thuế và các giới hạn vận hành khác chưa được đưa vào bài toán.

### Phase 7 — Kiểm định mô hình cầu và độ bền kết luận giá (2026-11-15)

**Ý nghĩa:** Phase 6 cho thấy một quy tắc chọn giá có thể tăng doanh thu
**trong simulator**, nhưng simulator lại dự báo kém mốc đơn giản trên test.
Phase 7 kiểm tra xem kết luận đó còn đứng vững khi chọn mô hình cầu bằng
chính train và thay giả định phản ứng cầu theo giá. Kết quả được chuẩn bị
cho trang “Phương pháp và đánh giá” ở phase sau; Phase 7 chưa nối API hay
frontend.

`experiments.phase7_demand_validation` tạo ba fold cuốn chiếu **hoàn toàn
trong khoảng train**. Mỗi fold fit ở 105/133/162 tuần đầu rồi chấm 20 tuần
kế tiếp theo dự báo một bước với lịch sử đã quan sát. Cùng các SKU–tuần và
giá đã bán, script so naive tuần trước, kernel gốc, kernel có trọng số nhu
cầu gần đây và phiên bản **cầu không đổi khi thay giá ứng viên**. Tham số
được chọn bằng RMSE gộp trên các giá nằm trong miền giá của phần fit. Đây
là phép chọn có điều kiện trên **20 SKU đã được xếp hạng bằng toàn bộ train**;
chưa lồng lại bước chọn SKU trong từng fold. Validation/test đã được xem ở
các phase trước và không dùng để chọn tham số ở Phase 7.

| Phạm vi kiểm tra tại giá đã bán | Số tuần | RMSE mô hình được CV chọn | RMSE naive tuần trước |
|---|---:|---:|---:|
| Ba fold trong train, giá trong miền fit | 1.070 | 61,82 | 66,63 |
| Validation, giá trong miền train | 778 | 60,54 | 64,06 |
| Test, giá trong miền train | 603 | 34,94 | **31,41** |
| Đúng cohort Phase 6 test, cả giá tuần đích trong miền train | 173 | 43,46 | **41,69** |

Mô hình được CV chọn là `flat_response_0.75`: nó dùng lịch sử gần đây để
dự báo mức cầu nhưng **không đổi lượng cầu khi thử một giá khác**. Trên test,
nó vẫn thua naive. Ở validation, dù RMSE tốt hơn naive, MAE lại **36,33
so với 33,86**; một chỉ số duy nhất không đủ để xác nhận mô hình. Khoảng
cách giữa cầu phẳng và kernel có phản ứng giá cũng thay đổi theo
fold: fold đầu kernel nhỉnh hơn rất ít, fold thứ hai cầu phẳng tốt hơn,
fold thứ ba gần như hòa. Vì vậy không xem lựa chọn từ CV là bằng chứng về
độ co giãn giá. Ở phép kiểm tra dốc cục bộ, kernel 0,75 còn cho cầu **tăng
khi tăng giá ở 6/20 SKU**; đây là quan hệ quan sát trong simulator, không
phải tác động nhân quả.

Cohort Phase 6 ban đầu có 174 tuần test vì **giá tham chiếu** trong miền train;
chỉ 173 tuần có thêm **giá đã bán ở tuần đích** trong miền train. Không gộp
hai điều kiện lọc này khi diễn giải sai số.

`experiments.phase7_policy_robustness` giữ **nguyên mọi hành động giá và
quy tắc chọn SKU của Phase 6** rồi chấm lại cùng SKU–tuần dưới sáu giả định
cầu không dùng nhãn của tuần đích; các bề mặt mô hình chỉ fit bằng train.
Sai khác doanh thu khi phát lại simulator gốc bằng
**0**; dự báo của nhánh cầu phẳng được CV chọn cũng khớp artifact kiểm định
cầu với sai khác **0**. Không huấn luyện lại PPO/DDPG/SAC và không dùng
kết quả test để chọn lại chính sách.

| Giả định cầu khi chấm cùng 174 giá test đã chọn | Doanh thu kỳ vọng so với giữ giá |
|---|---:|
| Kernel gốc kiểu Song | +1,82% |
| Kernel hiệu chỉnh trọng số nhu cầu gần đây 0,50 | +1,74% |
| Kernel hiệu chỉnh 0,75 đã dùng ở Phase 6 | +1,69% |
| Kernel hiệu chỉnh 1,00 | +1,64% |
| **Mô hình cầu phẳng 0,75 được CV chọn** | **−1,21%** |
| Giữ lượng bán tuần trước cố định ở mọi giá | −1,37% |

Giữ giá vẫn là **chính sách chung đứng đầu** trong cả sáu phép chấm. Dấu
của lợi ích từ quy tắc riêng SKU đổi khi bỏ giả định cầu thay đổi theo giá:
kết quả +1,69% của Phase 6 **không bền** trước một mô hình cầu khác có cơ
sở kiểm định trong train. Cả hai phía vẫn là doanh thu mô phỏng; dữ liệu
M5 không quan sát lượng cầu tại giá chưa bán và không xác định tác động
nhân quả của việc đổi giá.

[Bảng đối chiếu bài Song và các nguồn cùng M5](experiments/phase7_paper_comparison.md)
ghi nguồn, dữ liệu, cách chia, chỉ số và mức độ so sánh. Các điểm số đã công
bố **chỉ so sánh phương pháp**, vì pilot 20 chuỗi CA_1 theo tuần không khớp
đúng cohort, horizon, cách chuẩn hóa/chấm điểm hoặc simulator của các bài
đó. Không tính phần trăm “hơn bài báo” từ các điểm số khác đơn vị.

**Chạy lại thủ công** từ thư mục gốc dự án, sau khi có artifact M5 Phase 6:

```powershell
python -m experiments.phase7_demand_validation
python -m experiments.phase7_policy_robustness
python -m unittest discover -s experiments -p 'test_*.py' -q
```

Kết quả và manifest nằm trong
`experiments/artifacts/song_m5_pilot/phase7_demand_validation/` và
`experiments/artifacts/song_m5_pilot/phase7_policy_robustness/`. Tệp
`observed_price_metrics.csv` và `train_cv_fold_metrics.csv` giải thích chất
lượng dự báo cầu; `phase6_cohort_observed_price_metrics.csv` kiểm tra đúng
cohort định giá; `frozen_selected_rule_by_assumption.csv`,
`frozen_selected_rule_by_sku.csv` và `ranking_stability.csv` thể hiện độ
bền của quyết định giá. Phase 7 kết luận **chưa đủ bằng chứng để triển khai
giá tự động cho doanh nghiệp**; giao diện tiếp theo chỉ nên trình bày giá
và doanh thu dưới nhãn dự báo/mô phỏng, kèm giới hạn kiểm định.

### Phase 8 — Giao diện ra quyết định và bằng chứng M5 (2026-11-15)

Phase 8 dùng lại **checkpoint và kết quả đã đóng băng của Phase 6–7**; không
huấn luyện lại và không ghi giá vào SQL Server. Hai trang mới:

- `/price-m5.html`: chọn SKU M5, tuần test, giá hiện tại trong miền train và
  cách chọn giá (quy tắc đã chọn trên validation, giữ giá, trung bình train,
  heuristic, lưới 5 Transformer, PPO, DDPG hoặc SAC). API suy luận lại từ
  **12 tuần lịch sử trước quyết định**, rồi trả giá đề xuất, doanh thu dự báo
  bởi Transformer, doanh thu mô phỏng và phép thử nếu cầu không đổi theo giá.
  Giá nhập chỉ thay giá tham chiếu của kịch bản; 12 giá và lượng bán lịch sử
  vẫn là số đã quan sát. Chỉ 9/10 SKU có 174 tuần test thỏa miền giá train;
  26 tuần còn lại không được đưa ra làm lựa chọn.
- `/method.html`: giải thích giao thức, RMSE/MAE dự báo của Transformer so với
  mốc dùng tuần trước, RMSE của mô hình cầu ở giá đã bán, kết quả 7 phương
  pháp giá trên validation/test và độ nhạy khi đổi giả định cầu. Sáu nguồn
  cùng dữ liệu M5 được dẫn đến bài gốc và đánh dấu **chỉ đối chiếu phương
  pháp** vì khác cohort, thước đo hoặc simulator.

API chỉ đọc artifact nằm trong `experiments/phase8_api.py`:
`GET /research/price/options`, `POST /research/price/recommend`,
`GET /research/method/overview`. API đối chiếu SHA của dữ liệu, checkpoint
và manifest với Phase 6–7; với giá lịch sử và quy tắc validation, giá tính
lại phải khớp quyết định đã lưu ở Phase 6. `frontend/serve.py` chuyển tiếp
ba route này qua `/api/`.

Chạy lại thủ công từ thư mục gốc dự án sau khi đã tạo đủ artifact theo các
phase trước (đặc biệt Phase 6 và 7):

```powershell
python -m unittest experiments.test_phase6_optimize experiments.test_phase8_api -v
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

Ở terminal thứ hai:

```powershell
python frontend/serve.py
```

Mở <http://127.0.0.1:8765/price-m5.html> và
<http://127.0.0.1:8765/method.html>. Nếu backend cũ vẫn chiếm cổng 8000,
hãy dừng tiến trình cũ rồi chạy lại lệnh `uvicorn` để nạp route Phase 8.
M5 ở hai trang này đọc artifact thí nghiệm, **không cần SQL Server**. Bản
demo Online Retail II ở trang tổng quan và danh sách sản phẩm vẫn dùng luồng
riêng như mô tả trước đó.

Ý nghĩa kết quả: chính sách **chung** được chọn bằng validation là giữ giá;
quy tắc **riêng từng SKU** đạt +1,69% doanh thu test trong simulator đã chọn
nhưng còn −1,21% khi giữ nguyên các giá đó và chấm bằng mô hình cầu phẳng
được chọn trong train-CV. RMSE test của mô hình cầu này là 34,94, cao hơn
mốc dùng tuần trước 31,41 trên 603 tuần có giá nằm trong miền train. Do M5
không ghi nhận cầu ở giá chưa bán và chưa có giá vốn, giao diện chỉ đưa ra
**giá tối đa doanh thu dự báo cho một kịch bản**, chưa chứng minh được giá
tối ưu hay lợi nhuận thực tế. Giai đoạn tiếp theo chờ nhóm duyệt Phase 8.
