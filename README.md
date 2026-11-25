# Dynamic Pricing — cách chạy dự án

## Ứng dụng cửa hàng M5

Giao diện **Quầy Giá** hiển thị 20 sản phẩm mẫu, mỗi sản phẩm có 10 dòng giá theo cửa hàng. Mở chi tiết để xem biểu đồ lịch sử hoặc chỉnh giá. Giá mới có hiệu lực khi lưu thành công vào SQL Server `DynamicPricingM5`. Trang **Dự đoán giá** dùng API mô hình M5 để tính một mức giá đề xuất bằng lưới giá và ARIMAX(1,0,0).

Chạy các lệnh ở thư mục gốc dự án. Máy cần Python, SQL Server, ODBC Driver 18 và `sqlcmd`. Có thể mở `database/m5_schema.sql` trong SSMS và chạy toàn bộ tệp nếu dùng SSMS thay cho `sqlcmd`.

1. Cài thư viện và dựng panel tuần. Lệnh tải tự dùng lại các tệp nguồn có hash hợp lệ.

   ```powershell
   python -m pip install -r final_system/requirements.txt "pyodbc>=5.3,<6"
   python -m final_system.data_pipeline download
   python -m final_system.data_pipeline prepare
   ```

2. Nạp bảng SQL Server. Chỉnh tên máy chủ nếu dùng instance khác `.\SQLEXPRESS`.

   ```powershell
   $env:M5_SQL_SERVER = '.\SQLEXPRESS'
   sqlcmd -S $env:M5_SQL_SERVER -E -b -i database/m5_schema.sql
   python -m database.import_m5 --all
   sqlcmd -S $env:M5_SQL_SERVER -E -b -d DynamicPricingM5 -f 65001 -i database/m5_storefront.sql
   ```

   Danh mục cần dữ liệu của đủ 10 cửa hàng. Nếu đã nhập dữ liệu, chỉ cần chạy lệnh SQL cuối để tạo danh mục và bảng lưu lịch sử chỉnh giá. Backend dùng `M5_SQL_SERVER` hoặc `M5_DB_CONNECTION_STRING` từ cửa sổ nó được khởi động.

3. Chuẩn bị mô hình. Nếu nhận artifact từ nhóm, chép `final_system/runtime/models/` rồi chuyển sang bước 4. Nếu dựng lại từ đầu, chạy lệnh huấn luyện ARIMAX sau.

   ```powershell
   python -m final_system.training train --workers 6
   ```

   Khi cần thử PPO trong API, chép `final_system/runtime/ppo/m5-policy.json` tương ứng với bộ ARIMAX hoặc tạo lại bằng các lệnh sau.

   ```powershell
   python -m pip install -r requirements-experiments.txt
   python -m experiments.train_m5_serving_ppo
   ```

4. Mở hai cửa sổ PowerShell từ thư mục gốc, chạy lần lượt mỗi cửa sổ một lệnh.

   ```powershell
   python -m uvicorn final_system.api:app --host 127.0.0.1 --port 8766
   python -m uvicorn backend.m5_api:app --host 127.0.0.1 --port 18767
   ```

   Mở [ứng dụng M5](http://127.0.0.1:18767/) và thử **Sản phẩm → Mứt quả mọng → Chỉnh giá** tại một cửa hàng. Dùng **Xem giá đề xuất** để chuyển sang trang dự đoán. Giá đề xuất dùng để tham khảo trước khi người dùng tự lưu giá mới. Nhóm có thể thử PPO và phản hồi ở [Swagger của backend](http://127.0.0.1:18767/docs).

Trên 10 cặp SKU–cửa hàng, 20 tuần test và 5 seed nhiễu, chỉ số doanh thu mô phỏng so với giá trung vị train (= 100) là **120,48 ± 2,41** cho lưới có phản hồi và **94,97 ± 0,82** cho PPO có phản hồi. Lưới là lựa chọn mặc định. Chi tiết nằm tại `final_system/runtime/ppo/m5-policy.evaluation.json` sau khi chạy huấn luyện PPO.

## Thí nghiệm ngoài ứng dụng · ARIMAX trên UCI

```powershell
python -m experiments.uci_arimax_comparison --workers 4
```

Lệnh dùng panel đã chuẩn bị ở bước 1 và ghi `scores.csv`, `predictions.csv`, `fit_status.csv`, `run.json` trong `experiments/artifacts/uci_arimax_v2/`. ARIMAX dùng cùng cấu hình M5 và chỉ học từ 37 tuần train của UCI. Trong 3.387 SKU được kiểm tra, 1.585 SKU có mô hình ARIMAX. Đánh giá dùng giá lịch sử đã quan sát và giữ nguyên 18.542 khóa test V2.

| Phạm vi test | Phương pháp | RMSE | MAE | R² | MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| 9.676 hàng có ARIMAX | Trung bình 4 tuần | 122,50 | 47,58 | 0,606 | 160,92% |
| 9.676 hàng có ARIMAX | ARIMAX | 158,59 | 49,57 | 0,340 | 124,77% |
| 18.542 hàng | Trung bình 4 tuần | 115,82 | 41,62 | 0,537 | 174,56% |
| 18.542 hàng | ARIMAX + dự phòng trung bình 4 tuần | 136,78 | 42,66 | 0,354 | 155,70% |

ARIMAX giảm MAPE trong lượt này. Trung bình 4 tuần có RMSE, MAE và R² tốt hơn trên cùng các khóa, nên vẫn là mốc chính cho UCI.

[Final system](final_system/README.md) hướng dẫn dựng dữ liệu, tạo bộ mô hình và gọi API dự báo lượng bán hoặc đề xuất giá.

[Thiết kế cuối cho thí nghiệm và hệ thống](docs/dynamic-pricing-final-design.md) ghi phạm vi UCI/M5, quy tắc giữ test, các phase và đầu ra cần bàn giao.

[Giao thức Phase 0](experiments/PROTOCOL_V2.md), [biên bản kiểm tra nguồn](docs/phase0-audit.md) và [mẫu báo cáo](docs/report-template-v2.md) là điểm bắt đầu cho các lượt chạy toàn phạm vi tiếp theo.

Dự án có ba nguồn dữ liệu. **M5** cung cấp dữ liệu cho ứng dụng cửa hàng, dự báo lượng bán và thử chính sách giá. **UCI Online Retail (2010–2011)** dùng để so sánh dự báo theo bài Song. **Online Retail II (2009–2011)** dùng trong bản demo gốc.

## Phase 0 — kiểm tra nguồn và khóa giao thức

Đã ghi nguồn, SHA256, ranh giới train/validation/test, phạm vi đủ điều kiện, chỉ số, seed và quy tắc giữ nguyên nhãn test trong [giao thức V2](experiments/PROTOCOL_V2.md). [Biên bản Phase 0](docs/phase0-audit.md) nêu bằng chứng và lệnh PowerShell để kiểm tra lại bốn tệp nguồn sau khi tải về. [Mẫu báo cáo](docs/report-template-v2.md) dành cho các lượt benchmark sau; hiện chưa chứa kết quả mới. Kiểm tra cú pháp cấu hình bằng `python -m json.tool experiments/protocol_v2.json` từ thư mục gốc. Nhóm đã cho phép tiếp tục Phase 1; Phase 0 không train hoặc thay đổi code mô hình.

## Phase 1 — dựng và xác minh dữ liệu tuần V2

Mục đích: tạo panel UCI giữ nguyên lượng bán hợp lệ ở validation/test, dựng panel M5 đủ 10 cửa hàng từ bản `sales_train_evaluation.csv`, rồi khóa chính xác các hàng test mà mọi mô hình phải chấm. Các lệnh chạy từ thư mục gốc, sau khi cài `requirements-experiments.txt` và tải dữ liệu bằng `python -m experiments.download_paper_data --dataset all` (nguồn tải ở bảng bên dưới):

```powershell
python -u -m experiments.song_online_retail --outlier-application train_only --output-dir experiments/artifacts/phase1_v2/uci
python -u -m experiments.run --stage data --weekly-path experiments/artifacts/phase1_v2/uci/weekly.csv --selected-path experiments/artifacts/phase1_v2/uci/selected_products.csv --output-root experiments/artifacts/phase1_v2/uci/panel
python -u -m experiments.full_m5_prepare --data-dir data/song_replication/m5 --output-root experiments/artifacts/phase1_v2/m5
python -u -m experiments.phase1_finalize
python -m unittest experiments.test_retail_ii_preprocess experiments.test_data experiments.test_phase1_finalize -v
```

Kết quả nằm trong `experiments/artifacts/phase1_v2/`: panel và manifest UCI, 10 panel/manifest M5, `test_keys_uci.csv`, `test_keys_m5.csv`, `m5_pricing_cohort.csv` và `phase1_manifest.json`. [Biên bản Phase 1](docs/phase1-audit.md) giải thích ý nghĩa từng tệp, số hàng/hash quan sát và các giới hạn. Thư mục `data/` và `experiments/artifacts/` không được Git lưu, nên thành viên khác cần tải nguồn và chạy lại bốn lệnh tạo dữ liệu. Phase 1 chưa train mô hình trên dữ liệu V2 hoặc chạy PPO đánh giá.

Lượt chạy đã xác minh trên máy nhóm: **18.542** khóa test UCI, **1.220.898** khóa test M5 trên 29.069 chuỗi đủ điều kiện, và 10 chuỗi định giá × 20 tuần = **200** quyết định dự kiến. Hash chính xác của các tệp khóa nằm trong giao thức V2 và biên bản Phase 1.

## Phase 2 — huấn luyện Song Transformer V2

Chạy từ thư mục gốc sau Phase 1. UCI dùng toàn bộ cửa sổ hợp lệ từ cohort đủ điều kiện trong panel. M5 chạy thử trên tối đa 12 chuỗi cho mỗi cửa hàng, chọn theo tổng lượng bán train và giữ chuỗi định giá đã khóa của cửa hàng. Một mô hình M5 học các lô dữ liệu từ cả 10 cửa hàng. Hai lượt dưới đây dùng seed 42 và giới hạn 3 epoch để kiểm tra luồng hoàn chỉnh.

```powershell
python -u -m experiments.phase2_song train --dataset uci --seed 42 --epochs 3
python -u -m experiments.phase2_song train --dataset m5 --seed 42 --epochs 3 --m5-skus-per-store 12
python -m experiments.phase2_song predict --run-dir experiments/artifacts/phase2_v2/uci_seed42
python -m experiments.phase2_song predict --run-dir experiments/artifacts/phase2_v2/m5_seed42
```

- **Chuẩn bị:** script đối chiếu SHA256 với manifest Phase 1, giữ lịch sử 12 tuần và tham số chuẩn hóa đã tính từ train. Với M5, script tạo đặc trưng từng cửa hàng và đọc từng lô từ cache tạm trên đĩa khi train và validation. M5 dùng lịch sự kiện và SNAP. Cờ khuyến mãi thiếu được đặt bằng 0, theo giao thức V2.
- **Huấn luyện:** Transformer 4 lớp, 8 attention head, chiều 128, Adam với learning rate `0.001`. Loss là MSE trên lượng bán chuẩn hóa. Chọn epoch có RMSE validation thấp nhất ở đơn vị sản phẩm/tuần.
- **Nạp lại:** hai lệnh `predict` giữ nguyên lịch sử của một dòng validation, đổi giá ứng viên rồi trả về `predicted_qty` và `dq_dp` (mức thay đổi cục bộ của lượng bán dự kiến khi giá tăng một đơn vị tiền). Có thể truyền mức giá khác bằng `--prices`. Ví dụ mặc định dùng ba giá trong khoảng đã quan sát ở các cửa sổ train của sản phẩm.

Mỗi thư mục chạy có `song_transformer.pt` (trọng số, scaler và metadata), `run.json` (phạm vi, split, hash nguồn, epoch và thời gian), `training.csv` (chỉ số theo epoch), `example.json` (lịch sử và giá đầu vào) và `prediction_check.json` (kết quả khi nạp lại). Để chạy lại và giữ kết quả cũ, truyền `--output-dir` với một đường dẫn mới. Thư mục artifact được Git bỏ qua, vì vậy thành viên trong nhóm dùng các lệnh trên để tạo lại trên máy của mình. Benchmark toàn phạm vi trên các khóa test đã khóa thuộc Phase 5.

| Lượt chạy CPU | Chuỗi có scaler | Chuỗi có cửa sổ train / validation | Số cửa sổ train / validation | Epoch được chọn | RMSE validation (sản phẩm/tuần) | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI V2 | 3.387 | 2.998 / 2.699 | 39.366 / 14.574 | 1 | 112,86 | 1 phút 54 giây |
| M5 V2 (10 cửa hàng) | 120 | 120 / 120 | 21.601 / 5.040 | 3 | 71,83 | 1 phút 15 giây |

Đã nạp lại hai checkpoint, dự báo ở ba mức giá và đối chiếu đạo hàm tự động với sai phân quanh giá giữa. Ví dụ, SKU `FOODS_3_090@CA_1` tuần `2014-10-13` cho khoảng 758, 680 và 620 sản phẩm dự báo ở ba giá 1,00, 1,24 và 1,48. Đây là kiểm tra đầu ra của mô hình tại một cửa sổ validation. RMSE trong bảng là số của hai lượt thử riêng, mỗi lượt trên validation của bộ dữ liệu tương ứng. Hash hai tệp khóa test Phase 1 đã được xác minh lại.

```powershell
python -m unittest discover -s experiments -p 'test_*.py'
```

## Phase 3 — thử định giá Song PPO trên M5 V2

Chạy từ thư mục gốc sau khi đã tạo panel Phase 1 và checkpoint M5 Phase 2. Luồng này dùng 10 chuỗi M5 đã khóa (mỗi cửa hàng một chuỗi), 20 tuần validation đầu tiên và seed 42. Mỗi thuật toán có huấn luyện dùng 512 tương tác mô phỏng trên train.

```powershell
python -u -m experiments.phase3_song_pricing run --seed 42 --timesteps 512 --validation-weeks 20
python -m experiments.phase3_song_pricing replay --run-dir experiments/artifacts/phase3_v2/m5_seed42
```

Simulator Gaussian kernel học từ các tuần train có giá. Script kiểm tra riêng sai số lượng bán ở giá lịch sử trên validation. Trong lượt định giá mô phỏng, mỗi chuỗi bắt đầu từ 12 tuần lịch sử train, sau đó dùng giá và lượng bán do mô phỏng tạo. Các phương pháp dùng cùng lịch, miền giá train và luồng nhiễu cho mỗi chuỗi. Các biến thể PPO dùng phản hồi cập nhật Transformer sau mỗi 10 tương tác và bắt đầu lại từ cùng trạng thái đã huấn luyện khi chuyển sang chuỗi validation khác. Transformer dùng luồng số ngẫu nhiên riêng khi học từ phản hồi. Trạng thái luồng này được lưu cùng trọng số để chạy lại validation.

Tám phương pháp gồm giá cố định theo giá trung bình train, heuristic, DDPG, SAC và bốn tổ hợp PPO bật/tắt gradient và phản hồi. Với các chính sách ngoài PPO, lượng bán **dự báo** trong log đến từ Transformer Phase 2 dùng riêng để đối chiếu. Doanh thu và regret của mọi chính sách đều tính bằng cùng simulator.

**Kiểm tra simulator:** 392 trong 420 bản ghi SKU–tuần validation có giá lịch sử trong miền train. Trên 392 bản ghi đó, Gaussian kernel có RMSE 218,08 sản phẩm/tuần, so với 135,34 khi dùng lượng bán tuần trước. Simulator chưa đạt mốc kiểm tra tại giá lịch sử. Kết quả định giá của lượt thử phục vụ kiểm tra luồng và làm mốc cho Phase 4.

Artifact trong `experiments/artifacts/phase3_v2/m5_seed42/`:

- `simulator_train_support.csv` và `simulator_observed_price_metrics.csv`: miền giá train và kiểm tra tại giá lịch sử validation.
- `policy_summary.csv` và `policy_episodes.csv`: doanh thu mô phỏng, chênh lệch so với giá cố định và PPO đóng băng, regret và chi tiết giá/lượng bán theo SKU–tuần. Regret là chênh lệch doanh thu kỳ vọng với giá tốt nhất trong lưới 100 giá của simulator.
- `models/`, `forecasters/` và `run.json`: các checkpoint, cấu hình, SHA256, số lần cập nhật và kết quả kiểm tra nạp lại. Lệnh `replay` nạp cặp PPO–Transformer chính và phát lại validation từ các tệp mô phỏng đã lưu trong lượt chạy.

Lượt chạy seed 42 hoàn tất trong khoảng 4 phút 57 giây. Tám phương pháp tạo 1.600 quyết định, 200 cho mỗi phương pháp. Tóm tắt bốn mốc từ `policy_summary.csv`:

| Phương pháp | Doanh thu mô phỏng | So với giá cố định | So với PPO đóng băng |
| --- | ---: | ---: | ---: |
| DDPG | 121.444,47 | +4,65% | +5,57% |
| Giá cố định | 116.053,38 | 0,00% | +0,89% |
| PPO có phản hồi và gradient | 115.827,56 | −0,19% | +0,69% |
| PPO đóng băng Transformer | 115.031,89 | −0,88% | 0,00% |

PPO có phản hồi và gradient cập nhật Transformer 51 lần khi huấn luyện và 2 lần mỗi chuỗi khi chạy validation. Lệnh nạp lại tái tạo đúng cả 200 quyết định. Ví dụ, `FOODS_3_090@CA_1` ở tuần `2014-10-13` nhận giá đề xuất khoảng 1,119 USD. Transformer dự báo 589,79 sản phẩm tại giá đó, còn lượt lấy mẫu của simulator trả 302,64 sản phẩm và 338,63 USD doanh thu mô phỏng.

Truyền `--output-dir` với đường dẫn mới nếu muốn tạo lại kết quả song song. Lượt đánh giá test định giá 200 quyết định × 5 seed thuộc Phase 5.

## Phase 4 — chọn cấu hình trên validation V2

Chạy từ thư mục gốc sau Phase 1–3. Mỗi mô hình dự báo học từ train của bộ dữ liệu tương ứng. UCI dùng 120 SKU chia theo ba nhóm lượng bán train, M5 dùng đúng 120 chuỗi từ Phase 2. Trong từng bộ dữ liệu, tất cả mô hình được đối chiếu trên cùng khóa SKU–tuần và chọn theo RMSE validation.

```powershell
python -u -m experiments.phase4_forecast --dataset uci
python -u -m experiments.phase4_forecast --dataset m5
python -u -m experiments.phase4_pricing
python -m unittest discover -s experiments -p 'test_*.py'
```

XGBoost thử bốn cặp độ sâu và số cây. LSTM và Transformer thử hai learning rate với tối đa 3 epoch cho mỗi lượt. Cấu hình Song giữ nguyên. TFT và Informer được ghi riêng là chưa có lượt V2 trong ngân sách sàng lọc.

| Dự báo validation | SKU có cửa sổ train / validation | Số hàng validation mỗi mô hình | RMSE thấp nhất (sản phẩm/tuần) |
| --- | ---: | ---: | --- |
| UCI V2 | 103 / 88 trong 120 SKU | 493 | Tuần trước 149,47. Song Transformer đạt 149,62. |
| M5 V2 | 120 / 120 | 5.040 | Song Transformer 71,83. ARIMA có fallback đạt 74,48. |

Mỗi bảng có 9 mô hình. Cả 120 SKU UCI được chọn từ train trước khi kiểm tra số cửa sổ hợp lệ. Số liệu sàng lọc này áp dụng cho phạm vi ghi trong bảng. Bảng test toàn phạm vi thuộc Phase 5.

Định giá M5 dùng 20 tuần validation đầu để chọn hệ số của simulator có điều chỉnh theo lượng bán tuần trước. So simulator với mốc tuần trước trên phần validation sau đó. Bảng chính sách dùng 20 tuần từ đầu phần sau, 10 chuỗi định giá đã khóa và 3 seed nhiễu mô phỏng. Mỗi lượt bắt đầu với lịch sử đã biết trước ngày bắt đầu rồi dùng phản hồi mô phỏng. DDPG, SAC và PPO dùng checkpoint huấn luyện seed 42 từ Phase 3 trong cả hai simulator. Ba seed trong bảng phản ánh biến động nhiễu mô phỏng của cùng chính sách.

Trên 204 bản ghi có giá lịch sử trong miền train ở phần validation sau, RMSE là 204,48 cho Gaussian kernel Song, 109,65 cho lag-adjusted kernel (trọng số 0,75) và 110,52 cho mốc tuần trước. Lag-adjusted vượt mốc lịch sử ở lượt này. Giữ Gaussian làm mốc theo giao thức Song và báo kết quả lag-adjusted riêng.

Mỗi simulator chấm 10 chính sách × 3 seed × 200 quyết định, tổng cộng 12.000 quyết định. DDPG đứng đầu về doanh thu mô phỏng trung bình, hơn giá cố định theo trung bình train 10,39% với Gaussian và 9,55% với lag-adjusted. DDPG chọn đúng giá train cao nhất của SKU trong cả 1.200 quyết định của hai simulator. Lưới giá Song gần như trùng kết quả. Vì vậy, giữ DDPG làm ứng viên để kiểm tra và thêm mốc luôn giữ giá ở cận trên khi chấm Phase 5. Ba lượt nhiễu vẫn dùng một bộ trọng số chính sách đã huấn luyện.

Kết quả nằm trong `experiments/artifacts/phase4_v2/`. Hai thư mục `forecast_uci/` và `forecast_m5/` có bảng chỉ số, cấu hình thử và dự báo từng SKU–tuần. Thư mục `pricing/` có bảng kiểm tra simulator, đường phản ứng lượng bán theo giá, giá đề xuất, doanh thu và regret theo simulator, phương pháp, seed và SKU. Các lệnh mặc định tạo thư mục đầu ra mới. Có thể dùng `--output-dir` và truyền lại đường dẫn qua `--forecast-uci`, `--forecast-m5` khi chạy sang thư mục khác.

[Cấu hình bàn giao Phase 5](experiments/phase5_settings_v2.json) lưu danh sách mô hình, siêu tham số, phạm vi sàng lọc, dấu vân tay của khóa test và điều kiện kiểm tra lại DDPG. Lượt Phase 4 dùng validation. Tập test toàn phạm vi được dành cho Phase 5 sau khi nhóm duyệt.

[Kế hoạch Phase 5](docs/dynamic-pricing-final-design.md#83-kế-hoạch-và-kết-quả-cần-có-của-phase-5) ghi thứ tự chạy, điều kiện đủ khóa test và kết quả cần bàn giao cho dự báo UCI/M5 và định giá M5.

## Phase 5 — đánh giá test V2 và tiếp tục lượt chạy

Chạy từ thư mục gốc sau Phase 1 và dùng `experiments/phase5_settings_v2.json`. Dự báo UCI/M5 dùng đủ khóa test của từng bộ dữ liệu. Định giá dùng 10 chuỗi M5, 20 tuần test và năm seed độc lập.

```powershell
python -m experiments.phase5_batch --status
python -u -m experiments.phase5_batch --dataset uci --cache-dir experiments/artifacts/phase5_v2/data_cache
python -u -m experiments.phase5_pricing
python -u -m experiments.phase5_batch --dataset m5 --cache-dir experiments/artifacts/phase5_v2/data_cache
python -m experiments.phase5_verify
```

Chạy lại cùng lệnh để sử dụng thư mục kết quả của lượt trước. Có thể giới hạn một lượt bằng `--models` và `--seeds`. Khi môi trường có CUDA, thêm `--cuda-python` với đường dẫn Python của môi trường đó để chạy các mô hình PyTorch bằng GPU. Các lệnh kiểm tra và tổng hợp lưu dưới `experiments/artifacts/phase5_v2/audit/`. Bảng tổng hợp chỉ đưa vào những mô hình đủ toàn bộ khóa và seed theo cấu hình. Số lượt đang có và các lượt còn thiếu được ghi trong `audit.json` từng bộ dữ liệu. Kết quả giá và đối chứng DDPG với giá cố định cận trên nằm trong thư mục `pricing/` và `audit/`.

Trên máy của nhóm, môi trường Python 3.14 dùng PyTorch 2.14.1 CUDA 12.6 đã chạy được trên RTX 3070 Ti. Môi trường này nằm riêng dưới `experiments/artifacts/phase5_v2/runtime_cuda/`. Dùng cùng môi trường và tùy chọn khi tiếp tục một lượt đang dở. Mô hình PyTorch lưu cả trọng số, optimizer và vị trí lô dữ liệu, ARIMA lưu dự báo từng cửa hàng, còn lượt XGBoost bị ngắt giữa quá trình fit sẽ chạy lại riêng mô hình và seed đó. Lệnh dưới đây dùng để tiếp tục khi tiến trình hiện tại đã dừng, hoặc để chạy trên máy khác có môi trường CUDA phù hợp.

```powershell
$phase5Cuda = 'experiments/artifacts/phase5_v2/runtime_cuda/Scripts/python.exe'
& $phase5Cuda -u -m experiments.phase5_forecast --dataset m5 --model song_transformer --device cuda --cuda-graph --calibration-batches 264 --calibration-shard CA_1 --output-dir experiments/artifacts/phase5_v2/calibration/graph_song_transformer
python -u -m experiments.phase5_batch --dataset m5 --cuda-python $phase5Cuda --cuda-graph --parallel-cpu-cuda
```

Đo Song Transformer trên CA_1 với batch 64 đạt khoảng 5.387 dòng/giây ở chế độ thường và 20.032 dòng/giây với CUDA Graph và fused Adam. Kiểm thử ngắt và tiếp tục từ checkpoint CUDA Graph tái tạo cùng trọng số. Dựa trên các phép đo ngắn cho ba họ mô hình, riêng 25 lượt PyTorch của M5 cần khoảng 3 giờ 10 phút huấn luyện, trước thời gian chuẩn bị, kiểm tra và chấm điểm. Tốc độ thực tế có thể thay đổi khi chạy lâu. Các phép đo ở `calibration/` được ghi riêng với điểm test. Nhật ký CPU/GPU trên máy nhóm nằm dưới `logs/`.

Trong lượt M5 đầy đủ trên máy Windows, bộ nhân attention tối ưu bộ nhớ của PyTorch tạo gradient `NaN` tại một batch Transformer seed 42. Cách tính attention `math` cho gradient hữu hạn trên chính batch đó và đã qua kiểm thử checkpoint. Các lượt Transformer M5 dùng `math` và lưu tên backend trong checkpoint. Lượt seed 42 bị ngắt được lưu riêng trong `diagnostic_archive/` và chạy lại từ đầu với cùng cấu hình huấn luyện. Kiểm tra nhanh sau khi đổi backend đạt khoảng 16.828 dòng/giây với Transformer trên CA_3, vì vậy thời gian thực tế có thể cao hơn ước lượng ban đầu.

**Kết quả đã xác minh.** UCI và M5 đều đủ 37/37 lượt, với 18.542 và 1.220.898 khóa test mỗi lượt. Mốc tuần trước có RMSE thấp nhất ở UCI, 140,84 sản phẩm/tuần. ARIMA + fallback dẫn ở M5, 8,15 sản phẩm/tuần. Hai mô hình UCI có giá là LSTM và Transformer có một số dự báo cực lớn. Ví dụ, SKU `90180A` có giá test 2,95, thấp hơn khoảng 627 độ lệch chuẩn so với giá train của SKU. Kết quả test giữ nguyên để phản ánh giới hạn ngoại suy này.

Định giá M5 đủ 22.000 quyết định trên hai simulator và năm seed. DDPG chọn giá cận trên trong 1.194/2.000 quyết định. Trong cả 100 tổ hợp SKU, seed và simulator, biên độ giá theo tuần của DDPG nhỏ hơn một cent. Mức tăng doanh thu mô phỏng trung bình so với luôn giữ cận trên là 1,89% trên Gaussian và 0,41% trên lag-adjusted. Doanh thu này đến từ cách chọn mức giá theo SKU trong từng seed, với giá gần như giữ cố định qua các tuần.

## Phase 6 — báo cáo và biểu đồ V2

Chạy từ thư mục gốc sau khi hoàn tất sàng lọc validation Phase 4 và có kết quả test Phase 5 đã kiểm tra cho cả UCI, M5 và định giá. Cài thư viện nếu môi trường chưa có và tạo lại báo cáo:

```powershell
python -m pip install -r requirements-experiments.txt
python -m experiments.phase6_report_v2
```

[Báo cáo bảng số liệu](experiments/artifacts/phase6_v2/report.md) chứa RMSE, MAE, R², WAPE, MAPE, sai số theo nhóm giá, hai bảng doanh thu mô phỏng và ablation PPO theo seed. Sáu biểu đồ và CSV nằm trong cùng thư mục. Máy mới tạo lại thư mục này từ kết quả Phase 4–5. Lệnh báo cáo đọc kết quả đã lưu nên có thể chạy bằng CPU. Có thể truyền `--output-dir` để ghi ở chỗ khác.

[Kết luận Phase 6 và đối chiếu bài báo](docs/phase6-report-v2.md) giải thích cách chọn mô hình, giới hạn của DDPG và PPO, mức tái dựng các bảng 1, 2, 3, 6 của Song và điều kiện so sánh với các bài dùng cùng nguồn.

## Bước 1 — chạy lại mốc định giá M5 trước khi tune

Mục đích của lượt này là tái tạo mốc Phase 5 để đo hiệu quả của các thay đổi sau này. Luồng Song dùng Transformer dự báo lượng bán và gradient giá, PPO chọn giá liên tục và lượng bán mô phỏng để cập nhật Transformer mỗi 10 quyết định. Chạy cùng lưới giá và các đối chứng đã có theo cấu hình đã khóa. Tổng phạm vi là 10 chuỗi M5, 20 tuần test, 11 phương pháp, hai simulator và năm seed `42, 123, 456, 789, 1024`.

Chạy từ thư mục gốc sau khi chuẩn bị dữ liệu Phase 1. Dùng thư mục đầu ra mới để huấn luyện lại từ đầu. Nếu bị gián đoạn, chạy lại cùng lệnh để tiếp tục từ những lượt đã lưu:

```powershell
python -u -m experiments.phase5_pricing --output-dir experiments/artifacts/pricing_baseline_v2/pricing
python -m experiments.phase5_verify --part pricing --output-root experiments/artifacts/pricing_baseline_v2
```

**Kết quả Bước 1.** Lượt chạy lại hoàn thành 55 tổ hợp phương pháp–seed và xác minh đủ 22.000 quyết định. PPO đầy đủ cập nhật Transformer 20 lần trong mỗi lượt test 200 quyết định. Doanh thu mô phỏng trung bình của PPO là 127.279,74 trên Gaussian và 109.589,28 trên lag-adjusted. Lưới giá Song đạt 138.417,71 và 118.557,57 theo cùng thứ tự.

Khi đối chiếu theo khóa, 20.000 dòng thuộc 10 phương pháp khớp mọi giá trị với Phase 5. Lưới giá XGBoost + giá chạy hai luồng có 435 giá đề xuất thay đổi trong 2.000 quyết định. Kiểm tra riêng bằng `n_jobs=1` tái tạo đúng cây XGBoost, giá và doanh thu mô phỏng Phase 5 trên cả năm seed.

Đọc [bảng tổng hợp](experiments/artifacts/pricing_baseline_v2/pricing/pricing_test.csv), [kết quả từng seed](experiments/artifacts/pricing_baseline_v2/pricing/pricing_test_runs.csv), [biên bản kiểm tra](experiments/artifacts/pricing_baseline_v2/audit_pricing.json), [bảng đối chiếu Phase 5](experiments/artifacts/pricing_baseline_v2/phase5_comparison.csv) và [kiểm tra số luồng XGBoost](experiments/artifacts/pricing_baseline_v2/xgboost_thread_check.json).

## Bước 2 — tune Transformer định giá trên M5

Sàng lọc trên 120 chuỗi của luồng định giá đã khóa, gồm 21.601 cửa sổ train và 5.040 khóa validation. Thử lịch sử 8 hoặc 12 tuần, learning rate 0,0003, 0,0005 hoặc 0,001, tối đa 4 epoch, batch 64 và seed 42, 123. Giữ kiến trúc Song 4 lớp, 8 head và chiều 128. Chọn bằng RMSE validation trung bình trên hai seed. Dùng thêm 50 trường hợp validation để kiểm tra lượng bán dự báo và gradient tại các mức trong miền giá train.

Chạy từ thư mục gốc sau Phase 1. Lệnh dưới đây dùng môi trường Python có CUDA, sàng lọc, huấn luyện cấu hình đã chọn trên tập train M5 của cả 10 cửa hàng, rồi kiểm tra dự báo trên các khóa test đã khóa:

```powershell
python -u -m experiments.pricing_transformer_tune --device cuda
```

Có thể dùng `--stage screen` để chỉ sàng lọc và `--stage full` để tiếp tục huấn luyện toàn phạm vi và chấm test. Chạy lại với cùng môi trường và cùng thư mục để tiếp tục từ checkpoint. Dùng `--output-dir` với đường dẫn mới để chạy một bản riêng. Lượt huấn luyện cuối dùng seed 42 và giữ phạm vi fit là train để checkpoint có thể dùng làm điểm khởi đầu cho định giá.

**Kết quả Bước 2.** Cấu hình chọn là 8 tuần, learning rate 0,0005 và 4 epoch. So với cấu hình 12 tuần, learning rate 0,001 và 3 epoch trên cùng validation, RMSE trung bình giảm từ 76,79 xuống 73,39 và MAE từ 44,77 xuống 43,04. Cấu hình thắng dùng hết ngân sách bốn epoch đã chốt.

Checkpoint cuối học từ 3.826.026 cửa sổ train. Khi chấm trên 1.220.898 khóa test đã xác minh của 29.069 chuỗi M5, kết quả seed 42 là **RMSE 9,64, MAE 3,76, R² 0,809, WAPE 40,14% và MAPE 59,37%** trên 962.285 nhãn dương. Mốc ARIMA + fallback có RMSE 8,15. Transformer theo Song seed 42 ở Phase 5 có RMSE 9,89 và dùng cả train lẫn validation để fit, vì vậy hai điểm Transformer mô tả hai phạm vi fit khác nhau.

Nạp checkpoint trên CPU để thử giá cho 50 trường hợp validation thuộc 10 chuỗi định giá. Cả 50 trường hợp đều cho lượng bán dự báo thay đổi theo giá. Gradient tại điểm giữa miền giá âm trong 45 trường hợp. Đây là phản ứng của mô hình. Hiệu quả giá đề xuất sẽ được đo bằng benchmark định giá.

[Bảng thử trên validation](experiments/artifacts/pricing_transformer_tune_v2/validation_trials.csv) · [Cấu hình đã chọn](experiments/artifacts/pricing_transformer_tune_v2/selection.json) · [Biên bản test M5](experiments/artifacts/pricing_transformer_tune_v2/m5_test_audit.json) · [Chỉ số từng cửa hàng](experiments/artifacts/pricing_transformer_tune_v2/m5_test_by_store.csv) · [Kiểm tra giá với checkpoint cuối](experiments/artifacts/pricing_transformer_tune_v2/full_checkpoint_price_check.json) · [Checkpoint](experiments/artifacts/pricing_transformer_tune_v2/full_m5/song_transformer.pt)

## Bước 3 — đánh giá ARIMAX có giá trên M5

Thử ARIMAX(1,0,0) với giá đầu vào và một biến thể thêm cặp sin/cos theo chu kỳ 52,18 tuần. Cả hai được ước lượng bằng bộ lọc trạng thái của `statsmodels.SARIMAX`. Mốc đối chứng thống kê là AR(1) cùng phạm vi fit và Transformer từ Bước 2. Mỗi mô hình thống kê fit riêng trên train của từng SKU–cửa hàng. Khi dự báo tuần hiện tại, các quan sát validation hoặc test chỉ đi vào lịch sử kể từ tuần sau.

Mô hình giá cần ít nhất 12 cặp tuần train quan sát liền kề và có giá thay đổi trong các cặp đó. Bảng trên mọi khóa dùng AR(1) cho chuỗi thiếu mô hình giá và dùng lượng bán quan sát gần nhất khi AR(1) không fit được. Bảng trên khóa chung chỉ lấy nơi có dự báo trực tiếp của các biến thể ARIMAX cần so. Có thêm bảng chỉ lấy giá ghi nhận nằm trong miền giá train. Chọn biến thể bằng micro RMSE trên cùng khóa validation có dự báo trực tiếp của cả hai biến thể, rồi chấm biến thể được chọn trên test V2.

Chạy sau khi có checkpoint và dự báo test của Bước 2. Lệnh đầu dùng Python đã cài PyTorch phù hợp với thiết bị. Lệnh sau chạy CPU và có thể giảm `--workers` trên máy ít RAM:

```powershell
python -u -m experiments.pricing_arimax_transformer_reference --device auto
python -u -m experiments.pricing_arimax --workers 6
```

Nếu đã có các kết quả từng cửa hàng, chạy lại cùng lệnh ARIMAX để kiểm tra SHA256 và dùng lại các lượt đã lưu. Thử độ nhạy bằng chín giá ứng viên trong miền train và 50 trường hợp validation thuộc 10 chuỗi định giá. Các số liệu đó mô tả phản ứng của mô hình. Benchmark doanh thu thuộc bước tích hợp engine kế tiếp.

**Kết quả Bước 3.** ARIMAX giá cơ bản được chọn với RMSE validation **9,54**, so với **10,42** của bản thêm Fourier trên cùng **766.164 khóa**. Test đã xác minh đủ **1.220.898 khóa**. ARIMAX dự báo trực tiếp cho **18.244/29.069 chuỗi** và **766.248 khóa test**, khoảng **62,76%**. Các khóa còn lại được chấm với dự báo dự phòng.

| Mô hình | RMSE trên 766.248 khóa chung | MAE trên 766.248 khóa chung | RMSE trên toàn bộ test |
| --- | ---: | ---: | ---: |
| AR(1) không giá | 8,80 | 4,15 | 8,79 |
| ARIMAX có giá | 9,66 | 4,35 | 9,34 |
| Transformer đã tune | 10,14 | 3,96 | 9,64 |

ARIMAX phản ứng với giá ở 50/50 trường hợp validation thử giá. Hệ số giá dương ở 5/10 chuỗi, tức mô hình dự báo lượng bán tăng khi tăng giá ở các chuỗi đó. Đề xuất dùng ARIMAX làm đối chứng lưới giá và đánh giá doanh thu cùng simulator trước khi chọn mô hình định giá.

[Điểm validation](experiments/artifacts/pricing_arimax_v2/validation_scores.csv) · [Điểm test](experiments/artifacts/pricing_arimax_v2/test_scores.csv) · [Độ bao phủ](experiments/artifacts/pricing_arimax_v2/coverage.csv) · [Phản ứng theo giá](experiments/artifacts/pricing_arimax_v2/validation_price_response.csv) · [Biên bản lượt chạy](experiments/artifacts/pricing_arimax_v2/run.json)

## Bước 4 — so sánh luồng chọn giá trên M5

Nạp checkpoint Transformer 8 tuần của Bước 2 và hệ số ARIMAX đã fit trên train ở Bước 3. Transformer và ARIMAX thử cùng lưới 41 mức trong miền giá train, cộng giá tham chiếu nếu khác các mức trong lưới. ARIMAX dùng lượng bán và giá của tuần trước để tính dự báo ứng với từng giá mới. Sau khi bắt đầu một lượt mô phỏng, tuần trước là đầu ra của simulator.

Huấn luyện lại PPO có gradient với 512 tương tác trên Gaussian cho mỗi seed của Bước 1. Hai nhánh dùng cùng checkpoint Transformer khởi đầu, một nhánh giữ trọng số và một nhánh cập nhật mỗi 10 quyết định. PPO vẫn có 12 tuần lịch sử trong trạng thái, còn Transformer có 8 tuần đầu vào như cấu hình đã chọn. Năm seed thay đổi huấn luyện PPO và nhiễu simulator. Checkpoint Transformer ban đầu dùng cùng seed 42.

Chạy từ thư mục gốc sau khi hoàn tất và giữ artifact Bước 1 đến Bước 3:

```powershell
python -u -m experiments.pricing_policy_compare
```

Dùng `--scope core` để chạy nhóm 10 chuỗi gốc, hoặc `--scope expanded` cho nhóm mở rộng. Chạy lại cùng lệnh để dùng lại các checkpoint và dữ liệu mô phỏng đã kiểm tra. Nhóm mở rộng gồm 50 chuỗi mới, năm chuỗi mỗi cửa hàng, chọn theo các nhóm tổng lượng bán train trong số chuỗi có ARIMAX, có ít nhất ba mức giá train và đủ lịch sử cùng lịch 20 tuần đã định. Lượt mở rộng dùng các chính sách PPO đã học trên 10 chuỗi gốc để kiểm tra khả năng áp dụng sang chuỗi khác.

**Kết quả Bước 4.** Máy chạy 5 phút 43 giây và xác minh **80.000 quyết định mô phỏng**, gồm 20.000 ở nhóm gốc và 60.000 ở nhóm mở rộng. Mức chênh dưới đây tính bằng trung bình phần trăm theo từng cặp seed so với **giữ giá cao nhất trong train**.

| Phương pháp | 10 chuỗi · Gaussian | 10 chuỗi · Lag-adjusted | 50 chuỗi · Gaussian | 50 chuỗi · Lag-adjusted |
| --- | ---: | ---: | ---: | ---: |
| Lưới giá ARIMAX | **+11,55%** | **+7,08%** | **+6,29%** | **+3,79%** |
| Lưới giá Transformer đã tune | +0,25% | +1,70% | −7,16% | −5,51% |
| PPO với Transformer đã tune và phản hồi | −6,73% | −7,36% | −3,35% | −3,19% |

Lưới giá ARIMAX dẫn doanh thu mô phỏng trung bình và thắng mốc cận trên ở 5/5 seed trong cả bốn phép đối chiếu. Tỷ lệ chọn cận thấp hoặc cận cao của ARIMAX là 94,1–94,2% ở nhóm gốc và 97,8–98,0% ở nhóm mở rộng. Chỉ 8,8–9,6% tổ hợp SKU–seed ở nhóm mở rộng đổi giá hơn 0,01 USD qua các tuần. Người xem có thể đọc đây là kết quả chọn mức giá trong miền đã quan sát, với phần lớn đề xuất nằm ở hai biên.

Trên nhóm gốc, PPO mới cải thiện 0,08% và 0,04% so với PPO cũ nhưng vẫn thấp hơn mốc cận trên ở cả hai simulator. Lưới giá Transformer mới giảm 0,73% trên Gaussian và tăng 1,43% trên lag-adjusted so với lưới Transformer cũ. Checkpoint mới khác cả phạm vi huấn luyện và cấu hình so với bản cũ, nên phép thử đo tác động của việc thay checkpoint. Đề xuất chọn lưới giá ARIMAX làm ứng viên chính cho SKU đủ điều kiện và giữ PPO để minh họa luồng Song. Các mức doanh thu đều là số mô phỏng.

[Kết quả 10 chuỗi](experiments/artifacts/pricing_policy_compare_v2/core/summary.csv) · [Kết quả 50 chuỗi](experiments/artifacts/pricing_policy_compare_v2/expanded/summary.csv) · [So sánh từng cặp trên nhóm gốc](experiments/artifacts/pricing_policy_compare_v2/core/paired_comparisons.csv) · [Danh sách 50 chuỗi](experiments/artifacts/pricing_policy_compare_v2/expanded_cohort.csv) · [Biên bản kiểm tra](experiments/artifacts/pricing_policy_compare_v2/run.json)

## Bổ sung Bước 4 — tune ARIMAX và đối chiếu PPO có phản hồi

Tái sử dụng 10 chuỗi gốc và 50 chuỗi mở rộng của Bước 4. Trên train của mỗi chuỗi, thử ARIMAX(p,0,0) với p bằng 0, 1 hoặc 2 và giá gốc hoặc log giá. Bậc 0 được ước lượng trực tiếp bằng hồi quy bình phương tối thiểu. Bậc 1 và 2 dùng `statsmodels.SARIMAX`. Chọn trên 20 tuần validation cuối, từ 16-03-2015 đến 27-07-2015, bằng doanh thu mô phỏng so với giá cố định theo SKU, ghép trên ba seed, hai simulator và hai nhóm chuỗi. Bảng cũng ghi RMSE và MAE ở giá lịch sử trong miền train.

Mốc giá cố định thử 41 mức giá train và giữ một mức riêng cho từng SKU. Chọn mức có doanh thu kỳ vọng cao nhất khi mô phỏng toàn kỳ validation, lấy trung bình giữa hai simulator. Giữ mức đó qua tất cả các tuần test.

Huấn luyện PPO trên nhóm 10 chuỗi với 512 tương tác mỗi seed. So nhánh giữ hệ số ARIMAX với nhánh cập nhật hệ số giá mỗi 10 quyết định. Nhánh phản hồi dùng hồi quy có phạt về hệ số train, với mức phạt 32 trên thang chuẩn hóa. Cần ít nhất bốn phản hồi cho SKU có biến động giá để cập nhật. Giữ hệ số tự hồi quy và các scaler của train. Cả hai nhánh đều đưa lượng bán mô phỏng vào lịch sử tuần sau.

Chạy từ thư mục gốc sau khi có kết quả Bước 3 và cả hai nhóm Bước 4:

```powershell
python -u -m experiments.pricing_arimax_feedback
```

Chấm sáu phương pháp trên 20 tuần test tiếp theo từ 21-12-2015 đến 02-05-2016, với năm seed và hai simulator. Hai nhóm dùng chung chính sách PPO đã huấn luyện trên 10 chuỗi gốc. Dữ liệu bán ghi nhận chỉ khởi tạo lịch sử trước kỳ mô phỏng. Các phương pháp dùng chung nhiễu theo SKU–tuần–seed trong từng nhóm và simulator. Chạy lại cùng lệnh để dùng lại mô hình và các lượt đã được xác minh.

**Kết quả tune.** Cả sáu cấu hình đều fit đủ 60 chuỗi. ARIMAX(0,0,0) dùng giá gốc đạt doanh thu mô phỏng validation cao nhất trong các cấu hình lưới, thấp hơn mốc giá cố định theo SKU 2,62%. RMSE validation của nó là 80,10 sản phẩm/tuần. ARIMAX(1,0,0) giá gốc có RMSE thấp nhất, 43,20, trên cùng 1.068 khóa trong miền giá train. Cấu hình bậc 0 không có thành phần tự hồi quy, nên lưới giá với hệ số giữ cố định đã chọn cùng mức giá qua các tuần test cho từng SKU.

**Kết quả test.** Lượt chính chạy 7 phút 43 giây trên CPU và xác minh đủ 72.000 quyết định. Các tỷ lệ dưới đây là trung bình chênh doanh thu mô phỏng theo từng cặp seed so với giá cố định riêng theo SKU được chọn trên validation.

| Phương pháp | 10 chuỗi · Gaussian | 10 chuỗi · Lag-adjusted | 50 chuỗi · Gaussian | 50 chuỗi · Lag-adjusted |
| --- | ---: | ---: | ---: | ---: |
| Giá cố định theo SKU | **0%** | **0%** | **0%** | **0%** |
| Giá cận trên train | −10,85% | −8,21% | −8,58% | −6,41% |
| Lưới ARIMAX Bước 3 | −1,87% | −2,58% | −2,90% | −2,57% |
| Lưới ARIMAX sau tune | −2,50% | −2,25% | −2,28% | −2,35% |
| ARIMAX + PPO giữ hệ số | −19,84% | −16,14% | −11,67% | −10,44% |
| ARIMAX + PPO có phản hồi | −20,38% | −16,47% | −11,66% | −10,70% |

Giá cố định theo SKU thắng từng phương pháp còn lại ở cả 20 tổ hợp nhóm–simulator–seed. Nhánh phản hồi thực hiện 1.011 lượt cập nhật hệ số giá trên test. So với PPO giữ hệ số, thay đổi doanh thu ghép seed nằm trong khoảng −0,71% đến +0,004%, với mức dương gần bằng 0 chỉ xuất hiện ở Gaussian trên 50 chuỗi. Kết quả ủng hộ giá cố định theo SKU làm mốc mặc định nếu mục tiêu là doanh thu trong mô phỏng này. Lưới ARIMAX bậc 1 và PPO có thể giữ làm hai cách thử giá để đối chiếu. Phạm vi tune trong bước bổ sung là 60 chuỗi. Số liệu toàn phạm vi dự báo M5 vẫn nằm ở Bước 2 và Bước 3.

[Bảng tune sáu cấu hình](experiments/artifacts/pricing_arimax_feedback_v2/validation_trials.csv) · [Giá cố định chọn từ validation](experiments/artifacts/pricing_arimax_feedback_v2/validation_fixed_prices.csv) · [Kết quả 10 chuỗi](experiments/artifacts/pricing_arimax_feedback_v2/core/summary.csv) · [Kết quả 50 chuỗi](experiments/artifacts/pricing_arimax_feedback_v2/expanded/summary.csv) · [Biên bản lượt chạy](experiments/artifacts/pricing_arimax_feedback_v2/run.json)

## Báo cáo doanh thu chuẩn hóa cho Bảng 3 và Bảng 6

Chạy từ thư mục gốc sau khi có kết quả Phase 5 và kiểm thử ARIMAX bổ sung:

```powershell
python -m experiments.pricing_normalized_report
```

Điểm của một phương pháp bằng 100 nhân doanh thu mô phỏng của phương pháp chia cho doanh thu của mốc **cùng seed và simulator**. Bảng 3 dùng giá trung bình lịch sử train làm mốc. Bảng 6 dùng PPO đóng băng Transformer. Bảng ARIMAX bổ sung dùng giá cố định riêng theo SKU, chọn trên validation. Lệnh đọc kết quả đã lưu và xuất doanh thu thô, điểm theo seed, trung bình, độ lệch chuẩn mẫu, min và max.

[Báo cáo và các bảng CSV](experiments/artifacts/pricing_normalized_v2/report.md) có thêm phép đối chiếu với Song. Bảng 3 của Song chỉ có số tổng hợp theo phương pháp nên dùng tỷ lệ của các trung bình cho cột đối chiếu. Bảng 6 có số từng seed, cho phép chia lại từng seed theo PPO đóng băng. Các điểm đối chiếu sử dụng mốc riêng của từng nghiên cứu.

## Tái tạo benchmark dự báo M5 toàn phạm vi (luồng cũ)

Các lệnh trong mục này thuộc luồng đã viết trước giao thức V2. Chúng chưa tạo bảng so sánh đủ điều kiện theo quy tắc test và cohort mới; không dùng điểm của chúng làm kết quả chính V2.

Các file dữ liệu và kết quả M5 nằm trong `data/` và `experiments/artifacts/`; hai thư mục này bị Git bỏ qua nên thành viên mới cần tải và tạo lại. Chạy các lệnh sau từ thư mục gốc sau khi pull code:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-experiments.txt
python -m experiments.download_paper_data --dataset m5
python -m experiments.full_m5_prepare --data-dir data/song_replication/m5 --output-root experiments/artifacts/song_m5_full
python -u -m experiments.full_m5_benchmark --prepared-root experiments/artifacts/song_m5_full --output-root experiments/artifacts/song_m5_full/full_benchmark
```

Lệnh tải kiểm tra kích thước và checksum của bộ M5. Bước chuẩn bị tạo panel tuần riêng cho đủ 10 cửa hàng, dùng `sales_train_evaluation.csv`, chia tuần theo `paper_70_15_15` và lookback 12 tuần. Benchmark chạy 8 phương pháp theo thứ tự từng cửa hàng để giới hạn RAM; mặc định seed 42, tối đa 25 epoch cho mô hình mạng nơ-ron. Có thể thêm `--stores CA_1` để chạy thử một cửa hàng; đó không phải kết quả toàn M5. Nếu bị gián đoạn, chạy lại cùng lệnh benchmark: cửa hàng đã hoàn tất được tái sử dụng.

Kết quả nằm trong `experiments/artifacts/song_m5_full/full_benchmark/`: mỗi thư mục cửa hàng chứa chỉ số, dự báo, chỉ số theo SKU và manifest lần chạy; `forecast_metrics_full_scope.csv` tổng hợp các chỉ số toàn phạm vi. Dữ liệu và kết quả không được commit/push vì dung lượng lớn; thành viên muốn có cùng artifact cần chạy lại hoặc trao đổi riêng các file kết quả. Dành vài GB dung lượng trống và lưu ý các mô hình ARIMA/Transformer có thể chạy nhiều giờ trên CPU.

### Link tải dữ liệu

| Dữ liệu | Trang nguồn | Tải trực tiếp |
| --- | --- | --- |
| **M5 Forecasting Accuracy** — nguồn chính thức | [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) | Cần đăng nhập Kaggle và chấp nhận điều khoản cuộc thi. |
| `sales_train_evaluation.csv` | [Mirror sktime](https://huggingface.co/datasets/sktime/tsf-datasets/blob/main/m5-forecasting-accuracy/sales_train_evaluation.csv) | [CSV](https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/sales_train_evaluation.csv) |
| `sales_train_validation.csv` | [Mirror sktime](https://huggingface.co/datasets/sktime/tsf-datasets/blob/main/m5-forecasting-accuracy/sales_train_validation.csv) | [CSV](https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/sales_train_validation.csv) |
| `sell_prices.csv` | [Mirror sktime](https://huggingface.co/datasets/sktime/tsf-datasets/blob/main/m5-forecasting-accuracy/sell_prices.csv) | [CSV](https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/sell_prices.csv) |
| `calendar.csv` | [Mirror sktime](https://huggingface.co/datasets/sktime/tsf-datasets/blob/main/m5-forecasting-accuracy/calendar.csv) | [CSV](https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/calendar.csv) |
| `sample_submission.csv` | [Mirror sktime](https://huggingface.co/datasets/sktime/tsf-datasets/blob/main/m5-forecasting-accuracy/sample_submission.csv) | [CSV](https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/sample_submission.csv) |
| **Online Retail** (UCI, 2010–2011; bài Song) | [Trang UCI](https://archive.ics.uci.edu/dataset/352/online%2Bretail) | [ZIP chính thức](https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip) |
| **Online Retail II** (UCI, 2009–2011; demo gốc) | [Trang UCI](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii) | [ZIP chính thức](https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip) |

Benchmark M5 toàn phạm vi cần `sales_train_evaluation.csv`, `sell_prices.csv` và `calendar.csv`. Lệnh `download_paper_data` tải năm tệp M5 từ mirror và kiểm tra kích thước/MD5 theo [bản ghi checksum Zenodo](https://zenodo.org/records/10203108). Script cũng có thể tải Online Retail một năm; Online Retail II là bộ dữ liệu khác và tải riêng nếu cần dựng lại demo cũ.

## 1. Chạy giao diện với dữ liệu và mô hình đã có

Mở PowerShell tại thư mục dự án:

```powershell
cd 'E:\File From CC\Data Pricing\dynamic-pricing'
python -m pip install -r requirements-experiments.txt
```

Cần giữ `models_2y/` cho demo Online Retail II và `experiments/artifacts/song_m5_pilot/` cho thí nghiệm M5. Nếu thiếu các tệp này, xem mục **Chạy lại từ đầu** bên dưới.

Mở **hai terminal** tại cùng thư mục. Terminal 1 chạy backend:

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

Terminal 2 chạy frontend:

```powershell
python frontend/serve.py
```

Mở các trang:

| Trang | Địa chỉ | Dữ liệu |
| --- | --- | --- |
| Đề xuất giá M5 | <http://127.0.0.1:8765/price-m5.html> | Mô hình và kết quả Phase 6–7 |
| Phương pháp và chỉ số | <http://127.0.0.1:8765/method.html> | Kết quả kiểm định và nguồn bài báo |
| Thí nghiệm mô hình | <http://127.0.0.1:8765/research.html> | Dự báo, chính sách giá M5/UCI đã lưu |
| Demo Online Retail II | <http://127.0.0.1:8765/> | `models_2y/`; trang sản phẩm cần SQL Server |
| Tài liệu API | <http://127.0.0.1:8000/docs> | FastAPI |

Tại trang M5, chọn SKU và tuần test; để trống giá hiện tại để dùng giá tuần trước, hoặc nhập giá trong miền train hiển thị dưới ô. Chọn phương pháp rồi bấm **Tính giá đề xuất**. Bảng bên dưới so sánh 7 phương pháp trên cùng SKU, tuần và giá đầu vào; seed của các chính sách học được chọn từ validation, trừ seed bạn chọn trực tiếp. Đồ thị và lịch sử 12 tuần nằm trong phần có thể mở rộng. Giá mới và doanh thu ở giá chưa bán là **dự báo/mô phỏng**, chưa phải giá tối ưu đã được xác nhận ngoài thực tế. Trang M5 không ghi vào SQL Server.

Nếu cổng 8000 đã có backend cũ, dừng tiến trình đó và chạy lại lệnh `uvicorn` để nạp các API mới.

## 2. Chạy lại từ đầu khi thiếu dữ liệu hoặc mô hình

**Demo Online Retail II:** đặt `online_retail_II.xlsx` trong `data/`, rồi chạy theo thứ tự:

```powershell
python 1_data_processing_online_retail_2y.py
python 2_data_processing_online_retail_ml_2y.py
python 3_select_products_2y.py
python 4_train_final_model_online_retail_2y.py
python 5_fit_pricing_model_2y.py
```

**SQL Server cho trang danh sách và lịch sử sản phẩm:** chỉ chạy khi muốn tạo/nhập database `DynamicPricingDemo`. Máy cần SQL Server, `sqlcmd` và kết nối Windows Authentication; đổi tên instance nếu cần:

```powershell
.\database\setup.ps1 -Server '.\SQLEXPRESS'
```

**Các phase M5 pilot và định giá:** dữ liệu/kết quả được lưu riêng trong `experiments/artifacts/song_m5_pilot/`. Cách chạy lại các phase và ý nghĩa từng bước nằm trong [nhật ký dự án](PROJECT_HISTORY.md); giao thức nghiên cứu ở [experiments/PROTOCOL.md](experiments/PROTOCOL.md). Hướng dẫn benchmark M5 toàn phạm vi ở [experiments/README.md](experiments/README.md).

## 3. Kiểm tra

```powershell
python -m unittest discover -s experiments -p 'test_*.py' -q
```

Hướng dẫn chi tiết từng trang nằm trong [frontend/README.md](frontend/README.md). README trước khi rút gọn đã được lưu nguyên nội dung tại [PROJECT_HISTORY.md](PROJECT_HISTORY.md) để nhóm tra cứu số liệu và lệnh chạy của các phase.
