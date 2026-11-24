# Phase 6 · Đánh giá dự báo lượng bán và đề xuất giá

## Kết luận

| Bài toán | Kết quả trên test V2 | Cách dùng |
| --- | --- | --- |
| Dự báo lượng bán UCI | Tuần trước có RMSE thấp nhất, **140,84** sản phẩm/tuần. ARIMA + fallback có MAE và WAPE thấp nhất, **47,13** và **68,42%**. | Dùng tuần trước làm mốc dự báo theo tiêu chí RMSE đã định. |
| Dự báo lượng bán M5 | ARIMA + fallback có RMSE **8,15**, MAE **3,56**, R² **0,864**, WAPE **38,01%** và MAPE dương **56,45%**. Trong các mô hình có đầu vào giá, Transformer theo Song có RMSE thấp nhất, **13,09 ± 1,80**. | Dùng ARIMA + fallback làm mốc dự báo thông thường. Giữ Transformer theo Song cho thí nghiệm dự báo cầu ở các giá ứng viên. |
| Định giá mô phỏng M5 | DDPG có doanh thu trung bình cao nhất trong cả hai simulator. Giá của từng SKU gần như cố định qua 20 tuần trong mỗi lượt. | Dùng giá cận trên train làm mốc mạnh. Lưới giá Song cho phép trình bày bảng giá, lượng bán dự báo và doanh thu dự kiến của từng ứng viên trong bản demo. |

[Toàn bộ bảng chỉ số và sáu biểu đồ](../experiments/artifacts/phase6_v2/report.md) được tạo từ các kết quả Phase 5 đã kiểm tra. [Lệnh chạy lại trong README](../README.md#phase-6--báo-cáo-và-biểu-đồ-v2).

## Dự báo lượng bán

- **Phạm vi.** UCI Online Retail bản một năm (ID 352) lấy từ 541.909 dòng hóa đơn, cho 3.387 SKU đủ điều kiện và 18.542 khóa test. M5 dùng tệp `sales_train_evaluation.csv`, `calendar.csv` và `sell_prices.csv`, giữ 29.069/30.490 chuỗi SKU–cửa hàng đủ điều kiện, với 1.220.898 khóa test. Mỗi dataset có 37 lượt dự báo đã qua kiểm tra đủ khóa. Đây là toàn phạm vi đủ điều kiện theo V2. [Nguồn UCI](https://archive.ics.uci.edu/dataset/352/online%2Bretail), [giao thức V2](../experiments/PROTOCOL_V2.md).
- **Chia theo thời gian.** UCI dùng 37/8/9 tuần train/validation/test, test bắt đầu 10-10-2011. M5 dùng 193/42/42 tuần, test bắt đầu 03-08-2015. Chốt ranh giới theo tuần lịch trước khi tính thống kê. Nhãn ở cả ba tập đều được tổng hợp theo tuần. Chỉ train dùng để tính ngưỡng ngoại lệ, chuẩn hóa và xét đủ điều kiện. Test giữ giao dịch hợp lệ kể cả vượt ngưỡng IQR. M5 chấm tuần bán 0 nếu đủ giá, với 962.285 khóa có lượng bán dương dùng để tính MAPE. UCI có 18.542 khóa dương. Dự báo một tuần cuốn chiếu có thể sử dụng các tuần test đã xảy ra làm lịch sử. Các mô hình có giá nhận giá ghi nhận của tuần cần dự báo như một đầu vào điều kiện.
- **Thử siêu tham số trên validation.** Trên mỗi dataset, mỗi biến thể XGBoost thử bốn cặp độ sâu `{3, 5}` và số cây `{150, 300}`. Các biến thể LSTM và Transformer thử learning rate `{0,0005; 0,001}` với tối đa ba epoch. Cấu hình XGBoost có giá chọn độ sâu 3 và 300 cây cho UCI, độ sâu 5 và 300 cây cho M5. Transformer theo Song giữ kiến trúc 4 lớp, 8 heads, chiều 128 và learning rate 0,001. Toàn bộ thử nghiệm, chỉ số, thời gian và cấu hình được lưu. [Cấu hình đã khóa](../experiments/phase5_settings_v2.json), [bảng thử UCI](../experiments/artifacts/phase6_v2/uci_forecast_trials.csv), [bảng thử M5](../experiments/artifacts/phase6_v2/m5_forecast_trials.csv).
- **Từ validation đến test.** UCI giữ nguyên thứ hạng dẫn đầu của mốc tuần trước. Trên M5, Transformer theo Song dẫn validation với RMSE 71,83 trên 5.040 khóa từ mẫu sàng lọc. ARIMA + fallback dẫn ở tập test rộng hơn. Hai tập có quy mô và phân bố lượng bán khác nhau, vì vậy không lấy chênh lệch 71,83 và 8,15 làm mức cải thiện. ARIMA dùng lượng bán đã biết gần nhất ở 1.895 khóa UCI và 6.132 khóa M5 khi thiếu lịch sử đủ điều kiện hoặc fit không đạt. Hai mốc xác định chạy một lần. Bảy phương pháp ngẫu nhiên chạy năm seed và báo độ lệch chuẩn mẫu.
- **Giá ngoài miền đã học.** UCI tuần 17-10-2011, SKU `90180A` bán 6 sản phẩm ở giá 2,95 GBP. Giá train của SKU trung bình 9,955 GBP và độ lệch chuẩn khoảng 0,01118 GBP. Giá test thấp hơn trung bình train khoảng 627 độ lệch chuẩn. Ở khóa này, Transformer + giá seed 42 dự báo khoảng 2,34 × 10¹⁷ sản phẩm và LSTM + giá seed 123 dự báo khoảng 1,24 × 10²¹. Các lần này đẩy RMSE trung bình lên 3,44 × 10¹⁴ và 1,85 × 10¹⁸ tương ứng. Báo cáo giữ các điểm đó. Hai mô hình có giá này chưa đủ ổn định để dùng cho UCI.
- **Đọc sai số.** RMSE nhạy với tuần sai số lớn. MAE là sai lệch tuyệt đối bình quân tính bằng sản phẩm. R² đánh giá phần biến thiên mô hình giải thích trên tập test. WAPE tính tổng lỗi tuyệt đối chia tổng lượng bán, còn MAPE chia lỗi từng hàng cho lượng bán thực tế dương trước khi lấy trung bình. MAPE có thể vượt 100% khi lượng bán thấp và dự báo lệch nhiều.

Lượt đầy đủ trên máy nhóm dùng Windows, RTX 3070 Ti cho các lượt M5 PyTorch. Tổng thời gian trong manifest của 37 lượt thành công khoảng **3,66 giờ M5** và **11,4 phút UCI**. Đây là tổng thời gian từng lượt. CPU và GPU có chạy song song, và thời gian này chưa cộng lần lỗi phải chạy lại hoặc lần kiểm tra kết quả độc lập.

## Định giá mô phỏng M5

Cohort chọn theo lịch sử train gồm 10 chuỗi, mỗi cửa hàng một SKU, chấm 20 tuần đầu của test, từ 03-08-2015 đến 14-12-2015. Mỗi chính sách có 200 quyết định mỗi seed. Hai simulator, 11 chính sách và năm seed tạo **22.000 quyết định**. Các chính sách học tăng cường dùng ngân sách 512 tương tác mỗi seed. Học trên Gaussian rồi dùng cùng chính sách đã học để kiểm tra độ nhạy với lag-adjusted 0,75. Hai mô hình cầu dùng để thử lưới giá là Transformer theo Song và XGBoost + giá. Chúng khác checkpoint dự báo dùng để chấm điểm toàn bộ test về phạm vi học hoặc tập SKU. Vì vậy RMSE 13,09 của dự báo toàn phạm vi không phải sai số của checkpoint dùng trong định giá. [Giao thức định giá](../experiments/PROTOCOL_V2.md#5-giao-thức-định-giá-m5), [cấu hình](../experiments/phase5_settings_v2.json).

| Chính sách | Doanh thu TB Gaussian | Doanh thu TB lag-adjusted | Uplift ghép seed so với giá cận trên (Gaussian / lag) |
| --- | ---: | ---: | ---: |
| Giá trung bình train | 129.786,45 | 110.789,43 | −5,23% / −6,32% |
| Giá cận trên train | 136.911,80 | 118.192,39 | 0% / 0% |
| DDPG | **139.541,53** | **118.763,54** | **+1,89% / +0,41%** |
| Lưới giá Song | 138.417,71 | 118.557,57 | +1,03% / +0,28% |
| PPO cập nhật + gradient | 127.279,74 | 109.589,28 | −6,81% / −7,40% |

- **DDPG.** 1.194/2.000 quyết định gần bằng giá cận trên train. Cả 100 tổ hợp SKU–seed–simulator đều có biên độ giá trong 20 tuần nhỏ hơn 0,01 USD. Với lag-adjusted seed 123, DDPG thấp hơn mốc cận trên 1,78%. Phần hơn trung bình cho thấy lợi thế chọn mức giá theo SKU trong môi trường mô phỏng. Khả năng điều chỉnh thực sự qua tuần chưa được chứng minh.
- **Ablation PPO.** Phương án có cả gradient giá và cập nhật Transformer dẫn trung bình trong bốn biến thể PPO ở cả hai simulator. So với PPO đóng băng, mức tăng ghép seed chỉ **0,14%** trên Gaussian và **0,12%** trên lag-adjusted. Gaussian chỉ có 2/5 seed cho chênh lệch dương. Thí nghiệm xác nhận vòng phản hồi hoạt động với 20 lần cập nhật Transformer trong mỗi lượt test có phản hồi, nhưng chưa thấy mức tăng ổn định qua hai simulator. Các giá trị này là doanh thu mô phỏng `giá × lượng bán mô phỏng`. M5 không có lượng bán phản thực tế ở các giá ứng viên và dự án chưa có giá vốn, nên các bảng chỉ đánh giá doanh thu trong simulator. [Bảng từng seed](../experiments/artifacts/phase6_v2/pricing_test_runs.csv).

## Đối chiếu với bài báo Song

| Bảng | Số liệu hoặc giao thức của Song | Kết quả tái dựng V2 và mức so sánh |
| --- | --- | --- |
| Table 1 · Dự báo lượng bán | Transformer của tác giả đạt RMSE 15,7 / 22,3, MAPE 8,2% / 10,9% trên UCI / M5. | Cùng hai nguồn, tần suất tuần và tỷ lệ chia thời gian gần 70/15/15. V2 dùng khóa test riêng, giữ ngoại lệ hợp lệ, tính RMSE trên số lượng gộp và MAPE theo từng khóa dương. Song mô tả chuẩn hóa từng mặt hàng cho M5 và diễn giải thang UCI khác nhau giữa §4.1.3 và §4.2. Vì vậy chỉ đối chiếu phương pháp và xu hướng. |
| Table 2 · Vùng giá | Song báo RMSE mô hình đầy đủ thấp/trung bình/cao là 16,2 / 20,5 / 25,8. | V2 dùng phân vị giá train cho từng dataset. Trên M5, Transformer theo Song đạt 19,24 / 8,43 / 8,29 sản phẩm/tuần, tức sai số cao nhất ở nhóm giá thấp. Thiếu khóa test và ngưỡng chính xác của bài nên không quy chênh lệch về kiến trúc mô hình. |
| Table 3 · Doanh thu tích lũy | Song báo 883 cho mô hình đầy đủ, 841 cho PPO đóng băng và 610 cho giá trung bình lịch sử, trên thang chuẩn hóa của simulator. | Cùng so sánh 200 quyết định mỗi lượt, năm seed và các baseline tương ứng. V2 chọn một SKU mỗi cửa hàng, bổ sung mốc cận trên và giữ đơn vị thô. Cohort, cách tạo lượng bán và đơn vị doanh thu khác nhau, nên không so trực tiếp 883 với doanh thu V2. |
| Table 6 · Ablation qua seed | Bốn cấu hình dùng seed 42, 123, 456, 789, 1024. Ví dụ, Song báo trung bình 883 cho phương án đầy đủ và 841 khi đóng băng Transformer. | V2 tái hiện bốn cấu hình và cùng danh sách seed. Kết quả của nhóm cho thấy lợi ích trung bình nhỏ so với đóng băng. Bảng từng seed ghi theo hai simulator riêng. |

Bài báo mô tả hai nguồn và cách chia, nhưng không công bố danh sách khóa `(SKU, tuần)` của test hoặc code thực nghiệm đủ để đối chiếu từng khóa. V2 đã tái dựng Transformer có đầu vào giá, gradient giá, PPO giá liên tục và cập nhật Transformer từ lượng bán mô phỏng mỗi 10 tương tác. TFT, Informer và Informer + giá chưa có lượt V2. Dòng **Transformer theo Song** ở bảng dự báo là bộ dự báo lượng bán, còn **PPO cập nhật + gradient** ở bảng định giá thể hiện vòng lặp phối hợp. [Song, 2026, §3.5, §4.1–4.3 và Table 6](https://link.springer.com/article/10.1007/s10791-026-10745-2).

## Đối chiếu nghiên cứu khác dùng cùng nguồn

| Nghiên cứu | Thiết kế đánh giá được công bố | Điều kiện để đối chiếu trực tiếp |
| --- | --- | --- |
| Bhavikatta và Goluguri, UCI Online Retail 352 | Gộp dữ liệu ở nhiều mức thời gian và nêu tỷ lệ chia 80/10/10 theo thời gian cho nhiệm vụ chuỗi. Thử các biến thể LSTM trong nghiên cứu nhiều dataset. | Cần xác định cấp gộp và khóa test của riêng UCI, quy tắc ngoại lệ, mục tiêu và mẫu số từng chỉ số trước khi ghép bảng. [Bài báo, §3–4](https://link.springer.com/article/10.1007/s43621-026-02828-3). |
| Makridakis, Spiliotis và Assimakopoulos, M5 Accuracy | Dự báo 28 ngày, xếp hạng bằng WRMSSE trên 12 cấp phân cấp. | Cần benchmark bổ sung theo đúng 28 ngày, cùng phân cấp và trọng số của cuộc thi. Điểm RMSE SKU–tuần V2 dùng cho câu hỏi nghiên cứu về giá và lượng bán theo tuần. [Bài báo M5 Accuracy](https://doi.org/10.1016/j.ijforecast.2021.11.013). |

## Phương án bàn giao cho Phase 7

Với UCI, dùng mốc tuần trước để hiển thị dự báo lượng bán và kèm ARIMA + fallback trong bảng đối chiếu. Với M5, tách mốc dự báo lượng bán tại giá đã biết (ARIMA + fallback) khỏi tính năng thử giá. Khi người dùng chọn một SKU và một tuần, tính `giá ứng viên × lượng bán dự báo theo giá đó` trên lưới 41 mức trong miền giá train, có thêm mức giá hiện tại khi hợp lệ. Cho thấy giá được chọn, mức giá cận trên, lượng bán dự báo và doanh thu dự kiến để người dùng giải thích quyết định. DDPG và PPO có thể xuất hiện ở phần so sánh mô phỏng. Điều này giữ được mục tiêu trả ra một giá đề xuất có thể giải thích, với chất lượng còn phụ thuộc vào dữ liệu giá của SKU và độ chính xác của mô hình cầu.
