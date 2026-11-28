# Giao thức Phase 0 cho thí nghiệm toàn phạm vi

**Trạng thái:** nhóm đã cho phép triển khai Phase 1 ngày 2026-11-20; các quy tắc thí nghiệm V2 vẫn được khóa. Tài liệu này thay thế [giao thức pilot cũ](PROTOCOL.md) cho các lượt chạy mới. Nguồn quyết định là [thiết kế cuối](../docs/dynamic-pricing-final-design.md); các giá trị cố định để chương trình đọc nằm trong [protocol_v2.json](protocol_v2.json), bằng chứng kiểm tra tệp ở [biên bản Phase 0](../docs/phase0-audit.md), và bố cục kết quả ở [mẫu báo cáo](../docs/report-template-v2.md).

## 1. Câu hỏi nghiên cứu và nguồn dữ liệu

- **UCI Online Retail ID 352 (2010–2011):** đánh giá dự báo lượng bán sản phẩm/tuần. Dùng `Online Retail.xlsx` từ UCI chính thức. Không gộp với Online Retail II 2009–2011 của demo cũ.
- **M5 Forecasting Accuracy:** đánh giá dự báo sản phẩm–cửa hàng/tuần và chọn giá. Bản chính dùng `sales_train_evaluation.csv` (1.941 ngày), `sell_prices.csv`, `calendar.csv`, đủ 10 cửa hàng. `sales_train_validation.csv` (1.913 ngày) chỉ thuộc pilot lịch sử; không gộp điểm với bản chính.
- UCI và M5 được xử lý, chuẩn hóa, train và lưu checkpoint **riêng**. So mô hình **trong từng dataset** trên cùng khóa SKU–tuần test; không so trực tiếp RMSE UCI với RMSE M5.
- Bản gốc Song mô tả hai dataset cho dự báo và chỉ M5 cho thí nghiệm PPO định giá. Bài không cung cấp đầy đủ cohort/test và công thức chuẩn hóa doanh thu; báo cáo phải phân biệt tái dựng phương pháp với tái tạo đúng số công bố.

Nguồn và SHA256 của bốn tệp đầu vào đã được kiểm tra lại trong `data/song_replication/download_manifest.json`; xem biên bản Phase 0. Không đưa file dữ liệu gốc vào Git.

## 2. Ranh giới thời gian và đơn vị nhãn

- Một hàng nhãn dự báo là `(dataset, sku, week)`; với M5, `sku` chứa cả `item_id@store_id`. Tuần bắt đầu **thứ Hai**. UCI cộng lượng bán từ giao dịch hợp lệ trong tuần; M5 cộng lượng bán ngày trong tuần đủ bảy ngày. Giá tuần theo quy tắc nguồn đã ghi trong manifest. Dữ liệu thô được giữ riêng để truy nguyên.
- Mỗi dataset chia theo **cùng các tuần lịch cho mọi SKU**: `floor(70% số tuần)` đầu cho train, tiếp đến `floor(85% số tuần) - floor(70%)` cho validation, phần còn lại cho test. Không xáo trộn dòng và không tính ranh giới từ số hàng còn lại sau lọc.
- UCI hiện có 54 tuần: 37 train (đến 2011-08-08), 8 validation (2011-08-15 đến 2011-10-03), 9 test (từ 2011-10-10). M5 bản chính có 277 tuần: 193 train (đến 2014-10-06), 42 validation (2014-10-13 đến 2015-07-27), 42 test (từ 2015-08-03). Các mốc này cố định theo nguồn đã khóa.
- Bài toán dự báo là **một tuần tới theo thời gian cuốn chiếu** với 12 tuần quá khứ. Tại tuần `t`, các tuần test trước `t` đã xảy ra có thể làm lịch sử; lượng bán của `t` hoặc tương lai không thể đi vào đặc trưng. Giá tuần `t` là giá điều kiện/giá ứng viên cho mô hình cầu, không phải nhãn giá cần dự báo.

## 3. Quy tắc làm sạch và bảo vệ test

1. Lưu raw và SHA256. Xác định mốc chia từ lịch nguồn trước khi fit bất kỳ thống kê nào. Quy tắc hợp lệ cố định được áp dụng nhất quán cho mọi split: kiểu dữ liệu/ngày, hóa đơn hủy hoặc trả hàng UCI, lượng bán/giá không hợp lệ, và tuần M5 không đủ ngày/giá. Tuần M5 có giá đủ bảy ngày và bán 0 là nhãn hợp lệ.
2. **Chỉ train được dùng để fit:** ngưỡng 3×IQR theo SKU, điều kiện chọn SKU, trung bình/độ lệch chuẩn, miền giá, và tham số điền khuyết. Validation chỉ chọn siêu tham số/mô hình; test chỉ chấm cuối.
3. Điểm test chính giữ **mọi lượng bán hợp lệ sau quy tắc cố định**. Không xóa giao dịch hoặc tuần validation/test chỉ vì vượt IQR train. Có thể xuất thêm bảng phụ theo lọc 3×IQR đã fit từ train để đối chiếu Song, nhưng phải ghi số dòng/tuần bị loại và không thay bảng chính.
4. UCI chọn mọi SKU có ít nhất hai tuần bán dương được quan sát ở train; hiện có 3.387 SKU. Tuần UCI không có giao dịch là ngữ cảnh với lượng bán 0 nhưng không được chấm như nhãn cầu xác nhận. M5 giữ mọi chuỗi có ít nhất hai tuần train đủ giá; hiện có 29.069 trên 30.490 chuỗi gốc. Số hàng test UCI và hash khóa test **chưa chốt** vì Phase 1 phải tạo lại panel sau khi bỏ việc xóa ngoại lệ test.
5. Cửa sổ 12 tuần, giá điền tiến và đặc trưng lịch chỉ dùng thông tin có sẵn đến thời điểm quyết định. Không điền ngược từ tương lai. Mọi mô hình trong bảng chính phải trả dự báo cho đúng cùng tập khóa test sau Phase 1; lưu hash của danh sách khóa đã sắp xếp. Nếu thiếu khóa, lượt đó là **chưa đầy đủ**, không được xếp cạnh mô hình đủ phạm vi.

## 4. Mô hình dự báo và phạm vi so sánh

- Bản Song giữ cấu hình 12 tuần, Transformer 4 lớp/8 heads/kích thước 128, Adam `1e-3`, MSE trên lượng bán chuẩn hóa; mô hình có giá trả `Q̂(s,p)` và `∂Q̂/∂p`. Cấu hình này không tune lại khi gọi là bản tái dựng Song.
- Bảng chính đặt ARIMA, XGBoost, LSTM, Transformer và các mô hình hoàn thành khác trên **toàn cohort train-eligible của từng dataset**, chấm chính cùng khóa test. TFT/Informer và biến thể có giá cần bổ sung để đối chiếu đủ Table 1 Song; nếu chưa chạy xong thì đánh dấu chưa có, không thay bằng số pilot 100/20 SKU.
- ARIMA hiện đòi ít nhất 12 tuần train có quan sát, trong khi UCI nhận SKU từ hai tuần. Để giữ bảng toàn phạm vi, phương pháp chính sẽ được ghi tên **`ARIMA + fallback chuỗi thưa`**: ARIMA khi đủ lịch sử; dùng giá trị lượng bán đã biết gần nhất trước tuần dự báo khi dưới 12 quan sát hoặc fit lỗi. Quy tắc fallback phải cố định từ train/validation, áp dụng trước khi xem test, ghi số SKU/hàng dùng fallback. ARIMA thuần chỉ được báo ở bảng phụ nếu toàn bộ mô hình trong bảng phụ dùng cùng cohort đủ điều kiện 12 tuần.
- Cấu hình cải tiến và siêu tham số được chọn theo validation với ngân sách công bố trước. Mô hình dự báo để chấm test có thể refit bằng train+validation **sau khi** khóa siêu tham số; checkpoint định giá Song chỉ học train (validation chọn epoch) và được lưu riêng. Không trao đổi hai checkpoint này hoặc gán điểm test của checkpoint này cho checkpoint kia.
- Seed của mô hình ngẫu nhiên: `42, 123, 456, 789, 1024`; báo trung bình và độ lệch chuẩn mẫu qua 5 seed khi đủ. ARIMA và các baseline xác định báo một lượt, không tạo độ lệch chuẩn giả.

**Điểm chính trong từng dataset:** micro RMSE `sqrt(mean((ŷ−y)²))` và micro MAE `mean(|ŷ−y|)` ở đơn vị sản phẩm/tuần trên cùng khóa test. Báo thêm WAPE `100 × Σ|ŷ−y|/Σy`, MAPE `100 × mean(|ŷ−y|/y)` chỉ trên nhãn `y>0` kèm số hàng dương, và R² gộp `1−Σ(y−ŷ)²/Σ(y−mean(y_test))²`. Nếu mẫu số WAPE/R² bằng 0 thì ghi không xác định, không ép về 0. RMSE chuẩn hóa chia cho trung bình và độ lệch chuẩn lượng bán **train** của cohort, ghi không xác định khi mẫu số bằng 0. Báo thêm kết quả theo SKU và nhóm giá. Chọn siêu tham số dự báo bằng **validation micro RMSE**, kiểm tra WAPE và phản ứng giá trước khi dùng mô hình cho định giá. Không dùng test để chọn cấu hình.

**Ba nhóm giá cho bảng tương ứng Table 2:** trong từng dataset, tính hai ngưỡng phân vị 1/3 và 2/3 từ các giá tuần hợp lệ **train** của cohort đã khóa. Giá test `p ≤ q1` thuộc nhóm thấp, `q1 < p ≤ q2` thuộc nhóm giữa, `p > q2` thuộc nhóm cao. Ghi ngưỡng, số khóa test và RMSE từng nhóm; nếu giá trùng nhiều, ba nhóm có thể không cân bằng. Đây là quy tắc của nhóm vì bài Song không công bố ngưỡng chia chính xác.

## 5. Giao thức định giá M5

- **Bản Song chính:** simulator Gaussian kernel với băng thông Silverman fit từ cặp giá–lượng bán train, giá liên tục trong miền giá quan sát train. Lag-adjusted kernel hiện có là phân tích độ nhạy, không gộp với số bản chính. Simulator được kiểm tra/hiệu chỉnh trên validation rồi đóng băng trước test; nó không xác nhận quan hệ nhân quả ở giá chưa từng bán.
- PPO Song dùng dự báo lượng bán và gradient giá của Transformer làm đầu vào; action là giá trong miền train, reward là doanh thu mô phỏng `p × q`. Transformer fine-tune mỗi 10 tương tác từ tối đa 200 phản hồi gần nhất, batch 32; PPO học từ rollout. Lưu cặp checkpoint PPO–Transformer cùng trạng thái.
- **Cohort 200 chu kỳ đề xuất:** từ mỗi trong 10 cửa hàng chọn **một** chuỗi xếp hạng cao nhất theo tổng lượng bán train, có ít nhất hai mức giá train và đủ lịch sử 12 tuần train; chốt thứ tự bằng `sku` khi hòa. Lấy 20 tuần test lịch đầu tiên của mỗi chuỗi, tổng 10 × 20 = 200 quyết định, chạy 5 seed trên cùng cohort. Không dùng lượng bán hay kết quả giá ở test để chọn chuỗi. Nếu chuỗi được chọn không thể tạo đủ 20 quyết định hợp lệ, Phase 1 phải báo lỗi/gap để nhóm xem lại, không âm thầm thay SKU bằng kết quả test.
- Mọi chính sách giá dùng cùng cohort, miền giá, simulator, ngân sách tương tác, thứ tự tuần và nhiễu theo seed. So giá cố định, heuristic, DDPG, SAC, PPO đóng băng, PPO có phản hồi, các phép bỏ gradient/cập nhật; các cách lưới giá của nhóm báo riêng. Chọn phương án cuối bằng validation, xác nhận một lần trên test.
- Trong test, không dùng lượng bán thật của tuần đang quyết định làm phản hồi cho giá PPO. Chỉ dùng phản hồi **mô phỏng** của các quyết định trước trong chuỗi thời gian; nhãn test thật tại giá quan sát chỉ dùng cho bảng dự báo tại giá lịch sử.

Chỉ số giá: giá từng quyết định, lượng bán và doanh thu mô phỏng, doanh thu tích lũy, mean/std/min/max qua 5 seed, uplift so với giá cố định và PPO đóng băng, regret theo oracle của cùng simulator. Doanh thu 883 của Song là số simulator chuẩn hóa, không so trực tiếp với doanh thu thô của nhóm. Không gọi đây là lợi nhuận hoặc doanh thu thực.

## 6. Điều kiện hoàn tất Phase 0 và việc chưa được làm

- Nguồn, SHA256, dataset, ranh giới tuần, cohort rule, metric, seed, fit scope, cách xử lý test và các giới hạn so bài Song được ghi vào tài liệu/config và đối chiếu với artifact thực tế.
- Phase 1 sẽ tạo panel test UCI giữ ngoại lệ hợp lệ, tính lại số hàng/hash khóa test, kiểm tra ARIMA fallback và tạo danh sách cohort M5 theo quy tắc train-only. Những giá trị này **không được giả là đã chốt ở Phase 0**.
- Phase 0 không sửa hàm tiền xử lý, không train mô hình, không chạy benchmark và không đưa kết quả pilot cũ vào bảng toàn phạm vi.

## 7. Kết quả khóa dữ liệu Phase 1

Sau khi nhóm cho phép Phase 1, đã tạo lại panel UCI và đủ 10 panel M5 trong `experiments/artifacts/phase1_v2/`. Tệp cấu hình đi kèm đã được điền số khóa/hash test và danh sách cohort M5; lệnh xác minh sẽ báo lỗi nếu artifact khác các giá trị đó. Xem [biên bản Phase 1](../docs/phase1-audit.md) để kiểm tra số hàng, SHA256, chính sách feature và cách chạy lại. Chưa có chỉ số mô hình hoặc doanh thu chính sách mới ở bước này.
