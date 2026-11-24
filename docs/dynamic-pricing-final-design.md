# Thiết kế cuối: tái hiện Song và xây dựng hệ thống dynamic pricing

Ngày chốt bản thiết kế: 2026-11-20. Đây là đặc tả để triển khai theo từng phase; tài liệu này không phải kết quả đã chạy. Bài gốc: [Song (2026)](https://link.springer.com/article/10.1007/s10791-026-10745-2). Sau mỗi phase, cập nhật README với mục đích, đầu vào, đầu ra, lệnh chạy lại và giới hạn kết quả; chờ nhóm duyệt trước khi sang phase kế tiếp.

## 1. Mục tiêu và hai tầng sản phẩm

1. Tái dựng phương pháp Song: Transformer dự báo nhu cầu có điều kiện theo giá và đạo hàm theo giá; PPO chọn giá liên tục; phản hồi nhu cầu cập nhật Transformer và PPO trong môi trường mô phỏng.
2. Trên cùng giao thức của nhóm, so sánh các mô hình dự báo lượng bán và các chính sách giá, chọn cấu hình cuối bằng validation rồi xác nhận trên test.
3. Tạo bảng số đo và biểu đồ, đồng thời đối chiếu các bảng của Song. Tách rõ số có thể so trong cùng thí nghiệm nhóm với số chỉ có thể đối chiếu phương pháp do thiếu mã nguồn, danh sách chuỗi và cách chuẩn hóa của tác giả.
4. Đóng gói phương án đã chọn thành hệ thống chạy lại được. Thư mục hệ thống cuối chỉ dùng cấu hình và siêu tham số cố định; code tune, benchmark và tạo báo cáo nằm ở thư mục nghiên cứu.

Giữ nguyên các script gốc và demo Online Retail II để nhóm có thể đối chiếu lịch sử dự án. Module nghiên cứu và hệ thống cuối tái sử dụng hoặc gọi các thành phần phù hợp; không ghi đè code gốc chỉ để sắp lại thư mục.

Mục tiêu kinh doanh là **tối đa doanh thu** `giá × lượng bán`. Dữ liệu không có giá vốn hoặc tồn kho đáng tin cậy nên không gọi kết quả là tối ưu lợi nhuận. Doanh thu ở giá chưa áp dụng là kết quả **mô phỏng**, không phải doanh thu thực hoặc bằng chứng tác động nhân quả.

## 2. Dữ liệu và phạm vi

| Bộ dữ liệu | Vai trò | Phạm vi mục tiêu | Mô hình lưu riêng |
| --- | --- | --- | --- |
| UCI Online Retail ID 352, 2010–2011 | So sánh **dự báo lượng bán** | Toàn bộ SKU đủ điều kiện xác định từ train; panel hiện chuẩn bị 3.387 SKU | Checkpoint và scaler UCI |
| M5 Forecasting Accuracy | So sánh dự báo và **định giá PPO** | Đủ 10 cửa hàng; panel hiện có 29.069 chuỗi đủ điều kiện từ 30.490 chuỗi gốc | Checkpoint, scaler và chính sách giá M5 |
| Online Retail II, 2009–2011 | Demo cũ của dự án | Giữ riêng khỏi bảng Song | Mô hình demo cũ |

Không gộp UCI và M5 thành một tập train hoặc một checkpoint. Mỗi bộ có đơn vị giá, sản phẩm và đặc trưng riêng. Chạy cùng **quy tắc đánh giá** trên hai bộ, nhưng báo hai bảng riêng; không xếp hạng UCI với M5 bằng RMSE thô.

Chọn `sales_train_evaluation.csv` (1.941 ngày) làm nguồn M5 chính cho lượt toàn phạm vi vì đây là bản công khai có toàn bộ ngày bán đã gắn nhãn. Pilot cũ dùng `sales_train_validation.csv` (1.913 ngày) chỉ là kiểm tra kỹ thuật; kết quả của hai bản không gộp. Trước khi chạy chính thức, ghi mã băm nguồn, số tuần và ranh giới thời gian vào manifest. Bài Song không nêu đủ phiên bản/tập chuỗi để khẳng định tập test của nhóm trùng tác giả.

**Toàn phạm vi** có nghĩa là mọi SKU/chuỗi đủ điều kiện theo train, sau quy tắc hợp lệ đã công bố. Không coi mọi dòng hóa đơn thô hoặc chuỗi M5 thiếu lịch sử/giá là nhãn huấn luyện hợp lệ. Dự báo toàn phạm vi là mục tiêu bắt buộc; định giá trên M5 trước tiên theo 200 chu kỳ × 5 seed của bài, sau đó có thể đánh giá rộng hơn theo cửa hàng/chuỗi như thí nghiệm mở rộng riêng nếu tài nguyên cho phép.

## 3. Giao thức dữ liệu, làm sạch và tập test

1. Lưu nguyên tệp nguồn và mã băm. Ấn định ranh giới **70% tuần đầu train, 15% validation, 15% cuối test** theo thời gian, không xáo trộn dòng. Ranh giới tính từ lịch nguồn, không từ số dòng còn lại sau lọc.
2. Áp dụng cùng các quy tắc hợp lệ **không học tham số** trên mọi giai đoạn: kiểu dữ liệu/ngày, hóa đơn hủy hoặc trả hàng của UCI, giá không hợp lệ; với M5, chỉ tuần có đủ ngày và xác định rõ tuần có đủ giá. Gộp UCI từ giao dịch và M5 từ ngày thành nhãn **sản phẩm–tuần**. Lượng bán tuần là tổng đơn vị; giá tuần theo quy tắc đã chốt và được ghi trong manifest. Tuần M5 bán 0 nhưng có giá đầy đủ vẫn là nhãn hợp lệ.
3. **Chỉ fit trên train:** ngưỡng 3×IQR, điều kiện chọn SKU, trung bình/độ lệch chuẩn từng SKU, miền giá có thể chọn, mọi tham số điền khuyết và tham số simulator. Validation dùng để chọn siêu tham số; test không tham gia các bước này.
4. Ngoại lệ lượng bán của validation/test **không bị xóa chỉ vì vượt ngưỡng IQR train** trong điểm chính. Giữ nhãn tuần hợp lệ sau quy tắc cố định. Nếu muốn đối chiếu cách lọc ngoại lệ của Song, xuất thêm điểm trên một tập đã lọc bằng ngưỡng train, kèm số giao dịch/tuần bị loại; không thay thế điểm test chính.
5. Tạo đặc trưng từ 12 tuần trước thời điểm dự báo. Điền giá lịch sử chỉ theo chiều tiến thời gian; không dùng giá/lượng bán từ tuần tương lai. UCI thiếu giao dịch trong một tuần có thể tạo lượng bán 0 làm **ngữ cảnh**, nhưng không tự coi đó là nhãn cầu xác nhận để chấm điểm chính. M5 có tuần bán 0 được chấm nếu giá đủ 7 ngày.
6. Chấm dự báo **cuốn chiếu một tuần**: tại tuần `t`, chỉ thông tin đã đến trước `t` được dùng làm lịch sử. Nếu sử dụng lượng bán của các tuần test trước đó, đánh dấu đây là đánh giá online một bước, không phải dự báo toàn bộ test từ một thời điểm duy nhất. Giá tuần đang xét là giá điều kiện/giá ứng viên được cho trước, không phải đầu ra của mô hình dự báo lượng bán.

Đặc biệt, code UCI hiện tính IQR từ train nhưng áp dụng xóa giao dịch ngoại lệ cả ở test trước khi gộp tuần. Việc fit ngưỡng này không rò rỉ tham số test, song làm thay đổi nhãn đánh giá; phải sửa trước lượt benchmark UCI toàn phạm vi. M5 hiện chưa có lọc 3×IQR. Biến `promotion_proxy` của M5 hiện chưa nối với `promotion_flag` mà Song Transformer đọc; phải kiểm tra và quyết định rõ giữ bằng 0 hay dùng proxy như một giả định mới, rồi ghi vào báo cáo.

## 4. Mô hình dự báo lượng bán

### 4.1 Bản Song cố định

- Train **một Transformer riêng cho UCI và một Transformer riêng cho M5** trên giá quan sát, lịch sử 12 tuần và đặc trưng lịch có sẵn. Giữ cấu hình bài báo làm mốc: 4 lớp, 8 attention heads, kích thước 128, Adam `1e-3`, MSE lượng bán chuẩn hóa.
- Tại giá ứng viên `p`, mô hình trả lượng bán dự báo `Q̂(s,p)` và đạo hàm `∂Q̂/∂p` bằng tự động vi phân. Đạo hàm là độ nhạy **của mô hình**, chưa xác nhận quan hệ nhân quả ngoài đời.
- Lưu checkpoint cùng scaler train, schema đặc trưng, nguồn/split, seed và cấu hình. Checkpoint dùng cho định giá không được học nhãn test.

### 4.2 Baseline và bản cải tiến của nhóm

- Baseline để đối chiếu Song: ARIMA, LSTM, Transformer, TFT, Informer và các biến thể có giá được bài nêu. XGBoost, cách thử lưới giá và mô hình bổ sung là **phương án của nhóm**, không gán cho Song. Nếu một baseline quá nặng trên máy nhóm, ghi thiếu sót rõ thay vì điền số thay thế.
- Giữ cùng SKU–tuần test, nhãn và thông tin sẵn có tại thời điểm dự báo. Công bố rõ baseline nào có giá làm đầu vào. ARIMA không có giá ứng viên không được thay trực tiếp thành mô hình cầu theo giá của PPO.
- **Bảng so sánh chính trên mỗi dataset dùng toàn phạm vi đủ điều kiện cho từng phương pháp:** UCI hiện có 3.387 SKU; M5 hiện có 29.069 chuỗi ở 10 cửa hàng. ARIMA, XGBoost, LSTM, Transformer và các baseline khác phải được chấm trên **chính cùng tập khóa `(dataset, SKU, tuần test)`**. Mỗi phương pháp được huấn luyện từ phần train tương ứng của toàn cohort, dù cách huấn luyện có thể khác (ví dụ ARIMA theo từng chuỗi, XGBoost dùng nhiều chuỗi chung). Không đặt RMSE của mô hình chạy 100 SKU bên cạnh RMSE của mô hình chạy toàn phạm vi để kết luận hơn/kém.
- Nếu một phương pháp chưa hoàn tất toàn phạm vi hoặc không dự báo được một số khóa test, đánh dấu kết quả **chưa đầy đủ** và không đưa vào bảng chính. Có thể tạo bảng phụ trên một cohort chung được khóa trước, nhưng tất cả phương pháp trong bảng phụ cũng phải chấm đúng cùng hàng; ghi rõ đây không phải kết quả toàn dữ liệu. Ưu tiên chạy theo cửa hàng/SKU/seed, lưu tiến độ và tối ưu bộ nhớ trước khi thu hẹp phạm vi.
- Giữ bản Song với siêu tham số cố định. Mỗi phương án cải tiến dùng tập ứng viên và ngân sách tune xác định trước, chọn bằng **validation**, sau đó khóa cấu hình trước khi xem bảng test cuối. Ghi mọi thử nghiệm, thời gian chạy và seed.
- Chọn mô hình cầu cho định giá dựa trên sai số tại giá đã quan sát **và** kiểm tra phản ứng khi đổi giá trong miền train; RMSE thấp nhất một mình không bảo đảm chọn giá tốt.

Chỉ số dự báo: số hàng được chấm, MAE, RMSE, MAPE trên nhãn dương (ghi mẫu số), WAPE và điểm theo SKU/nhóm giá. Xuất cả thang đơn vị sản phẩm/tuần và thang chuẩn hóa theo train. Mô hình ngẫu nhiên báo trung bình ± độ lệch chuẩn qua 5 seed; mô hình xác định ghi rõ không có biến thiên seed. Không so số UCI với M5 nếu đơn vị/chuẩn hóa khác nhau.

## 5. Môi trường và thuật toán chọn giá trên M5

### 5.1 Tái hiện Song trước

1. Fit simulator nhu cầu–giá bằng các cặp quan sát **train**, dùng hồi quy kernel Gaussian và miền giá quan sát ở train. Chọn/kiểm tra cấu hình bằng validation rồi đóng băng. Kiểm tra sai số tại giá quan sát, độ dốc và độ nhạy với giả định simulator; không coi dự báo phản thực tế là nhãn thật.
2. PPO nhận lịch sử, giá hiện tại, `Q̂` và `∂Q̂/∂p`; action liên tục được ánh xạ thành giá trong miền đã chốt. Reward là `giá × lượng bán mô phỏng` (chuẩn hóa reward phục vụ huấn luyện phải được ghi riêng với đơn vị doanh thu báo cáo).
3. Giữ cấu hình Song: actor/critic 3 lớp 256 ReLU; PPO `clip=0.2`, `gamma=0.99`, `GAE lambda=0.95`, learning rate `3e-4`, batch 64. Sau mỗi 10 tương tác, fine-tune Transformer từ tối đa 200 phản hồi gần nhất, batch 32. PPO tiếp tục học từ rollout. Lưu **cặp** PPO và Transformer đúng cùng trạng thái huấn luyện.
4. Chạy giá cố định, heuristic, DDPG, SAC, PPO với Transformer đóng băng, PPO cập nhật và các phép bỏ gradient/cập nhật. Mỗi chính sách cùng SKU–tuần, miền giá, simulator, ngân sách tương tác và seed. Bài Song dùng 200 chu kỳ × 5 seed; danh sách chuỗi của nhóm phải chốt trước và công bố.

Trong đánh giá giá theo thời gian, lượng bán thật của tuần test đang xét được giữ kín. Phản hồi cho giá do PPO chọn đến từ simulator đã fit trên train; chỉ phản hồi từ **các quyết định trước đó** mới có thể đi vào trạng thái và lần cập nhật kế tiếp. Không lấy lượng bán thật ở giá lịch sử của tuần test làm nhãn cầu cho mức giá PPO chưa áp dụng.

### 5.2 Thí nghiệm cải tiến

Thử các mô hình cầu **nhận giá ứng viên** và thuật toán giá khác, trong đó có grid search của nhóm. Khi đổi mô hình cầu trong trạng thái PPO phải **huấn luyện lại PPO**, không lắp checkpoint cầu mới vào chính sách đã học với Transformer cũ. Tune chính sách trên validation, chọn phương án cuối trước khi chấm test. Dùng cùng simulator và nhiễu ngẫu nhiên theo cặp để so công bằng; thử simulator thay thế chỉ là phân tích độ bền, báo riêng.

Chỉ số định giá: giá từng quyết định, lượng bán và doanh thu mô phỏng, doanh thu tích lũy qua thời gian, mean/std/min/max qua 5 seed, mức tăng so với giá cố định và PPO đóng băng, regret so với oracle **của simulator**, tỉ lệ quyết định chạm biên giá, và độ ổn định theo SKU/nhóm giá. Chỉ tính lợi nhuận khi có giá vốn hợp lệ.

## 6. Tập test cuối, bảng bài báo và biểu đồ

- Khóa nguồn, cohort, split, feature, simulator, siêu tham số và quy tắc chọn mô hình **trước** khi chạy bảng test cuối. Không tune lại sau khi nhìn số test; nếu nghiên cứu thêm, coi đó là thí nghiệm mới và ghi rõ.
- Bảng dự báo riêng UCI và M5: phương pháp, số SKU–tuần, MAE/RMSE/MAPE/WAPE, RMSE std, thang đo và seed. Bảng nhóm giá thấp/vừa/cao dùng ngưỡng học từ train.
- Bảng định giá M5: kết quả từng seed và tổng hợp mean/std/min/max, uplift, regret; có baseline cố định, PPO đóng băng và phép bỏ thành phần.
- Biểu đồ: lượng bán thật–dự báo theo thời gian; sai số theo mô hình/nhóm giá; dự báo lượng bán và doanh thu theo giá ứng viên; đường giá PPO; doanh thu tích lũy và khoảng biến thiên qua seed; tác động của cập nhật Transformer và gradient.
- Bảng đối chiếu Song phân biệt **cùng định nghĩa và phạm vi** với **chỉ giống phương pháp**. Có thể đối chiếu cấu trúc Table 1, 2, 3, 5, 6; không ép RMSE hay doanh thu của nhóm trùng số công bố. Doanh thu 883 của Song là số simulator chuẩn hóa, không so trực tiếp với doanh thu thô của nhóm. Bài không công bố đủ mã nguồn, cohort và chuẩn hóa để xác nhận cùng test.
- Bài Song chỉ mô tả thí nghiệm PPO định giá trên M5 nhưng chưa giải thích đầy đủ cách tạo kết quả `Ours` có phản hồi cho UCI trong Table 1. Báo cáo UCI của nhóm phải ghi rõ đây là phần dự báo trên UCI, không tuyên bố đã tái hiện đúng cơ chế phản hồi UCI của tác giả.

## 7. Phân tách code nghiên cứu và hệ thống cuối

### Code nghiên cứu

Giữ trong `experiments/` hoặc một package nghiên cứu kế thừa, gồm chuẩn bị dữ liệu, tune trên validation, benchmark, kiểm tra simulator, so sánh chính sách, vẽ biểu đồ và báo cáo. Mỗi lần chạy lưu manifest, mã băm dữ liệu, cấu hình, seed, dự đoán/giá từng hàng và metric tổng hợp. Các lượt dài chạy theo cửa hàng, mô hình và seed; lưu artifact hoàn tất để tiếp tục sau khi gián đoạn. Không đánh dấu “toàn M5” nếu thiếu cửa hàng hoặc mô hình trong bảng.

### Thư mục hệ thống cuối

Sau khi chọn phương pháp, tạo một thư mục riêng (dự kiến `final_system/`) chỉ gồm:

- Cấu hình cố định cho nguồn UCI và M5, feature schema, cách xử lý dữ liệu và ranh giới train/validation/test đã chốt.
- Lệnh chuẩn bị dữ liệu và **train lại bằng siêu tham số cố định**; không có grid search, tune hoặc benchmark trong luồng chạy cuối.
- Checkpoint đã chọn hoặc hướng dẫn tạo lại, cùng scaler và metadata của checkpoint.
- Engine định giá **đã chọn** trên M5. API nạp checkpoint để trả dự báo lượng bán cho UCI/M5 và trả giá đề xuất, lượng bán dự báo, doanh thu kỳ vọng, miền giá hợp lệ, tên phương pháp/phiên bản mô hình cho M5.
- README cài đặt, tải dữ liệu, lệnh chạy thủ công, cách khởi động backend/frontend và giới hạn của phản hồi mô phỏng.

Checkpoint phục vụ API là **đúng checkpoint đã chọn bằng validation và xác nhận trên test**. Lệnh train lại dùng cùng cấu hình cố định và tạo một phiên bản artifact mới. Nếu sau này refit bằng train+validation, đó là phiên bản mô hình khác và không được gán lại điểm test của checkpoint cũ nếu chưa có đánh giá độc lập.

Frontend gọi API của thư mục cuối và hiển thị giá đề xuất cùng các chỉ số so sánh đã lưu từ nghiên cứu. SQL Server lưu sản phẩm, lịch sử bán theo tuần, quyết định giá, phản hồi bán hàng và phiên bản mô hình cho luồng ứng dụng. Dữ liệu UCI/M5 gốc và artifact nghiên cứu có thể giữ ở tệp đã phiên bản hóa; không bắt buộc nhét toàn bộ M5 vào SQL Server để huấn luyện. Trong demo lịch sử, phản hồi giá mới là do simulator tạo và phải được gắn nhãn. Chỉ khi một giá được áp dụng thực tế và phát sinh doanh số mới có phản hồi bán hàng thật để cập nhật mô hình.

## 8. Thứ tự phase và điều kiện chuyển bước

| Phase | Trạng thái V2 | Việc chính và điều kiện hoàn thành |
| --- | --- | --- |
| 0. Giao thức | Hoàn thành | Đã chốt nguồn, mốc chia dữ liệu theo tuần, chỉ số và phạm vi đánh giá. |
| 1. Dữ liệu | Hoàn thành | Đã dựng panel UCI và 10 cửa hàng M5, xác minh scaler từ train và khóa tập test. |
| 2. Song dự báo | Hoàn thành trên phạm vi thử | Đã nạp lại hai checkpoint Transformer riêng và kiểm tra dự báo lượng bán cùng đạo hàm theo giá. |
| 3. Song định giá | Hoàn thành lượt thử validation | Đã chạy 8 phương pháp trên cùng lịch M5 V2 và tái tạo 200 quyết định từ cặp checkpoint PPO–Transformer. Simulator chưa đạt mốc kiểm tra tại giá lịch sử. |
| 4. Phương án thay thế | Hoàn thành trên validation | Đã sàng lọc 9 mô hình dự báo cho mỗi bộ dữ liệu, kiểm tra hai simulator và so 10 chính sách giá M5. Đã khóa cấu hình để bàn giao. |
| 5. Đánh giá cuối | Đang chạy dự báo M5 | Đã xác minh 37/37 lượt dự báo UCI và 22.000 quyết định định giá M5. Dự báo M5 đủ 10 cửa hàng đang chạy ma trận mô hình và seed. |
| 6. Báo cáo | Theo kế hoạch | Tổng hợp bảng, biểu đồ và đối chiếu các kết quả có cùng định nghĩa, phạm vi và thang đo với bài Song. |
| 7. Hệ thống | Theo kế hoạch | Đóng gói cấu hình đã chọn trong `final_system/`, nối API và frontend, hướng dẫn cài đặt và nạp lại mô hình. |

**Trạng thái khi viết tài liệu:** UCI toàn phạm vi đã có panel nhưng so sánh nhiều mô hình mới chạy trên 100 SKU; M5 đã có panel 10 cửa hàng nhưng benchmark toàn phạm vi chưa xong. Vòng Song đã chạy trên pilot CA_1 với phản hồi mô phỏng; trang giá chính hiện chưa dùng luồng PPO cập nhật Transformer. Phase 0 và sửa giao thức test UCI là bước đầu tiên khi bắt đầu triển khai.

Phase 1 đã khóa **18.542** khóa test UCI, **1.220.898** khóa test M5 và danh sách 10 chuỗi định giá M5. Phase 2 đã dùng toàn bộ cửa sổ hợp lệ của UCI (2.998 SKU có cửa sổ train) và 120 chuỗi thử M5 từ 10 cửa hàng. Cả hai checkpoint được huấn luyện từ train, chọn epoch trên validation và xác minh đầu ra sau khi nạp lại. Xem [biên bản Phase 1](phase1-audit.md) và [cách chạy Phase 2](../README.md#phase-2--huấn-luyện-song-transformer-v2).

### 8.1 Phạm vi triển khai Phase 3

Phase 3 dùng **M5 V2**. Đầu vào là checkpoint M5 Phase 2 và 10 chuỗi định giá đã khóa ở Phase 1, mỗi cửa hàng một chuỗi. Checkpoint thử M5 đã bao gồm đủ 10 chuỗi này. Có thể tái sử dụng simulator Gaussian kernel, môi trường tương tác và cấu hình PPO của pilot cũ. Luồng V2 phải đọc đúng cohort đã khóa và dùng validation cho lượt kiểm tra.

1. **Chuẩn bị simulator.** Học đường cầu theo giá từ các cặp giá–lượng bán train của từng chuỗi. Lưu biên giá đã quan sát, cấu hình và kiểm tra sai số simulator tại giá lịch sử trên validation. Nhãn lượng bán thật ở validation dùng để kiểm tra simulator tại giá lịch sử đó.
2. **Nối vòng đề xuất giá.** Transformer cho lượng bán dự báo và gradient ở giá tham chiếu. PPO dùng hai tín hiệu này cùng lịch sử để đề xuất một giá trong biên train. Simulator trả lượng bán ở giá đề xuất, rồi môi trường tính doanh thu mô phỏng và thêm phản hồi vào lịch sử. Fine-tune Transformer sau mỗi 10 tương tác, giữ tối đa 200 phản hồi gần nhất và lấy tối đa 32 mẫu mỗi lần. PPO học từ các lượt mô phỏng.
3. **Chạy các đối chứng trên validation.** Bắt đầu bằng một seed (42). Thử giá cố định, heuristic, DDPG, SAC và các biến thể PPO đóng băng Transformer, bỏ gradient hoặc bỏ cả hai. Dùng cùng chuỗi, tuần validation, miền giá và nguồn nhiễu cho các lượt đối chiếu. Cố định cùng ngân sách tương tác cho các chính sách có huấn luyện. Lượt test 200 quyết định × 5 seed được thực hiện ở Phase 5.
4. **Lưu và kiểm tra chạy lại.** Lưu cặp PPO–Transformer tại cùng trạng thái cùng cấu hình simulator, seed và dấu vết tương tác. Nạp lại cặp đã lưu với cùng đầu vào để kiểm tra tái tạo giá đề xuất. Xác minh giá luôn trong biên train và số lần cập nhật Transformer khớp lịch.

**Kết quả cần bàn giao** gồm một lượt định giá M5 V2 chạy từ đầu đến cuối, cặp checkpoint đã kiểm tra, log theo SKU–tuần và bảng đối chứng validation. Mỗi dòng ghi giá đề xuất, lượng bán dự báo, lượng bán mô phỏng, doanh thu mô phỏng và số lần cập nhật. Bảng đối chứng ghi doanh thu tích lũy, mức thay đổi so với giá cố định và regret so với giá tốt nhất trong cùng simulator. Các giá đề xuất và doanh thu ở bước này là kết quả mô phỏng phục vụ kiểm tra mô hình. Giá đề xuất của hệ thống cuối sẽ dùng cấu hình được chọn ở Phase 4 và xác nhận ở Phase 5.

**Kết quả Phase 3:** lượt seed 42 dùng 512 tương tác train cho mỗi chính sách học và 200 quyết định validation cho mỗi phương pháp. Cặp PPO–Transformer chính đã được nạp lại để tái tạo 200 quyết định. Trên 392 bản ghi validation có giá trong miền train, RMSE của simulator là 218,08 sản phẩm/tuần và của mốc tuần trước là 135,34. Dùng kết quả lượt này để xác minh luồng định giá và làm mốc khi kiểm tra các phương án trên validation. Xem [lệnh chạy lại và bảng kết quả](../README.md#phase-3--thử-định-giá-song-ppo-trên-m5-v2).

### 8.2 Kế hoạch và kết quả cần có của Phase 4

Phase 4 bắt đầu từ dữ liệu V2 đã khóa, các checkpoint Phase 2 và log Phase 3. Mỗi mô hình học từ train của bộ dữ liệu tương ứng. Dùng validation để kiểm tra và chọn cấu hình, giữ các khóa test của Phase 1 cho Phase 5. Tái sử dụng logic benchmark dự báo, chọn simulator và thử lưới giá đã có. Các lệnh pilot cũ có lượt chấm test, vì vậy luồng Phase 4 cần giới hạn rõ ở train và validation.

1. **Kiểm tra simulator M5.** Giữ Gaussian kernel Silverman làm mốc Song. Thử đối chứng `lag-adjusted kernel` dùng lượng bán tuần trước làm ngữ cảnh. Fit từ train. So RMSE và MAE của từng simulator tại cùng các giá lịch sử validation nằm trong miền train, với mốc dùng lượng bán tuần trước. Kiểm tra thêm độ phủ giá và đường phản ứng trong miền train. Kết quả của hai simulator được báo riêng. Mốc RMSE là một kiểm tra cần thiết tại giá lịch sử. Các mức giá chưa áp dụng vẫn là giả định mô phỏng.
2. **So dự báo lượng bán riêng cho UCI và M5.** Ưu tiên mốc tuần trước, ARIMA theo quy tắc fallback V2, XGBoost, LSTM, Transformer và các biến thể nhận giá đã có. Giữ cấu hình Song cố định làm mốc. Xem xét TFT/Informer theo chi phí chạy và ghi rõ các mô hình chưa hoàn tất. Chốt trước tập cấu hình, seed, phạm vi và ngân sách. Có thể chạy sàng lọc trên phạm vi nhỏ để đo thời gian và bộ nhớ. Trong từng lượt so sánh, mọi phương pháp dùng cùng khóa SKU–tuần, cùng nhãn, cùng quy tắc dự báo theo thời gian. Xếp cấu hình bằng micro RMSE validation, báo kèm MAE, MAPE trên nhãn dương, WAPE, R² và số hàng. Kết quả trên phạm vi nhỏ được ghi rõ phạm vi. Phase 5 thực hiện bảng toàn phạm vi đủ điều kiện cho từng bộ dữ liệu.
3. **So phương án đề xuất giá trên M5.** Chọn các mô hình cầu thực sự nhận giá ứng viên, ví dụ Song Transformer và XGBoost có giá, rồi kiểm tra lượng bán dự báo khi đổi giá trong biên train. Dùng chúng cho phương án lưới giá chọn mức có `giá × lượng bán dự báo` cao nhất. So với PPO Song và các mốc Phase 3 bằng doanh thu trả về từ cùng một simulator mô phỏng, trên cùng SKU–tuần, miền giá và nhiễu. Nếu thay mô hình cầu trong trạng thái PPO thì huấn luyện lại PPO. Ghi doanh thu mô phỏng, mức chênh so với giá cố định, regret trong simulator, tỉ lệ chạm biên giá và độ ổn định qua SKU/seed. Lặp thêm seed validation cho các ứng viên sát nhau trước khi chốt. Nếu đổi simulator làm đổi thứ hạng, báo kết quả đó như phân tích độ nhạy.
4. **Khóa đầu ra để chuyển Phase 5.** Lưu bảng validation theo từng bộ dữ liệu, cấu hình siêu tham số và seed, phạm vi thực tế, thời gian chạy và đường dẫn artifact. Ghi rõ mô hình nào được chọn hoặc giữ làm mốc cho bảng test, cùng lý do lựa chọn. Nếu mọi simulator thử đều kém mốc tuần trước hoặc kết quả định giá thiếu ổn định, ghi kết luận định giá còn giới hạn và giữ phép đối chiếu Song làm kết quả mô phỏng. Phần dự báo lượng bán vẫn có thể chuyển sang Phase 5 theo tiêu chí riêng.

**Kết quả cần bàn giao** gồm hai bảng dự báo validation (UCI và M5), một bảng kiểm tra simulator, một bảng định giá M5 và một cấu hình tái chạy đã khóa cho Phase 5. Cấu hình ghi ứng viên mô hình cầu, cách chọn giá và mức bằng chứng hiện có để nhóm giải thích lý do chọn. Phase 5 mới xác nhận trên test cố định. Kết quả Phase 4 là căn cứ để chọn ứng viên, với mức độ chắc chắn dựa trên sai số simulator và độ ổn định qua các lượt chạy.

**Kết quả Phase 4:** UCI có mốc tuần trước dẫn theo RMSE 149,47 trên 493 bản ghi, sát Song Transformer 149,62. M5 có Song Transformer dẫn với RMSE 71,83 trên 5.040 bản ghi. Simulator lag-adjusted trọng số 0,75 đạt RMSE 109,65 trên phần validation sau, so với 110,52 của mốc tuần trước. DDPG đứng đầu về doanh thu tại cả hai simulator, nhưng cả 1.200 giá DDPG đều đúng cận trên của SKU. Dùng DDPG làm ứng viên đối chiếu với mốc luôn giữ giá cận trên ở Phase 5. Xem [cách chạy lại và kết quả](../README.md#phase-4--chọn-cấu-hình-trên-validation-v2) và [cấu hình đã khóa](../experiments/phase5_settings_v2.json).

### 8.3 Kế hoạch và kết quả cần có của Phase 5

Phase 5 là lượt đánh giá cuối trên test V2. Đầu vào là dữ liệu Phase 1 và [cấu hình đã khóa từ validation](../experiments/phase5_settings_v2.json). Mỗi lượt ghi dấu vân tay của cấu hình, nguồn dữ liệu và khóa test. Các kết quả test xác nhận hoặc giới hạn kết luận từ validation. Mọi siêu tham số, số epoch đã chọn, quy tắc chọn SKU, tiêu chí và ngân sách vẫn lấy từ cấu hình đã khóa.

| Luồng | Phạm vi phải xác minh |
| --- | --- |
| Dự báo UCI | Panel chứa 3.387 SKU đủ điều kiện. Chấm 18.542 khóa test quan sát từ 2.774 SKU, trong 9 tuần từ 2011-10-10. |
| Dự báo M5 | Đủ 29.069 chuỗi sản phẩm–cửa hàng tại 10 cửa hàng. Chấm 1.220.898 khóa test trong 42 tuần từ 2015-08-03. Tuần bán 0 có đủ giá được tính. |
| Định giá M5 | Một chuỗi đã khóa cho mỗi cửa hàng. Dùng đúng 20 tuần test lịch đầu tiên, từ 2015-08-03 đến 2015-12-14. Mỗi phương pháp có 200 quyết định mỗi seed. |

1. **Chuẩn bị lệnh V2 và kiểm tra khả năng chạy.** Tái sử dụng mô hình, bộ đọc panel theo cửa hàng, code mô phỏng và nạp checkpoint đã kiểm tra ở các phase trước. Luồng Phase 4 chỉ chấm validation. Lệnh M5 cũ huấn luyện riêng theo từng cửa hàng và không xác minh các khóa chung theo V2. Dựng lệnh V2 đọc cấu hình đã khóa, kiểm tra nguồn, tuần, khóa và mã băm. Dùng dữ liệu tổng hợp hoặc validation để kiểm tra dự báo cuốn chiếu, quy tắc ARIMA, nạp lại mô hình và tiếp tục khi bị ngắt. Đo một lượt M5 giới hạn để ước lượng thời gian, RAM và dung lượng đĩa trước khi chạy quy mô lớn.
2. **Chấm dự báo UCI trước để kiểm tra luồng, sau đó chạy M5 đủ 10 cửa hàng.** Mỗi bộ dữ liệu có 9 phương pháp đã khóa. Hai mốc xác định là tuần trước và ARIMA có fallback, mỗi mốc chạy một lượt. Bảy phương pháp có huấn luyện ngẫu nhiên chạy các seed `42, 123, 456, 789, 1024`, tổng 37 lượt dự báo mỗi bộ dữ liệu. Huấn luyện lại checkpoint dự báo trên train + validation theo cấu hình đã chọn, giữ scaler và các quy tắc tiền xử lý đã tính từ train. Dự báo từng tuần với lịch sử đã biết đến tuần trước. Mỗi mô hình học chung có một mô hình cho toàn bộ M5 mỗi seed, với dữ liệu đọc và ghi theo lô hoặc cửa hàng. Hai bộ dữ liệu sẽ tạo khoảng 45,86 triệu dòng nếu lưu toàn bộ dự báo. Lưu theo bộ dữ liệu, mô hình và seed để có thể tiếp tục từng lượt. Chỉ xếp hạng các phương pháp đã đủ seed, đủ khóa, đúng nhãn và không có khóa thừa hoặc trùng. Ghi riêng TFT/Informer và các lượt thiếu.
3. **Chấm định giá M5 theo 5 seed độc lập.** Huấn luyện mô hình cầu định giá trên train của 120 chuỗi M5 đã khóa. Với mỗi seed, huấn luyện 6 chính sách có học (DDPG, SAC và 4 biến thể PPO) trên simulator Gaussian với 512 tương tác mỗi chính sách. Lưu cặp PPO–Transformer cùng trạng thái phản hồi. Chấm 10 phương pháp của Phase 4 cộng mốc luôn giữ mức giá train cao nhất của SKU. Các phương pháp dùng chung khóa, giá trong miền train và cùng nguồn nhiễu cho từng SKU và seed. Lịch sử ban đầu chỉ có các tuần đã qua. Khi mô phỏng tiến sang tuần mới, giá và lượng bán mô phỏng được đưa vào lịch sử và phản hồi. Tổng quy mô là **11.000 quyết định mỗi simulator** nếu đủ tất cả lượt. Giữ Gaussian Silverman cho bảng Song, dùng cùng chính sách đã học để chạy riêng kiểm tra độ nhạy với lag-adjusted trọng số 0,75. Báo chênh lệch DDPG so với mốc giữ giá cận trên theo từng seed và số giá khác cận trên để xem DDPG có thực sự đổi giá theo tuần hay không.
4. **Xác minh và bàn giao.** Tổng hợp chỉ số từ các dự báo đã qua kiểm tra khóa. Xuất hai bảng UCI/M5 gồm RMSE, MAE, R², WAPE, MAPE trên nhãn dương kèm mẫu số, RMSE theo thang train và độ biến động qua seed. Lưu bảng theo SKU và ba vùng giá với ngưỡng tính từ train. Với định giá, lưu giá, dự báo và lượng bán mô phỏng, doanh thu, biên giá và lần cập nhật từng tuần. Xuất doanh thu tích lũy, trung bình, độ lệch chuẩn mẫu, min/max, regret, và mức tăng so với các mốc cố định và PPO đóng băng. Giữ riêng hai bảng simulator. Bàn giao dự báo theo khóa, checkpoint, manifest, log và lệnh chạy lại để làm biểu đồ và đối chiếu bài báo ở Phase 6.

**Điều kiện hoàn thành:** hai bảng dự báo dùng đúng khóa test V2 đã khóa. Mỗi hàng định giá trong bảng chính có đủ 200 quyết định × 5 seed trên cùng cohort. Manifest phân biệt rõ lượt đã xác minh và lượt thiếu. Kết quả định giá là doanh thu mô phỏng. Nếu DDPG lặp lại mức giá cận trên, kết quả chỉ ra hành vi giữ giá đó trong phạm vi thử.
