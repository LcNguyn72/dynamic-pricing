# Biên bản Phase 1 — dữ liệu tuần UCI và M5 theo giao thức V2

Ngày thực hiện: 2026-11-20. **Trạng thái:** Phase 1 đã tạo và xác minh dữ liệu V2; chờ nhóm kiểm tra trước Phase 2. Bản cuối của `experiments/protocol_v2.json` có SHA256 `a73ad476b478a2583406f94b0f1215add26da6fef536d6215cb017b5214d1567`.

## Mục đích và phạm vi

Phase 1 dựng dữ liệu để các phase sau huấn luyện và đánh giá trên **cùng hàng test**. UCI Online Retail ID 352 và M5 Forecasting Accuracy được xử lý riêng. Chưa train mô hình thật trên panel V2 hoặc chạy PPO đánh giá; không sửa API/frontend. Bộ kiểm thử toàn thư mục có chạy benchmark nhỏ trên dữ liệu tổng hợp. Code gốc của demo Online Retail II giữ cách xử lý cũ. [Giao thức V2](../experiments/PROTOCOL_V2.md) quy định các bước; [README gốc](../README.md#phase-1--dựng-và-xác-minh-dữ-liệu-tuần-v2) có lệnh chạy lại đầy đủ từ nguồn đã tải.

## UCI Online Retail ID 352

- Nguồn `Online Retail.xlsx`: 541.909 dòng giao dịch, SHA256 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.
- Tính ranh giới từ 54 tuần lịch nguồn **trước khi chọn SKU**: 37 train, 8 validation, 9 test; test bắt đầu 2011-10-10. Chọn 3.387 SKU từ train; scaler theo SKU chỉ fit ở tuần train quan sát.
- Quy tắc hợp lệ cố định loại hóa đơn hủy/hoàn, lượng bán hoặc giá không hợp lệ. Ngưỡng 3×IQR học từ train chỉ xóa dòng train. Có 14.742 dòng train bị xóa theo IQR; 11.978 dòng validation/test bị gắn cờ IQR nhưng được giữ để không cắt nhãn đánh giá. Mọi tuần quan sát ngoài train đều có `qty == qty_all`.
- Tuần đầu bắt đầu 2010-11-29 nhưng giao dịch đầu tiên là 2010-12-01; tuần cuối bắt đầu 2011-12-05 nhưng dữ liệu chỉ đến 2011-12-09. Đây là hai tuần nguồn không đủ bảy ngày; giao thức giữ chúng và ghi rõ giới hạn.
- Panel có 18.573 tuần SKU quan sát trong test; 18.542 hàng có đủ đặc trưng 12 tuần quá khứ và trở thành khóa test chính. So với cửa sổ pilot cũ, thêm 237 khóa test và giữ thêm 483.889 đơn vị bán trong các tuần test trên cùng lưới SKU. Các giá trị cực lớn có thể làm RMSE tăng, nhưng không được xóa chỉ vì nằm trong test.
- Cửa sổ được đối chiếu với panel cho đủ 12 lag `qty`, `price`, `qty_z`, `price_z`; hash `windows.csv` là `63ce25b44bf8cce32ee1cc2fa4922ebe9620822ffddfb5177d5ffc7b88f03a38`. Có 1.176 SKU với dưới 12 tuần train quan sát, còn 869 SKU với dưới 12 tuần train+validation quan sát; các lượt ARIMA sau cần áp dụng và báo số lần fallback theo giao thức.

## M5 Forecasting Accuracy

- Nguồn chính: `sales_train_evaluation.csv` 1.941 ngày, `sell_prices.csv`, `calendar.csv`. Tuần M5 phải đủ bảy ngày; lượng bán tuần là tổng bảy ngày, giá tuần là trung bình giá niêm yết ngày. Tuần bán 0 vẫn là nhãn nếu cả bảy ngày có giá.
- Tạo lại panel theo từng cửa hàng trong `experiments/artifacts/phase1_v2/m5/`; manifest mỗi cửa hàng ghi hash của ba nguồn, hash panel, số chuỗi, mốc split và scaler fit từ train. `full_m5_preparation.json` tổng hợp mười cửa hàng.
- Mốc chung: 277 tuần, 193 train, 42 validation, 42 test; test bắt đầu 2015-08-03. Kiểm tra lưới SKU–tuần đầy đủ, không trùng khóa, cờ `observed_sales` đúng với bảy ngày có giá, scaler `qty_z`/`price_z` đúng thống kê train, và mọi test key có đủ lịch sử 12 tuần.
- `promotion_proxy` chỉ là dấu hiệu giảm giá suy ra từ lịch sử niêm yết. M5 không có `promotion_flag` quan sát; đặc trưng cờ khuyến mãi của Song Transformer hiện nhận 0. Không biến proxy giảm giá thành cờ cố định khi thử giá ứng viên.
- Bản V2 đã tạo **8.052.113** hàng SKU–tuần cho 29.069 chuỗi đủ điều kiện từ 30.490 chuỗi gốc. Có 1.220.898 khóa test, tức 42 tuần cho từng chuỗi; **258.613** khóa là tuần bán 0 có đủ giá. Trong train có 463 chuỗi với dưới 12 tuần quan sát, nhưng sau train+validation không còn chuỗi nào dưới 12 tuần; vẫn cần ghi số lượt fallback nếu mô hình ARIMA gặp lỗi fit.
- Manifest tổng hợp mới `full_m5_preparation.json` có SHA256 `779c78257342e009879484cfdd3a922a435c6b52097866dfd065cbd868f897bc`. Mỗi manifest cửa hàng V2 có scaler và hash nguồn/panel, thay cho manifest cũ thiếu các thông tin này. Các panel cũ không bị ghi đè.

## Định dạng khóa test và cohort giá

`test_keys_uci.csv` và `test_keys_m5.csv` có header `dataset,sku,week`, được sắp theo dataset, SKU, thứ Hai ISO `YYYY-MM-DD`, mã UTF-8 và dòng kết thúc LF. SHA256 tính trên **đúng byte của tệp**. Chỉ chấm `observed_sales=True` sau khi tạo cửa sổ 12 tuần; UCI không tự coi tuần không giao dịch là cầu xác nhận, còn tuần M5 bán 0 có đủ giá được chấm.

| Dataset | Chuỗi đủ điều kiện từ train | Khóa test đã khóa | SHA256 `test_keys_*.csv` |
| --- | ---: | ---: | --- |
| UCI Online Retail ID 352 | 3.387 SKU; 2.774 có nhãn test sau lookback | 18.542 | `6c7c37bbae2adb0cfcd75d98f3553c7b4f41b569ca2d0f2cb7df0f154279c689` |
| M5 đủ 10 cửa hàng | 29.069 chuỗi | 1.220.898 | `306da95af8a5553312ab27760e0c1d306834749dcc5149b51040368c5d4f1af3` |

`m5_pricing_cohort.csv` chọn một chuỗi mỗi cửa hàng theo tổng lượng bán train cao nhất trong số chuỗi có ít nhất hai giá train và 12 tuần lịch sử; hòa thì chọn SKU tăng dần. Chỉ sau khi chọn mới kiểm tra 20 tuần test lịch đầu; nếu thiếu thì báo lỗi, không chọn lại bằng test. Đây là cohort **do nhóm đề xuất**, không phải danh sách SKU được Song công bố.

| Cửa hàng | Chuỗi được chọn bằng train |
| --- | --- |
| CA_1 | `FOODS_3_090@CA_1` |
| CA_2 | `FOODS_3_586@CA_2` |
| CA_3 | `FOODS_3_090@CA_3` |
| CA_4 | `FOODS_3_090@CA_4` |
| TX_1 / TX_2 / TX_3 | `FOODS_3_586` tại từng cửa hàng |
| WI_1 / WI_2 | `FOODS_3_226` tại từng cửa hàng |
| WI_3 | `FOODS_3_090@WI_3` |

Cả 10 chuỗi có đủ 20 tuần test lịch đầu từ 2015-08-03 đến 2015-12-14, tổng **200 quyết định**. SHA256 của `m5_pricing_cohort.csv`: `340daf493d7f03fcfc87c320d3da934705efb83fad7090668a224aeb51920ade`.

## File code thay đổi và lý do

- `retail_ii_preprocess.py`: thêm chế độ IQR `train_only`; mặc định `all` giữ nguyên hành vi demo cũ.
- `song_online_retail.py`: cho phép chọn chế độ V2 bằng CLI và ghi chính sách xử lý, ranh giới tuần vào manifest.
- `data.py`: chia tuần theo toàn tệp nguồn trước khi lọc SKU, tránh dịch mốc khi SKU được chọn thiếu tuần đầu/cuối.
- `full_m5_prepare.py`: ghi hash ba nguồn, panel và manifest cho từng cửa hàng, cùng manifest tổng hợp.
- `phase1_finalize.py`: xác minh nguồn/panel/scaler/feature, xuất khóa test chuẩn và cohort định giá train-only. Các kiểm thử hành vi nằm trong `test_retail_ii_preprocess.py`, `test_data.py`, `test_phase1_finalize.py`.

## Đầu ra và điều kiện chuyển Phase 2

Tất cả tệp dưới `experiments/artifacts/phase1_v2/` bị Git bỏ qua và cần tạo lại trên máy khác. `phase1_manifest.json` giữ hash nguồn, hash panel, số hàng/khóa test và cohort. Các kết quả pilot cũ trong `experiments/artifacts/song_online_retail/` và `experiments/artifacts/song_m5_full/` không được thay bằng bản V2. Tuần thiếu giá M5 có thể còn trong panel làm ngữ cảnh lịch sử, nhưng không là nhãn chấm; cửa sổ có lag giá thiếu sẽ bị loại.

Phase 2 chỉ dùng panel/khóa test có hash trong manifest Phase 1, lưu checkpoint và scaler riêng cho UCI/M5. Mô hình phải chấm đủ cùng khóa; bảng dự báo V2 chưa có số RMSE/MAE/R²/MAPE vì Phase 1 chưa huấn luyện.

**Kiểm tra cuối:** `python -m unittest discover -s experiments -p 'test_*.py'` hoàn tất **56/56 test**; `phase1_finalize` chạy lại sau khi khóa cấu hình và khớp hash UCI/M5; JSON đọc được, link tài liệu nội bộ tồn tại và `git diff --check` không báo lỗi định dạng (chỉ có cảnh báo chuyển LF/CRLF của Git trên Windows).
