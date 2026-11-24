# Giao thức pilot lịch sử: bài Song và các bài dùng cùng dữ liệu

> Các lượt chạy mới dùng [giao thức Phase 0 toàn phạm vi](PROTOCOL_V2.md). Nội dung bên dưới ghi lại phạm vi pilot chốt ngày 2026-11-15 để đối chiếu kết quả cũ; 100 SKU UCI, 20 chuỗi M5 và `sales_train_validation.csv` không còn là phạm vi bảng chính.

Ngày chốt bản đầu: 2026-11-15. Tài liệu này phân biệt **tái dựng phương pháp** với **tái tạo đúng số công bố**. Phạm vi, cách chia và thang đo phải được ghi cùng mỗi kết quả; chỉ dùng kết quả test sau khi đã chốt lựa chọn trên validation.

## 1. Nguồn và khả năng đối chiếu

| Nguồn | Dữ liệu gốc | Bài toán và thang thời gian | Vai trò trong dự án | So trực tiếp điểm công bố? |
| --- | --- | --- | --- | --- |
| [Song (2026)](https://link.springer.com/article/10.1007/s10791-026-10745-2) | [UCI Online Retail ID 352](https://archive.ics.uci.edu/dataset/352/online%2Bretail) và [M5](https://zenodo.org/records/10203108) | Nhu cầu SKU/tuần, giá liên tục trên M5; Transformer + PPO | Bài gốc để tái dựng luồng | Chưa: không có mã, SKU và một số chi tiết chuẩn hóa/simulator chưa được công bố đầy đủ |
| [IEEE: Tree Models and SHAP (2025)](https://ieeexplore.ieee.org/abstract/document/11203339) | M5 | Nhu cầu sản phẩm/ngày, CatBoost, thử giá ứng viên và SHAP | Ứng viên thuật toán dự báo và cách giải thích giá | Không: đơn vị thời gian và giao thức khác; hiện chỉ xác minh được tóm tắt công khai, cần toàn văn trước khi nhận là tái dựng đúng bài |
| [M5 Accuracy competition: Results, findings, and conclusions (2022)](https://doi.org/10.1016/j.ijforecast.2021.11.013) | M5 | Dự báo 28 ngày, WRMSSE; phân tích các cách dùng LightGBM | Nguồn chọn baseline LightGBM | Không: chân trời, số chuỗi và chỉ số khác; đây không phải một mô hình LightGBM duy nhất để sao chép |
| [Retail Data Warehouse validation (2026)](https://ejurnal.jejaringppm.org/index.php/jitcsa/article/download/247/183/1115) | UCI Online Retail ID 352 | Kho dữ liệu giao dịch và dự báo doanh thu tổng/ngày | Đối chiếu thiết kế dữ liệu và báo cáo BI | Không: biến mục tiêu và mức tổng hợp khác |

Online Retail II ID 502 (2009–2011), dữ liệu của demo cũ, **không phải** Online Retail ID 352 (2010–2011) mà Song dùng. Danh sách 20 SKU demo cũ không được chuyển nguyên sang bảng so sánh chính.

## 2. Giao thức chung của nhóm

- **UCI:** một hàng là SKU–tuần; dự báo số sản phẩm ở tuần hiện tại từ 12 tuần quá khứ, giá hiện tại là đầu vào kịch bản. Nguồn có 541.909 dòng hóa đơn gốc. Chọn SKU đủ điều kiện chỉ từ train. Nhánh Song có thể chạy tất cả 3.387 SKU đủ điều kiện; bảng đa thuật toán trên máy nhóm dùng cohort 100 SKU được chốt bằng train.
- **M5:** một hàng là sản phẩm–cửa hàng–tuần; dùng `sales_train_validation.csv`, `sell_prices.csv`, `calendar.csv`. Pilot định giá gồm 20 chuỗi ở CA_1, chọn bằng thông tin train. Đây là **tập con**, không phải toàn bộ 30.490 chuỗi sản phẩm–cửa hàng của M5.
- **Chia thời gian:** 70% tuần đầu train, 15% kế validation, 15% cuối test; chung ranh giới tuần trong mỗi dataset. Không xáo trộn theo dòng. Ngưỡng ngoại lệ, chọn cohort, chuẩn hóa và siêu tham số chỉ học từ train/validation theo vai trò của chúng; test chỉ để chấm cuối.
- **Làm sạch:** quy tắc cố định như nhận diện hóa đơn hủy và giá không hợp lệ có thể áp dụng trước chia. Ngưỡng 3×IQR theo SKU được ước lượng bằng train. UCI không có nhãn khuyến mãi thật. Tuần UCI không có hóa đơn là ngữ cảnh với số lượng 0 nhưng không được xem là nhãn cầu xác nhận trong điểm chính. M5 có giá niêm yết đủ 7 ngày và bán 0 là nhãn thật.
- **Chỉ số dự báo:** báo RMSE/MAE theo đơn vị sản phẩm/tuần và thêm thang chuẩn hóa theo SKU; số hàng được chấm phải giống nhau giữa thuật toán. Nêu riêng kết quả UCI và M5. MAPE loại hàng có cầu bằng 0 và phải ghi mẫu số.
- **Định giá:** mọi chính sách dùng cùng tập M5, khoảng giá, số quyết định, seed và simulator. Báo giá đề xuất, cầu dự báo, doanh thu/regret **mô phỏng**; không gọi là doanh thu thật hay lợi nhuận vì không có quan sát cầu phản thực tế và giá vốn.
- **Siêu tham số:** Song có một cấu hình giữ nguyên theo bài; các mô hình khác dùng phạm vi và ngân sách tìm kiếm công bố trước. Chọn bằng validation, lưu mọi thử nghiệm và thời gian chạy. Dùng nhiều seed cho kết quả cuối khi ngân sách CPU cho phép; nếu ít hơn năm lần chạy của Song thì báo rõ.

## 3. Hai tầng kết quả trong báo cáo

1. **So sánh công bằng do nhóm chạy lại:** cùng cohort, split, đặc trưng được phép, mục tiêu, simulator và thang đo. Đây là nơi kết luận thuật toán nào tốt hơn trong thí nghiệm của nhóm.
2. **Đối chiếu bài công bố:** nêu cấu trúc, cách chọn tham số, dữ liệu và kết quả của từng bài cùng những sai khác. Không lấy RMSE ngày hoặc WRMSSE 28 ngày so số học với RMSE SKU/tuần.

## 4. Điểm chưa biết cần ghi khi triển khai

- Song không cung cấp mã nguồn, danh sách SKU, đầy đủ quy tắc chọn chuỗi M5 và công thức chuẩn hóa doanh thu. Bài có hai mô tả thang RMSE ở các mục thực nghiệm; xuất cả thang thô và thang chuẩn hóa để tránh gán sai cho số bài báo.
- IEEE: cần toàn văn để xác minh chính xác split, đặc trưng, tìm siêu tham số, phạm vi giá và cách tính RMSE; tóm tắt công khai chưa đủ để gọi phần triển khai sau này là bản sao của tác giả.
- M5 competition dùng đánh giá chính thức 28 ngày/WRMSSE. LightGBM trên panel 20 chuỗi/tuần là baseline **lấy cảm hứng từ** bài tổng kết, không phải điểm thi M5.
- Dữ liệu giao dịch và M5 không cho biết cầu ở giá chưa áp dụng. Chất lượng dự báo tại giá quan sát không xác nhận chất lượng simulator ở giá giả định.

## 5. Cách kiểm tra lại Phase 0

1. Đọc bốn nguồn ở bảng trên, đối chiếu cột dữ liệu, đơn vị dự báo và chỉ số trước khi thêm một bài khác.
2. Từ thư mục gốc dự án, chạy `python -m experiments.download_paper_data --dataset all`. Chương trình kiểm tra lại kích thước/MD5 M5 theo Zenodo, tệp Excel UCI theo ZIP gốc và ghi `data/song_replication/download_manifest.json`; nếu tệp đã có và đúng thì không tải lại.
3. Mở manifest vừa ghi và xác nhận `Online Retail.xlsx`, `sales_train_validation.csv`, `sell_prices.csv`, `calendar.csv` có nguồn và mã băm. Đây chỉ là kiểm tra nguồn, chưa chạy mô hình.

**Ý nghĩa:** Phase 0 khóa câu hỏi nghiên cứu và ngăn việc so nhầm bài Song với demo Online Retail II hoặc so nhầm điểm giữa các bài toán khác nhau.
