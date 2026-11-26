# Đối sánh mô hình Song với mô hình dự án

Code và biểu đồ của lượt chạy mới được giữ riêng, không sửa Word hoặc ghi đè Phase 5/6.

Xem [METHOD.md](METHOD.md) để đọc luồng xử lý, kiến trúc, siêu tham số, độ phức tạp, công thức và giới hạn phép so sánh.

## Phạm vi

- UCI: toàn bộ 18.542 khóa test V2.
- M5 dự báo: toàn bộ 29.069 chuỗi đủ điều kiện và 1.220.898 khóa test V2.
- M5 định giá: toàn bộ chuỗi có ARIMAX fit từ train, đủ biến thiên giá cho kernel và có 12 tuần lịch sử trước kỳ đánh giá. Lưu danh sách/độ phủ và lý do không đủ điều kiện; không gọi đây là tất cả sản phẩm M5 đều có thể tối ưu giá.
- Khảo sát bổ sung: test V2 đã được xem trong các bước trước. Kết quả không phải thử nghiệm xác nhận trên một holdout mới chưa từng sử dụng.

## Chạy

Môi trường đã tạo cho lượt chạy hiện tại nằm tại `tmp/comparison_venv` (Python 3.10, có thư viện hệ thống đọc được; phiên bản thực tế được ghi trong artifact). Nếu tạo môi trường mới, dùng Python >=3.11 và cài `requirements.txt` của folder này. Có thể cài bản PyTorch CUDA theo máy từ nguồn chính thức trước khi cài các gói còn lại.

```powershell
$comparisonPython = ".\tmp\comparison_venv\Scripts\python.exe"
& $comparisonPython -m unittest experiments.test_comparison_v3 -v
& $comparisonPython -u -m experiments.comparison_v3 prepare
& $comparisonPython -u -m experiments.comparison_v3 arimax --workers 6
& $comparisonPython -u -m experiments.comparison_v3 forecast --datasets uci --device cuda
& $comparisonPython -u -m experiments.comparison_v3 forecast --datasets m5 --device cuda
& $comparisonPython -u -m experiments.comparison_v3 simulator-audit --workers 6
& $comparisonPython -u -m experiments.comparison_v3 pricing --workers 6
& $comparisonPython -u -m experiments.comparison_v3 charts
& $comparisonPython -u -m experiments.comparison_v3 verify
```

Hoặc dùng `python -u -m experiments.comparison_v3 all --device cuda --workers 6`. `--output` cho phép tạo lượt độc lập; không dùng cùng output khi thay cấu hình. `--seeds 42` dùng để thử một seed, không được gọi là kết quả đủ năm seed.

Có thể chạy từ PowerShell bằng `./experiments/comparison_v3/run.ps1`; script tự tìm môi trường riêng đã tạo và tiếp tục toàn bộ pipeline từ checkpoint đã hoàn tất.

Lệnh có thể chạy lại để tiếp tục. Mô hình dự báo lưu checkpoint sau mỗi epoch và trạng thái RNG; các đơn vị hoàn tất được tái sử dụng sau kiểm tra identity/hash. PPO lưu checkpoint được chọn theo validation; nếu tiến trình ngắt trước khi đơn vị PPO hoàn tất, đơn vị đó huấn luyện lại. Không khôi phục đường học hoặc kết quả còn thiếu bằng dữ liệu giả.

Với batch dài, có thể đặt `{"pricing_workers":6}` trong `execution.json` dưới thư mục output để lượt pricing tiếp theo đổi số worker mà không khởi động lại controller. Đây là cấu hình tài nguyên, không đổi model/dữ liệu/RNG. Giá trị thực dùng được ghi vào `execution_used_pricing.json`; lượt pricing đang chạy không bị thay đổi.

## Thiết kế

1. Đọc panel chuẩn bị sẵn, xác minh SHA256 và từng khóa test; thống kê chuẩn hóa chỉ lấy từ train. Dữ liệu neural được lưu bằng các array NPY theo cửa hàng, không nạp toàn M5 vào GPU cùng lúc.
2. Huấn luyện SongTransformer 4 encoder layer, 8 head, width 128, lịch sử 12 tuần. Thử learning rate 0,001 và 0,0003 ở seed 42; chọn theo RMSE validation rồi chạy năm seed. Train-only, tối đa 30 epoch, patience 5 và tối thiểu 5 epoch; lưu toàn bộ train MSE và validation RMSE. UCI batch 512. M5 batch 4096 và BF16 autocast lúc train để phù hợp GPU 6 GB; checkpoint và chấm validation/test dùng float32. Đây là lựa chọn triển khai được công bố, không nhận là code tác giả.
3. Fit ARIMAX bằng chính estimator của `final_system`. Full-scope forecast báo `ARIMAX + fallback`; bảng `arimax_available` chấm mọi mô hình trên cùng các chuỗi có mô hình giá, không trộn độ phủ khác nhau trong một bảng.
4. Huấn luyện lại PPO của hai kiến trúc trên phạm vi M5 đủ điều kiện: mạng 3×256 ReLU của Song, mạng 2×32 Tanh và các feature của PPO phục vụ. Cùng 8.192 tương tác train, reward scale chung, miền giá train và simulator Gaussian. Kiến trúc phục vụ được mở rộng one-hot theo roster mới; đây không phải policy mười chuỗi cũ.
5. Chọn checkpoint PPO theo doanh thu validation trên đại diện được xếp hạng từ lượng bán train, một chuỗi cho mỗi cửa hàng/nhóm hàng. Evaluation dùng cùng ngân sách 20 tuần cho **mọi** chuỗi đủ điều kiện và cả hai simulator. Từng SKU được khởi tạo lại từ cùng lịch sử trước kỳ, mỗi policy có lịch sử mô phỏng riêng. Song feedback cập nhật mô hình cầu; ARIMAX feedback cập nhật beta theo engine hiện tại. Không dùng lượng bán test thật làm nhãn ở giá mới.
6. Cùng một stream nhiễu theo seed/SKU được dùng cho các policy; việc chọn giá và fine-tune không tiêu thụ stream đó. Lưu từng quyết định và tổng theo store/method/seed/simulator.
7. Khoảng bootstrap 95% ghép cặp trên **tổng từng seed**, điều kiện theo bộ SKU và simulator cố định. Năm seed không cung cấp bằng chứng rộng về mọi thị trường; 8.192 bước là ngân sách cố định, không tự chứng minh policy đã hội tụ hoặc đã thấy mọi SKU train. Báo thêm số chuỗi đã thấy trong đường học.

## Artifact

Mặc định dưới `experiments/artifacts/comparison_v3`:

- `data/`: array train/validation/test, hàng nhãn, panel và manifest hash theo cửa hàng.
- `arimax/`: trạng thái model train-only và lý do fit không đạt.
- `forecast/`: cấu hình lựa chọn, learning curve, checkpoint, dự báo chi tiết theo seed/cửa hàng.
- `pricing/`: phạm vi đủ điều kiện, policy, cặp model cầu, quyết định giá và tổng theo seed.
- `forecast_metrics.csv`, `forecast_by_store.csv`: chỉ số gộp tính từ lỗi từng hàng, không lấy trung bình RMSE cửa hàng.
- `pricing_paired_intervals.csv`: uplift ghép cặp và khoảng bất định; so sánh chính là lưới ARIMAX với Song PPO, các so sánh khác là khảo sát.
- `pricing_summary.csv`: doanh thu mô phỏng trung bình, độ lệch chuẩn mẫu, min/max và số seed thực có; độ lệch chuẩn để trống nếu mới có một seed.
- `figures/`: ảnh PNG 180 dpi và SVG; có thể tái tạo bằng stage `charts`.
- `README.md`, `comparison_status.json`: trạng thái đủ/thiếu seed và cách diễn giải.
- `verification.json`, `contract_tests.txt`, `source_manifest.json`, `source_snapshot/`: kiểm tra cuối và bản mã nguồn dùng tái lập; chỉ có chứng nhận hoàn tất khi stage `verify` chạy thành công.

Không so trực tiếp điểm mới với 15,7/22,3 hoặc doanh thu 883 trong bài vì chưa có cùng mẫu/thang đo. Muốn kết luận tốt hơn phương pháp gốc cần xác nhận thêm tính tương đương của bản tái dựng; muốn kết luận tăng doanh thu thực cần bằng chứng ngoài simulator.
