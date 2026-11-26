# Thiết kế và kiểm thử giao diện M5

Bố cục danh mục mua sắm với ảnh giỏ hàng ở trang chủ, bộ lọc theo nhóm sản phẩm và giá ưu tiên trong thẻ. Các trang chi tiết dùng bảng giá 10 cửa hàng làm nội dung chính. Căn trái tên và thao tác, căn phải giá. Trang dự đoán dùng hai cột nhập và kết quả trên máy tính, xếp dọc trên màn hình nhỏ.

Màu nền #F8FAFF, mực #142948, xanh cobalt #204BD7, xanh phấn #D9E8FF, cam #FFC051 và viền #DFE7F3. Be Vietnam Pro dùng cho toàn bộ giao diện, dùng độ đậm và khoảng trắng để phân cấp. Ảnh minh họa chụp như danh mục bán lẻ. Trang chủ dùng giỏ hàng thay cho banner số liệu.

- Tạo bảng danh mục, giá chỉnh và view tổng hợp theo SKU–cửa hàng. Kiểm thử đủ 20 × 10 dòng và dữ liệu M5 nguồn.
- Nối danh mục và thao tác chỉnh giá vào backend. Kiểm thử giá mới xuất hiện sau khi lưu và lịch sử ghi nhận.
- Làm lại giao diện M5 gồm trang chủ, sản phẩm, chi tiết và dự đoán. Dùng API mô hình sẵn có cho giá đề xuất.
- Kiểm tra luồng chính với SQL Server, API mô hình và trình duyệt ở kích thước máy tính và điện thoại.
