# Biểu đồ

Tạo bởi `ve_du_lieu.py` (biểu đồ dữ liệu) và `ve_ket_qua.py` (biểu đồ kết quả mô hình).
Mỗi biểu đồ có một file CSV đi kèm chứa số liệu đã vẽ (mở bằng Excel được).

| Mã | Biểu đồ | Nguồn dữ liệu | Ý nghĩa |
|---|---|---|---|
| A2 | `figA2_lam_sach_du_lieu.png` - Thác nước làm sạch dữ liệu | data/online_retail_II_sheets.csv | Mỗi bước làm sạch loại bao nhiêu dòng; sau đó lọc sản phẩm. |
| B1 | `figB1_phan_bo_luong_ban.png` - Phân phối lượng bán theo tuần | models_2y/weekly_online_retail.csv | Đuôi dài: vài tuần bán rất nhiều làm RMSE cao hơn MAE. |
| B3 | `figB3_so_muc_gia.png` - Số mức giá và số lần đổi giá của mỗi sản phẩm | models_2y/weekly_online_retail.csv | Mỗi sản phẩm đổi giá ít, nên elasticity riêng từng sản phẩm yếu. |
| D3 | `figD3_dai_dien_20_san_pham.png` - 20 sản phẩm được chọn so với toàn bộ | models_2y/products_online_retail.csv + danh sách 20 sản phẩm | 20 sản phẩm có đại diện cho 1306 sản phẩm không. |
| 1 | `fig01_thuc_te_du_bao_theo_tuan.png` - Thực tế và dự báo theo tuần (3 sản phẩm) | model_table + demand_model | Mô hình có bám xu hướng của sản phẩm ổn định, đột biến, điển hình không. |
| 2 | `fig02_thuc_te_vs_du_bao.png` - Thực tế so với dự báo (toàn bộ tập test) | model_table + demand_model | Điểm nằm trên đường chéo là dự báo đúng; thang thường và thang log. |
| 3 | `fig03_phan_bo_sai_so.png` - Phân phối sai số dự báo | model_table + demand_model | Sai số lệch về phía nào, đuôi dài đến đâu. |
| 4 | `fig04_sai_so_theo_nhom.png` - Sai số theo nhóm giá và nhóm mức bán | model_table + demand_model | Mô hình kém ở nhóm nào; có cả sai số chuẩn hóa theo sản phẩm. |
| 5 | `fig05_sai_so_theo_thang.png` - Sai số theo tháng | model_table + demand_model | Mô hình kém ở giai đoạn nào (mùa lễ, đầu năm), và dự báo thiên cao hay thấp. |
| 6 | `fig06_duong_cau.png` - Đường cầu dự báo của 4 sản phẩm | pricing_engine (models_2y) | Lượng bán theo giá, kèm dải 10-90% theo bất định của elasticity. |
| 7 | `fig07_phan_ung_gia.png` - Phản ứng giá: dự báo so với thực tế | weekly + model_table + demand_model + pricing_params | Ở các lần đổi giá thật, mô hình có dự báo đúng hướng và độ lớn không. |
| 8 | `fig08_phan_bo_elasticity.png` - Phân phối elasticity và xác suất cầu co giãn | models_2y/pricing_params_online_retail.csv | Phần lớn sản phẩm nằm đâu so với ngưỡng -1. |
| 9 | `fig09_de_xuat_gia.png` - Đề xuất giá cho 20 sản phẩm | pricing_engine (models_2y) | Đổi giá bao nhiêu phần trăm, theo doanh thu và theo lợi nhuận (giá vốn giả định). |
| 10 | `fig10_duong_doanh_thu.png` - Doanh thu kỳ vọng theo giá (4 sản phẩm) | pricing_engine (models_2y) | Giá hiện tại và giá đề xuất nằm đâu trên đường doanh thu. |

Lưu ý khi dùng trong báo cáo:

- Biểu đồ 9 và 10 phụ thuộc vào ngày áp dụng giá và (ở chế độ lợi nhuận) giá vốn GIẢ ĐỊNH 60%. Không dùng làm kết luận về lợi nhuận thật.
- Biểu đồ 7: tuần xảy ra đổi giá thuộc tập test, nhưng hồi quy elasticity được ước lượng trên cả 2 năm, nên đây là so sánh mô tả, không phải kiểm định ngoài mẫu. Kiểm định ngoài mẫu nằm ở `5_fit_pricing_model_2y.py`.
- Tập test là mỗi tuần thứ 5 (xen kẽ), không phải kiểm định ngoài năm.
