Phân tích bổ sung cho báo cáo Nền tảng Tối ưu giá bán động.
Chạy từ gốc repo với Python >=3.11, numpy, pandas, matplotlib:
python output/report_revision_support/analyze_v2.py
Cần panel tại final_system/runtime/prepared.
Script kiểm tra hash, khóa test, tính các baseline, tạo histogram và không ghi đè kết quả Phase 5.
Các baseline được chạy bổ sung; không huấn luyện lại neural.
Đọc analysis_manifest.json và report_verification.json để xem phạm vi kiểm tra.
