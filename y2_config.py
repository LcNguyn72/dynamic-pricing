"""
Cấu hình chung cho pipeline 2 NĂM (Online Retail II)
====================================================
Các file *_2y.py đọc cấu hình ở đây. Bản 1 năm trong ./models KHÔNG bị động vào:
pipeline 2 năm ghi hết vào ./models_2y.
"""
import os

RAW_XLSX = "data/online_retail_II.xlsx"
RAW_CACHE = "data/online_retail_II_sheets.csv"   # cache đọc nhanh (có cột 'sheet'), tự tạo ở lần chạy đầu
OUT_DIR = "./models_2y"                          # nơi ghi kết quả của pipeline 2 năm
ONE_YEAR_DIR = "./models"                        # bản 1 năm (chỉ ĐỌC để so sánh)

MIN_WEEKS_FILTER = 60    # lọc sản phẩm: >= 60 tuần có bán (~57% của ~105 tuần, giống 30/53 của bản 1 năm)
MIN_CV = 0.05            # độ biến thiên tối thiểu của giá tuần
MIN_TRAIN_WEEKS = 20     # số tuần train tối thiểu của một sản phẩm (bản 1 năm: 10)
TEST_EVERY = 5           # giữ ra mỗi tuần thứ 5 làm test
OOT_CUT_WEEK = "2010-11-29"   # mốc cắt kiểm định ngoài thời gian: trước mốc = năm 1, từ mốc = năm 2


def path(name):
    return os.path.join(OUT_DIR, name)
