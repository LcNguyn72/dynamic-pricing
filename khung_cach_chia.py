"""
BƯỚC 1: KHUNG THƯ MỤC THEO CÁCH CHIA
====================================
Chỉ tạo và quản lý cấu trúc thư mục. Chưa huấn luyện, chưa di chuyển file cũ.

Cấu trúc (bên trong ./models_2y):
    chung/        file dùng chung cho mọi cách chia (một bản)
    xen_ke/       bản của cách chia xen kẽ mỗi tuần thứ 5
    chieu_1_2/    bản của cách chia train năm 1 -> test năm 2
    chieu_2_1/    bản của cách chia train năm 2 -> test năm 1

Quy ước:
  - Tên thư mục dùng ASCII (không dấu) để tránh lỗi trên Windows.
  - Mỗi thư mục cách chia có manifest.json ghi cách chia và danh sách file dự kiến.
  - Các file gốc còn nằm phẳng trong ./models_2y (engine hiện tại đang đọc chỗ đó). Không đụng vào
    cho đến bước 7, khi engine được trỏ sang thư mục mới.

Chạy:  python khung_cach_chia.py
"""
import json
import os
from datetime import datetime

from y2_config import OUT_DIR

CACH_CHIA = {
    "xen_ke": "Xen kẽ (mỗi tuần thứ 5 là test)",
    "chieu_1_2": "Train năm 1 -> test năm 2",
    "chieu_2_1": "Train năm 2 -> test năm 1",
}
THU_MUC_CHUNG = "chung"

# File dùng chung: không phụ thuộc cách chia
FILE_CHUNG = [
    "weekly_online_retail.csv",
    "products_online_retail.csv",
    "pricing_params_online_retail.csv",   # elasticity từ toàn bộ dữ liệu (quyết định 1)
    "pricing_model_online_retail.json",
]

# File theo từng cách chia: mỗi cách chia một bản
FILE_THEO_CACH = [
    "model_table_online_retail.csv",
    "product_stats_online_retail.json",
    "demand_model_online_retail.pkl",
    "model_metrics_online_retail.json",
    "gia_de_xuat_online_retail.csv",
]


def kiem_tra_cach_chia(cach_chia):
    if cach_chia not in CACH_CHIA:
        raise ValueError(f"Cách chia không hợp lệ: {cach_chia}. Chọn một trong {list(CACH_CHIA)}")
    return cach_chia


def thu_muc_chung():
    """Đường dẫn thư mục dùng chung, tạo nếu chưa có."""
    p = os.path.join(OUT_DIR, THU_MUC_CHUNG)
    os.makedirs(p, exist_ok=True)
    return p


def thu_muc_cach_chia(cach_chia):
    """Đường dẫn thư mục của một cách chia, tạo nếu chưa có."""
    kiem_tra_cach_chia(cach_chia)
    p = os.path.join(OUT_DIR, cach_chia)
    os.makedirs(p, exist_ok=True)
    return p


def duong_dan(cach_chia, ten_file):
    """Đường dẫn đầy đủ của một file. File chung tự động đặt trong thư mục chung."""
    if ten_file in FILE_CHUNG:
        return os.path.join(thu_muc_chung(), ten_file)
    if ten_file in FILE_THEO_CACH:
        return os.path.join(thu_muc_cach_chia(cach_chia), ten_file)
    raise ValueError(f"File không nằm trong danh sách khung: {ten_file}")


def tao_manifest(cach_chia):
    """Ghi manifest.json cho một cách chia. Không ghi đè nếu đã có, chỉ cập nhật thời gian."""
    p = os.path.join(thu_muc_cach_chia(cach_chia), "manifest.json")
    du_lieu = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            du_lieu = json.load(f)
    du_lieu.update({
        "cach_chia": cach_chia,
        "mo_ta": CACH_CHIA[cach_chia],
        "file_du_kien": FILE_THEO_CACH,
        "cap_nhat_luc": datetime.now().isoformat(timespec="seconds"),
    })
    with open(p, "w", encoding="utf-8") as f:
        json.dump(du_lieu, f, indent=2, ensure_ascii=False)
    return p


def kiem_tra_ghi_duoc():
    """Thử ghi và xóa một file tạm trong từng thư mục để chắc chắn quyền ghi."""
    loi = []
    for ten, thu_muc in [(THU_MUC_CHUNG, thu_muc_chung())] + [(c, thu_muc_cach_chia(c)) for c in CACH_CHIA]:
        p = os.path.join(thu_muc, ".kiem_tra_ghi")
        try:
            with open(p, "w") as f:
                f.write("ok")
            os.remove(p)
        except OSError as e:
            loi.append(f"{ten}: {e}")
    return loi


def main():
    print("=" * 70)
    print("BƯỚC 1: KHUNG THƯ MỤC THEO CÁCH CHIA")
    print("=" * 70)
    print(f"Thư mục gốc: {os.path.abspath(OUT_DIR)}")
    thu_muc_chung()
    for c in CACH_CHIA:
        tao_manifest(c)
    loi = kiem_tra_ghi_duoc()
    print("\nCấu trúc:")
    print(f"  {THU_MUC_CHUNG}/  (một bản)")
    for f in FILE_CHUNG:
        print(f"      - {f}")
    for c, mo_ta in CACH_CHIA.items():
        print(f"  {c}/  ({mo_ta})")
        for f in FILE_THEO_CACH:
            print(f"      - {f}")
    if loi:
        print("\nLỖI quyền ghi:")
        for l in loi:
            print("  " + l)
    else:
        print("\nKiểm tra quyền ghi: OK cho cả bốn thư mục.")
    print("Các file phẳng hiện có trong ./models_2y không bị thay đổi.")


if __name__ == "__main__":
    main()
