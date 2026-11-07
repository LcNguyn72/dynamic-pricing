"""
BƯỚC 2: BẢN DÙNG CHUNG (models_2y/chung)
========================================
Sao chép bốn file dùng chung từ thư mục phẳng hiện có sang models_2y/chung, rồi kiểm tra:
  - file nguồn tồn tại, không rỗng
  - nội dung giống hệt sau khi sao chép (so sánh SHA-256)
  - số dòng CSV khớp với nguồn
  - pricing_model_online_retail.json ghi phương pháp elasticity là 'global'

Không huấn luyện lại. Không sửa nội dung file. Không ghi đè nếu file đích đã khớp nguồn.
Chạy lại an toàn: nếu nguồn chưa đổi, bước này không làm gì thêm.

Chạy:  python buoc2_ban_dung_chung.py
Cần khung_cach_chia.py trong cùng thư mục.
"""
import hashlib
import json
import os
import shutil
import sys

import khung_cach_chia as KC
from y2_config import OUT_DIR


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for doan in iter(lambda: f.read(1 << 20), b""):
            h.update(doan)
    return h.hexdigest()


def so_dong_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return max(sum(1 for _ in f) - 1, 0)


def sao_chep_mot_file(ten):
    nguon = os.path.join(OUT_DIR, ten)
    dich = KC.duong_dan("xen_ke", ten)   # trả về models_2y/chung/<ten> vì là file dùng chung
    if not os.path.exists(nguon):
        return {"file": ten, "trang_thai": "THIẾU NGUỒN", "ghi_chu": f"không thấy {nguon}"}
    if os.path.getsize(nguon) == 0:
        return {"file": ten, "trang_thai": "LỖI", "ghi_chu": "file nguồn rỗng"}

    h_nguon = sha256(nguon)
    if os.path.exists(dich) and sha256(dich) == h_nguon:
        trang_thai = "ĐÃ KHỚP (bỏ qua)"
    else:
        shutil.copy2(nguon, dich)
        trang_thai = "ĐÃ SAO CHÉP"

    if sha256(dich) != h_nguon:
        return {"file": ten, "trang_thai": "LỖI", "ghi_chu": "hash đích khác nguồn sau khi sao chép"}

    ghi_chu = ""
    if ten.endswith(".csv"):
        n_nguon, n_dich = so_dong_csv(nguon), so_dong_csv(dich)
        ghi_chu = f"{n_dich:,} dòng"
        if n_nguon != n_dich:
            return {"file": ten, "trang_thai": "LỖI", "ghi_chu": f"số dòng {n_dich} khác nguồn {n_nguon}"}
    return {"file": ten, "trang_thai": trang_thai, "ghi_chu": ghi_chu, "sha256": h_nguon[:12]}


def kiem_tra_phuong_phap_elasticity():
    """Kiểm tra cảnh báo: nếu elasticity được chọn không phải 'global', nó có thể phụ thuộc XGBoost,
    và khi đó cần chạy lại bước 5 cho từng cách chia (hiện chỉ dùng chung một bản)."""
    p = os.path.join(KC.thu_muc_chung(), "pricing_model_online_retail.json")
    if not os.path.exists(p):
        return "THIẾU", "chưa có pricing_model trong chung/"
    with open(p, encoding="utf-8") as f:
        du = json.load(f)
    phuong_phap = str(du.get("method", du.get("phuong_phap", "")))
    if phuong_phap == "global":
        return "OK", "phương pháp = global (không phụ thuộc XGBoost)"
    return "CẢNH BÁO", f"phương pháp = '{phuong_phap}', cần kiểm tra có dùng XGBoost không"


def main():
    print("=" * 70)
    print("BƯỚC 2: BẢN DÙNG CHUNG (models_2y/chung)")
    print("=" * 70)
    KC.thu_muc_chung()
    ket = [sao_chep_mot_file(f) for f in KC.FILE_CHUNG]
    print(f"\n{'File':<36}{'Trạng thái':<22}Ghi chú")
    for r in ket:
        print(f"{r['file']:<36}{r['trang_thai']:<22}{r.get('ghi_chu', '')}")

    tt, ghi = kiem_tra_phuong_phap_elasticity()
    print(f"\nKiểm tra elasticity: {tt} - {ghi}")

    loi = [r for r in ket if r["trang_thai"] in ("LỖI", "THIẾU NGUỒN")]
    if loi or tt == "THIẾU":
        print("\nBƯỚC 2 CHƯA HOÀN TẤT. Sửa các mục trên rồi chạy lại.")
        sys.exit(1)
    if tt == "CẢNH BÁO":
        print("\nBƯỚC 2 HOÀN TẤT NHƯNG CÒN CẢNH BÁO: xem mục elasticity trước khi sang bước 3.")
    else:
        print("\nBƯỚC 2 HOÀN TẤT.")


if __name__ == "__main__":
    main()
