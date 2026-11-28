# Quầy Giá

Ứng dụng hỗ trợ chủ cửa hàng so sánh giá, lượng bán và doanh thu dự kiến cho 20 sản phẩm tại 10 cửa hàng M5. ARIMAX ước lượng lượng bán ở từng giá ứng viên. Ứng dụng đề xuất giá có doanh thu dự kiến cao nhất trong các mức đã thử. Chủ cửa hàng có thể áp dụng giá cho một hoặc nhiều cửa hàng.

## Chạy demo

Trên máy đã cài đặt, mở **hai cửa sổ PowerShell tại thư mục có README này**. Giữ cả hai cửa sổ mở.

**Cửa sổ 1 — API mô hình**

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn final_system.api:app --host 127.0.0.1 --port 8766
```

**Cửa sổ 2 — Quầy Giá**

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn backend.m5_api:app --host 127.0.0.1 --port 18767
```

Mở [Quầy Giá](http://127.0.0.1:18767/). Để thử, vào **Dự đoán giá**, chọn một sản phẩm, cửa hàng và tuần **23/05/2016**, rồi bấm **Xem giá đề xuất**. Chuyển giữa **Lượng bán dự kiến** và **Doanh thu dự kiến** để so sánh. Nút **Áp dụng** lưu giá vào SQL Server. Giá gốc của danh mục lấy từ tuần 16/05/2016.

## Cài lần đầu trên Windows

Cần Python 3.11 trở lên, SQL Server đang chạy, ODBC Driver 18 và công cụ `sqlcmd`. Ví dụ dưới đây dùng Windows Authentication và SQL Server Express trên máy cá nhân. Tài khoản cài đặt cần quyền tạo cơ sở dữ liệu.

**1. Tải mã và cài thư viện.** Bộ mô hình đã huấn luyện đi kèm nhánh này.

```powershell
git clone --branch final_version --single-branch https://github.com/LcNguyn72/dynamic-pricing.git
cd dynamic-pricing
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

**2. Tải dữ liệu đã xử lý** và giải nén từ thư mục gốc dự án. [Gói dữ liệu V2](https://github.com/LcNguyn72/dynamic-pricing/releases/tag/du-lieu-v2)

```powershell
$zip = Join-Path $env:TEMP 'du-lieu-v2.zip'
curl.exe -fL --retry 2 -o $zip https://github.com/LcNguyn72/dynamic-pricing/releases/download/du-lieu-v2/du-lieu-v2.zip
if ((Get-FileHash $zip).Hash -ne 'f7341149c5291a44d92853fbd04648a50e3859b4c825ca086a7584464c9a2f6e') { throw 'Gói dữ liệu chưa đầy đủ' }
tar.exe -xf $zip
```

**3. Tạo và nạp cơ sở dữ liệu.** Nếu dùng SQL Server mặc định, đổi `.\SQLEXPRESS` thành `localhost` và đặt cùng biến `M5_SQL_SERVER` trong cửa sổ chạy Quầy Giá.

```powershell
$env:M5_SQL_SERVER = '.\SQLEXPRESS'
sqlcmd -S $env:M5_SQL_SERVER -E -C -b -i database/m5_schema.sql
& '.\.venv\Scripts\python.exe' -m database.import_m5 --all
sqlcmd -S $env:M5_SQL_SERVER -E -C -b -d DynamicPricingM5 -f 65001 -i database/m5_storefront.sql
```

Bước nhập đọc dữ liệu của cả 10 cửa hàng. Khi cả ba lệnh thành công, chạy hai lệnh ở mục **Chạy demo**.

## Phần kèm theo để kiểm tra bài làm

- **Lazada** — `crawler/` chứa mã thu thập và ba CSV gốc. Lượng bán được ước tính từ chênh lệch bộ đếm giữa hai lần quan sát và được gộp theo kỳ ba ngày. Chạy `& '.\.venv\Scripts\python.exe' -m final_system.training source-train --source lazada` để chuẩn hóa và huấn luyện lại. Các bảng Lazada trong báo cáo thuộc lần chạy gốc. Kết quả của lần chạy mới nằm trong `final_system/runtime/source_runs/lazada/`. [Biểu đồ lượng bán và giá](experiments/report/comparison_word/word_lazada_distributions.png)
- **Thực nghiệm UCI/M5** — bảng điểm và biểu đồ của báo cáo nằm trong [experiments/report](experiments/report). Để chạy lại đối sánh sau khi giải nén dữ liệu, cài `& '.\.venv\Scripts\python.exe' -m pip install -r requirements-experiments.txt`, rồi chạy `& '.\.venv\Scripts\python.exe' -m experiments.comparison_v3 all --device cpu`. Đây là bước tùy chọn, có thể mất nhiều thời gian. [Phương pháp đối sánh](experiments/comparison_v3/METHOD.md)
