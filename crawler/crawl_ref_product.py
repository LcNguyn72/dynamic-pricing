import csv
import re

from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

from config import (
    BASE_URL,
    INPUT_FILE,
    REF_FILE,
    PROFILE_DIR,
    MAX_PAGES,
    TOP_N,
    MIN_SOLD,
    ID_RE,
)

from proxy_manager import (
    get_vietnam_proxies,
    select_working_proxy,
)

from product_parse import (
    parse_sold,
    filter_products,
    pick_references,
)

from browser_interact import (
    click_next,
    wait_for_items,
    wait_and_scroll,
    check_challenge,
    fetch_current,
    collect_items,
    search_products,
)

# INPUT
def load_products():
    products = []

    with open(
        INPUT_FILE,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()

            price_text = re.sub(
                r"\D",
                "",
                row.get("price") or ""
            )

            if not name or not price_text:
                continue

            products.append({
                "name": name,
                "price": int(price_text),
                "url": row.get("url", ""),
            })

    return products


def get_own_ids(products):
    ids = set()

    for product in products:
        match = ID_RE.search(
            product.get("url", "")
        )

        if match:
            ids.add(match.group(1))

    return ids

# OUTPUT
FIELDS = [
    "ref_product",
    "id",
    "name",
    "location",
    "review",
    "ratingScore",
    "sellerName",
    "isSponsored",
    "url",
    "sold",
]

def save_csv(rows):
    with open(
        REF_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(rows)

def make_row(own, item):
    try:
        rating = (
            round(float(item["ratingScore"]), 2)
            if item.get("ratingScore")
            else ""
        )
    except (ValueError, TypeError):
        rating = ""

    return {
        "ref_product": own["name"],
        "id": item.get("itemId", ""),
        "name": item.get("name", ""),
        "location": item.get("location", ""),
        "review": item.get("review", ""),
        "ratingScore": rating,
        "sellerName": item.get("sellerName", ""),
        "isSponsored": item.get("isSponsored", False),
        "url": urljoin(
            BASE_URL,
            item.get("itemUrl", "")
        ),
        "sold": item.get("_sold", 0),
    }

# MAIN

def main():
    # -----------------------------------------------------
    # 1. Đọc sản phẩm
    # -----------------------------------------------------

    try:
        products = load_products()
    except FileNotFoundError:
        print(
            f"Không tìm thấy file {INPUT_FILE}"
        )
        return

    if not products:
        print(
            f"Không có sản phẩm hợp lệ trong {INPUT_FILE}"
        )
        return

    own_ids = get_own_ids(products)

    print(
        f"[MAIN] Đã đọc {len(products)} sản phẩm."
    )

    # -----------------------------------------------------
    # 2. Đọc và kiểm tra proxy
    # -----------------------------------------------------

    proxies = get_vietnam_proxies()

    if not proxies:
        print(
            "[PROXY] Không có proxy hợp lệ. "
            "Dừng chương trình."
        )
        return

    proxy = select_working_proxy(
        proxies
    )

    if not proxy:
        print(
            "[PROXY] Không tìm thấy proxy kết nối được "
            "trong số proxy đã thử."
        )
        return

    print(
        f"[PROXY] Sử dụng: {proxy}"
    )

    # -----------------------------------------------------
    # 3. Khởi chạy trình duyệt qua proxy
    # -----------------------------------------------------

    results = []

    with sync_playwright() as p:
        context = None

        try:
            context = p.chromium.launch_persistent_context(
                PROFILE_DIR,
                channel="chrome",
                headless=False,
                locale="vi-VN",
                timezone_id="Asia/Ho_Chi_Minh",
                viewport={
                    "width": 1366,
                    "height": 768,
                },
                proxy={"server": proxy},
            )

            page = (
                context.pages[0]
                if context.pages
                else context.new_page()
            )

            # -------------------------------------------------
            # 4. Kiểm tra IP egress qua proxy
            # -------------------------------------------------

            try:
                page.goto(
                    "https://api.ipify.org/",
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                current_ip = page.locator(
                    "body"
                ).inner_text(timeout=5000).strip()

                print(
                    f"[PROXY] IP kết nối hiện tại: {current_ip}"
                )

            except Exception as e:
                print(
                    f"[PROXY] Không xác minh được IP: {e}"
                )
                print(
                    "[PROXY] Dừng để tránh tiếp tục "
                    "khi chưa xác nhận được kết nối."
                )
                return

            # -------------------------------------------------
            # 5. Tìm kiếm từng sản phẩm
            # -------------------------------------------------

            for index, own in enumerate(
                products,
                start=1
            ):
                print(
                    f"\n[{index}/{len(products)}] "
                    f"{own['name']} | {own['price']}"
                )

                try:
                    page.goto(
                        BASE_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    items = search_products(
                        page,
                        own["name"],
                        own,
                        own_ids
                    )

                    if not items:
                        print(
                            "[MAIN] Không lấy được dữ liệu "
                            "sản phẩm."
                        )
                        continue

                    filtered = filter_products(
                        items,
                        own,
                        own_ids
                    )

                    selected = pick_references(
                        filtered
                    )

                    results.extend(
                        make_row(own, item)
                        for item in selected
                    )

                    save_csv(results)

                    print(
                        f"[MAIN] Phù hợp: {len(filtered)}"
                        f" | Đã chọn: {len(selected)}"
                        + (
                            " (thiếu shop)"
                            if len(selected) < TOP_N
                            else ""
                        )
                    )

                except Exception as e:
                    print(
                        f"[MAIN] Lỗi sản phẩm: {e}"
                    )

                    save_csv(results)

        except Exception as e:
            print(
                f"[BROWSER] Không thể khởi chạy "
                f"hoặc sử dụng Chromium: {e}"
            )

        finally:
            save_csv(results)

            if context is not None:
                context.close()

    print(
        f"\nHoàn tất: {len(results)} dòng "
        f"đã lưu vào {REF_FILE}"
    )


if __name__ == "__main__":
    main()