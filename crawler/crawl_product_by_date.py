import csv
from datetime import datetime
from pathlib import Path
import time
import tempfile

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from config import (
    BASE_URL,
    REF_FILE,
    DAILY_FILE,
    MAX_PAGES,
    MAX_IDLE_ROUNDS,
)

from proxy_manager import (
    get_vietnam_proxies,
    select_working_proxy,
)

from product_parse import (
    parse_sold,
    today_string
)

from browser_interact import (
    click_next,
    wait_for_items,
    wait_and_scroll,
    check_challenge,
    search_by_ids,
)


# CONFIG

FIELDS = [
    "crawl_date",
    "id",
    "price",
    "review",
    "rating_score",
    "sold",
    "is_sponsored",
    "daily_sold",
]

# INPUT

def load_reference_products():
    products = []
    seen_ids = set()

    with open(
        REF_FILE,
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            item_id = str(
                row.get("id") or ""
            ).strip()

            ref_product = str(
                row.get("ref_product") or ""
            ).strip()

            name = str(
                row.get("name") or ""
            ).strip()

            if not item_id or not ref_product or not name:
                continue

            if item_id in seen_ids:
                continue

            products.append({
                "id": item_id,
                "ref_product": ref_product,
                "name": name,
            })

            seen_ids.add(item_id)

    return products


# HISTORY CSV

def load_history():
    history = {}

    path = Path(DAILY_FILE)

    if not path.exists():
        return history

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            item_id = str(
                row.get("id") or ""
            ).strip()

            crawl_date = str(
                row.get("crawl_date") or ""
            ).strip()

            if not item_id or not crawl_date:
                continue

            history[(crawl_date, item_id)] = {
                field: row.get(field, "")
                for field in FIELDS
            }

    return history


def save_history(history):
    with open(
        DAILY_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for key in sorted(history):
            writer.writerow(history[key])


def get_previous_record(history, item_id, current_date):

    previous_records = []

    for (crawl_date, saved_id), row in history.items():
        if saved_id != item_id:
            continue

        if crawl_date >= current_date:
            continue

        sold = row.get("sold", "")

        if sold in ("", None):
            continue

        try:
            sold_value = int(float(sold))
        except (ValueError, TypeError):
            continue

        previous_records.append((
            crawl_date,
            sold_value,
        ))

    if not previous_records:
        return None

    previous_records.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return previous_records[0][1]

def get_crawled_ids_today(history, crawl_date):
    crawled_ids = set()

    for (saved_date, item_id), row in history.items():
        if saved_date != crawl_date:
            continue

        if item_id and row.get("sold", "") != "":
            crawled_ids.add(item_id)

    return crawled_ids


# PARSE

def make_product_data(item):
    return {
        "id": str(
            item.get("itemId") or ""
        ).strip(),

        "price": item.get("price"),

        "review": item.get("review", ""),

        "rating_score": item.get("ratingScore", ""),

        "sold": parse_sold(item),

        "is_sponsored": item.get(
            "isSponsored", ""
        ),
    }


# SAVE DAILY DATA

def make_row(item, crawl_date, daily_sold=""):
    try:
        rating = (
            round(float(item["rating_score"]), 2)
            if item.get("rating_score") not in ("", None)
            else ""
        )
    except (ValueError, TypeError):
        rating = ""

    return {
        "crawl_date": crawl_date,
        "id": str(item.get("id") or "").strip(),
        "price": item.get("price", ""),
        "review": item.get("review", ""),
        "rating_score": rating,
        "sold": item.get("sold", ""),
        "is_sponsored": item.get("is_sponsored", ""),
        "daily_sold": daily_sold,
    }

def update_product_history(history, item, crawl_date):

    item_id = item.get("id", "")

    if not item_id:
        return False

    previous_sold = get_previous_record(
        history,
        item_id,
        crawl_date,
    )

    current_sold = item.get("sold", "")

    daily_sold = ""

    if (
        previous_sold is not None
        and current_sold not in ("", None)
    ):
        try:
            difference = (
                int(current_sold) - previous_sold
            )

            if difference >= 0:
                daily_sold = difference
            else:
                print(
                    f"[WARN] Sold giảm với ID {item_id}: "
                    f"{previous_sold} -> {current_sold}"
                )

        except (ValueError, TypeError):
            daily_sold = ""

    row = make_row(
        item,
        crawl_date,
        daily_sold,
    )

    history[(crawl_date, item_id)] = row

    return True


# PROXY

def create_browser_context(playwright, proxy):
    # profile_dir = tempfile.mkdtemp(prefix="chrome_profile_")
    # return playwright.chromium.launch_persistent_context(
    #     user_data_dir=profile_dir,
    #     channel="chrome",
    #     headless=True,
    #     locale="vi-VN",
    #     timezone_id="Asia/Ho_Chi_Minh",
    #     viewport={
    #         "width": 1366,
    #         "height": 768,
    #     },
    #     proxy={"server": proxy},
    #     args=[
    #         "--no-sandbox",
    #         "--disable-dev-shm-usage",
    #     ]
    # )
    profile_manager = tempfile.TemporaryDirectory(
        prefix="firefox_profile_"
    )

    profile_dir = profile_manager.name

    context_options = {
        "user_data_dir": profile_dir,
        "headless": False,
        "locale": "vi-VN",
        "timezone_id": "Asia/Ho_Chi_Minh",
        "viewport": {
            "width": 1366,
            "height": 768,
        },
    }

    if PROXY_MODE == MODE.PROXY:
        if not proxy:
            profile_manager.cleanup()
            raise ValueError("Proxy mode yêu cầu proxy")
        context_options["proxy"] = {"server": proxy}
        
    return playwright.firefox.launch_persistent_context(
            **context_options
        )

def get_working_proxy_forever():
    while True:
        print("[PROXY] Đang lấy danh sách proxy Việt Nam...")

        try:
            proxies = get_vietnam_proxies()

            if not proxies:
                print(
                    "[PROXY] Không có proxy khả dụng. "
                    "Thử lại sau 5 phút."
                )
                time.sleep(300)
                continue

            proxy = select_working_proxy(proxies)

            if not proxy:
                print(
                    "[PROXY] Không tìm được proxy hoạt động. "
                    "Thử lại sau 5 phút."
                )
                time.sleep(300)
                continue

            print(f"[PROXY] Đã chọn: {proxy}")
            return proxy

        except Exception as e:
            print(f"[PROXY] Lỗi lấy hoặc kiểm tra proxy: {e}")
            print("[PROXY] Thử lại sau 5 phút.")
            time.sleep(300)

# MAIN

def main():
    crawl_date = today_string()

    try:
        products = load_reference_products()

    except FileNotFoundError:
        print(f"[INPUT] Không tìm thấy {REF_FILE}")
        return

    except Exception as e:
        print(f"[INPUT] Lỗi: {e}")
        return

    if not products:
        print("[INPUT] Không có sản phẩm hợp lệ.")
        return

    print(f"[MAIN] Ngày crawl: {crawl_date}")
    print(f"[MAIN] Tổng sản phẩm ref: {len(products)}")

    # -----------------------------------------------------
    # 2. Đọc lịch sử từ chính file daily_product_info.csv
    # -----------------------------------------------------

    history = load_history()

    print(f"[HISTORY] Đã đọc {len(history)} bản ghi ngày-ID.")

    groups = {}

    for product in products:
        groups.setdefault(
            product["ref_product"],
            [],
        ).append(product)

    # -----------------------------------------------------
    # 4. Khởi chạy browser
    # -----------------------------------------------------

    with sync_playwright() as p:
        all_ids = {
            product["id"]
            for product in products
        }
        found_ids = get_crawled_ids_today(
            history,
            crawl_date,
        )
        found_ids &= all_ids

        while (True):
            context = None
            captcha_detected = False
            previous_found_count = len(found_ids)
            try:
                # proxy = get_working_proxy_forever()

                # print(f"[PROXY] Sử dụng: {proxy}")

                # context = create_browser_context(p, proxy)
                context = p.firefox.launch_persistent_context(
                    user_data_dir=tempfile.mkdtemp(
                        prefix="chrome_profile_"
                    ),
                    headless=False,
                    locale="vi-VN",
                    timezone_id="Asia/Ho_Chi_Minh",
                    viewport={
                        "width": 1366,
                        "height": 768,
                    }
                )

                page = (
                    context.pages[0]
                    if context.pages
                    else context.new_page()
                )
                page.set_default_timeout(30000)
                page.set_default_navigation_timeout(60000)
                page.goto(
                    BASE_URL,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
            # -------------------------------------------------
            # 6. Search theo ref_product
            # -------------------------------------------------
                for group_index, (keyword, group) in enumerate(
                    groups.items(),
                    start=1,
                ):
                    target_ids = {
                        product["id"]
                        for product in group
                    }

                    remaining_ids = target_ids - found_ids
                    if not remaining_ids:
                        print(
                            f"[SKIP] Group đã crawl hôm nay. "
                            f"Bỏ qua {len(target_ids)} sản phẩm."
                        )
                        continue
                    if len(remaining_ids) < len(group)/3:
                        print(
                            f"[SKIP] Group còn ít sản phẩm chưa crawl. "
                            f"Bỏ qua {len(remaining_ids)} sản phẩm."
                        )
                        continue
                    print(
                        f"\n[GROUP {group_index}/{len(groups)}] "
                        f"{keyword}"
                    )

                    found, challenge = search_by_ids(
                        page,
                        keyword,
                        remaining_ids,
                        MAX_PAGES,
                    )

                    if challenge:
                        print("[CHALLENGE] Phát hiện CAPTCHA.")
                        # input("Nhấn Enter để tiếp tục sau khi giải CAPTCHA...")
                        captcha_detected = True
                        time.sleep(300)
                        break

                    for item_id, raw_item in found.items():
                        data = make_product_data(raw_item)

                        if not data["id"]:
                            continue

                        update_product_history(
                            history,
                            data,
                            crawl_date,
                        )

                        found_ids.add(item_id)

                        save_history(history)

                        print(
                            f"[SAVE] {item_id} | "
                            f"Price={data['price']} | "
                            f"Review={data['review']} | "
                            f"Sold={data['sold']}"
                        )

                if captcha_detected:
                    continue
            # -------------------------------------------------
            # 7. Fallback theo name cho ID còn thiếu
            # -------------------------------------------------

                missing_products = [
                    product
                    for product in products
                    if product["id"] not in found_ids
                ]

                print(
                    f"\n[FALLBACK] Còn "
                    f"{len(missing_products)} sản phẩm."
                )

                for index, product in enumerate(
                    missing_products,
                    start=1,
                ):
                    item_id = product["id"]
                    keyword = product["name"]

                    print(
                        f"\n[FALLBACK {index}/"
                        f"{len(missing_products)}] "
                        f"ID={item_id} | {keyword}"
                    )

                    page.goto(
                        BASE_URL,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )

                    found, challenge = search_by_ids(
                        page,
                        item_id,
                        {item_id},
                        MAX_PAGES,
                    )

                    if challenge:
                        print("[CHALLENGE] Phát hiện CAPTCHA.")
                        # input("Nhấn Enter để tiếp tục sau khi giải CAPTCHA...")
                        captcha_detected = True
                        time.sleep(300)
                        break

                    raw_item = found.get(item_id)

                    if not raw_item:
                        print(f"[NOT FOUND] ID={item_id}")
                        continue

                    data = make_product_data(raw_item)

                    if not data["id"]:
                        continue

                    update_product_history(
                        history,
                        data,
                        crawl_date,
                    )

                    found_ids.add(item_id)

                    save_history(history)

                    print(
                        f"[SAVE] {item_id} | "
                        f"Price={data['price']} | "
                        f"Review={data['review']} | "
                        f"Sold={data['sold']}"
                    )
                if captcha_detected:
                    continue
                break
            except PlaywrightTimeoutError as e:
                print(f"[BROWSER] Timeout: {e}")
                continue
            except Exception as e:
                print(f"[BROWSER] Lỗi: {e}")

            finally:
                try:
                    save_history(history)
                except Exception as e:
                    print(f"[CSV] Lỗi lưu lịch sử: {e}")

                if context is not None:
                    try:
                        context.close()
                    except Exception as e:
                        print(f"[BROWSER] Lỗi đóng context: {e}")

    print("\n========== HOÀN TẤT ==========")
    print(f"[MAIN] Số sản phẩm tìm thấy: {len(found_ids)}")
    print(f"[MAIN] Tổng bản ghi lịch sử: {len(history)}")
    print(f"[MAIN] File đầu ra: {DAILY_FILE}")


if __name__ == "__main__":
    main()
