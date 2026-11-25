import random
import time
from config import MAX_PAGES

from product_parse import (
    filter_products,
    pick_references,
)

from config import (
    BASE_URL,
    INPUT_FILE,
    REF_FILE,
    MAX_PAGES,
    TOP_N,
    MIN_SOLD,
    ID_RE,
    MAX_PROXY_ATTEMPTS,
    TIMEZONE
)

NEXT_SELECTORS = [
    "li.ant-pagination-next:not(.ant-pagination-disabled)",
    "li[title='Next Page']:not(.ant-pagination-disabled)",
    "li[title='Trang sau']:not(.ant-pagination-disabled)",
]

def click_next(page):
    page.evaluate(
        "window.scrollTo(0, document.body.scrollHeight)"
    )
    page.wait_for_timeout(
        random.randint(1000, 2000)
    )
    for selector in NEXT_SELECTORS:
        locator = page.locator(selector).first
        try:
            if locator.count() == 0:
                continue
            locator.scroll_into_view_if_needed(
                timeout=3000
            )
            page.wait_for_timeout(
                random.randint(500, 1200)
            )
            locator.click(timeout=5000)
            return True

        except Exception as e:
            print(
                f"[SEARCH] Click thất bại "
                f"({selector}): {e}"
            )
    if page.locator(".ant-pagination").count():
        print(
            "[SEARCH] Không thể chuyển trang "
            "hoặc đã hết trang."
        )
    else:
        print(
            "[SEARCH] Không thấy thanh phân trang."
        )

    return False

def wait_for_items(page, bucket, timeout_ms):
    waited = 0

    while waited < timeout_ms:
        if bucket:
            return True
        page.wait_for_timeout(500)
        waited += 500

    return bool(bucket)

def wait_and_scroll(page, minimum=4, maximum=10):
    try:
        page.wait_for_timeout(
            random.randint(minimum * 1000, maximum * 1000)
        )

        for _ in range(random.randint(2, 4)):
            distance = random.randint(250, 600)

            page.evaluate(
                "(d) => window.scrollBy(0, d)",
                distance
            )

            page.wait_for_timeout(
                random.randint(minimum * 1000, maximum * 1000)
            )

        page.evaluate(
            "(d) => window.scrollBy(0, d)",
            random.randint(250, 600)
        )

        page.wait_for_timeout(
            random.randint(minimum * 1000, maximum * 1000)
        )

    except Exception as e:
        print(f"[WAIT & SCROLL] {e}")

def check_challenge(page):
    try:
        url = page.url.lower()
        title = page.title().lower()

        # 1. Kiểm tra URL trang chính
        challenge_url_indicators = [
            "captcha",
            "_____tmd_____",
            "action=captcha",
            "/punish",
            "third_validate_close_page",
        ]

        if any(keyword in url for keyword in challenge_url_indicators):
            return True

        # 2. Kiểm tra title
        indicators = [
            "captcha",
            "security check",
            "verify you are",
            "verification",
            "verify to continue",
            "security verification",
        ]

        if any(keyword in title for keyword in indicators):
            return True

        # 3. Kiểm tra nội dung trang
        try:
            body = page.locator("body").inner_text(
                timeout=3000
            ).lower()

            body_indicators = [
                "captcha",
                "security check",
                "verify you are",
                "verification",
                "verify to continue",
                "security verification",
                "slide to verify",
                "drag to verify",
                "complete the puzzle",
                "xác minh bảo mật",
                "trượt để xác minh",
            ]

            if any(keyword in body for keyword in body_indicators):
                return True

        except Exception:
            pass

        # 4. Kiểm tra src của tất cả iframe, kể cả iframe ẩn
        try:
            iframe_locator = page.locator("iframe")
            iframe_count = iframe_locator.count()

            iframe_url_indicators = [
                "_____tmd_____",
                "recaptcha=1",
                "action=captcha",
                "/punish",
                "third_validate_close_page",
            ]

            iframe_title_indicators = [
                "captcha",
                "verification",
                "challenge",
            ]

            iframe_content_indicators = [
                "captcha",
                "security check",
                "verify you are",
                "verification",
                "verify to continue",
                "security verification",
                "slide to verify",
                "drag to verify",
                "complete the puzzle",
                "xác minh bảo mật",
                "trượt để xác minh",
            ]

            for i in range(iframe_count):
                iframe = iframe_locator.nth(i)

                try:
                    src = (
                        iframe.get_attribute("src") or ""
                    ).lower()

                    srcdoc = (
                        iframe.get_attribute("srcdoc") or ""
                    ).lower()

                    iframe_title = (
                        iframe.get_attribute("title") or ""
                    ).lower()

                    if any(
                        keyword in src
                        for keyword in iframe_url_indicators
                    ):
                        return True

                    if any(
                        keyword in iframe_title
                        for keyword in iframe_title_indicators
                    ):
                        return True

                    if any(
                        keyword in srcdoc 
                        for keyword in iframe_content_indicators
                    ):
                        return True

                except Exception:
                    continue

        except Exception as e:
            print(f"[CHALLENGE] Lỗi kiểm tra iframe: {e}")

        # 5. Kiểm tra popup
        popup_selectors = [
            '[role="dialog"]',
            '[aria-modal="true"]',
            '[class*="captcha" i]',
            '[id*="captcha" i]',
            '[class*="challenge" i]',
            '[id*="challenge" i]',
            '[class*="verification" i]',
            '[id*="verification" i]',
        ]

        popup_indicators = [
            "captcha",
            "security check",
            "verify you are",
            "verify to continue",
            "security verification",
            "slide to verify",
            "drag to verify",
            "complete the puzzle",
            "xác minh bảo mật",
            "trượt để xác minh",
        ]

        for selector in popup_selectors:
            try:
                locator = page.locator(selector)

                for i in range(locator.count()):
                    element = locator.nth(i)

                    if not element.is_visible():
                        continue

                    try:
                        text = element.inner_text(
                            timeout=1000
                        ).strip().lower()
                    except Exception:
                        text = ""

                    if any(
                        keyword in text
                        for keyword in popup_indicators
                    ):
                        return True

            except Exception:
                continue

        return False

    except Exception as e:
        print(f"[CHALLENGE] Lỗi kiểm tra: {e}")
        return False

def fetch_current(page, timeout=10000):
    try:
        data = page.evaluate("""
            async (timeout) => {
                const controller = new AbortController();

                const timer = setTimeout(
                    () => controller.abort(),
                    timeout
                );

                try {
                    const u = new URL(location.href);
                    u.searchParams.set('ajax', 'true');

                    const r = await fetch(u.toString(), {
                        credentials: 'include',
                        signal: controller.signal
                    });

                    if (!r.ok) {
                        return {};
                    }

                    return await r.json();

                } finally {
                    clearTimeout(timer);
                }
            }
        """, timeout)

        return (
            data.get("mods", {}).get("listItems", [])
            or []
        )

    except Exception as e:
        print(
            f"[SEARCH] Fetch dự phòng thất bại: {e}"
        )
        return []

def collect_items(page, bucket):
    print("[collect items] Bắt đầu thu thập dữ liệu sản phẩm...")
    if wait_for_items(page, bucket, 8000):
        return list(bucket), False
    print("[collect items] Không thu thập được dữ liệu sản phẩm, thử fetch dự phòng...")
    if check_challenge(page):
        return [], True
    print("[collect items] Phát hiện challenge, dừng thu thập dữ liệu.")
    if wait_for_items(page, bucket, 3000):
        return list(bucket), False
    print("[collect items] Không thu thập được dữ liệu sản phẩm, thử fetch dự phòng...")
    return fetch_current(page), False

def search_products(page, keyword, own, own_ids):
    all_items = {}
    bucket = []

    def handle_response(response):
        if (
            "catalog" not in response.url.lower()
            or response.status != 200
        ):
            return

        try:
            items = (
                response.json()
                .get("mods", {})
                .get("listItems", [])
            )

            if items:
                bucket.extend(items)

        except Exception:
            pass

    page.on(
        "response",
        handle_response
    )

    try:
        if check_challenge(page):
            return []

        search_box = page.locator(
            'input[name="q"], input[type="search"]'
        ).first

        search_box.wait_for(timeout=15000)
        search_box.fill(keyword)

        bucket.clear()

        search_box.press("Enter")

        page.wait_for_load_state(
            "domcontentloaded",
            timeout=30000
        )

        for page_number in range(
            1,
            MAX_PAGES + 1
        ):
            print(
                f"[SEARCH] Trang "
                f"{page_number}/{MAX_PAGES}"
            )

            if page_number > 1:
                bucket.clear()

                if not click_next(page):
                    break

            wait_and_scroll(page, 2, 3)

            items, challenge_detected = collect_items(
                page,
                bucket
            )

            if not items:
                print(
                    "[SEARCH] Không lấy được dữ liệu "
                    "trang hiện tại."
                )
                break

            for item in items:
                item_id = str(
                    item.get("itemId", "")
                )

                if item_id:
                    all_items[item_id] = item

            filtered = filter_products(
                list(all_items.values()),
                own,
                own_ids
            )

            selected = pick_references(
                filtered
            )

            print(
                f"[SEARCH] Tổng: {len(all_items)}"
                f" | Đạt: {len(filtered)}"
                f" | Shop khác nhau: {len(selected)}"
            )

            if len(selected) >= TOP_N:
                print(
                    "[SEARCH] Đã đủ số sản phẩm tham chiếu."
                )
                break

    finally:
        page.remove_listener(
            "response",
            handle_response
        )

    return list(all_items.values())

def search_by_ids(page, keyword, target_ids, max_pages=MAX_PAGES):
    target_ids = {
        str(item_id).strip()
        for item_id in target_ids
        if str(item_id).strip()
    }

    if not target_ids:
        return {}, False

    found_items = {}
    bucket = []

    def handle_response(response):
        if (
            "catalog" not in response.url.lower()
            or response.status != 200
        ):
            return

        try:
            data = response.json()

            items = (
                data.get("mods", {})
                .get("listItems", [])
            )

            if items:
                bucket.extend(items)

        except Exception:
            pass

    page.on("response", handle_response)

    try:
        page.wait_for_load_state(
            "domcontentloaded",
            timeout=30000,
        )
        if check_challenge(page):
            return {}, True

        # Tìm ô tìm kiếm.
        search_box = page.locator(
            'input[name="q"], input[type="search"]'
        ).first

        search_box.wait_for(timeout=15000)
        search_box.fill(keyword)

        # Xóa dữ liệu cũ trước khi tìm kiếm.
        bucket.clear()

        search_box.press("Enter")

        # Duyệt các trang kết quả.
        for page_number in range(1, max_pages + 1):

            print(
                f"[SEARCH_ID] Trang "
                f"{page_number}/{max_pages} | "
                f"Từ khóa: {keyword}"
            )
            page.wait_for_load_state(
                "domcontentloaded",
                timeout=30000,
            )
            if check_challenge(page):
                return {}, True
            if page_number > 1:
                bucket.clear()

                if not click_next(page):
                    break
            wait_and_scroll(page, 2, 3)

            items, challenge_detected = collect_items(
                page,
                bucket,
            )
            if challenge_detected:
                print(
                    "[SEARCH_ID] Phát hiện CAPTCHA "
                )
                return {}, True
            if not items:
                print(
                    "[SEARCH_ID] Không lấy được "
                    "dữ liệu trang hiện tại."
                )
                break
                
            for item in items:
                item_id = str(
                    item.get("itemId") or ""
                ).strip()

                if item_id in target_ids:
                    found_items[item_id] = item

            print(
                f"[SEARCH_ID] Tìm thấy "
                f"{len(found_items)}/{len(target_ids)} ID"
            )

            if target_ids.issubset(found_items.keys()):
                print("[SEARCH_ID] Đã tìm đủ ID.")
                break

    except Exception as e:
        print(
            f"[SEARCH_ID] Lỗi tìm kiếm "
            f"'{keyword}': {e}"
        )

    finally:
        page.remove_listener(
            "response",
            handle_response,
        )

    missing_ids = target_ids - found_items.keys()

    if missing_ids:
        print(
            f"[SEARCH_ID] Chưa tìm thấy "
            f"{len(missing_ids)} ID: "
            f"{', '.join(sorted(missing_ids))}"
        )

    return found_items, False
