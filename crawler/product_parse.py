import re
from datetime import datetime
from zoneinfo import ZoneInfo

from config import MIN_SOLD, TOP_N, TIMEZONE, MAX_PAGES

def get_price_range(price):
    delta = max(price * 0.4, 20000)

    return (
        max(0, price - delta),
        price + delta
    )

def parse_sold(item):

    querystring = item.get("querystring", "") or ""

    match = re.search(
        r"(?:^|&)sale=(\d+)",
        querystring
    )

    if match:
        return int(match.group(1))

    text = str(
        item.get("itemSoldCntShow", "")
    ).lower()

    match = re.search(
        r"([\d.,]+)\s*(k|m)?",
        text
    )

    if not match:
        return 0

    try:
        number = float(
            match.group(1).replace(",", ".")
        )
    except ValueError:
        return 0

    multiplier = {
        "k": 1000,
        "m": 1000000,
    }.get(match.group(2), 1)

    return int(number * multiplier)

def filter_products(items, own, own_ids):
    low, high = get_price_range(
        own["price"]
    )

    candidates = {}

    for item in items:
        item_id = str(
            item.get("itemId", "")
        )

        if not item_id or item_id in own_ids:
            continue

        try:
            price = float(
                item.get("price", 0)
            )
        except (ValueError, TypeError):
            continue

        sold = parse_sold(item)

        if low <= price <= high and sold >= MIN_SOLD:
            item["_sold"] = sold
            candidates[item_id] = item

    return sorted(
        candidates.values(),
        key=lambda item: item["_sold"],
        reverse=True
    )

def pick_references(items):
    used_shops = set()
    selected = []

    for item in items:
        shop = (
            item.get("sellerName") or ""
        ).strip()

        if not shop or shop in used_shops:
            continue

        used_shops.add(shop)
        selected.append(item)

        if len(selected) >= TOP_N:
            break

    return selected

def today_string():
    return datetime.now(TIMEZONE).date().isoformat()
