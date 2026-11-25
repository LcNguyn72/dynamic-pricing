import csv
import random
import re
import urllib.request
import requests

from config import (
    PROXY_TIMEOUT,
    MAX_PROXY_ATTEMPTS,
)

def get_vietnam_proxies():
    url = "https://api.proxyscrape.com/v4/free-proxy-list/get"

    params = {
        "protocol":"http,https",
        "timeout":"3000",
        "country":"VN",
        "ssl":"all",
        "anonymity":"elite",
        "limit":"2000",
        "request":"displayproxies",
    }

    try:
        response = requests.get(url, params=params, timeout=(5, 20))
        response.raise_for_status()

        proxies = []

        for line in response.text.splitlines():
            line = line.strip()

            if not line or ":" not in line:
                continue

            ip, port = line.rsplit(":", 1)

            try:
                port = int(port)
            except ValueError:
                continue

            if ip and 1 <= port <= 65535:
                proxies.append(f"http://{ip}:{port}")

        print(f"[PROXY] Lấy được {len(proxies)} proxy")
        return proxies

    except requests.RequestException as e:
        print(f"[PROXY] Lỗi lấy danh sách: {e}")
        return []

def check_proxy(proxy):
    proxy_url = proxy
    try:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_url,
            "https": proxy_url,
        })
        opener = urllib.request.build_opener(proxy_handler)
        request = urllib.request.Request(
            "https://www.lazada.vn/",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                )
            }
        )
        with opener.open(
            request,
            timeout=PROXY_TIMEOUT
        ) as response:
            if response.status < 500:
                return True
    except Exception as e:
        print(f"[PROXY] Proxy lỗi: {proxy} | {e}")
        return False
    return False

def select_working_proxy(proxies):
    if not proxies:
        return None
    attempts = min(
        len(proxies),
        MAX_PROXY_ATTEMPTS
    )
    for index, proxy in enumerate(
        proxies[:attempts],
        start=1
    ):
        print(
            f"[PROXY] Kiểm tra {index}/{attempts}: "
            f"{proxy}"
        )
        if check_proxy(proxy):
            print(
                f"[PROXY] Kết nối kiểm tra thành công: "
                f"{proxy}"
            )
            return proxy
        print("[PROXY] Không kết nối được, thử proxy tiếp theo.")
    return None
