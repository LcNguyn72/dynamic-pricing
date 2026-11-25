import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo 
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent

load_dotenv(ROOT_DIR / ".env")


def get_path(name, default):
    path = Path(os.getenv(name, default))

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path


BASE_URL = os.getenv(
    "BASE_URL",
    "https://www.lazada.vn/"
)

INPUT_FILE = get_path(
    "INPUT_FILE",
    "data/products.csv"
)

# Output
REF_FILE = get_path(
    "REF_FILE",
    "data/ref_products.csv"
)

DAILY_FILE = get_path(
    "DAILY_FILE",
    "data/daily_products.csv"
)

LOG_FILE = get_path(
    "LOG_FILE",
    "log/app.log"
)
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))
TOP_N = int(os.getenv("TOP_N", "5"))
MIN_SOLD = int(os.getenv("MIN_SOLD", "10"))

PROXY_TIMEOUT = int(
    os.getenv("PROXY_TIMEOUT", "15")
)

MAX_PROXY_ATTEMPTS = int(
    os.getenv("MAX_PROXY_ATTEMPTS", "5")
)

ID_RE = re.compile(r"-?i(\d{6,})")

TIMEZONE = ZoneInfo(
    os.getenv("TIMEZONE", "Asia/Ho_Chi_Minh")
)

MAX_IDLE_ROUNDS = int(
    os.getenv("MAX_IDLE_ROUNDS", "2")
)
