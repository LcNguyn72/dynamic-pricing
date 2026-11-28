"""Download and verify the frozen UCI and M5 source files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib


UCI_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
M5_MIRROR = "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/{name}"
M5_FILES = {
    "sales_train_evaluation.csv": (121736518, "b806dfc9f30a745102b708c09951f6aa"),
    "sell_prices.csv": (203395785, "08c591caa99e55daf3e0ccac913f7c85"),
    "calendar.csv": (103469, "3ffeab2991b0c8e861d008b39ea4c95c"),
}


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, *, size: int | None = None, md5: str | None = None) -> None:
    """Stream to a temporary file and publish only a complete verified copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if (size is None or target.stat().st_size == size) and (md5 is None or file_hash(target, "md5") == md5):
            print(f"[verified existing] {target.name}", flush=True)
            return
        raise ValueError(f"Existing file failed verification; inspect it before replacing: {target}")
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "dynamic-pricing-research/1.0"})
    received = 0
    next_notice = 25 * 1024 * 1024
    digest = hashlib.md5()
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(256 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received >= next_notice:
                    print(f"[download] {target.name}: {received / 1_000_000:.1f} MB", flush=True)
                    next_notice += 25 * 1024 * 1024
        if size is not None and received != size:
            raise ValueError(f"Size mismatch for {target.name}: {received} != {size}")
        if md5 is not None and digest.hexdigest() != md5:
            raise ValueError(f"MD5 mismatch for {target.name}: {digest.hexdigest()} != {md5}")
        temporary.replace(target)
        print(f"[verified] {target.name}: {received:,} bytes", flush=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def extract_online_retail(archive: Path, target: Path) -> None:
    with ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name.lower() == "online retail.xlsx"]
        if len(matches) != 1:
            raise ValueError(f"Expected one Online Retail.xlsx in UCI archive, got {matches}")
        member = zipped.getinfo(matches[0])
    if target.is_file():
        crc = 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                crc = zlib.crc32(block, crc)
        if target.stat().st_size != member.file_size or crc != member.CRC:
            raise ValueError(f"Existing UCI workbook does not match the source ZIP: {target}")
        print(f"[verified existing] {target.name}: {target.stat().st_size:,} bytes", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        with ZipFile(archive) as zipped:
            with zipped.open(matches[0]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[extracted] {target.name}: {target.stat().st_size:,} bytes", flush=True)
