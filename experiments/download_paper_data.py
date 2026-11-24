"""Download the two source datasets used by Song (2026).

The paper uses UCI Online Retail for demand forecasting and M5 for pricing.
M5 is fetched from a public copy and verified against the MD5 checksums and
sizes in the University of Nicosia's Zenodo deposit of the Kaggle CSVs.
Downloads are kept outside Git under data/song_replication/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "song_replication"
UCI_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
ZENODO_RECORD = "https://zenodo.org/records/10203108"
ZENODO_API = "https://zenodo.org/api/records/10203108/files/{name}/content"
M5_MIRROR = "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/m5-forecasting-accuracy/{name}"
M5_FILES = {
    "sales_train_validation.csv": (120007726, "26a366a25beb57b0a8f4c7b148758f94"),
    "sales_train_evaluation.csv": (121736518, "b806dfc9f30a745102b708c09951f6aa"),
    "sell_prices.csv": (203395785, "08c591caa99e55daf3e0ccac913f7c85"),
    "calendar.csv": (103469, "3ffeab2991b0c8e861d008b39ea4c95c"),
    "sample_submission.csv": (5228786, "c281a69d7c011274899d92020a66e25b"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "online-retail", "m5"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    root = args.output_dir
    files: list[dict] = []
    if args.dataset in {"all", "online-retail"}:
        archive = root / "online_retail" / "online_retail_uci.zip"
        workbook = root / "online_retail" / "Online Retail.xlsx"
        download(UCI_URL, archive)
        extract_online_retail(archive, workbook)
        files.extend([
            {"path": str(archive.resolve()), "source": UCI_URL, "bytes": archive.stat().st_size,
             "sha256": file_hash(archive, "sha256")},
            {"path": str(workbook.resolve()), "source": UCI_URL + "#Online Retail.xlsx",
             "bytes": workbook.stat().st_size, "sha256": file_hash(workbook, "sha256")},
        ])
    if args.dataset in {"all", "m5"}:
        for name, (size, md5) in M5_FILES.items():
            target = root / "m5" / name
            url = M5_MIRROR.format(name=name)
            download(url, target, size=size, md5=md5)
            files.append({"path": str(target.resolve()), "source": url, "bytes": size,
                          "md5": md5, "checksum_source": ZENODO_API.format(name=name),
                          "sha256": file_hash(target, "sha256")})
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "Song (2026), https://doi.org/10.1007/s10791-026-10745-2",
        "online_retail_source": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "m5_checksum_source": ZENODO_RECORD,
        "m5_source_note": "Downloaded from the public Hugging Face copy; each file is checked against the University of Nicosia Zenodo deposit's MD5 and size",
        "files": files,
    }
    with (root / "download_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(root / "download_manifest.json", flush=True)


if __name__ == "__main__":
    main()
