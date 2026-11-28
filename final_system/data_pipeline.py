"""Download or prepare the frozen UCI and M5 weekly datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bundle import DEFAULT_PREPARED_ROOT, DEFAULT_RAW_ROOT, PROTOCOL_PATH, read_json, verify_file
from .pipeline.sources import (
    M5_FILES, M5_MIRROR, UCI_URL, download, extract_online_retail,
)

M5_SOURCES = {"sales_train_evaluation.csv": "sales_sha256", "sell_prices.csv": "price_sha256",
              "calendar.csv": "calendar_sha256"}


def verify_raw(raw):
    protocol = read_json(PROTOCOL_PATH)["datasets"]
    workbook = raw / "online_retail" / "Online Retail.xlsx"
    verify_file(workbook, protocol["uci_online_retail"]["source_sha256"])
    for filename, key in M5_SOURCES.items():
        verify_file(raw / "m5" / filename, protocol["m5"][key])
    return workbook


def download_sources(raw):
    archive = raw / "online_retail" / "online_retail_uci.zip"
    workbook = raw / "online_retail" / "Online Retail.xlsx"
    if not workbook.exists():
        download(UCI_URL, archive)
        extract_online_retail(archive, workbook)
    for name in M5_SOURCES:
        size, md5 = M5_FILES[name]
        download(M5_MIRROR.format(name=name), raw / "m5" / name, size=size, md5=md5)
    verify_raw(raw)
    print(f"[final_system] Verified sources: {raw.resolve()}", flush=True)


def _verify_existing_panels(output, contract):
    uci = contract["phase1"]["uci"]
    verify_file(output / "uci" / "weekly.csv", uci["weekly_sha256"])
    verify_file(output / "uci" / "panel" / "manifest.json", uci["panel_manifest_sha256"])
    verify_file(output / "uci" / "panel" / "windows.csv", uci["windows_sha256"])
    for store, record in contract["m5_stores"].items():
        verify_file(output / "m5" / f"manifest_{store}.json", record["manifest_sha256"])
        verify_file(output / "m5" / f"weekly_{store}.csv", record["weekly_sha256"])


def prepare(raw, output):
    from .pipeline.m5_panel import prepare as prepare_m5
    from .pipeline.panel_audit import finalize
    from .pipeline.uci import prepare as prepare_uci
    from .pipeline.uci_panel import prepare_experiment
    from .pipeline.verified_weekly import load_contract

    output = output.resolve()
    workbook = verify_raw(raw.resolve())
    if (output / "phase1_manifest.json").is_file():
        _verify_existing_panels(output, load_contract(output))
        print(f"[final_system] Existing verified weekly panels: {output}", flush=True)
        return
    uci_dir, m5_dir = output / "uci", output / "m5"
    print("[final_system] Preparing UCI Online Retail 352", flush=True)
    prepare_uci(workbook, uci_dir, outlier_application="train_only")
    prepare_experiment(uci_dir / "panel", uci_dir / "weekly.csv", uci_dir / "selected_products.csv")
    prepare_m5(raw.resolve() / "m5", m5_dir)
    finalize(uci_dir, m5_dir, raw.resolve() / "m5", output)
    load_contract(output)
    print(f"[final_system] Verified weekly panels: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("download", help="Download and verify UCI and M5 raw data")
    panels = sub.add_parser("prepare", help="Build and verify the two weekly datasets")
    for command in (fetch, panels):
        command.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    panels.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    args = parser.parse_args()
    if args.action == "download":
        download_sources(args.raw_root)
    else:
        prepare(args.raw_root, args.prepared_root)


if __name__ == "__main__":
    main()
